# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flexflow.core import *
from .base import FlexFlowModel
import random, shutil


class FalconConfig:
    def __init__(self, hf_config):
        self.max_seq_len = 256
        self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.bias = hf_config.bias
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.multi_query = hf_config.multi_query
        self.n_head = hf_config.n_head
        self.n_layer = hf_config.n_layer
        self.parallel_attn = hf_config.parallel_attn
        self.vocab_size = hf_config.vocab_size


class FlexFlowFalcon(FlexFlowModel):
    def __init__(
        self,
        mode,
        sampling_config,
        ffconfig,
        hf_config,
        data_type,
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        weights_filepath="",
        tokenizer_filepath="",
    ):
        self.mode = mode
        self.sampling_config = sampling_config
        self.ffconfig = ffconfig
        self.max_batch_size = max_batch_size
        self.data_type = data_type
        self.falcon_config = FalconConfig(hf_config)
        self.falcon_config.max_seq_length = max_seq_length
        self.falcon_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1

        self.build_model()

    def build_model(self):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [self.falcon_config.max_num_tokens, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = ffmodel.embedding(
            input_tensor,
            self.falcon_config.vocab_size,
            self.falcon_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="word_embeddings_weight",
        )
        axes = [
            0,
        ]

        for i in range(self.falcon_config.n_layer):
            ffmodel.set_transformer_layer_id(i)

            att_norm = ffmodel.layer_norm(
                token,
                axes,
                True,
                self.falcon_config.layer_norm_epsilon,
                name=f"layers_{i}_input_layernorm_weight",
            )

            if self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multihead_self_attention(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    1,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    0.0,  # dropout
                    False,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    name=f"layers_{i}_self_attention_dense_weight",
                )
            else:
                assert False

            dense_h_to_4h = ffmodel.dense(
                att_norm,
                self.falcon_config.hidden_size * 4,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_mlp_dense_h_to_4h_weight",
            )
            dense_h_to_4h = ffmodel.gelu(dense_h_to_4h)
            mlp_output = ffmodel.dense(
                dense_h_to_4h,
                self.falcon_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_mlp_dense_4h_to_h_weight",
            )

            token = ffmodel.add(token, mha)
            token = ffmodel.add(token, mlp_output)

        ln_f = ffmodel.layer_norm(
            token, axes, True, self.falcon_config.layer_norm_epsilon, name="ln_f_weight"
        )
        lm_head = ffmodel.dense(
            ln_f,
            self.falcon_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head_weight",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(lm_head, -1)
            # output = ffmodel.beam_top_k(softmax, self.falcon_config.max_beam_width, False)
            output = ffmodel.argmax(softmax, True)
        else:
            if self.sampling_config.do_sample:
                dense = ffmodel.scalar_true_divide(
                    lm_head, self.sampling_config.temperature, False
                )
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.sampling(softmax, self.sampling_config.topp)
            else:
                # output = ffmodel.arg_top_k(lm_head, 1, False)
                output = ffmodel.argmax(lm_head, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("transformer_h_", "layers_")
                .replace("transformer_", "")
            )
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
        # copy embedding weights
        shutil.copy(
            os.path.join(dst_folder, "word_embeddings_weight"),
            os.path.join(dst_folder, "lm_head_weight"),
        )

    def get_layers_with_weights(self):
        layer_names = [
            "word_embeddings_weight",
            "ln_f_weight",
            "lm_head_weight",
        ] + [
            expr
            for i in range(self.falcon_config.n_layer)
            for expr in (
                f"layers_{i}_input_layernorm_weight",
                f"layers_{i}_self_attention_dense_weight",
                f"layers_{i}_mlp_dense_h_to_4h_weight",
                f"layers_{i}_mlp_dense_4h_to_h_weight",
            )
        ]
        layers_with_weights = {
            layer_name: self.ffmodel.get_layer_by_name(layer_name)
            for layer_name in layer_names
        }

        return layers_with_weights