/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opt.h"

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void OPT::create_opt_model(FFModel &ff,
                           std::string const &model_config_file_path,
                           std::string const &weight_file_path,
                           InferenceMode mode,
                           bool use_full_precision) {
  OPTConfig opt_config(model_config_file_path);
  opt_config.print();

  if (ff.config.tensor_parallelism_degree > opt_config.num_attention_heads ||
      opt_config.num_attention_heads % ff.config.tensor_parallelism_degree !=
          0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

  //------------------------------ build the model --------------------------
  Tensor input;
  Tensor position_input;
  ff.set_position_offset(2);
  {
    int const token_dims[] = {
        (mode == TREE_VERIFY_MODE || mode == BEAM_SEARCH_MODE)
            ? BatchConfig::max_verify_tokens_per_batch()
            : BatchConfig::max_tokens_per_batch(),
        1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
    position_input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);
  std::vector<int> axes = {0};

  Tensor token = ff.embedding(input,
                              opt_config.vocab_size,
                              opt_config.word_embed_proj_dim,
                              AGGR_MODE_NONE,
                              use_full_precision ? DT_FLOAT : DT_HALF,
                              NULL,
                              embed_init,
                              "embed_tokens");

  Tensor positional_embedding =
      ff.embedding(position_input,
                   opt_config.max_position_embeddings,
                   opt_config.hidden_size,
                   AGGR_MODE_NONE,
                   use_full_precision ? DT_FLOAT : DT_HALF,
                   NULL,
                   embed_init,
                   "embed_positions");

  Tensor fc2 = nullptr, added = nullptr;
  Tensor res_ln_outputs[2] = {nullptr, nullptr};

  for (int i = 0; i < opt_config.num_hidden_layers; i++) {
    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // 125m, 1.7B, ..., 175B applies layer norm BEFORE attention,
    // 350m applies layer norm AFTER attention
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#LL324C1-L325C1
    // this version is before normalization
    ff.residual_layer_norm(
        (i == 0) ? token : added,
        (i == 0) ? positional_embedding : fc2,
        nullptr,
        res_ln_outputs,
        false,
        axes,
        opt_config.layer_norm_elementwise_affine,
        1e-05,
        true,
        false,
        DT_NONE,
        std::string("layers." + std::to_string(i) + ".self_attn_layer_norm")
            .c_str());
    Tensor residual = res_ln_outputs[0];
    Tensor hidden_states = res_ln_outputs[1];

    Tensor qkv_proj = ff.dense(
        hidden_states,
        opt_config.hidden_size *
            3, // q, k, v. need to change if want to remove replication.
               // (q_heads + 2 * kv_heads) * proj_size
        AC_MODE_NONE,
        true,          // seems like it does not use bias
        DT_NONE,       // what is this
        nullptr,       // ?
        nullptr,       // ?
        nullptr,       // ?
        REG_MODE_NONE, // no regularization
        0.0f,          // no dropout
        std::string("layers." + std::to_string(i) + ".self_attn.qkv_proj")
            .c_str());

    Tensor o_proj;
    switch (mode) {
      case BEAM_SEARCH_MODE: {
        o_proj = ff.spec_inc_multihead_self_attention(
            qkv_proj,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
            opt_config.rotary_embedding_meta,
            true, /*scaling query*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5), /*scaling factor*/
            false,     /*qk_prod_scaling*/
            false,     /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attn")
                .c_str() /*name*/
        );
        break;
      }
      case TREE_VERIFY_MODE: {
        o_proj = ff.inc_multihead_self_attention_verify(
            qkv_proj,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
            opt_config.rotary_embedding_meta,
            true, /*scaling query*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5), /*scaling factor*/
            false,     /*qk_prod_scaling*/
            false,     /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attn")
                .c_str() /*name*/
        );
        break;
      }
      case INC_DECODING_MODE: {
        o_proj = ff.inc_multihead_self_attention(
            qkv_proj,
            opt_config.hidden_size,
            opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            opt_config.hidden_size / opt_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
            opt_config.rotary_embedding_meta,
            true, /*scaling query*/
            pow((opt_config.hidden_size / opt_config.num_attention_heads),
                -0.5), /*scaling factor*/
            false,     /*qk_prod_scaling*/
            false,     /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attn")
                .c_str() /*name*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }

    Tensor mha = ff.dense(
        o_proj,
        opt_config.hidden_size,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers." + std::to_string(i) + ".self_attn.o_proj")
            .c_str());

    ff.add_bias_residual_layer_norm(mha,
                                    residual,
                                    res_ln_outputs,
                                    axes,
                                    opt_config.layer_norm_elementwise_affine,
                                    1e-05,
                                    true,
                                    false,
                                    DT_NONE,
                                    std::string("layers." + std::to_string(i) +
                                                ".add_bias_residual_layer_norm")
                                        .c_str());
    added = res_ln_outputs[0];
    Tensor final_norm = res_ln_outputs[1];

    //--------linear fc1 fc2 ----------
    Tensor fc1 =
        ff.dense(final_norm,
                 opt_config.ffn_dim,
                 AC_MODE_RELU,
                 true,
                 DT_NONE,
                 nullptr,
                 nullptr,
                 nullptr,
                 REG_MODE_NONE,
                 0.0f,
                 std::string("layers." + std::to_string(i) + ".fc1").c_str());
    fc2 = ff.dense(fc1,
                   opt_config.hidden_size,
                   AC_MODE_NONE,
                   true,
                   DT_NONE,
                   nullptr,
                   nullptr,
                   nullptr,
                   REG_MODE_NONE,
                   0.0f,
                   std::string("layers." + std::to_string(i) + ".fc2").c_str());
    // Low-Rank Adapter (LoRA) for the second linear layer
    // ff.lora_linear(std::string("fc2"), std::string("layers." +
    // std::to_string(i) + ".fc2.lora").c_str());
  }

  // final
  ff.residual_layer_norm(added,
                         fc2,
                         nullptr,
                         res_ln_outputs,
                         false,
                         axes,
                         opt_config.layer_norm_elementwise_affine,
                         1e-05,
                         true,
                         false,
                         DT_NONE,
                         "final_layer_norm");
  Tensor all_final_norm = res_ln_outputs[1];

  Tensor lm_head = ff.dense(all_final_norm,
                            opt_config.vocab_size,
                            AC_MODE_NONE,
                            false,
                            DT_NONE,
                            nullptr,
                            nullptr,
                            nullptr,
                            REG_MODE_NONE,
                            0.0f,
                            "lm_head");

  Tensor output;
  if (mode == BEAM_SEARCH_MODE) {
    Tensor softmax = ff.softmax(lm_head, -1);
    // output = ff.beam_top_k(softmax, opt_config.max_beam_width, false);
    output = ff.argmax(softmax, /*beam_Search*/ true);
  } else {
    // output = ff.arg_top_k(lm_head, /*k=*/1, false);
    Tensor softmax = ff.softmax(lm_head, -1);
    output = ff.argmax(softmax, /*beam_Search*/ false);
  }

  // If PEFT is enabled, add LoRA layers
  if (ff.config.enable_peft) {
    // todo: add attention projections
    std::vector<std::string> target_modules = {"fc1", "fc2"};
    ff.add_lora_layers(target_modules);
  }

  FileDataLoader *fileloader = new FileDataLoader(
      "",
      weight_file_path,
      opt_config.num_attention_heads,
      opt_config.num_attention_heads,
      opt_config.hidden_size,
      opt_config.hidden_size / opt_config.num_attention_heads,
      ff.config.tensor_parallelism_degree,
      use_full_precision);
  InferenceManager *im = InferenceManager::get_inference_manager();
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
