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

#include "mixtral.h"

//#define MIXTRAL_DEBUG
#ifdef MIXTRAL_DEBUG
#define dbg_printf(...) printf(__VA_ARGS__)
#else
#define dbg_printf(...)
#endif

namespace FlexFlow {

using namespace Legion;
using json = nlohmann::json;

void MIXTRAL::create_mixtral_model(FFModel &ff,
                                   std::string const &model_config_file_path,
                                   std::string const &weight_file_path,
                                   InferenceMode mode,
                                   GenerationConfig generation_config,
                                   bool use_full_precision) {

  MixtralConfig mixtral_config(model_config_file_path);
  mixtral_config.print();

  if (ff.config.tensor_parallelism_degree > mixtral_config.num_attention_heads ||
      mixtral_config.num_attention_heads % ff.config.tensor_parallelism_degree !=
          0) {
    assert(false && "The number of attention heads is smaller, or it is not "
                    "divisible by the tensor parallelism degree");
  }

  std::unordered_map<std::string, Layer *> weights_layers;

  Tensor input;
  {
    int const token_dims[] = {
        (mode == TREE_VERIFY_MODE || mode == BEAM_SEARCH_MODE)
            ? BatchConfig::max_verify_tokens_per_batch()
            : BatchConfig::max_tokens_per_batch(),
        1};
    input = ff.create_tensor<2>(token_dims, DT_INT32);
  }

  Initializer *embed_init = new UniformInitializer(std::rand(), 0, 0);

  Tensor token = ff.embedding(input,
                              mixtral_config.vocab_size,
                              mixtral_config.hidden_size,
                              AGGR_MODE_NONE,
                              use_full_precision ? DT_FLOAT : DT_HALF,
                              NULL,
                              embed_init,
                              "embed_tokens");

  Tensor mlp_out = nullptr;

  for (int i = 0; i < mixtral_config.num_hidden_layers; i++) {
    dbg_printf("mixtral hidden layer %d\n", i);

    // set transformer layer id
    ff.set_transformer_layer_id(i);

    // step 1: attention
    Tensor att_norm = nullptr;
    Tensor token_att_norm[2] = {nullptr, nullptr};
    if (i == 0) {
      att_norm = ff.rms_norm(
          token,
          mixtral_config.rms_norm_eps,
          mixtral_config.hidden_size,
          DT_NONE,
          std::string("layers." + std::to_string(i) + ".input_layernorm")
              .c_str());
    } else {
      ff.residual_rms_norm(
          token,
          mlp_out,
          token_att_norm,
          mixtral_config.rms_norm_eps,
          mixtral_config.hidden_size,
          false, // inplace_residual
          DT_NONE,
          std::string("layers." + std::to_string(i) + ".input_layernorm")
              .c_str());
      token = token_att_norm[0];
      att_norm = token_att_norm[1];
    }
    Tensor qkv_proj = ff.dense(
          att_norm,
          mixtral_config.hidden_size *
              3, // q, k, v. need to change if want to remove replication.
                 // (q_heads + 2 * kv_heads) * proj_size
          AC_MODE_NONE,
          false,         // seems like llama does not use bias
          DT_NONE,       // what is this
          nullptr,       // ?
          nullptr,       // ?
          nullptr,       // ?
          REG_MODE_NONE, // no regularization
          0.0f,          // no dropout
          std::string("layers." + std::to_string(i) + ".self_attn.qkv_proj")
              .c_str());

    Tensor mha;
    switch (mode) {
      case INC_DECODING_MODE: {
        mha = ff.inc_multiquery_self_attention(
            qkv_proj,
            mixtral_config.hidden_size,
            mixtral_config.num_attention_heads,
            mixtral_config.num_key_value_heads,
            mixtral_config.hidden_size / mixtral_config.num_attention_heads,
            mixtral_config.hidden_size / mixtral_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            nullptr, /*kernel_initializer*/
            mixtral_config.rotary_embedding_meta,
            false, /*scaling query*/
            1.0f,  /*scaling factor*/
            true,  /*qk_prod_scaling*/
            false, /*position_bias*/
            std::string("layers." + std::to_string(i) + ".self_attn")
                .c_str() /*name*/
        );
        break;
      }
      default: {
        assert(false);
      }
    }

    Tensor mha_input = mha;
    mha = ff.dense(
        mha_input,
        mixtral_config.hidden_size,
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

    // step 2: SILU activaion
    Tensor token_ff_norm[2] = {nullptr, nullptr};
    ff.residual_rms_norm(
        token,
        mha,
        token_ff_norm,
        mixtral_config.rms_norm_eps,
        mixtral_config.hidden_size,
        false, // inplace_residual
        DT_NONE,
        std::string("layers." + std::to_string(i) + ".post_attention_layernorm")
            .c_str());
    token = token_ff_norm[0];
    Tensor ff_norm = token_ff_norm[1];

    // MoE
    Tensor gate = ff.dense(
        ff_norm, // (hidden_size, 1, 128)
        mixtral_config.num_local_experts,
        AC_MODE_NONE,
        false,
        DT_NONE,
        nullptr,
        nullptr,
        nullptr,
        REG_MODE_NONE,
        0.0f,
        std::string("layers." + std::to_string(i) + ".block_sparse_moe_gate")
            .c_str());

    // gate = ff.softmax( // This operation fails!
    //     gate, // (num_experts, 1, 128)
    //     0,
    //     DT_NONE,
    //     std::string("layers." + std::to_string(i) + ".block_sparse_moe_softmax")
    //         .c_str());

    Tensor topk_out[2] = {nullptr, nullptr};
    ff.top_k(
        gate, // (num_experts, 1, 128)
        topk_out,
        mixtral_config.num_experts_per_tok,
        false,
        std::string("layers." + std::to_string(i) + ".block_sparse_moe_topk")
            .c_str());
    Tensor topk_values = topk_out[0]; // (experts_per_tok, 1, 128) (confirmed 3 dims)
    Tensor topk_indices = topk_out[1]; // (experts_per_tok, 1, 128) (confirmed 3 dims)

    Tensor grouped_tokens[mixtral_config.num_local_experts] = {nullptr};
    ff.group_by(
        ff_norm,
        topk_indices,
        grouped_tokens,
        mixtral_config.num_local_experts,
        0.0f,
        std::string("layers." + std::to_string(i) + ".block_sparse_moe_groupby")
            .c_str());

    // grouped_tokens[0] has dims (1024, 1, 0)
    Tensor aggregate_inputs[4 + mixtral_config.num_local_experts] = {nullptr};
    for (int expert_idx = 0; expert_idx < mixtral_config.num_local_experts; expert_idx++) {
      Tensor w1 = ff.dense(grouped_tokens[expert_idx],
                           mixtral_config.intermediate_size,
                           AC_MODE_NONE,
                           false,
                           DT_NONE,
                           nullptr,
                           nullptr,
                           nullptr,
                           REG_MODE_NONE,
                           0.0f,
                           std::string("layers." + std::to_string(i) +
                                       ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w1")
                               .c_str());

      Tensor w3 = ff.dense(grouped_tokens[expert_idx],
                           mixtral_config.intermediate_size,
                           AC_MODE_NONE,
                           false,
                           DT_NONE,
                           nullptr,
                           nullptr,
                           nullptr,
                           REG_MODE_NONE,
                           0.0f,
                           std::string("layers." + std::to_string(i) +
                                       ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w3")
                               .c_str());

      Tensor multi =
          ff.sigmoid_silu_multi(w1,
                                w3,
                                DT_NONE,
                                std::string("layers." + std::to_string(i) +
                                            ".block_sparse_moe_experts_" +
                                            std::to_string(expert_idx) + "ssm")
                                    .c_str());

      Tensor w2 = ff.dense(multi,
                           mixtral_config.hidden_size,
                           AC_MODE_NONE,
                           false,
                           DT_NONE,
                           nullptr,
                           nullptr,
                           nullptr,
                           REG_MODE_NONE,
                           0.0f,
                           std::string("layers." + std::to_string(i) +
                                       ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w2")
                               .c_str());
      // w2 and aggreagte_inputs[4+] have dims (1024, 1, 0)

      aggregate_inputs[4 + expert_idx] = w2; // w2 has 3 dimensions
    }

    Tensor topk_values_reduced = ff.reduce_sum(topk_values, {0}, true); // (2, 1, 1)
    topk_values = ff.divide(topk_values, topk_values_reduced); // (2, 1, 128)

    aggregate_inputs[0] = topk_values; // (experts_per_tok, 1, 128) (3 dims confirmed)
    aggregate_inputs[1] = topk_indices; // (experts_per_tok, 1, 128) (3 dims confirmed)
    aggregate_inputs[2] = topk_values; // TODO this is a tmp fix
    aggregate_inputs[3] = gate;  // TODO this is a tmp fix
//    aggregate_inputs[2] = aggregate_inputs[3] = nullptr;
    mlp_out = ff.aggregate(aggregate_inputs,
                           mixtral_config.num_local_experts,
                           0.0f,
                           std::string("layers." + std::to_string(i) +
                                       ".block_sparse_moe_experts_aggregate")
                               .c_str());
  }

  // final normalization and linear
  Tensor final_rms_norm_output[2] = {nullptr, nullptr};
  ff.residual_rms_norm(token,
                       mlp_out,
                       final_rms_norm_output,
                       mixtral_config.rms_norm_eps,
                       mixtral_config.hidden_size,
                       false,
                       DT_NONE,
                       "norm");

  Tensor dense = ff.dense(final_rms_norm_output[1],
                          mixtral_config.vocab_size,
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
    Tensor softmax = ff.softmax(dense, -1);
    // output = ff.beam_top_k(softmax, mixtral_config.max_beam_width, false);
    // output = ff.argmax(softmax, /*beam_Search*/ true);
    output = ff.arg_top_k(softmax, mixtral_config.max_beam_width, false, true);
    // output = ff.top_k(softmax, )
  } else {
    // Tensor softmax = ff.softmax(dense, -1);
    if (generation_config.do_sample) {
      dense = ff.scalar_truediv(dense, generation_config.temperature, false);
      Tensor softmax = ff.softmax(dense, -1);
      output = ff.sampling(softmax, generation_config.topp);
    } else {
      // output = ff.arg_top_k(dense, /*k=*/1, false);
      output = ff.argmax(dense, /*beam_Search*/ false);
    }
  }

  FileDataLoader *fileloader = new FileDataLoader(
      "",
      weight_file_path,
      mixtral_config.num_attention_heads,
      mixtral_config.num_key_value_heads,
      mixtral_config.hidden_size,
      mixtral_config.hidden_size / mixtral_config.num_attention_heads,
      ff.config.tensor_parallelism_degree,
      use_full_precision);

  InferenceManager *im = InferenceManager::get_inference_manager();
  im->register_model_weights_loader(&ff, fileloader);
}

}; // namespace FlexFlow
