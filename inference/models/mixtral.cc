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
  // token has dimensions (hidden_size, 1, 128)

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
      printf("before rms norm in layer %d token has %d dims\n",i, token->num_dims);
      printf("before rms norm in layer %d mlp_out has %d dims\n",i, token->num_dims);
      printf("before rms norm in layer %d token dims are %d %d %d %d\n",i, token->dims[0], token->dims[1], token->dims[2], token->dims[3]);
      printf("before rms norm in layer %d, mlp_out dims are %d %d %d %d\n",i, mlp_out->dims[0], mlp_out->dims[1], mlp_out->dims[2], mlp_out->dims[3]);
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
    // token has dimensions (hidden_size, 1, 128)



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
      case BEAM_SEARCH_MODE: {
        mha = ff.spec_inc_multiquery_self_attention(
            qkv_proj,
            mixtral_config.hidden_size,
            mixtral_config.num_attention_heads,
            mixtral_config.num_key_value_heads,
            mixtral_config.hidden_size / mixtral_config.num_attention_heads,
            mixtral_config.hidden_size / mixtral_config.num_attention_heads,
            0.0f,    /*dropout*/
            false,   /*add_zero_attn*/
            DT_NONE, /*data_type*/
            NULL,    /*kernel_initializer*/
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
      case TREE_VERIFY_MODE: {
        mha = ff.inc_multiquery_self_attention_verify(
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
    token = token_ff_norm[0];   // token has dimensions (hidden_size, 1, 128)
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

    gate = ff.softmax( // TODO This sfotmax is wrong! not taking across last dim, which is not supported by ff!
        gate, // (num_experts, 1, 128)
        0,
        DT_NONE,
        std::string("layers." + std::to_string(i) + ".block_sparse_moe_softmax")
            .c_str());

    Tensor topk_out[2] = {nullptr, nullptr};
    printf("gate data_type %d\n", gate->data_type);
    ff.top_k(
        gate, // (num_experts, 1, 128)
        topk_out,
        mixtral_config.num_experts_per_tok,
        false,
        std::string("layers." + std::to_string(i) + ".block_sparse_moe_topk").c_str()
        );
    Tensor topk_values = topk_out[0]; // (experts_per_tok, 1, 128) (confirmed 3 dims)
    Tensor topk_indices = topk_out[1]; // (experts_per_tok, 1, 128) (confirmed 3 dims)

//    TODO understand why graph.cc complains that last layer has multiple inputs
      Tensor grouped_tokens[mixtral_config.num_local_experts] = {nullptr};
      ff.group_by(
          ff_norm,
          topk_indices,
          grouped_tokens,
          mixtral_config.num_local_experts,
          0.0f,
          std::string("layers." + std::to_string(i) + ".block_sparse_moe_groupby").c_str());

    // grouped_tokens[0] has dims (1024, 1, 0)

// TODO all experts
//    Tensor aggregate_inputs[4 + mixtral_config.num_local_experts] = {nullptr};
    Tensor aggregate_inputs[4 + 1] = {nullptr};
    Tensor one_aggregate_inputs[1] = {nullptr};

    // TODO don't use only one expert
//      for (int expert_idx = 0; expert_idx < mixtral_config.num_local_experts; expert_idx++) {
    for (int expert_idx = 1; expert_idx < 2; expert_idx++) {
	Tensor w1 = ff.dense(
        			   ff_norm,
                       mixtral_config.intermediate_size,
                       AC_MODE_NONE,
                       false,
                       DT_NONE,
                       nullptr,
                       nullptr,
                       nullptr,
                       REG_MODE_NONE,
                       0.0f,
                       std::string("layers." + std::to_string(i) + ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w1").c_str());

  	Tensor w3 = ff.dense(
            		   ff_norm,
                       mixtral_config.intermediate_size,
                       AC_MODE_NONE,
                       false,
                       DT_NONE,
                       nullptr,
                       nullptr,
                       nullptr,
                       REG_MODE_NONE,
                       0.0f,
                       std::string("layers." + std::to_string(i) + ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w3").c_str());

  Tensor multi = ff.sigmoid_silu_multi(w1, w3); //DT_NONE,std::string("layers." + std::to_string(i) +".block_sparse_moe_experts." +std::to_string(expert_idx) + "ssm").c_str());

  Tensor w2 = ff.dense( // output has dims (1024, 1, 0), 3 dims confirmed
      				   multi,
                       mixtral_config.hidden_size,
                       AC_MODE_NONE,
                       false,
                       DT_NONE,
                       nullptr,
                       nullptr,
                       nullptr,
                       REG_MODE_NONE,
                       0.0f,
                       std::string("layers." + std::to_string(i) + ".block_sparse_moe_experts_" +
                                       std::to_string(expert_idx) + "_w2").c_str());
    aggregate_inputs[4 + expert_idx-1] = w2; // (1024, 1, 0), 3 dims confirmed
    }

    // TODO those two lines are techincally nice-to-haves!! skip for now, but it fails if we uncomment
//       Tensor topk_values_reduced = ff.reduce_sum(topk_values, {0}, true); // (2, 1, 1)
//    topk_values = ff.divide(topk_values, topk_values_reduced); // (2, 1, 128)


//    Tensor dummy_gate = ff.dense( // TODO try uncommenting the whole block
//        ff_norm,
//        mixtral_config.num_local_experts,
//        AC_MODE_NONE,
//        false,
//        DT_NONE,
//        nullptr,
//        nullptr,
//        nullptr,
//        REG_MODE_NONE,
//        0.0f,
//        std::string("layers." + std::to_string(i) + ".block_sparse_moe_gate")
//            .c_str());
//    dummy_gate = ff.softmax(
//        gate,
//        0,
//        DT_NONE,
//        std::string("dummy_gate").c_str());
//
        aggregate_inputs[0] = topk_values; // (experts_per_tok, 1, 128) (3 dims confirmed)
    aggregate_inputs[1] = topk_indices; // (experts_per_tok, 1, 128) (3 dims confirmed)
    aggregate_inputs[2] = topk_values; // TODO this is a tmp fix
    aggregate_inputs[3] = gate;  // TODO this is a tmp fix TODO decide vs dummygate

//        mlp_out = aggregate_inputs[5]; // TODO don't use just one expert
    mlp_out = ff.aggregate(aggregate_inputs,
//                           topk_values->dims[2],
//                           mixtral_config.num_local_experts, // TODO don't use just one expert
                           1,
                           0.0f,
                           std::string("layers." + std::to_string(i) +
                                       ".block_sparse_moe_experts_aggregate")
                               .c_str());

  // mlp_out has dimensions (hidden_size, 1, 128)
  printf("mlp_out in layer %d dims are %d %d %d %d\n",i, mlp_out->dims[0], mlp_out->dims[1], mlp_out->dims[2], mlp_out->dims[3]);
  assert(mlp_out->dims[0] == mixtral_config.hidden_size && "mlp_out dims[0] != hidden_size");
  assert(mlp_out->dims[1] == 1 && "mlp_out dims[1] != 1");
//  assert(mlp_out->dims[2] == 128 && "mlp_out dims[2] != 128"); // TODO update with new seq len
  printf("seq length is now %d\n", mlp_out->dims[2]);

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
    // Tensor softmax = ff.softmax(dense, -1);
    if (generation_config.do_sample) {
      dense = ff.scalar_truediv(dense, generation_config.temperature, false);
      Tensor softmax = ff.softmax(dense, -1);
      output = ff.sampling(softmax, generation_config.topp);
    } else {
      Tensor softmax = ff.softmax(dense, -1); // TODO added that to copy llama, see if needed in HF transformers impl.
      output = ff.argmax(softmax, /*beam_Search*/ false);
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