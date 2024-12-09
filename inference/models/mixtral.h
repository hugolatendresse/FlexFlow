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
#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/request_manager.h"
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

namespace FlexFlow {

class MIXTRAL {
public:
  struct MixtralConfig {
    MixtralConfig(std::string const &model_config_file_path) {
      std::ifstream config_file(model_config_file_path);
      if (config_file.is_open()) {
        try {
          json model_config;
          config_file >> model_config;

          try {
            max_position_embeddings = model_config.at("max_position_embeddings");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'max_position_embeddings': " << e.what() << std::endl;
            assert(false);
          }

          try {
            num_hidden_layers = model_config.at("num_hidden_layers");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'num_hidden_layers': " << e.what() << std::endl;
            assert(false);
          }

          try {
            vocab_size = model_config.at("vocab_size");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'vocab_size': " << e.what() << std::endl;
            assert(false);
          }

          try {
            num_attention_heads = model_config.at("num_attention_heads");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'num_attention_heads': " << e.what() << std::endl;
            assert(false);
          }

          try {
            if (model_config.find("num_key_value_heads") != model_config.end()) {
              num_key_value_heads = model_config["num_key_value_heads"];
            } else {
              num_key_value_heads = num_attention_heads;
            }
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'num_key_value_heads': " << e.what() << std::endl;
            assert(false);
          }

          try {
            hidden_size = model_config.at("hidden_size");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'hidden_size': " << e.what() << std::endl;
            assert(false);
          }

          try {
            rms_norm_eps = model_config.at("rms_norm_eps");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'rms_norm_eps': " << e.what() << std::endl;
            assert(false);
          }

          try {
            intermediate_size = model_config.at("intermediate_size");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'intermediate_size': " << e.what() << std::endl;
            assert(false);
          }

          try {
            rotary_embedding_meta.apply_rotary_embedding = true;
            if (model_config.find("rope_theta") != model_config.end()) {
              rotary_embedding_meta.rope_theta = model_config["rope_theta"];
            } else {
              rotary_embedding_meta.rope_theta = 10000.0f;
            }
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'rope_theta': " << e.what() << std::endl;
            assert(false);
          }

          try {
            if (model_config.find("scaling_factor") != model_config.end() &&
                !model_config["scaling_factor"].is_null()) {
              rotary_embedding_meta.rope_type = model_config["scaling_factor"]["rope_type"];
              rotary_embedding_meta.factor = model_config["scaling_factor"]["factor"];
              rotary_embedding_meta.low_freq_factor = model_config["scaling_factor"]["low_freq_factor"];
              rotary_embedding_meta.high_freq_factor = model_config["scaling_factor"]["high_freq_factor"];
              rotary_embedding_meta.original_max_position_embeddings =
                  model_config["scaling_factor"]["original_max_position_embeddings"];
            }
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'scaling_factor': " << e.what() << std::endl;
            assert(false);
          }

          try {
            num_experts_per_tok = model_config.at("num_experts_per_tok");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'num_experts_per_tok': " << e.what() << std::endl;
            assert(false);
          }

          try {
            num_local_experts = model_config.at("num_local_experts");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'num_local_experts': " << e.what() << std::endl;
            assert(false);
          }

          try {
            output_router_logits = model_config.at("output_router_logits");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'output_router_logits': " << e.what() << std::endl;
            assert(false);
          }

          try {
            router_aux_loss_coef = model_config.at("router_aux_loss_coef");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'router_aux_loss_coef': " << e.what() << std::endl;
            assert(false);
          }

//          try { // TODO sliding_window is null in the JSON config file of mistralai/Mixtral-8x7B-v0.1
//            sliding_window = model_config.at("sliding_window");
//          } catch (json::exception const &e) {
//            std::cerr << "Error reading 'sliding_window': " << e.what() << std::endl;
//            assert(false);
//          }

          try {
            tie_word_embeddings = model_config.at("tie_word_embeddings");
          } catch (json::exception const &e) {
            std::cerr << "Error reading 'tie_word_embeddings': " << e.what() << std::endl;
            assert(false);
          }

        } catch (json::exception const &e) {
          std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
          assert(false);
        }
      } else {
        std::cerr << "Error opening JSON file " << model_config_file_path
                  << std::endl;
        assert(false);
      }
      max_beam_width = BeamSearchBatchConfig::MAX_BEAM_WIDTH;
      max_beam_depth = BeamSearchBatchConfig::MAX_BEAM_DEPTH;
    }

    void print() const {
      std::cout << "Mixtral Config:" << std::endl;
      std::cout << "\thidden_size: " << hidden_size << std::endl;
      std::cout << "\tintermediate_size: " << intermediate_size << std::endl;
      std::cout << "\tmax_position_embeddings: " << max_position_embeddings
                << std::endl;
      std::cout << "\tnum_attention_heads: " << num_attention_heads
                << std::endl;
      std::cout << "\tnum_key_value_heads: " << num_key_value_heads
                << std::endl;
      std::cout << "\tnum_experts_per_tok: " << num_experts_per_tok
                << std::endl;
      std::cout << "\tnum_local_experts: " << num_local_experts << std::endl;
      std::cout << "\tnum_hidden_layers: " << num_hidden_layers << std::endl;
      std::cout << "\toutput_router_logits: " << output_router_logits
                << std::endl;
      std::cout << "\trms_norm_eps: " << rms_norm_eps << std::endl;
      std::cout << "\trope_theta: " << rope_theta << std::endl;
      std::cout << "\trouter_aux_loss_coef: " << router_aux_loss_coef
                << std::endl;
//      std::cout << "\tsliding_window: " << sliding_window << std::endl; // TODO
      std::cout << "\ttie_word_embeddings: " << tie_word_embeddings
                << std::endl;
      std::cout << "\tvocab_size: " << vocab_size << std::endl;
      std::cout << "\tmax_beam_width: " << max_beam_width << std::endl;
      std::cout << "\tmax_beam_depth: " << max_beam_depth << std::endl;
    }
    int hidden_size, intermediate_size;
    int max_position_embeddings;
    int num_attention_heads, num_key_value_heads;
    int num_experts_per_tok, num_local_experts;
    int num_hidden_layers;
    bool output_router_logits;
    float rms_norm_eps;
    float rope_theta;
    float router_aux_loss_coef;
//    int sliding_window; // TODO
    bool tie_word_embeddings;
    int vocab_size;
    int max_beam_width, max_beam_depth;
    RotaryEmbeddingMeta rotary_embedding_meta;
  };

  static void create_mixtral_model(FFModel &ff,
                                   std::string const &model_config_file_path,
                                   std::string const &weight_file_path,
                                   InferenceMode mode,
                                   GenerationConfig generation_config,
                                   bool use_full_precision = false);
};

}; // namespace FlexFlow
