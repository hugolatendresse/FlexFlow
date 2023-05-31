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
#include <string>
// #define MAX_SEQ_LEN 1024
static int const MAX_SEQ_LEN = FlexFlow::BatchConfig::MAX_SEQ_LENGTH;
#define BATCH_SIZE 16
#define MNIST_DIMS 28 * 28
#define DATA_DIM MNIST_DIMS
// #define DATA_DIM 3

struct InferenceConfig {
  InferenceConfig(void) {
    //----------------------- Input/output data ------------------------
    token_dim = DATA_DIM;
    sequence_length = MAX_SEQ_LEN;
    batch_size = BATCH_SIZE;
    out_dim = DATA_DIM;
    num_labels = out_dim;
    num_layers = 12;

    vocab_size = 50257;
    block_size = 1024;

    //----------------------- Inference parameters ---------------------
    // total number of requests processed as part of the simulation
    total_requests = 2560;
    poisson_distribution = true;
    // average number of request arrivals per second
    arrival_rate = 250;
    num_inflight_batches = 4;
    incremental_mode = true;
    //----------------------- Rest of model parameters ------------------
    hidden_size = DATA_DIM;
    // Encoder layer
    num_attention_heads = 16;
    attention_kdim = attention_vdim = hidden_size / num_attention_heads;
    num_encoder_layers = 12;
  }

  // Input/output data
  int token_dim;
  int sequence_length;
  int batch_size;
  int out_dim;
  int num_labels;
  int num_layers;

  int vocab_size;
  int block_size;

  std::string dataset_path;
  // Inference parameters
  int total_requests;
  bool poisson_distribution;
  double arrival_rate;
  int num_inflight_batches;
  bool incremental_mode;
  // Model parameters
  int hidden_size;
  int num_attention_heads;
  int attention_kdim;
  int attention_vdim;
  int num_encoder_layers;
};