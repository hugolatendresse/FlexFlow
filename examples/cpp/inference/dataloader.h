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

#include "data_generator.h"
#include "flexflow/batch_config.h"
#include "flexflow/model.h"
#include "inference_config.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Legion;
using namespace FlexFlow;

class DataLoader {
public:
  DataLoader(FFModel &ff,
             InferenceConfig const &inferenceConfig,
             DataGenerator &data_generator,
             std::vector<ParallelTensor> input);
  static void load_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  void next_batch(FFModel &ff,
                  int bid,
                  BatchConfig *bc,
                  std::map<size_t, int> &batch_predictions,
                  MachineView const *mv = nullptr);
  void store_outputs(BatchConfig *bc,
                     InferenceResult const &ir,
                     std::map<size_t, int> &batch_predictions);

public:
  size_t num_samples;
  ParallelTensor full_input;
  std::vector<ParallelTensor> batch_input;
  std::map<size_t, std::vector<int>> outputs;
  struct DataLoaderInput {
    InferenceConfig const &_inferenceConfig;
    DataGenerator &_data_generator;
  };
  struct DataLoaderNextBatchInput {
    BatchConfig::SampleIdxs const &meta;
    std::map<size_t, int> const &prev_batch_preds;
  };
};