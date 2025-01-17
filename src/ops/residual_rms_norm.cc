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

#include "flexflow/ops/residual_rms_norm.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/residual_rms_norm_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::ResidualRMSNorm;

bool operator==(ResidualRMSNormParams const &lhs,
                ResidualRMSNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.eps == rhs.eps &&
         lhs.dim == rhs.dim && lhs.inplace_residual == rhs.inplace_residual;
}

bool ResidualRMSNormParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  return input.first.is_valid() && input.second.is_valid();
}

ResidualRMSNormParams ResidualRMSNorm::get_params() const {
  ResidualRMSNormParams params;
  params.layer_guid = this->layer_guid;
  params.eps = this->eps;
  params.dim = this->dim;
  params.inplace_residual = this->inplace_residual;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

void FFModel::residual_rms_norm(const Tensor input1,
                                const Tensor input2,
                                Tensor *outputs,
                                float eps,
                                int dim,
                                bool inplace_residual,
                                DataType data_type,
                                char const *name) {
//  printf("input2 address: %p\n", input2);

  if (data_type == DT_NONE) {
    data_type = input1->data_type;
  }
  Tensor casted_input1 =
      (data_type != input1->data_type)
          ? cast(input1, data_type, "type cast for residual_rms_norm")
          : input1;
  Tensor casted_input2 =
      (data_type != input2->data_type)
          ? cast(input2, data_type, "type cast for residual_rms_norm")
          : input2;
//  printf("RRMSN %s: input1 dims: %d, input2 dims: %d\n",name, input1->num_dims, input2->num_dims);
//  printf("RRMSN %s: input1 dims[0] size: %d, input2 dims[0] size: %d\n",name, input1->dims[0], input2->dims[0]);
//  printf("RRMSN %s: input1 dims[1] size: %d, input2 dims[1] size: %d\n",name, input1->dims[1], input2->dims[1]);
//  printf("RRMSN %s: input1 dims[2] size: %d, input2 dims[2] size: %d\n",name, input1->dims[2], input2->dims[2]);
  Layer *rm = new Layer(this,
                        OP_RESIDUAL_RMS_NORM,
                        data_type,
                        name,
                        2 /*inputs*/,
                        1 /*weights*/,
                        2 /*outputs*/,
                        casted_input1,
                        casted_input2);

  rm->outputs[0] = create_tensor_legion_ordering(
      input1->num_dims, input1->dims, data_type, rm, 0, true /*create_grad*/);
  rm->outputs[1] = create_tensor_legion_ordering(
      input1->num_dims, input1->dims, data_type, rm, 1, true /*create_grad*/);

  // weights
  int weight_dims[1] = {dim};
  rm->weights[0] = create_weight_legion_ordering(1,
                                                 weight_dims,
                                                 data_type,
                                                 rm,
                                                 false /*create_grad*/,
                                                 nullptr,
                                                 CHOSEN_SYNC_TYPE);

  rm->add_float_property("eps", eps);
  rm->add_int_property("dim", dim);
  rm->add_int_property("inplace_residual", inplace_residual);
  layers.push_back(rm);
  outputs[0] = rm->outputs[0];
  outputs[1] = rm->outputs[1];
}

Op *ResidualRMSNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float eps;
  layer->get_float_property("eps", eps);
  long long value;
  layer->get_int_property("dim", value);
  int dim = value;
  layer->get_int_property("inplace_residual", value);
  bool inplace_residual = (bool)value;

  return new ResidualRMSNorm(model,
                             layer->layer_guid,
                             inputs[0],
                             inputs[1],
                             eps,
                             dim,
                             inplace_residual,
                             false,
                             layer->name);
}

ResidualRMSNorm::ResidualRMSNorm(
    FFModel &model,
    ResidualRMSNormParams const &params,
    std::pair<ParallelTensor, ParallelTensor> const &inputs,
    bool allocate_weights = false,
    char const *name)
    : ResidualRMSNorm(model,
                      params.layer_guid,
                      inputs.first,
                      inputs.second,
                      params.eps,
                      params.dim,
                      params.inplace_residual,
                      allocate_weights,
                      params.name) {}

ResidualRMSNorm::ResidualRMSNorm(
    FFModel &model,
    ResidualRMSNorm const &other,
    std::pair<ParallelTensor, ParallelTensor> const &inputs,
    bool allocate_weights)
    : ResidualRMSNorm(model,
                      other.layer_guid,
                      inputs.first,
                      inputs.second,
                      other.eps,
                      other.dim,
                      other.inplace_residual,
                      allocate_weights,
                      other.name) {}
ResidualRMSNorm::ResidualRMSNorm(FFModel &model,
                                 LayerID const &_layer_guid,
                                 const ParallelTensor _input1,
                                 const ParallelTensor _input2,
                                 float _eps,
                                 int dim,
                                 bool _inplace_residual,
                                 bool allocate_weights,
                                 char const *name)
    : Op(model,
         OP_RESIDUAL_RMS_NORM,
         _input1->data_type,
         name,
         2 /*num of inputs tensor */,
         1 /*num of weights tensor */,
         2 /*num of outputs tensor */,
         _input1,
         _input2) {
  eps = _eps;
  inplace_residual = _inplace_residual;
  inputs[0] = _input1;
  inputs[1] = _input2;
  layer_guid = _layer_guid;
  int num_dims = _input1->num_dims;
  this->dim = dim;
  data_dim = _input1->dims[0].size;
  effective_batch_size = 1;
  for (int i = 1; i <= num_dims - 2; i++) {
    effective_batch_size *= _input1->dims[i].size;
  }
  // Currently assert that all non-replica dims are not parallelized
  // We only support parallelism along the replica dim now
  for (int i = 0; i < _input1->num_dims - 1; i++) {
    assert(_input1->dims[i].degree == 1);
  }
  // Check that the two inputs have the same dimensions
//  printf("ResidualRMSNorm: input1 dims: %d, input2 dims: %d\n", _input1->num_dims, _input2->num_dims);
//  printf("ResidualRMSNorm: input1 dims[0] size: %d, input2 dims[0] size: %d\n", _input1->dims[0].size, _input2->dims[0].size);
//  printf("ResidualRMSNorm: input1 dims[1] size: %d, input2 dims[1] size: %d\n", _input1->dims[1].size, _input2->dims[1].size);
//  printf("ResidualRMSNorm: input1 dims[2] size: %d, input2 dims[2] size: %d\n", _input1->dims[2].size, _input2->dims[2].size); // Mistatch! 128 vs 1
  for (int i = 0; i < _input1->num_dims; i++) {
    assert(_input2->dims[i] == _input1->dims[i]);
  }
  // output has the same parallel dims as input
  ParallelDim output_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input1->num_dims; i++) {
    output_dims[i] = _input1->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(_input1->num_dims,
                                                            output_dims,
                                                            _input1->data_type,
                                                            this,
                                                            0 /*owner_idx*/);
  outputs[1] = model.create_parallel_tensor_legion_ordering(_input1->num_dims,
                                                            output_dims,
                                                            _input1->data_type,
                                                            this,
                                                            1 /*owner_idx*/);

  if (allocate_weights) {
    // weights should have the shape of (data_dim, data_dim)
    ParallelDim new_weight_dims[MAX_TENSOR_DIM];

    new_weight_dims[0].size = dim;
    new_weight_dims[0].degree = 1;
    new_weight_dims[0].parallel_idx = -1;
    new_weight_dims[1] = _input1->dims[_input1->num_dims - 1]; // replica dim

    // weights
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
    weights[0] =
        model.create_parallel_weight_legion_ordering(2,
                                                     new_weight_dims,
                                                     _input1->data_type,
                                                     nullptr /*owner_op*/,
                                                     false /*create_grad*/,
                                                     kernel_initializer,
                                                     CHOSEN_SYNC_TYPE);
  }
}

void ResidualRMSNorm::map_output_tensors(FFModel &ff) {
  assert(numOutputs == 2);
  assert(outputs[0]->get_volume() == inputs[0]->get_volume());
  if (inplace_residual) {
    outputs[0]->parallel_is = inputs[0]->parallel_is;
    outputs[0]->region = inputs[0]->region;
    outputs[0]->part = inputs[0]->part;
    outputs[0]->region_grad = inputs[0]->region_grad;
    outputs[0]->part_grad = inputs[0]->part_grad;
    // map output 1 to new region
    ff.map_tensor(outputs[1], this);
  } else {
    Op::map_output_tensors(ff);
  }
}

void ResidualRMSNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RESIDUAL_RMSNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ResidualRMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  if (inplace_residual) {
    assert(outputs[0]->part == inputs[0]->part);
    assert(outputs[0]->region == inputs[0]->region);
  }
  int fid = 0;
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part,
                        0 /*projection id*/,
                        inplace_residual ? READ_WRITE : READ_ONLY,
                        EXCLUSIVE,
                        inputs[0]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  if (!inplace_residual) {
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(fid++, FID_DATA);
  }
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(fid++, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void ResidualRMSNorm::init_inference(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);

  IndexLauncher launcher(RESIDUAL_RMSNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(ResidualRMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  if (inplace_residual) {
    assert(batch_outputs[0]->part == batch_inputs[0]->part);
    assert(batch_outputs[0]->region == batch_inputs[0]->region);
  }
  int fid = 0;
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part,
                        0 /*projection id*/,
                        inplace_residual ? READ_WRITE : READ_ONLY,
                        EXCLUSIVE,
                        batch_inputs[0]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  if (!inplace_residual) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(fid++, FID_DATA);
  }
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(fid++, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

OpMeta *ResidualRMSNorm::init_task(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  ResidualRMSNorm *rn = (ResidualRMSNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = get_proc_mem(Machine::get_machine(), task->target_proc);
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  ResidualRMSNormMeta *meta =
      new ResidualRMSNormMeta(handle, rn, gpu_mem_allocator);
  std::strcpy(meta->op_name, rn->name);
  meta->layer_guid = rn->layer_guid;
  return meta;
}

void ResidualRMSNorm::forward(FFModel const &ff) {
  assert(false);
}

FutureMap
    ResidualRMSNorm::inference(FFModel const &ff,
                               BatchConfigFuture const &bc,
                               std::vector<ParallelTensor> const &batch_inputs,
                               std::vector<ParallelTensor> const &batch_outputs,
                               MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();

  IndexLauncher launcher(RESIDUAL_RMSNORM_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  if (inplace_residual) {
    assert(batch_outputs[0]->part == batch_inputs[0]->part);
    assert(batch_outputs[0]->region == batch_inputs[0]->region);
  }
  int fid = 0;
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part,
                        0 /*projection id*/,
                        inplace_residual ? READ_WRITE : READ_ONLY,
                        EXCLUSIVE,
                        batch_inputs[0]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  if (!inplace_residual) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region));
    launcher.add_field(fid++, FID_DATA);
  }
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(fid++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(fid++, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I/O): input1 / residual output
  regions[1](I): input2
  regions[2](O): output
  regions[3](I): weight
*/
void ResidualRMSNorm::inference_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  ResidualRMSNormMeta *m = *((ResidualRMSNormMeta **)task->local_args);
  assert(task->regions.size() == 5 - m->inplace_residual);
  assert(regions.size() == 5 - m->inplace_residual);
  GenericTensorAccessorR input1 = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR input2 = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  GenericTensorAccessorW residual_output, output;
  GenericTensorAccessorR weight;
  if (m->inplace_residual) {
    // residual_output is mapped to the same region as the input
    residual_output = helperGetGenericTensorAccessorWO(m->output_type[0],
                                                       regions[0],
                                                       task->regions[0],
                                                       FID_DATA,
                                                       ctx,
                                                       runtime);
    output = helperGetGenericTensorAccessorWO(m->output_type[1],
                                              regions[2],
                                              task->regions[2],
                                              FID_DATA,
                                              ctx,
                                              runtime);
    weight = helperGetGenericTensorAccessorRO(m->weight_type[0],
                                              regions[3],
                                              task->regions[3],
                                              FID_DATA,
                                              ctx,
                                              runtime);
  } else {
    residual_output = helperGetGenericTensorAccessorWO(m->output_type[0],
                                                       regions[2],
                                                       task->regions[2],
                                                       FID_DATA,
                                                       ctx,
                                                       runtime);
    output = helperGetGenericTensorAccessorWO(m->output_type[1],
                                              regions[3],
                                              task->regions[3],
                                              FID_DATA,
                                              ctx,
                                              runtime);
    weight = helperGetGenericTensorAccessorRO(m->weight_type[0],
                                              regions[4],
                                              task->regions[4],
                                              FID_DATA,
                                              ctx,
                                              runtime);
  }

  inference_kernel_wrapper(
      m, bc, input1, input2, weight, residual_output, output);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    if (m->inplace_residual) {
      ResidualRMSNorm::save_inference_tensors_to_file(
          m, shard_id, bc, {input2}, {weight}, {residual_output, output});
    } else {
      ResidualRMSNorm::save_inference_tensors_to_file(
          m,
          shard_id,
          bc,
          {input1, input2},
          {weight},
          {residual_output, output});
    }
  }
}

void ResidualRMSNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->eps);
  sez.serialize(this->dim);
  sez.serialize(this->inplace_residual);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node ResidualRMSNorm::deserialize(FFModel &ff,
                                  Legion::Deserializer &dez,
                                  ParallelTensor inputs[],
                                  int num_inputs) {
  assert(num_inputs == 2);
  float eps;
  size_t id, transformer_layer_id, deserialized_model_id;
  int dim;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(eps);
  dez.deserialize(dim);
  int inplace_residual;
  dez.deserialize(inplace_residual);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  ResidualRMSNormParams params;
  params.layer_guid = layer_guid;
  params.eps = eps;
  params.dim = dim;
  params.inplace_residual = inplace_residual;
  strcpy(params.name, name);
  return ff.get_or_create_node<ResidualRMSNorm>({inputs[0], inputs[1]}, params);
}

void ResidualRMSNorm::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(RESIDUAL_RMSNORM_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): RMS output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): residual output / RMS input
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): residual input grad 0
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): residual input grad 1
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[1]->region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): gamma
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): gamma_grad
  launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): RMS output_grad
  regions[1](I): Residual output / RMS input
  regions[2](I/O): Residual input 0 grad
  regions[3](I/O): Residual input 1 grad
  regions[4](I): weight
  regions[5](I/O): weight_grad
*/
void ResidualRMSNorm::backward_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(task->regions.size() == 6);
  assert(regions.size() == 6);
  ResidualRMSNormMeta const *m = *((ResidualRMSNormMeta **)task->local_args);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW residual_output_rms_input =
      helperGetGenericTensorAccessorRW(m->input_type[0],
                                       regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW residual_input0_grad =
      helperGetGenericTensorAccessorRW(m->input_type[0],
                                       regions[2],
                                       task->regions[2],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW residual_input1_grad =
      helperGetGenericTensorAccessorRW(m->input_type[0],
                                       regions[3],
                                       task->regions[3],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[4], task->regions[4], FID_DATA, ctx, runtime);
  GenericTensorAccessorW weight_grad = helperGetGenericTensorAccessorRW(
      m->weight_type[0], regions[5], task->regions[5], FID_DATA, ctx, runtime);
  backward_kernel_wrapper(m,
                          output_grad,
                          residual_output_rms_input,
                          residual_input0_grad,
                          residual_input1_grad,
                          weight,
                          weight_grad);
}

Legion::FutureMap
    ResidualRMSNorm::peft_bwd(FFModel const &ff,
                              BatchConfigFuture const &bc,
                              std::vector<ParallelTensor> const &batch_inputs,
                              std::vector<ParallelTensor> const &batch_outputs,
                              MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  IndexLauncher launcher(RESIDUAL_RMSNORM_PEFT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  int fid = 0;
  // residual input grad 0
  launcher.add_region_requirement(RegionRequirement(
      batch_inputs[0]->part_grad,
      0 /*projection id*/,
      inplace_residual && !reset_input_grads[0] ? READ_WRITE : WRITE_ONLY,
      EXCLUSIVE,
      batch_inputs[0]->region_grad));
  launcher.add_field(fid++, FID_DATA);
  // residual input grad 1
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[1]->part_grad,
                        0 /*projection id*/,
                        reset_input_grads[1] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE,
                        batch_inputs[1]->region_grad));
  launcher.add_field(fid++, FID_DATA);
  if (!inplace_residual && !reset_input_grads[0]) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[0]->part_grad,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          batch_outputs[0]->region_grad));
    launcher.add_field(fid++, FID_DATA);
  }
  // RMS output_grad
  launcher.add_region_requirement(
      RegionRequirement(batch_outputs[1]->part_grad,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        batch_outputs[1]->region_grad));
  launcher.add_field(fid++, FID_DATA);
  // gamma
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(fid++, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): RMS output_grad
  regions[1](I/O): Residual input 0 grad
  regions[2](I/O): Residual input 1 grad
  regions[3](I): weight
*/
void ResidualRMSNorm::peft_bwd_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  ResidualRMSNormMeta *m = *((ResidualRMSNormMeta **)task->local_args);
  int expected_regions =
      (m->inplace_residual || m->reset_input_grads[0]) ? 4 : 5;
  assert(task->regions.size() == expected_regions);
  assert(regions.size() == expected_regions);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }

  int rid = 0, t_rid = 0;
  GenericTensorAccessorW input_grad_0 =
      helperGetGenericTensorAccessorRW(m->input_type[0],
                                       regions[rid++],
                                       task->regions[t_rid++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW input_grad_1 =
      helperGetGenericTensorAccessorRW(m->input_type[0],
                                       regions[rid++],
                                       task->regions[t_rid++],
                                       FID_DATA,
                                       ctx,
                                       runtime);

  GenericTensorAccessorR output_grad_0;
  if (!m->reset_input_grads[0]) {
    if (m->inplace_residual) {
      // mapped to input 0
      output_grad_0 = helperGetGenericTensorAccessorRO(m->output_type[0],
                                                       regions[0],
                                                       task->regions[0],
                                                       FID_DATA,
                                                       ctx,
                                                       runtime);
    } else {
      output_grad_0 = helperGetGenericTensorAccessorRO(m->output_type[0],
                                                       regions[rid++],
                                                       task->regions[t_rid++],
                                                       FID_DATA,
                                                       ctx,
                                                       runtime);
    }
  }
  GenericTensorAccessorR output_grad_1 =
      helperGetGenericTensorAccessorRO(m->output_type[0],
                                       regions[rid++],
                                       task->regions[t_rid++],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR weight =
      helperGetGenericTensorAccessorRO(m->weight_type[0],
                                       regions[rid++],
                                       task->regions[t_rid++],
                                       FID_DATA,
                                       ctx,
                                       runtime);

  peft_bwd_kernel_wrapper(
      m, bc, output_grad_0, output_grad_1, input_grad_0, input_grad_1, weight);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    if (!m->reset_input_grads[0]) {
      ResidualRMSNorm::save_inference_tensors_to_file(
          m,
          shard_id,
          bc,
          {input_grad_0, input_grad_1},
          {weight},
          {output_grad_0, output_grad_1},
          false);
    } else {
      ResidualRMSNorm::save_inference_tensors_to_file(
          m,
          shard_id,
          bc,
          {input_grad_0, input_grad_1},
          {weight},
          {output_grad_1},
          false);
    }
  }
}

Op *ResidualRMSNorm::materialize(FFModel &ff,
                                 ParallelTensor inputs[],
                                 int num_inputs) const {
  ResidualRMSNormParams params = get_params();
  return new ResidualRMSNorm(
      ff, params, {inputs[0], inputs[1]}, true, this->name);
}

bool ResidualRMSNorm::measure_operator_cost(Simulator *sim,
                                            MachineView const &mv,
                                            CostMetrics &cost_metrics) const {
  return false;
}

} // namespace FlexFlow
namespace std {
size_t hash<FlexFlow::ResidualRMSNormParams>::operator()(
    FlexFlow::ResidualRMSNormParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.eps);
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.dim);
  hash_combine(key, params.inplace_residual);
  return key;
}
}; // namespace std
