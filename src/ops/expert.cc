#include "flexflow/ops/expert.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
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

// TODO remove all instances of "Linear" in this script

//using namespace FlexFlow::Kernels::Linear;

static constexpr int W1_IDX = 0;
static constexpr int W2_IDX = 1;
static constexpr int W3_IDX = 2;

Tensor FFModel::expert(const Tensor input,
                      int outDim_intermediate, // used to get w1 and w3
                      int outDim_hidden, // used to get w2
                      ActiMode activation,
                      bool use_bias,
                      DataType data_type,
                      Layer const *shared_op,
                      Initializer *kernel_initializer,
                      Initializer *bias_initializer,
                      RegularizerMode kernel_reg_type,
                      float kernel_reg_lambda,
                      char const *name1,
                      char const *name3, // mimics order of usage of weights
                      char const *name2,
                       char const *name_expert // name for the entire expert
                       ) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  DataType quantization_type = cpu_offload ? config.quantization_type : DT_NONE;
  bool offload = cpu_offload;
  Layer *li = nullptr;
  if (data_type != input->data_type) {
  assert(false && "Not implemented");
    //    Tensor casted_input = cast(input, data_type, "type cast for dense");
//    li = new Layer(this,
//                   OP_LINEAR,
//                   data_type,
//                   name,
//                   1 /*inputs*/,
//                   use_bias ? 2 : 1 /*weights*/,
//                   1 /*outputs*/,
//                   casted_input);
  } else {
    li = new Layer(this,
                   OP_EXPERT,
                   data_type,
                   name_expert,
                   1 /*inputs*/,
                   use_bias ? 2 : 1 /*weights*/,
                   1 /*outputs*/,
                   input);
  }

  {

    int numdims_w1 = input->num_dims;
    int dims_w1[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims_w1; i++) {
      dims_w1[i] = input->dims[i];
    }
    dims_w1[0] = outDim_intermediate;

    int numdims_w2 = input->num_dims;
    int dims_w2[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims_w2; i++) {
      dims_w2[i] = input->dims[i];
    }
    dims_w2[0] = outDim_hidden;

    //    std::cout << "Dense " << name << " Creating output tensor with dims[2] = " << dims[0] << std::endl;
    // dims[2] is always 1024
    li->outputs[W1_IDX] = create_tensor_legion_ordering(
        numdims_w1, dims_w1, data_type, li, 0, true /*create_grad*/);
    li->outputs[W2_IDX] = create_tensor_legion_ordering(
    numdims_w2, dims_w2, data_type, li, 0, true /*create_grad*/);
  }
  {
    int dims_in_out_w1[2] = {input->dims[0], outDim_intermediate};
    int dims_in_out_w2[2] = {outDim_intermediate, outDim_hidden};

    if (quantization_type != DT_NONE) {
      assert(false && "Not implemented");
//      dims[0] =
//          get_quantization_to_byte_size(data_type, quantization_type, dims[0]);
    }
    li->weights[W1_IDX] = create_weight_legion_ordering(
        2,
        dims_in_out_w1,
        quantization_type == DT_NONE ? data_type : quantization_type,
        li,
        true /*create_grad*/,
        kernel_initializer,
        CHOSEN_SYNC_TYPE);

    li->weights[W2_IDX] = create_weight_legion_ordering(
    2,
    dims_in_out_w2,
    quantization_type == DT_NONE ? data_type : quantization_type,
    li,
    true /*create_grad*/,
    kernel_initializer,
    CHOSEN_SYNC_TYPE);
  }
  li->add_int_property("use_bias", use_bias);
  li->add_int_property("out_dim_intermediate", outDim_intermediate);
  li->add_int_property("out_dim_hidden", outDim_hidden);
  li->add_int_property("activation", activation);
  li->add_int_property("kernel_reg_type", kernel_reg_type);
  li->add_float_property("kernel_reg_lambda", kernel_reg_lambda);
  li->add_int_property("quantization_type", quantization_type);
  li->add_int_property("offload", offload);
  layers.push_back(li);
  return li->outputs[1];
}

Op *Expert::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  layer->get_int_property("out_dim_intermediate", value);
  int outdim_intermediate = value;
  layer->get_int_property("out_dim_hidden", value);
  int outdim_hidden = value;

  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode)value;
  layer->get_int_property("kernel_reg_type", value);
  RegularizerMode kernel_reg_type = (RegularizerMode)value;
  float kernel_reg_lambda;
  layer->get_float_property("kernel_reg_lambda", kernel_reg_lambda);
  layer->get_int_property("quantization_type", value);
  DataType quantization_type = (DataType)value;
  layer->get_int_property("offload", value);
  bool offload = (bool)value;
  return new Expert(model,
                    layer->layer_guid,
                    inputs[0],
                    outdim_dim_hidden,
                    activation,
                    kernel_reg_type,
                    kernel_reg_lambda,
                    use_bias,
                    layer->data_type,
                    quantization_type,
                    offload,
                    false /*allocate_weights*/,
                    layer->name);
}

// size_t Linear::get_params_hash() const {
//   return this->get_params().get_hash(this->inputs[0]);
// }

Expert::Expert(FFModel &model,
               Linear const &other,
               const ParallelTensor input,
               bool allocate_weights)
    : Expert(model,
             other.layer_guid,
             input,
             other.out_channels,
             other.activation,
             other.kernel_reg_type,
             other.kernel_reg_lambda,
             other.use_bias,
             other.data_type,
             other.quantization_type,
             other.offload,
             allocate_weights,
             other.name) {}

Expert::Expert(FFModel &model,
               LinearParams const &params,
               ParallelTensor const input,
               char const *name,
               bool allocate_weights)
    : Expert(model,
             params.layer_guid,
             input,
             params.out_channels,
             params.activation,
             params.kernel_reg_type,
             params.kernel_reg_lambda,
             params.use_bias,
             params.data_type,
             params.quantization_type,
             params.offload,
             allocate_weights,
             params.name) {}

Expert::Expert(FFModel &model,
               LayerID const &_layer_guid,
               const ParallelTensor _input,
               int out_dim_hidden, // TODO see if need intermediate too
               ActiMode _activation,
               RegularizerMode _kernel_reg_type,
               float _kernel_reg_lambda,
               bool _use_bias,
               DataType _data_type,
               DataType _quantization_type,
               bool _offload,
               bool allocate_weights,
               char const *name)
    : Op(model,
         OP_EXPERT,
         _data_type,
         name,
         1 /*inputs*/,
         _use_bias ? 2 : 1 /*weights*/,
         allocate_weights,
         2 /*outputs*/, // TODO add if we have more weights
         _input),
      out_channels(out_dim_hidden), activation(_activation), use_bias(_use_bias),
      kernel_reg_type(_kernel_reg_type), kernel_reg_lambda(_kernel_reg_lambda),
      quantization_type(_quantization_type), offload(_offload),
      replica(ParallelTensorBase::NO_TENSOR) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  data_type = _data_type;
  auto dimension_names =
      this->get_params().get_dimension_names(_input->get_shape());
  this->in_channels =
      _input->dims[dimension_names.at(LinearParams::INPUT_CHANNEL)].size;

  ParallelTensorShape input_shape = this->inputs[0]->get_shape();
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  ExpertParams params = this->get_params();
  params.construct_mappings(*this->parallel_dims_mapping, input_shape);
  params.solve_dims(input_shape, output_shape, kernel_shape, bias_shape);
  kernel_shape.dims[0].size = this->in_channels;
  bias_shape.dims[0].degree = _input->dims[_input->num_dims - 1].degree;
  bias_shape.dims[0].parallel_idx =
      _input->dims[_input->num_dims - 1].parallel_idx;
  bias_shape.dims[1].size = bias_shape.dims[1].degree = 1;
  bias_shape.dims[1].parallel_idx = -1;
  bias_shape.dims[bias_shape.num_dims - 1].size =
      bias_shape.dims[bias_shape.num_dims - 1].degree = 1;
  for (int i = 0; i < input_shape.num_dims - 1; i++) {
    if (_input->dims[i].degree > 1) {
      bias_shape.dims[bias_shape.num_dims - 1].size *= _input->dims[i].degree;
      bias_shape.dims[bias_shape.num_dims - 1].degree *= _input->dims[i].degree;
      bias_shape.dims[bias_shape.num_dims - 1].parallel_idx =
          _input->dims[i].parallel_idx;
    }
  }

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
    if (quantization_type != DT_NONE) {
      kernel_shape.dims[0].size = get_quantization_to_byte_size(
          data_type, quantization_type, kernel_shape.dims[0].size);
    }
    weights[KERNEL_IDX] = model.create_parallel_weight_legion_ordering(
        kernel_shape.num_dims,
        kernel_shape.dims,
        quantization_type == DT_NONE ? _data_type : quantization_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        kernel_initializer,
        CHOSEN_SYNC_TYPE);

  }

  // Create the output tensor
//  std::cout << "Linear::Linear " << name << " create_parallel_tensor has num_dims = " << output_shape.num_dims << std::endl;
//  std::cout << "Linear::Linear " << name << " create_parallel_tensor with dims[0] = " << output_shape.dims[0].size << std::endl;
//  std::cout << "Linear::Linear " << name << " create_parallel_tensor with dims[1] = " << output_shape.dims[1].size << std::endl;
//  std::cout << "Linear::Linear " << name << " create_parallel_tensor with dims[2] = " << output_shape.dims[2].size << std::endl;
//  std::cout << "Linear::Linear " << name << " create_parallel_tensor with dims[3] = " << output_shape.dims[3].size << std::endl;
  // TODO for w2, the dims are (1024,1,1,1) (4 dims confirmed) !! Should have sequence length somewhere...
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_shape.num_dims, output_shape.dims, _data_type, this);

  // assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Linear::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  // assert(check_output_input_weight_same_machine_view());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Linear)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // launcher.add_region_requirement(
  //     RegionRequirement(input_lps[0], 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    // launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    // launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Linear::init_inference(FFModel const &ff,
                            std::vector<ParallelTensor> const &batch_inputs,
                            std::vector<ParallelTensor> const &batch_outputs,
                            MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  // assert(check_output_input_weight_same_machine_view());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Linear)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // launcher.add_region_requirement(
  //     RegionRequirement(input_lps[0], 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    // launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    // launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta *Linear::init_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Linear const *linear = (Linear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorWO(linear->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  switch (output.domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    if (output.data_type == DT_HALF) {                                         \
      if (linear->quantization_type != DT_NONE) {                              \
        return init_task_with_dim<half, char, DIM>(                            \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return init_task_with_dim<half, half, DIM>(                            \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else if (output.data_type == DT_FLOAT) {                                 \
      if (linear->quantization_type != DT_NONE) {                              \
        return init_task_with_dim<float, char, DIM>(                           \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return init_task_with_dim<float, float, DIM>(                          \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else {                                                                   \
      assert(false && "Unsupported data type");                                \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  return NULL;
}

template <typename DT, typename WT, int NDIM>
OpMeta *Linear::init_task_with_dim(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  assert(regions.size() == task->regions.size());
  assert(regions.size() == 2 || regions.size() == 3);
  Linear const *linear = (Linear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  // TensorAccessorR<float, 2> acc_input(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<DT, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<DT, NDIM> acc_output(regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  TensorAccessorR<WT, NDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  // TensorAccessorR<float, 1> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  // int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  // printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
  //        in_dim,
  //        out_dim,
  //        batch_size);
  Memory gpu_mem = get_proc_mem(Machine::get_machine(), task->target_proc);
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  if (linear->offload) {
    // cpu-offload enabled
    // use offload_reserved_space
    gpu_mem_allocator.register_reserved_work_space(
        handle.offload_reserve_space, handle.offload_reserve_space_size);
  }

  LinearMeta *m = new LinearMeta(
      handle, batch_size, linear, gpu_mem_allocator, in_dim * out_dim);
  m->activation = linear->activation;
  m->kernel_reg_type = linear->kernel_reg_type;
  m->kernel_reg_lambda = linear->kernel_reg_lambda;
  m->use_bias = linear->use_bias;
  m->add_bias_only_once = linear->add_bias_only_once;
  m->profiling = linear->profiling;
  m->inference_debugging = linear->inference_debugging;
  m->trainable_inputs[0] = linear->trainable_inputs[0];
  m->weight_ptr_type = m->input_type[0];
  m->quantization_type = linear->quantization_type;
  m->offload = linear->offload;
  std::strcpy(m->op_name, linear->name);
  m->layer_guid = linear->layer_guid;

  init_kernel(m, batch_size, out_dim);

  return m;
}

void Linear::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(LINEAR_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

FutureMap Linear::inference(FFModel const &ff,
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
  /* std::cout << "Linear op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(LINEAR_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

void Linear::inference_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LinearMeta *m = *((LinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  assert(regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  if (m->quantization_type == DT_NONE) {
    assert(m->input_type[0] == m->weight_type[0]);
  }
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  assert((weight.domain.hi()[0] - weight.domain.lo()[0] + 1) == in_dim);
  assert((weight.domain.hi()[1] - weight.domain.lo()[1] + 1) == out_dim);
  assert(weight.domain.get_volume() == in_dim * out_dim);

  int batch_size = bc->num_active_infr_tokens();
  GenericTensorAccessorR bias;
  if (m->use_bias &&
      !(m->add_bias_only_once && task->index_point.point_data[0] != 0)) {
    bias = helperGetGenericTensorAccessorRO(m->weight_type[1],
                                            regions[3],
                                            task->regions[3],
                                            FID_DATA,
                                            ctx,
                                            runtime);
    assert(bias.domain.get_volume() == static_cast<size_t>(out_dim));
  }
  FlexFlow::Kernels::Linear::inference_kernel_wrapper(m,
                           bc,
                           input.ptr,
                           output.ptr,
                           weight.ptr,
                           bias.ptr,
                           in_dim,
                           out_dim,
                           batch_size);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    std::vector<GenericTensorAccessorR> weights_accessors;
    weights_accessors.push_back(weight);
    if (m->use_bias &&
        !(m->add_bias_only_once && task->index_point.point_data[0] != 0)) {
      weights_accessors.push_back(bias);
    }
    Linear::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, weights_accessors, {output});
    printf("\tw=[%i,%i].T @ in=[%i,%i] -> out=[%i,%i]\n",
           in_dim,
           out_dim,
           in_dim,
           bc->num_tokens,
           out_dim,
           bc->num_tokens);
  }
}

FutureMap Linear::peft_bwd(FFModel const &ff,
                           BatchConfigFuture const &bc,
                           std::vector<ParallelTensor> const &batch_inputs,
                           std::vector<ParallelTensor> const &batch_outputs,
                           MachineView const *mv) {
  assert(false && "Not implemented");
}

/*
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
template <typename DT, typename WT, int NDIM>
void Linear::forward_task_with_dim(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  // Linear* linear = (Linear*) task->args;
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  assert(regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (3 + static_cast<size_t>(m->use_bias)));

  TensorAccessorR<DT, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<DT, NDIM> acc_output(regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  TensorAccessorR<WT, NDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == static_cast<size_t>(out_dim * batch_size));
  assert(acc_input.rect.volume() == static_cast<size_t>(in_dim * batch_size));
  // assert(acc_kernel.rect.volume() == static_cast<size_t>(in_dim * out_dim));
  DT const *acc_bias_ptr = nullptr;
  if (m->use_bias &&
      !(m->add_bias_only_once && task->index_point.point_data[0] != 0)) {
    TensorAccessorR<DT, NDIM> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(acc_bias.rect.volume() == static_cast<size_t>(out_dim));
    acc_bias_ptr = acc_bias.ptr;
  }

  forward_kernel_wrapper(m,
                         acc_input.ptr,
                         acc_output.ptr,
                         acc_kernel.ptr,
                         acc_bias_ptr,
                         in_dim,
                         out_dim,
                         batch_size);
}

void Expert::backward(FFModel const &ff) {
  assert(false && "Not implemented");
}

void Expert::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(false && "Not implemented");
}

/*
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
template <typename DT, int NDIM>
void Expert::backward_task_with_dim(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(false && "Not implemented");
}

void Expert::print_layer(FFModel const &ff) {
  printf("linear layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  RegionRequirement kernel_req(
      weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

  RegionRequirement bias_req(
      weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();

  TensorAccessorW<float, 2> acc_kernel(
      kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
  TensorAccessorW<float, 1> acc_bias(
      bias_region, bias_req, FID_DATA, ctx, runtime, true);

  float const *kernel_ptr = acc_kernel.ptr;
  float const *bias_ptr = acc_bias.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %p, %zu, [%d, %d]\n",
         kernel_ptr,
         kernel_size,
         kernel_dim1,
         kernel_dim2);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);

  for (size_t i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

  for (size_t i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

  runtime->unmap_region(ctx, kernel_region);
  runtime->unmap_region(ctx, bias_region);
}

bool Expert::estimate_sync_cost(Simulator *sim,
                                MachineView const &view,
                                CostMetrics &cost_metrics) const {
  // Estimate the cost of sync weights
  ParallelTensorShape tensor_shape;
  tensor_shape.num_dims = 3;
  tensor_shape.data_type = inputs[0]->data_type;
  tensor_shape.dims[0] = inputs[0]->dims[0];
  tensor_shape.dims[1] = inputs[0]->dims[inputs[0]->num_dims - 1];
  tensor_shape.dims[2] = inputs[0]->dims[inputs[0]->num_dims - 2];
  tensor_shape.dims[1].size = out_channels;
  tensor_shape.dims[1].degree = 1;
  tensor_shape.dims[2].degree =
      inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  tensor_shape.dims[2].size =
      inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  cost_metrics.sync_time =
      sim->default_estimate_sync_cost(tensor_shape, view, 1);
  // printf("[Estimate Linear] name(%s) sync_time(%.4lf)\n", name,
  // cost_metrics.sync_time);
  return true;
}

ParallelConfig Expert::get_random_parallel_config(FFModel const &ff) const {
  if (!ff.config.enable_parameter_parallel) {
    return Op::get_random_parallel_config(ff);
  }
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0]->dims[outputs[0]->num_dims - 1].size;
  int channel = outputs[0]->dims[0].size;
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++) {
    if (channel % i == 0) {
      for (int j = 1; i * j <= total_devices; j++) {
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
      }
    }
  }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims - 1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++) {
    pc.dim[i] = 1;
  }
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++) {
    pc.device_ids[i] = start_idx + i;
  }
  return pc;
}

bool Expert::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_ACTI:
      *value = (int)activation;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Expert::is_valid_parallel_config(FFModel const &ff,
                                      ParallelConfig const &pc) const {
  if (!ff.config.enable_parameter_parallel) {
    return Op::is_valid_parallel_config(ff, pc);
  }
  // Support data and parameter parallel
  if (pc.nDims != outputs[0]->num_dims) {
    return false;
  }
  for (int i = 1; i < pc.nDims - 1; i++) {
    if (pc.dim[i] != 1) {
      return false;
    }
  }
  return true;
}

bool Expert::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  assert(false && "Not implemented");
  return false;
}

bool operator==(ExpertParams const &lhs, ExpertParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid &&
         lhs.out_channels == rhs.out_channels && lhs.use_bias == rhs.use_bias &&
         lhs.data_type == rhs.data_type && lhs.activation == rhs.activation &&
         lhs.kernel_reg_type == rhs.kernel_reg_type &&
         lhs.kernel_reg_lambda == rhs.kernel_reg_lambda;
}

void Expert::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->out_channels);
  sez.serialize(this->activation);
  sez.serialize(this->kernel_reg_type);
  sez.serialize(this->kernel_reg_lambda);
  sez.serialize(this->use_bias);
  sez.serialize(this->data_type);
  sez.serialize(this->quantization_type);
  sez.serialize(this->offload);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

/* static */
using PCG::Node;
Node Expert::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 1);
  int out_channels;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias;
  DataType data_type;
  DataType quantization_type;
  bool offload;
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(out_channels);
  dez.deserialize(activation);
  dez.deserialize(kernel_reg_type);
  dez.deserialize(kernel_reg_lambda);
  dez.deserialize(use_bias);
  dez.deserialize(data_type);
  dez.deserialize(quantization_type);
  dez.deserialize(offload);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  ExpertParams params;
  params.activation = activation;
  params.kernel_reg_type = kernel_reg_type;
  params.kernel_reg_lambda = kernel_reg_lambda;
  params.out_channels = out_channels;
  params.use_bias = use_bias;
  params.data_type = data_type;
  params.layer_guid = layer_guid;
  params.quantization_type = quantization_type;
  params.offload = offload;
  strcpy(params.name, name);
  return ff.get_or_create_node<Linear>(inputs[0], params);
}

ExpertParams Expert::get_params() const {
  ExpertParams params;
  params.layer_guid = this->layer_guid;
  params.out_channels = this->out_channels;
  params.use_bias = this->use_bias;
  params.data_type = this->data_type;
  params.activation = this->activation;
  params.kernel_reg_type = this->kernel_reg_type;
  params.kernel_reg_lambda = this->kernel_reg_lambda;
  params.quantization_type = this->quantization_type;
  params.offload = this->offload;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }

  return params;
}

bool ExpertParams::is_valid(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  this->solve_dims(input_shape,
                   output_shape.dims,
                   &output_shape.num_dims,
                   kernel_shape.dims,
                   &kernel_shape.num_dims,
                   bias_shape.dims,
                   &bias_shape.num_dims);
  bool is_valid = true;
  is_valid &= input_shape.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= kernel_shape.is_valid();
  if (use_bias) {
    is_valid &= bias_shape.is_valid();
  }
  return is_valid;
}

/** @brief  A wrapper around the main version of the solve_dims function.
 *
 * It takes a the input tensor as a parameter, instead of the input's
 * ParallelTensorShape.
 */
void ExpertParams::solve_dims(const ParallelTensor input,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  this->solve_dims(input->get_shape(),
                   output_dims,
                   output_ndims,
                   kernel_dims,
                   kernel_ndims,
                   bias_dims,
                   bias_ndims);
}

/** @brief  A wrapper around the main version of the solve_dims function.
 *
 * For each of the output, weights, and bias tensors, it takes a
 * ParallelTensorShape argument, instead of a pointer to an integer variable to
 * record the number of dimensions, plus a ParallelDim array to record all the
 * information regarding each dimension.
 */
void ExpertParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelTensorShape &output_shape,
                              ParallelTensorShape &kernel_shape,
                              ParallelTensorShape &bias_shape) const {
  this->solve_dims(input_shape,
                   output_shape.dims,
                   &output_shape.num_dims,
                   kernel_shape.dims,
                   &kernel_shape.num_dims,
                   bias_shape.dims,
                   &bias_shape.num_dims);
}

void ExpertParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  assert((output_dims == nullptr) == (output_ndims == nullptr));
  assert((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert((bias_dims == nullptr) == (bias_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  this->construct_mappings(mapping, input_shape);
  // sets the is_replica_dim field to true for the dimensions that are used to
  // record the number of replicas
  this->mark_replica_dims(input_shape, output_dims, kernel_dims, bias_dims);

  solve_parallel_dim_mappings(
      mapping, {input_shape.dims}, {kernel_dims, bias_dims}, {output_dims});

  // sets the dimension sizes of the output, weights, and bias tensors
  this->calculate_nonreplica_dim_sizes(input_shape,
                                       output_dims,
                                       output_ndims,
                                       kernel_dims,
                                       kernel_ndims,
                                       bias_dims,
                                       bias_ndims);
}

/** @brief  Create a map between each of a tensor's dimension name and its
 * corresponding index
 *
 * The tensor dimension names are defined as follows. For the input tensor, the
 * first dimension is called INPUT_CHANNEL, and generally corresponds to number
 * of floats needed to store a single element from the input dataset. For
 * example, when each element in the dataset is a flattened MNIST image, the
 * INPUT_CHANNEL dimension will have a size of 28x28=784. The second to last and
 * last dimensions in the input tensor are, respectively, the INPUT_SAMPLE and
 * INPUT_REPLICA dimensions. The size of the INPUT_SAMPLE dimension generally
 * corresponds to the batch size used for training. The size of the
 * INPUT_REPLICA tells us how many replicas of the tensors have been created.
 * The dimensions of the output tensor are named analogously: the first
 * dimension is OUTPUT_CHANNEL, the second to last is OUTPUT_SAMPLE, and the
 * last one is OUTPUT_REPLICA. Both the input and output tensor may have
 * additional dimensions, without a name, between {INPUT,OUTPUT}_CHANNEL and
 * {INPUT,OUTPUT}_SAMPLE. For instance, when the input data comes in textual
 * form, it is common to have an additional dimension representing the sequence
 * length. When it comes to the weights, the dimensions are named simply as
 * KERNEL_CHANNEL_IN (first dimension of a weight's tensor), KERNEL_CHANNEL_OUT
 * (second dimension) and BIAS_CHANNEL_OUT (first dimension of the bias tensor)
 *
 * @param[in] input_shape   A ParallelTensorShape object representing the shape
 * of the ParallelTensor used for the input to the operator
 * @return dimension_names  A map from each LinearParams::NamedDimensions to the
 * index corresponding to that dimension in the input, weight, (bias), or output
 * tensor.
 */
std::unordered_map<ExpertParams::NamedDimensions, int>
    ExpertParams::get_dimension_names(
        ParallelTensorShape const &input_shape) const {
  int num_dims = input_shape.num_dims;

  return {{INPUT_CHANNEL, 0},
          {INPUT_SAMPLE, num_dims - 2},
          {INPUT_REPLICA, num_dims - 1},
          {OUTPUT_CHANNEL, 0},
          {OUTPUT_SAMPLE, num_dims - 2},
          {OUTPUT_REPLICA, num_dims - 1},
          {KERNEL_CHANNEL_IN, 0},
          {KERNEL_CHANNEL_OUT, 1},
          {BIAS_CHANNEL_OUT, 0}};
}

/** @brief  Sets the size field of ParallelDim objects passed as arguments to
 * the expected (non-replica) dimensions of the output, weights, and bias
 * tensors. In addition, it sets the output_ndims, kernel_ndims and bias_ndims
 * variables to the number of dimensions (including the replica dimensions) of,
 * respectively, the ouput, weights, and bias tensors.
 *
 * The number of dimensions, and dimension sizes of the output, weights, and
 * bias dimensions are set as follows. The number of dimensions of all three
 * tensors are copied from the dimensions of the input tensor. The replica
 * dimensions are not subtracted or otherwise excluded. The size of the output
 * tensor dimensions are also copied from the input tensor, with the exception
 * of the last dimension (replica dimension), which is not set, and the first
 * dimension, whose size is set equal to the out_channels member of the
 * LinearParams struct, which in turn is set by the outDim parameter of the
 * FModel::dense function. When it comes to the size of the weights dimensions,
 * the first dimension is set to have size equal to the quotient of the size of
 * the INPUT_CHANNEL dimension of the input (first dimension) and the degree
 * (number of partitions) of the same input dimension. The second dimension of
 * the the weights tensor is set equal to out_channels, just like the first
 * dimension of the output tensor. Finally, the size of the first dimension of
 * the bias tensor is also set equal to the value of out_channels.
 *
 * @param[in]   input_shape   A required argument recording the dimensions of
 * the input tensor
 * @param[out]  output_dims   An array of ParallelDim objects representing the
 * dimensions of the output tensor
 * @param[out]  output_ndims  The number of dimensions (including the replica
 * dimension(s)) of the output tensor
 * @param[out]  kernel_dims   An array of ParallelDim objects representing the
 * dimensions of the weights tensor
 * @param[out]  kernel_ndims  The number of dimensions (including the replica
 * dimension(s)) of the weights tensor
 * @param[out]  bias_dims     An array of ParallelDim objects representing the
 * dimensions of the bias tensor
 * @param[out]  bias_ndims    The number of dimensions (including the replica
 * dimension(s)) of the bias tensor
 */
void ExpertParams::calculate_nonreplica_dim_sizes(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    int *output_ndims,
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    int *kernel_ndims,
    ParallelDim bias_dims[MAX_TENSOR_DIM],
    int *bias_ndims) const {
  auto dimension_names = this->get_dimension_names(input_shape);
  int num_dims = input_shape.num_dims;

  if (output_dims != nullptr) {
    for (int i = 1; i < input_shape.num_dims - 1; i++) {
      output_dims[i].size = input_shape.dims[i].size;
    }
    output_dims[dimension_names.at(OUTPUT_CHANNEL)].size = this->out_channels;
    *output_ndims = num_dims;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_IN)].size =
        input_shape.dims[INPUT_CHANNEL].size /
        input_shape.dims[INPUT_CHANNEL].degree;
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_OUT)].size =
        this->out_channels;
    *kernel_ndims = num_dims;
  }
  if (bias_dims != nullptr) {
    bias_dims[dimension_names.at(BIAS_CHANNEL_OUT)].size = this->out_channels;
    *bias_ndims = num_dims;
  }
}

/** @brief Switch the is_replica_dim field to true in each ParallelDim of
 *         the output, weight and bias tensor, if the corresponding dimension
 *         is used to keep track of the number of replicas
 *
 * @param[in]   input_shape   A required argument recording the dimensions of
 * the input tensor
 * @param[out]  output_dims   An array of ParallelDim objects representing the
 * dimensions of the output tensor
 * @param[out]  kernel_dims   An array of ParallelDim objects representing the
 * dimensions of the weights tensor
 * @param[out]  bias_dims     An array of ParallelDim objects representing the
 * dimensions of the bias tensor
 *
 */
void ExpertParams::mark_replica_dims(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    ParallelDim bias_dims[MAX_TENSOR_DIM]) const {
  int num_dims = input_shape.num_dims;
  auto dimension_names = this->get_dimension_names(input_shape);
  if (output_dims != nullptr) {
    output_dims[dimension_names.at(OUTPUT_REPLICA)].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    for (int i = 2; i < num_dims; i++) {
      kernel_dims[i].is_replica_dim = true;
    }
  }
  if (bias_dims != nullptr) {
    for (int i = 1; i < num_dims; i++) {
      bias_dims[i].is_replica_dim = true;
    }
  }
}

void ExpertParams::construct_mappings(
    std::vector<ParallelDimMappingRecord> &mappings,
    ParallelTensorShape const &input_shape) const {
  std::unordered_map<NamedDimensions, int> dimension_names =
      this->get_dimension_names(input_shape);

  Op::construct_output_parallel_dims(
      mappings,
      {{dimension_names.at(INPUT_CHANNEL), dimension_names.at(OUTPUT_REPLICA)},
       {dimension_names.at(INPUT_REPLICA),
        dimension_names.at(OUTPUT_CHANNEL)}});
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_output_parallel_dims(mappings, i, i);
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {{dimension_names.at(INPUT_CHANNEL),
                                       dimension_names.at(KERNEL_CHANNEL_IN)},
                                      {dimension_names.at(INPUT_REPLICA),
                                       dimension_names.at(KERNEL_CHANNEL_OUT)}},
                                     0 /*input_idx*/,
                                     W1_IDX); // TODO add W2_IDX somehow!!
  // map a bunch of replica dimensions for the unnamed dimensions in the input
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, W1_IDX); // TODO add W2_IDX somehow!!
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {
                                         {dimension_names.at(INPUT_REPLICA),
                                          dimension_names.at(BIAS_CHANNEL_OUT)},
                                     },
                                     0 /*input_idx*/,
                                     BIAS_IDX);
  for (int i = 0; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, BIAS_IDX);
  }
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ExpertParams>::operator()(
    FlexFlow::ExpertParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.out_channels);
  hash_combine(key, params.use_bias);
  hash_combine(key, params.data_type);
  hash_combine(key, params.activation);
  hash_combine(key, params.kernel_reg_type);
  hash_combine(key, params.kernel_reg_lambda);
  hash_combine(key, params.quantization_type);
  hash_combine(key, params.offload);
  return key;
}
}; // namespace std
