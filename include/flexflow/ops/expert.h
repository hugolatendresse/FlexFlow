#pragma once

#include "flexflow/batch_config.h"
#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/utils/memory_allocator.h"
#include "flexflow/ops/expert_params.h"


namespace FlexFlow {

class ExpertMeta;

class Expert : public Op {
public:
  using Params = ExpertParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;
  Expert(FFModel &model,
               Expert const &other,
               const ParallelTensor input,
               bool allocate_weights);
  Expert(FFModel &model,
                 ExpertParams const &params,
                 ParallelTensor const input,
                 char const *name,
                 bool allocate_weights);
  Expert(FFModel &model,
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
                 char const *name);

  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap peft_bwd(FFModel const &,
                             BatchConfigFuture const &,
                             std::vector<ParallelTensor> const &,
                             std::vector<ParallelTensor> const &,
                             MachineView const *mv = nullptr) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  ExpertParams get_params() const;

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  bool estimate_sync_cost(Simulator *sim,
                        MachineView const &pc,
                        CostMetrics &cost_metrics) const override;
  ParallelConfig get_random_parallel_config(FFModel const &ff) const override;
  bool is_valid_parallel_config(FFModel const &ff,
                                ParallelConfig const &pc) const override;


  template <typename T>
  static void inference_kernel(ExpertMeta const *m,
                               int num_elements,
                               T const *input1_ptr,
                               T const *input2_ptr,
                               T *output_ptr,
                               ffStream_t stream);
  static void inference_kernel_wrapper(ExpertMeta *m,
                                       BatchConfig const *bc,
                                       GenericTensorAccessorR const &input1,
                                       GenericTensorAccessorR const &input2,
                                       GenericTensorAccessorW const &output);
  static void
      backward_kernel_wrapper(ExpertMeta const *m,
                              GenericTensorAccessorR const &output_grad,
                              GenericTensorAccessorR const &input1,
                              GenericTensorAccessorR const &input2,
                              GenericTensorAccessorW const &input1_grad,
                              GenericTensorAccessorW const &input2_grad);
  static void
      peft_bwd_kernel_wrapper(ExpertMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &output_grad,
                              GenericTensorAccessorW const &input1_grad,
                              GenericTensorAccessorW const &input2_grad);

public:
  int in_channels, out_channels;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias;
  ParallelTensor replica;
  DataType quantization_type;
  bool offload;

};

class ExpertMeta : public OpMeta {
public:
  ExpertMeta(FFHandler handle,
             int batch_size,
             Linear const *li,
             MemoryAllocator gpu_mem_allocator,
             int weightSize);
  ~ExpertMeta(void);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
    void *one_ptr;
  void *weight_ptr;
  DataType weight_ptr_type;
  DataType quantization_type;
  bool offload;
  char *quantized_weight_ptr;
  size_t quantized_weightSize;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias, add_bias_only_once;
  Realm::RegionInstance reserveInst;
  // PEFT related fields
  void *input_activation;
  void *output_activation_buffer;
  size_t allocated_peft_buffer_size = 0;
};

}; // namespace FlexFlow
