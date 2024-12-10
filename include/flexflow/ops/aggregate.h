#ifndef _FLEXFLOW_AGGREGATE_H_
#define _FLEXFLOW_AGGREGATE_H_

#include "flexflow/inference.h"
#include "flexflow/model.h"
#include "flexflow/ops/aggregate_params.h"

namespace FlexFlow {

#define AGGREGATE_MAX_K 4
#define AGGREGATE_MAX_BATCH_SIZE 64
#define AGGREGATE_MAX_N 128

class Aggregate;

class AggregateMeta : public OpMeta {
public:
  AggregateMeta(FFHandler handle,
                Aggregate const *aggr,
                MemoryAllocator &gpu_mem_allocator);
  ~AggregateMeta(void);
  float **dev_exp_preds;
  float **dev_exp_grads;

public:
  Realm::RegionInstance reserveInst;
  // PEFT related fields
  void *input_activation;
  size_t allocated_peft_buffer_size = 0;
};

class Aggregate : public Op {
public:
  using Params = AggregateParams;
  using Input = std::vector<ParallelTensor>;
  Aggregate(FFModel &model,
            LayerID const &_layer_guid,
            ParallelTensor const *inputs,
            int _n,
            float _lambda_bal,
            char const *name = nullptr);
  Aggregate(FFModel &model,
            Aggregate const &other,
            std::vector<ParallelTensor> const &inputs);
  Aggregate(FFModel &model,
            Params const &params,
            Input const &inputs,
            char const *name = nullptr);
  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_kernel_wrapper(AggregateMeta const *m,
                                     float **exp_preds,
                                     int const *acc_gate_assign_ptr,
                                     float const *acc_gate_pred_ptr,
                                     float *acc_output_ptr,
                                     int n,
                                     int const k,
                                     int rows,
                                     int const batch_size,
                                     int out_dim);
  static void inference_kernel_wrapper(AggregateMeta const *m, // TODO never actually defined, I think
                                   float **exp_preds,
                                   int const *acc_gate_assign_ptr,
                                   float const *acc_gate_pred_ptr,
                                   float *acc_output_ptr,
                                   int n,
                                   int const k,
                                   int rows,
                                   int const batch_size,
                                   int out_dim);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void backward_kernel_wrapper(AggregateMeta const *m,
                                      float **exp_preds,
                                      float **exp_grads,
                                      int const *acc_gate_assign_ptr,
                                      int const *acc_true_gate_assign_ptr,
                                      float const *acc_gate_pred_ptr,
                                      float *full_acc_gate_grad_ptr,
                                      float const *acc_output_grad_ptr,
                                      int n,
                                      int const k,
                                      int rows,
                                      float lambda_bal,
                                      int const batch_size,
                                      int out_dim);
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               Input const &inputs,
                               int num_inputs);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &mv,
                             CostMetrics &cost_metrics) const override;
  Params get_params() const;

public:
  int n;
  float lambda_bal;
};

}; // namespace FlexFlow

#endif
