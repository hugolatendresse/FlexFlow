#ifndef _FLEXFLOW_LORA_LINEAR_FIRST_H
#define _FLEXFLOW_LORA_LINEAR_FIRST_H

#include "flexflow/inference.h"
#include "flexflow/node.h"
#include "flexflow/operator.h"
#include "flexflow/ops/lora_linear_params.h"
#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

class FFModel;
class Layer;

class LoraLinear : public Op {
public:
  using Params = LoraLinearParams;
  using Input = std::pair<ParallelTensor, ParallelTensor>;

  LoraLinear(FFModel &model,
             LayerID const &layer_guid,
             ParallelTensor const input,
             ParallelTensor const output,
             int rank,
             DataType _data_type,
             bool allocate_weights,
             char const *name);
  LoraLinear(FFModel &model,
             LoraLinear const &other,
             ParallelTensor const input,
             ParallelTensor const output,
             bool allocate_weights);
  LoraLinear(FFModel &model,
             Params const &params,
             Input const &inputs,
             bool allocate_weights = false,
             char const *name = nullptr);

  void init(FFModel const &) override;
  void init_inference(FFModel const &,
                      std::vector<ParallelTensor> const &,
                      std::vector<ParallelTensor> const &,
                      MachineView const *mv = nullptr) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  Legion::FutureMap inference(FFModel const &,
                              BatchConfigFuture const &,
                              std::vector<ParallelTensor> const &,
                              std::vector<ParallelTensor> const &,
                              MachineView const *mv = nullptr) override;
  Legion::FutureMap peft_bwd(FFModel const &,
                             BatchConfigFuture const &,
                             std::vector<ParallelTensor> const &,
                             std::vector<ParallelTensor> const &,
                             MachineView const *mv = nullptr) override;
  void print_layer(FFModel const &model) override;
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void inference_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void peft_bwd_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void serialize(Legion::Serializer &) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);

  // size_t get_params_hash() const override;
  LoraLinearParams get_params() const;

private:
  LoraLinear(int guid,
             bool profiling,
             ParallelTensor const input,
             ParallelTensor const output,
             int rank,
             bool allocate_weights,
             char const *name);

  void register_mappings();
  void register_output_mappings();
  void register_weight_mappings();

public:
  int rank;
};

}; // namespace FlexFlow

#endif // _FLEXLOW_LORA_LINEAR_FIRST_H
