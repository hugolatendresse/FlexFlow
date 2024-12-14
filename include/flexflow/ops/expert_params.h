#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ExpertParams {
  LayerID layer_guid;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(ExpertParams const &, ExpertParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ExpertParams> {
  size_t operator()(FlexFlow::ExpertParams const &) const;
};
} // namespace std
