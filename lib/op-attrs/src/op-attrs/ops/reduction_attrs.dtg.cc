// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/reduction_attrs.struct.toml
/* proj-data
{
  "generated_from": "1d2b5b7cf11ed04a27a6fd8215e4e2a5"
}
*/

#include "op-attrs/ops/reduction_attrs.dtg.h"

#include <sstream>

namespace FlexFlow {
ReductionAttrs::ReductionAttrs(int const &reduction_degree)
    : reduction_degree(reduction_degree) {}
bool ReductionAttrs::operator==(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) == std::tie(other.reduction_degree);
}
bool ReductionAttrs::operator!=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) != std::tie(other.reduction_degree);
}
bool ReductionAttrs::operator<(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) < std::tie(other.reduction_degree);
}
bool ReductionAttrs::operator>(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) > std::tie(other.reduction_degree);
}
bool ReductionAttrs::operator<=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) <= std::tie(other.reduction_degree);
}
bool ReductionAttrs::operator>=(ReductionAttrs const &other) const {
  return std::tie(this->reduction_degree) >= std::tie(other.reduction_degree);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReductionAttrs>::operator()(
    FlexFlow::ReductionAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<int>{}(x.reduction_degree) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ReductionAttrs
    adl_serializer<FlexFlow::ReductionAttrs>::from_json(json const &j) {
  return {j.at("reduction_degree").template get<int>()};
}
void adl_serializer<FlexFlow::ReductionAttrs>::to_json(
    json &j, FlexFlow::ReductionAttrs const &v) {
  j["__type"] = "ReductionAttrs";
  j["reduction_degree"] = v.reduction_degree;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::ReductionAttrs> Arbitrary<FlexFlow::ReductionAttrs>::arbitrary() {
  return gen::construct<FlexFlow::ReductionAttrs>(gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ReductionAttrs const &x) {
  std::ostringstream oss;
  oss << "<ReductionAttrs";
  oss << " reduction_degree=" << x.reduction_degree;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ReductionAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow