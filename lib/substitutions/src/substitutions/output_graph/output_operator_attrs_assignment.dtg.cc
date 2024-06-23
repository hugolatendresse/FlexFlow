// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/output_graph/output_operator_attrs_assignment.struct.toml
/* proj-data
{
  "generated_from": "bbfb309c5a39a729da23dace4df4a9de"
}
*/

#include "substitutions/output_graph/output_operator_attrs_assignment.dtg.h"

#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/output_graph/output_operator_attribute_expr.dtg.h"
#include <sstream>
#include <unordered_map>

namespace FlexFlow {
OutputOperatorAttrsAssignment::OutputOperatorAttrsAssignment(
    std::unordered_map<::FlexFlow::OperatorAttributeKey,
                       ::FlexFlow::OutputOperatorAttributeExpr> const
        &assignments)
    : assignments(assignments) {}
bool OutputOperatorAttrsAssignment::operator==(
    OutputOperatorAttrsAssignment const &other) const {
  return std::tie(this->assignments) == std::tie(other.assignments);
}
bool OutputOperatorAttrsAssignment::operator!=(
    OutputOperatorAttrsAssignment const &other) const {
  return std::tie(this->assignments) != std::tie(other.assignments);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::OutputOperatorAttrsAssignment>::operator()(
    FlexFlow::OutputOperatorAttrsAssignment const &x) const {
  size_t result = 0;
  result ^=
      std::hash<std::unordered_map<::FlexFlow::OperatorAttributeKey,
                                   ::FlexFlow::OutputOperatorAttributeExpr>>{}(
          x.assignments) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace FlexFlow {
std::string format_as(OutputOperatorAttrsAssignment const &x) {
  std::ostringstream oss;
  oss << "<OutputOperatorAttrsAssignment";
  oss << " assignments=" << x.assignments;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s,
                         OutputOperatorAttrsAssignment const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow