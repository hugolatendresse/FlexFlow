// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/unlabelled_graph_pattern.struct.toml
/* proj-data
{
  "generated_from": "f494ed79eb1ba4010155e456b452157f"
}
*/

#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"

#include "utils/graph.h"

namespace FlexFlow {
UnlabelledGraphPattern::UnlabelledGraphPattern(
    ::FlexFlow::OpenMultiDiGraphView const &raw_graph)
    : raw_graph(raw_graph) {}
} // namespace FlexFlow