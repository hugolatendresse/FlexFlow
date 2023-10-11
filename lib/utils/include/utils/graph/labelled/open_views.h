#ifndef _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H
#define _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H

#include "standard_labelled.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"
#include "output_labelled_open.h"

namespace FlexFlow {

template <typename SubgraphView, typename NodeLabel, typename EdgeLabel>
struct OutputLabelledOpenMultiDiSubgraphView :
  virtual IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {

  OutputLabelledOpenMultiDiSubgraphView(OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> const &g, std::unordered_set<Node> const &nodes)
    : g(g), nodes(nodes) {}

  NodeLabel const &at(Node const &n) const override {
    return g.at(n);
  }

  EdgeLabel const &at(MultiDiOutput const &o) const override {
    return g.at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return SubgraphView(g, nodes).query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &q) const override {
    return SubgraphView(g, nodes).query_edges(q);
  }

private:
  OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> const &g;
  std::unordered_set<Node> const &nodes;
};

}

#endif