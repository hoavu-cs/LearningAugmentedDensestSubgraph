// learning_augmented_ds.hpp

#pragma once

#include <vector>

namespace learning_augmented_ds {

    // Approximates the densest subgraph in an undirected graph.
    // - predictions: predictions whether a node is in the densest subgraph
    // - adjacency_list: adjacency list of the undirected graph
    // - epsilon: paramter that reflects the recall of the model
    void approximate_undirected_ds(
        const std::vector<float>& predictions,
        const std::vector<std::vector<int>>& adjacency_list,
        float epsilon
    );

    // Approximates the densest subgraph in a directed graph.
    // - predictions: predicted importance score for each node
    // - incoming: list of incoming neighbors for each node
    // - outgoing: list of outgoing neighbors for each node
    // - epsilon: parameter that reflects the recall of the model
    void approximate_directed_ds(
        const std::vector<float>& predictions,
        const std::vector<std::vector<int>>& incoming,
        const std::vector<std::vector<int>>& outgoing,
        float epsilon
    );

} // namespace learning_augmented_ds
