#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace learning_augmented_ds 
{

    void approximate_undirected_ds(
        const std::unordered_set<uint64_t>& predictions, 
        const std::vector<std::vector<uint64_t>>& adjacency_list, 
        float epsilon) {
        
        uint64_t n = adjacency_list.size(); // number of nodes in the graph
        float augmented_size = epsilon / (1 - epsilon); // number of nodes to be added to the predicted set
        std::vector<std::pair<uint64_t, uint64_t>> t_vector; // vector to store the count of neighbors in S

        std::unordered_map<uint64_t, uint64_t> t;

        for (uint64_t i = 0; i < n; ++i) {
            for (const auto& neighbor : adjacency_list[i]) {
                if (predictions.find(neighbor) != predictions.end()) {
                    t[i] += 1;
                }
            }
        }

        for (const auto& [node, count] : t) {
            t_vector.emplace_back(node, count);
        }

        // Sort in descending order based on the count of neighbors in S
        std::sort(t_vector.begin(), t_vector.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        return;
    }

    void approximate_directed_ds(const std::vector<bool>& predictions_1, 
                                const std::vector<bool>& predictions_2,
                                const std::vector<std::vector<int>>& incoming, 
                                const std::vector<std::vector<int>>& outgoing, float epsilon) {
        return;
    }
}