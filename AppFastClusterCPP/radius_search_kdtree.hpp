#pragma once

#include <vector>
#include <array>
#include <omp.h>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a batch radius search using a flattened KD-tree and OpenMP parallelism.
     * * This function iterates over a batch of query points, finding all neighbors within
     * a specified radius for each. It utilizes an iterative stack-based traversal to
     * avoid recursion overhead and leverages OpenMP for multi-threaded execution.
     * * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Dimensionality of the space.
     * @tparam Metric Distance calculation policy (e.g., SquaredEuclidean).
     * @tparam Alloc Allocator type for the input query vector.
     * * @param tree The pre-built flattened KD-tree.
     * @param dataset The original Array-of-Structures (AoS) dataset referenced by the tree.
     * @param queries A batch of points to perform the search for.
     * @param radius The search radius.
     * @return std::vector<std::vector<std::size_t>> A vector of neighbor indices for each query point.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric, typename Alloc>
        requires fc::metrics::ScalarMetric<Metric, T, Dim>
    auto radius_search_kdtree_batch(
        const fc::KDTreeFlat& tree,
        const fc::DatasetAoS<T, Dim>& dataset,
        const std::vector<fc::PointAoS<T, Dim>, Alloc>& queries,
        T radius
    ) -> std::vector<std::vector<std::size_t>>
    {
        if (tree.empty() || queries.empty()) return {};

        // Prepare results container for each query point
        std::vector<std::vector<std::size_t>> all_results(queries.size());

        // Optimization: Use squared distance for Euclidean metrics to avoid expensive sqrt() calls.
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        // dynamic scheduling is used because different queries may traverse 
        // different numbers of nodes, leading to potential load imbalance.
#pragma omp parallel for schedule(dynamic, 16)
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const auto& query = queries[i];
            auto& local_result = all_results[i];

            // Heuristic reserve to minimize reallocations within the parallel loop.
            local_result.reserve(128);

            // Thread-local stack for iterative tree traversal (prevents stack overflow).
            // A depth of 64 is sufficient for trees containing billions of points (depth ~ log2(N)).
            int32_t stack[64];
            int32_t stack_ptr = 0;

            stack[stack_ptr++] = static_cast<int32_t>(tree.root_idx);

            while (stack_ptr > 0) {
                int32_t node_idx = stack[--stack_ptr];
                const auto& node = tree.nodes[node_idx];

                // 1. Evaluate distance to the point stored in the current node.
                T dist;
                if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
                    dist = fc::metrics::SquaredEuclideanAoS::evaluate(dataset[node.point_idx], query);
                }
                else {
                    dist = Metric::evaluate(dataset[node.point_idx], query);
                }

                if (dist <= effective_radius) {
                    local_result.push_back(node.point_idx);
                }

                // 2. Branch traversal logic (splitting by hyperplane).
                T axis_diff = query[node.split_dim] - node.split_val;

                // Determine which child node is closer to the query point.
                int32_t first_child = (axis_diff <= 0) ? node.left_child : node.right_child;
                int32_t second_child = (axis_diff <= 0) ? node.right_child : node.left_child;

                // Always push the closer branch to the stack last to ensure it is processed first.
                // (This is a depth-first search optimization).
                if (first_child != -1) {
                    stack[stack_ptr++] = first_child;
                }

                // Only check the opposite branch if the search sphere intersects the splitting hyperplane.
                // For L2 (Euclidean) metrics, the intersection condition is: (axis_difference)^2 <= R^2.
                T axis_dist_sq = axis_diff * axis_diff;
                if (second_child != -1 && axis_dist_sq <= effective_radius) {
                    stack[stack_ptr++] = second_child;
                }
            }
        }

        return all_results;
    }

} // namespace fc::algorithms