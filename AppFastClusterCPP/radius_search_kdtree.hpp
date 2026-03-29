#pragma once

#include <vector>
#include <array>
#include <omp.h>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a high-performance batch radius search using a flattened KD-tree.
     * * This function executes a spatial query to find all points within a given distance
     * from each query point. It leverages OpenMP for multi-threaded processing and
     * implements an iterative traversal to maximize CPU cache efficiency and avoid
     * recursion overhead.
     * * @tparam T           Scalar type for coordinates (e.g., float, double).
     * @tparam Dim         Number of spatial dimensions.
     * @tparam Metric      Distance calculation policy (must satisfy MetricAoS concept).
     * @tparam Alloc       Allocator type for the query vector.
     * * @param tree         The pre-built flattened KD-tree structure.
     * @param dataset      The source dataset (AoS) referenced by the tree indices.
     * @param queries      A collection of points to perform the search for.
     * @param radius       The search radius (threshold).
     * * @return A nested vector where each entry contains indices of neighbors for the corresponding query.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric, typename Alloc>
        requires fc::metrics::MetricAoS<Metric, T, Dim>
    auto radius_search_kdtree_batch(
        const fc::KDTreeFlat& tree,
        const fc::DatasetAoS<T, Dim>& dataset,
        const std::vector<fc::PointAoS<T, Dim>, Alloc>& queries,
        T radius
    ) -> std::vector<std::vector<std::size_t>>
    {
        // Early exit for empty inputs to prevent unnecessary allocations or processing.
        if (tree.empty() || queries.empty()) return {};

        // Pre-allocate the results container. Inner vectors will be allocated dynamically.
        std::vector<std::vector<std::size_t>> all_results(queries.size());

        /**
         * Optimization: Use squared distance for Euclidean metrics to avoid
         * computationally expensive square root operations during every comparison.
         */
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        /**
         * Parallelize query processing using OpenMP.
         * Dynamic scheduling is preferred here because KD-tree traversal depth varies
         * depending on the local point density around each query, which can cause load imbalance.
         */
#pragma omp parallel for schedule(dynamic, 16)
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const auto& query = queries[i];
            auto& local_result = all_results[i];

            // Pre-reserve memory to reduce the number of heap reallocations within the hot loop.
            local_result.reserve(128);

            /**
             * Fixed-size thread-local stack for iterative DFS traversal.
             * A stack size of 64 is sufficient for datasets with up to 2^64 points,
             * effectively covering any practical use case.
             */
            int32_t stack[64];
            int32_t stack_ptr = 0;

            // Initialize traversal from the tree root.
            stack[stack_ptr++] = static_cast<int32_t>(tree.root_idx);

            while (stack_ptr > 0) {
                int32_t node_idx = stack[--stack_ptr];
                const auto& node = tree.nodes[node_idx];

                // 1. Calculate distance from query to the current node's point.
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

                // 2. Pruning and Branch Selection.
                // Determine the signed distance from the query to the splitting hyperplane.
                T axis_diff = query[node.split_dim] - node.split_val;

                // Identify the near and far child nodes relative to the query point.
                int32_t first_child = (axis_diff <= 0) ? node.left_child : node.right_child;
                int32_t second_child = (axis_diff <= 0) ? node.right_child : node.left_child;

                /**
                 * Push the 'near' branch to the stack. Note: To process the closer branch first,
                 * it should be pushed onto the stack AFTER the farther branch.
                 */
                if (first_child != -1) {
                    stack[stack_ptr++] = first_child;
                }

                /**
                 * Pruning logic: Only traverse the 'far' branch if the hypersphere defined
                 * by the search radius intersects the splitting hyperplane.
                 */
                T axis_dist_sq = axis_diff * axis_diff;
                if (second_child != -1 && axis_dist_sq <= effective_radius) {
                    stack[stack_ptr++] = second_child;
                }
            }
        }

        return all_results;
    }
} // namespace fc::algorithms