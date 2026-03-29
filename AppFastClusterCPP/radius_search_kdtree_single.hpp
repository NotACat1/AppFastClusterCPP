#pragma once

#include <vector>
#include <array>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a high-performance single-threaded radius search using a flattened KD-tree.
     * * This function implements an iterative depth-first search (DFS) to locate all points
     * within a specified search radius. By utilizing a manual stack, it eliminates the
     * overhead of recursive function calls and prevents stack overflow on deep trees.
     * * @tparam T         Coordinate scalar type (e.g., float, double).
     * @tparam Dim       Spatial dimensionality of the dataset.
     * @tparam Metric    Distance calculation policy (must satisfy MetricAoS concept).
     * * @param tree       The pre-built flattened KD-tree structure.
     * @param dataset    The source dataset (AoS) referenced by the tree indices.
     * @param query      The query point used as the center of the search sphere.
     * @param radius     The search radius threshold.
     * * @return std::vector<std::size_t> Indices of points found within the search radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric>
        requires fc::metrics::MetricAoS<Metric, T, Dim>
    auto radius_search_kdtree_single(
        const fc::KDTreeFlat& tree,
        const fc::DatasetAoS<T, Dim>& dataset,
        const fc::PointAoS<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        // Early exit: handle empty tree case to avoid unnecessary processing.
        if (tree.empty()) return {};

        std::vector<std::size_t> found_indices;
        // Pre-allocate initial capacity to mitigate frequent reallocations in the hot loop.
        found_indices.reserve(64);

        /**
         * Performance Optimization:
         * For Euclidean metrics, we operate in squared distance space to avoid the
         * high computational cost of the square root (sqrt) operation.
         */
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        /**
         * Fixed-size stack for iterative traversal.
         * A stack depth of 64 is sufficient for a balanced tree containing up to 2^64 elements,
         * covering virtually any practical dataset size.
         */
        int32_t stack[64];
        int32_t stack_ptr = 0;

        // Start traversal from the root node.
        stack[stack_ptr++] = static_cast<int32_t>(tree.root_idx);

        while (stack_ptr > 0) {
            int32_t node_idx = stack[--stack_ptr];
            const auto& node = tree.nodes[node_idx];

            // 1. Point Evaluation
            // Calculate distance from the query to the point stored at the current node.
            T dist;
            if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
                // Compile-time dispatch to SquaredEuclidean for Euclidean search.
                dist = fc::metrics::SquaredEuclideanAoS::evaluate(dataset[node.point_idx], query);
            }
            else {
                dist = Metric::evaluate(dataset[node.point_idx], query);
            }

            if (dist <= effective_radius) {
                found_indices.push_back(node.point_idx);
            }

            // 2. Traversal & Pruning
            // Determine the signed distance from the query point to the splitting hyperplane.
            T diff = query[node.split_dim] - node.split_val;

            // Heuristic: Identify "near" and "far" branches relative to the query position.
            int32_t near_child = (diff <= 0) ? node.left_child : node.right_child;
            int32_t far_child = (diff <= 0) ? node.right_child : node.left_child;

            /**
             * Branch Pruning Logic:
             * We only traverse the "far" child if the search hypersphere intersects
             * the splitting hyperplane. This significantly reduces the search space.
             */
            if (far_child != -1) {
                // Compute the squared distance to the splitting plane.
                T plane_dist_sq = diff * diff;
                if (plane_dist_sq <= effective_radius) {
                    stack[stack_ptr++] = far_child;
                }
            }

            // Always push the "near" child last to ensure it is processed first (LIFO order),
            // which improves the chances of early pruning in other search types.
            if (near_child != -1) {
                stack[stack_ptr++] = near_child;
            }
        }

        return found_indices;
    }
} // namespace fc::algorithms