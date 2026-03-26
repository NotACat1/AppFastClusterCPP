#pragma once

#include <vector>
#include <array>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a single-threaded radius search using a flattened KD-tree.
     * * This function implements an iterative depth-first search (DFS) to find all points
     * within a specified radius. It utilizes a manual stack to avoid recursion overhead
     * and includes performance optimizations for Euclidean distance metrics.
     *
     * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @tparam Metric Distance calculation policy (e.g., fc::metrics::EuclideanAoS).
     *
     * @param tree The pre-built flattened KD-tree structure.
     * @param dataset The original Array-of-Structures (AoS) dataset referenced by the tree.
     * @param query The query point for which neighbors are sought.
     * @param radius The search radius.
     * @return std::vector<std::size_t> A list of point indices located within the search radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric>
        requires fc::metrics::ScalarMetric<Metric, T, Dim>
    auto radius_search_kdtree_single(
        const fc::KDTreeFlat& tree,
        const fc::DatasetAoS<T, Dim>& dataset,
        const fc::PointAoS<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        if (tree.empty()) return {};

        std::vector<std::size_t> found_indices;
        // Initial buffer reservation to minimize reallocations during traversal
        found_indices.reserve(64);

        // Optimization for Euclidean metrics: compute comparisons in squared space
        // to eliminate expensive square root operations.
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        // Thread-local stack for iterative tree traversal.
        // A depth of 64 is sufficient for balanced trees containing up to $2^{64}$ points.
        int32_t stack[64];
        int32_t stack_ptr = 0;

        stack[stack_ptr++] = static_cast<int32_t>(tree.root_idx);

        while (stack_ptr > 0) {
            int32_t node_idx = stack[--stack_ptr];
            const auto& node = tree.nodes[node_idx];

            // 1. Evaluate distance from the query to the point stored in the current node
            T dist;
            if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
                // Directly use SquaredEuclidean for performance when the policy is Euclidean
                dist = fc::metrics::SquaredEuclideanAoS::evaluate(dataset[node.point_idx], query);
            }
            else {
                dist = Metric::evaluate(dataset[node.point_idx], query);
            }

            if (dist <= effective_radius) {
                found_indices.push_back(node.point_idx);
            }

            // 2. Determine branch traversal order based on the splitting hyperplane
            T diff = query[node.split_dim] - node.split_val;

            // Identify "near" and "far" child nodes relative to the query position
            int32_t near_child = (diff <= 0) ? node.left_child : node.right_child;
            int32_t far_child = (diff <= 0) ? node.right_child : node.left_child;

            // Branch Pruning: Only add the "far" child to the stack if the search sphere
            // actually intersects the splitting hyperplane.
            if (far_child != -1) {
                // Calculate distance to the plane (squared for L2 metrics)
                T plane_dist_sq = diff * diff;
                if (plane_dist_sq <= effective_radius) {
                    stack[stack_ptr++] = far_child;
                }
            }

            // Add the "near" child to the stack last so that it is processed next (LIFO)
            if (near_child != -1) {
                stack[stack_ptr++] = near_child;
            }
        }

        return found_indices;
    }

} // namespace fc::algorithms