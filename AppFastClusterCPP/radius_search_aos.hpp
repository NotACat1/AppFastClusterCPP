#pragma once

#include <vector>
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a linear-scan (brute-force) radius search on an AoS dataset.
     * * @details This function evaluates the distance between the query point and every
     * point in the dataset, yielding an O(N) time complexity. While not asymptotically
     * optimal for massive datasets, its contiguous memory access pattern and zero-overhead
     * metric dispatch make it highly performant for small-to-medium datasets. It also
     * serves as a reliable ground-truth baseline for validating spatial indices
     * (e.g., KD-Trees, Octrees, or BVH).
     * * @tparam T The numeric coordinate type (e.g., float, double).
     * @tparam Dim The spatial dimensionality of the points.
     * @tparam Metric The distance metric policy (must satisfy the MetricAoS concept).
     * * @param dataset The contiguous Array of Structures (AoS) to search.
     * @param query The reference point defining the center of the search sphere.
     * @param radius The maximum inclusive distance threshold.
     * @return std::vector<std::size_t> Indices of all points falling within the radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric>
        requires fc::metrics::MetricAoS<Metric, T, Dim>
    auto radius_search_brute_force_aos(
        const fc::DatasetAoS<T, Dim>& dataset,
        const fc::PointAoS<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        std::vector<std::size_t> indices;

        // Memory Optimization: Pre-allocate capacity to prevent dynamic buffer 
        // reallocations during the hot loop. 
        // @note The 1% capacity heuristic assumes a relatively sparse return set. 
        // For highly dense point clouds, consider exposing this as a tuning parameter.
        indices.reserve(dataset.size() / 100 + 1);

        // Algorithmic Optimization: If the caller requested standard Euclidean distance, 
        // we intercept it at compile-time and switch to Squared Euclidean logic. 
        // This avoids executing a costly hardware square root instruction per point, 
        // while preserving the user's expected API (passing a linear radius).
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        // Cache the size locally to prevent repeated function call overhead
        const std::size_t data_size = dataset.size();

        for (std::size_t i = 0; i < data_size; ++i) {
            T dist;

            // Zero-overhead dispatch: The compiler evaluates this branch at compile-time 
            // and completely eliminates the dead code path. This ensures only the optimal 
            // metric instruction set is evaluated in the inner loop.
            if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
                dist = fc::metrics::SquaredEuclideanAoS::evaluate(dataset[i], query);
            }
            else {
                dist = Metric::evaluate(dataset[i], query);
            }

            // Append the index if the point lies within the bounded hypersphere
            if (dist <= effective_radius) {
                indices.push_back(i);
            }
        }

        return indices;
    }

} // namespace fc::algorithms