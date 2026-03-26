#pragma once

#include <vector>
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"

namespace fc::algorithms {

    /**
     * @brief Performs a brute-force radius search for Array of Structures (AoS) datasets.
     * * This implementation has a linear time complexity of O(N). It serves as a robust
     * baseline for performance benchmarking against spatial indexing structures like KD-trees.
     * * @tparam T The coordinate data type (e.g., float, double).
     * @tparam Dim The spatial dimensionality.
     * @tparam Metric A distance metric policy satisfying the fc::metrics::ScalarMetric concept.
     * * @param dataset The contiguous AoS input dataset to search.
     * @param query The reference point for proximity calculation.
     * @param radius The search threshold distance.
     * @return std::vector<std::size_t> A list of indices for points within the specified radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Metric>
        requires fc::metrics::ScalarMetric<Metric, T, Dim>
    auto radius_search_brute_force_aos(
        const fc::DatasetAoS<T, Dim>& dataset,
        const fc::PointAoS<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        std::vector<std::size_t> indices;

        // Pre-allocate memory to minimize heap reallocations. 
        // Note: The heuristic (1% of dataset) can be tuned based on expected data density.
        indices.reserve(dataset.size() / 100 + 1);

        // Compile-time optimization: If the Metric is Euclidean, we compare squared distances 
        // to bypass the computationally expensive square root (sqrt) operation.
        T effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
            effective_radius = radius * radius;
        }

        for (std::size_t i = 0; i < dataset.size(); ++i) {
            T dist;

            // Resolve the metric logic at compile-time to maintain zero-overhead abstraction.
            if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoS>) {
                dist = fc::metrics::SquaredEuclideanAoS::evaluate(dataset[i], query);
            }
            else {
                dist = Metric::evaluate(dataset[i], query);
            }

            if (dist <= effective_radius) {
                indices.push_back(i);
            }
        }

        return indices;
    }

} // namespace fc::algorithms