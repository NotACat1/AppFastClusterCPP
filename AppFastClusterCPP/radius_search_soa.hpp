#pragma once

#include <vector>
#include <omp.h>
#include "dataset_soa.hpp"
#include "metrics_soa.hpp"

namespace fc::algorithms {

    /**
     * @brief High-performance brute-force radius search for SoA datasets.
     * * This implementation is optimized for modern CPU architectures by leveraging:
     * 1. Structure-of-Arrays (SoA) layout to facilitate SIMD auto-vectorization.
     * 2. Cache-friendly sequential memory access patterns.
     * 3. Multi-threaded execution via OpenMP with thread-local accumulation.
     * * @tparam T        Coordinate scalar type (e.g., float, double).
     * @tparam Dim      Spatial dimensionality.
     * @tparam Policy   Distance calculation policy (must satisfy MetricSoA concept).
     * * @param dataset   Input dataset in SoA format.
     * @param query     Target query point coordinates.
     * @param radius    Search radius threshold.
     * @return std::vector<std::size_t> Indices of points within the specified radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Policy>
        requires fc::metrics::MetricSoA<Policy, T>
    auto radius_search_brute_force_soa(
        const fc::DatasetSoA<T, Dim>& dataset,
        const std::array<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        const std::size_t n = dataset.size();
        if (n == 0) return {};

        // 1. Distance Computation Phase
        // The SoA layout ensures that coordinates are contiguous in memory, allowing 
        // the compiler to generate efficient SIMD instructions (AVX/AVX-512/NEON).
        std::vector<T> distances(n);
        fc::metrics::compute_distances_soa<Policy>(query, dataset, distances);

        // 2. Metric Optimization (Squared Space Comparison)
        // To maximize instruction throughput, we perform comparisons in squared space 
        // when using Euclidean metrics to avoid the overhead of square root operations.
        T effective_radius = radius;
        if constexpr (std::is_same_v<Policy, fc::metrics::EuclideanSoA>) {
            effective_radius = radius * radius;
            // Implementation Note: If 'compute_distances_soa' returns non-squared L2,
            // ensure the policy is switched to SquaredEuclidean for optimal performance.
        }

        // 3. Parallel Index Collection (Filtering)
        // We utilize a thread-local accumulation strategy to minimize synchronization 
        // overhead and prevent false sharing on the global results vector.
        std::vector<std::size_t> final_indices;

#pragma omp parallel
        {
            std::vector<std::size_t> local_indices;

            // Heuristic pre-allocation based on an assumed uniform distribution 
            // to mitigate frequency of heap reallocations within the parallel region.
            local_indices.reserve(n / (omp_get_num_threads() * 2));

#pragma omp for nowait
            for (std::size_t i = 0; i < n; ++i) {
                if (distances[i] <= effective_radius) {
                    local_indices.push_back(i);
                }
            }

            // Consolidate thread-local results into the shared output container.
            // Using a critical section here is efficient as it is invoked only 
            // once per thread after the main processing loop completes.
#pragma omp critical
            {
                final_indices.insert(final_indices.end(), local_indices.begin(), local_indices.end());
            }
        }

        return final_indices;
    }
} // namespace fc::algorithms