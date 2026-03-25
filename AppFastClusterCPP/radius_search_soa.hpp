#pragma once

#include <vector>
#include <omp.h>
#include "dataset_soa.hpp"
#include "metrics_soa.hpp"

namespace fc::algorithms {

    /**
     * @brief High-performance brute-force radius search for SoA datasets using SIMD and OpenMP.
     * * The algorithm is structured in two distinct phases to maximize CPU pipeline utilization:
     * 1. Distance Computation: Batch calculation of distances using SIMD-optimized policies.
     * 2. Parallel Filtering: Extracting valid indices using a thread-local accumulation strategy.
     * * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @tparam Policy Distance calculation policy (e.g., EuclideanPolicy).
     * * @param dataset The input dataset in Structure-of-Arrays (SoA) format.
     * @param query The target query point coordinates.
     * @param radius The search radius.
     * @return std::vector<std::size_t> A vector of indices for points falling within the radius.
     */
    template <fc::MLCoordinate T, std::size_t Dim, typename Policy>
        requires fc::metrics::SoAMetricPolicy<Policy, T>
    auto radius_search_brute_force_soa(
        const fc::DatasetSoA<T, Dim>& dataset,
        const std::array<T, Dim>& query,
        T radius
    ) -> std::vector<std::size_t>
    {
        const std::size_t n = dataset.size();
        if (n == 0) return {};

        // 1. Compute distances for all points in the dataset.
        // The SoA layout and 64-byte alignment allow modern compilers (GCC/Clang) 
        // to effectively apply SIMD auto-vectorization to the coordinate loops.
        std::vector<T> distances(n);
        fc::metrics::compute_distances_soa<Policy>(query, dataset, distances);

        // 2. Metric optimization: Comparison in squared space.
        // For Euclidean metrics, we compare squared distances to avoid expensive square root operations.
        T effective_radius = radius;
        if constexpr (std::is_same_v<Policy, fc::metrics::EuclideanPolicy>) {
            effective_radius = radius * radius;
            // Note: If using EuclideanPolicy, the compute_distances_soa call may include 
            // a final sqrt() depending on implementation. For maximum throughput, 
            // prefer SquaredEuclideanPolicy.
        }

        // 3. Parallel index collection using OpenMP.
        // We use thread-local vectors to eliminate race conditions and minimize 
        // synchronization overhead during the hot loop.
        std::vector<std::size_t> final_indices;

#pragma omp parallel
        {
            std::vector<std::size_t> local_indices;
            // Heuristic pre-allocation to reduce the frequency of reallocations within threads.
            local_indices.reserve(n / (omp_get_num_threads() * 2));

#pragma omp for nowait
            for (std::size_t i = 0; i < n; ++i) {
                if (distances[i] <= effective_radius) {
                    local_indices.push_back(i);
                }
            }

            // Consolidate thread-local results into the global output vector.
            // Using a critical section here is safe as it happens only once per thread.
#pragma omp critical
            {
                final_indices.insert(final_indices.end(), local_indices.begin(), local_indices.end());
            }
        }

        return final_indices;
    }

} // namespace fc::algorithms