#pragma once

#include <vector>
#include <omp.h>
#include "dataset_aosoa.hpp"
#include "metrics_aosoa.hpp"

namespace fc::algorithms {

    /**
     * @brief High-performance brute-force radius search optimized for AoSoA layout using AVX2 and OpenMP.
     * * This implementation leverages the hybrid Array of Structures of Arrays (AoSoA) format
     * to achieve maximum SIMD throughput. By processing 8-float lanes (AVX2) in parallel
     * across multiple CPU cores, it minimizes memory latency and maximizes FLOPs.
     * * @tparam Metric A SIMD-enabled metric policy (e.g., SquaredEuclidean, Manhattan, Chebyshev).
     * @param dataset The input dataset in AoSoA format (8-float SIMD width).
     * @param query The reference point coordinates (3D).
     * @param radius The search radius threshold.
     * @return std::vector<std::size_t> Global indices of points within the specified radius.
     */
    template <fc::metrics::SIMDMetric Metric>
    auto radius_search_brute_force_aosoa(
        const fc::DatasetAoSoA<float, 3, 8>& dataset,
        const std::array<float, 3>& query,
        float radius
    ) -> std::vector<std::size_t>
    {
        const std::size_t num_blocks = dataset.block_count();
        const std::size_t total_points = dataset.size();
        if (total_points == 0) return {};

        // Optimization: Use squared radius for Euclidean distance to bypass expensive sqrt() calls.
        float effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::Euclidean>) {
            effective_radius = radius * radius;
        }

        std::vector<std::size_t> final_indices;

#pragma omp parallel
        {
            // Thread-local buffer to prevent race conditions and minimize atomic contention.
            std::vector<std::size_t> local_indices;
            local_indices.reserve(total_points / (omp_get_num_threads() * 2));

            // Stack-allocated, 32-byte aligned buffer for intermediate SIMD distance results.
            alignas(32) float block_distances[8];

#pragma omp for nowait
            for (std::size_t b = 0; b < num_blocks; ++b) {
                // 1. Calculate distances for a full SIMD block (8 points) in a single pass.
                // Dispatches to SquaredEuclidean if the metric is Euclidean to maintain L2 performance.
                if constexpr (std::is_same_v<Metric, fc::metrics::Euclidean>) {
                    fc::metrics::SquaredEuclidean::evaluate(query, dataset.get_block(b), block_distances);
                }
                else {
                    Metric::evaluate(query, dataset.get_block(b), block_distances);
                }

                // 2. Identify points within range.
                // Handle the remainder points in the tail block to ensure data integrity.
                const std::size_t points_in_block = (b == num_blocks - 1) ?
                    (total_points % 8 == 0 ? 8 : total_points % 8) : 8;

                for (std::size_t lane = 0; lane < points_in_block; ++lane) {
                    if (block_distances[lane] <= effective_radius) {
                        local_indices.push_back(b * 8 + lane);
                    }
                }
            }

            // Synchronized merge of thread-local results into the final result set.
#pragma omp critical
            {
                final_indices.insert(final_indices.end(), local_indices.begin(), local_indices.end());
            }
        }

        return final_indices;
    }

} // namespace fc::algorithms