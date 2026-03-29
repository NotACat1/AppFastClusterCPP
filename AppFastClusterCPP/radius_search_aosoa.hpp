#pragma once

#include <vector>
#include <omp.h>
#include "dataset_aosoa.hpp"
#include "metrics_aosoa.hpp"

namespace fc::algorithms {

    /**
     * @brief High-throughput radius search optimized for AVX2 and multi-core execution.
     * * @details This implementation leverages the Array of Structures of Arrays (AoSoA)
     * memory layout to maximize data cache utilization and SIMD lane occupancy.
     * It combines data-level parallelism (processing 8 floats per instruction) with
     * thread-level parallelism (OpenMP) to achieve near peak FLOPs for distance queries.
     * * @tparam Metric A valid SIMD-accelerated distance policy (MetricAoSoA).
     * @param dataset The target point cloud structured in 8-wide AoSoA blocks.
     * @param query The 3D reference coordinate serving as the search origin.
     * @param radius The maximum inclusive distance boundary.
     * @return std::vector<std::size_t> Unsorted global indices of all points within the radius.
     */
    template <fc::metrics::MetricAoSoA Metric>
    auto radius_search_brute_force_aosoa(
        const fc::DatasetAoSoA<float, 3, 8>& dataset,
        const std::array<float, 3>& query,
        float radius
    ) -> std::vector<std::size_t>
    {
        const std::size_t num_blocks = dataset.block_count();
        const std::size_t total_points = dataset.size();
        if (total_points == 0) return {};

        // Compile-time Optimization: Transform Euclidean radius to squared Euclidean.
        // This ensures the inner SIMD loop avoids the high-latency _mm256_sqrt_ps 
        // instruction without altering the user-facing API.
        float effective_radius = radius;
        if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoSoA>) {
            effective_radius = radius * radius;
        }

        std::vector<std::size_t> final_indices;

        // Spawn a thread pool to process memory blocks concurrently.
#pragma omp parallel
        {
            // Thread-Local Storage (TLS): Maintain a private buffer for each thread.
            // This completely eliminates false sharing and the massive performance 
            // penalty of atomic locks/mutexes during the hot scanning loop.
            std::vector<std::size_t> local_indices;

            // Heuristic pre-allocation: Assumes a uniform distribution of results 
            // across the available OpenMP thread pool.
            local_indices.reserve(total_points / (omp_get_num_threads() * 2));

            // Stack alignment: Guarantee 32-byte alignment for the local float array.
            // This prevents cache-line splits and ensures zero-penalty YMM register stores.
            alignas(32) float block_distances[8];

            // Distribute blocks dynamically across threads. 'nowait' removes the 
            // implicit barrier at the end of the loop, allowing threads that finish 
            // early to proceed directly to the merge phase.
#pragma omp for nowait
            for (std::size_t b = 0; b < num_blocks; ++b) {

                // 1. Vectorized Distance Computation
                // The compiler guarantees dead-code elimination for the unused branch.
                if constexpr (std::is_same_v<Metric, fc::metrics::EuclideanAoSoA>) {
                    fc::metrics::SquaredEuclideanAoSoA::evaluate(query, dataset.get_block(b), block_distances);
                }
                else {
                    Metric::evaluate(query, dataset.get_block(b), block_distances);
                }

                // 2. Scalar Reduction and Fringe Handling
                // Determine valid lanes to process. For the final block, we clamp 
                // the iteration count to handle datasets not perfectly divisible by 8.
                const std::size_t points_in_block = (b == num_blocks - 1) ?
                    (total_points % 8 == 0 ? 8 : total_points % 8) : 8;

                for (std::size_t lane = 0; lane < points_in_block; ++lane) {
                    if (block_distances[lane] <= effective_radius) {
                        // Map the local block lane back to the global dataset index
                        local_indices.push_back(b * 8 + lane);
                    }
                }
            }

            // 3. Synchronized Result Aggregation
            // Serialized block where each thread merges its private buffer into the 
            // global result vector. The block is short-lived, minimizing thread stalling.
#pragma omp critical
            {
                final_indices.insert(final_indices.end(), local_indices.begin(), local_indices.end());
            }
        }

        return final_indices;
    }

} // namespace fc::algorithms