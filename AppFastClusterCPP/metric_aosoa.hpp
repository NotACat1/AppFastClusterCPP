#pragma once

#include <immintrin.h>
#include <array>
#include <vector>
#include <concepts>
#include "dataset_aosoa.hpp"

namespace fc::metrics {

    /**
     * @brief Concept defining the static interface for SIMD-accelerated distance metrics.
     * * Requirements for a Metric class:
     * 1. Must implement a static 'evaluate' method.
     * 2. Must process a query point against an 8-lane SimdBlock.
     * 3. Result must be stored in a pre-allocated float array of size 8.
     */
    template <typename M>
    concept SIMDMetric = requires(const std::array<float, 3>&q, const SimdBlock<float, 3, 8>&b, float* out) {
        { M::evaluate(q, b, out) } -> std::same_as<void>;
    };

    /**
     * @brief Internal hardware utilities for low-level bit manipulation.
     */
    namespace detail {
        /**
         * @brief Generates a mask to clear the sign bit of a 32-bit float.
         * Used to implement absolute value (ABS) functionality for metrics like L1 and L-inf.
         */
        inline __m256 get_abs_mask_ps() {
            return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        }
    }

    // --- Static Polymorphism Implementation (Metric Policies) ---

    /**
     * @brief Squared Euclidean Distance (L2 Squared).
     * * @details Optimization: Uses Fused Multiply-Add (FMA) instructions to compute
     * (a * b + c) in a single cycle. This is the preferred metric for K-Means
     * and KNN as it avoids the expensive square root operation.
     */
    struct SquaredEuclidean {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            // Broadcast query coordinates to SIMD registers
            __m256 qx = _mm256_set1_ps(query[0]);
            __m256 qy = _mm256_set1_ps(query[1]);
            __m256 qz = _mm256_set1_ps(query[2]);

            // Load 8-point coordinates from aligned block
            __m256 bx = _mm256_load_ps(block.lanes[0]);
            __m256 by = _mm256_load_ps(block.lanes[1]);
            __m256 bz = _mm256_load_ps(block.lanes[2]);

            // Compute delta
            __m256 dx = _mm256_sub_ps(qx, bx);
            __m256 dy = _mm256_sub_ps(qy, by);
            __m256 dz = _mm256_sub_ps(qz, bz);

            // Accumulate squared distances using FMA: dist = dx*dx + dy*dy + dz*dz
            __m256 distSq = _mm256_mul_ps(dx, dx);
            distSq = _mm256_fmadd_ps(dy, dy, distSq);
            distSq = _mm256_fmadd_ps(dz, dz, distSq);

            _mm256_storeu_ps(out_distances, distSq);
        }
    };

    /**
     * @brief Standard Euclidean Distance (L2 Norm).
     * * @note Performance overhead: Contains a hardware square root instruction (_mm256_sqrt_ps).
     */
    struct Euclidean {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            SquaredEuclidean::evaluate(query, block, out_distances);
            __m256 dist = _mm256_loadu_ps(out_distances);
            dist = _mm256_sqrt_ps(dist);
            _mm256_storeu_ps(out_distances, dist);
        }
    };

    /**
     * @brief Manhattan Distance (L1 Norm).
     * * @details Optimization: Computes absolute difference via bitwise AND with a sign-bit mask.
     * Ideal for high-dimensional data where distance contrast is a concern.
     */
    struct Manhattan {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            __m256 qx = _mm256_set1_ps(query[0]);
            __m256 qy = _mm256_set1_ps(query[1]);
            __m256 qz = _mm256_set1_ps(query[2]);

            __m256 bx = _mm256_load_ps(block.lanes[0]);
            __m256 by = _mm256_load_ps(block.lanes[1]);
            __m256 bz = _mm256_load_ps(block.lanes[2]);

            __m256 abs_mask = detail::get_abs_mask_ps();

            // Compute bitwise ABS: |qx - bx|
            __m256 dx = _mm256_and_ps(_mm256_sub_ps(qx, bx), abs_mask);
            __m256 dy = _mm256_and_ps(_mm256_sub_ps(qy, by), abs_mask);
            __m256 dz = _mm256_and_ps(_mm256_sub_ps(qz, bz), abs_mask);

            // Sum components: L1 = dx + dy + dz
            __m256 dist = _mm256_add_ps(_mm256_add_ps(dx, dy), dz);
            _mm256_storeu_ps(out_distances, dist);
        }
    };

    /**
     * @brief Chebyshev Distance (L-infinity Norm).
     * * @details Optimization: Uses the hardware 'max' instruction to find the
     * dominant dimension difference across 8 points simultaneously.
     */
    struct Chebyshev {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            __m256 qx = _mm256_set1_ps(query[0]);
            __m256 qy = _mm256_set1_ps(query[1]);
            __m256 qz = _mm256_set1_ps(query[2]);

            __m256 bx = _mm256_load_ps(block.lanes[0]);
            __m256 by = _mm256_load_ps(block.lanes[1]);
            __m256 bz = _mm256_load_ps(block.lanes[2]);

            __m256 abs_mask = detail::get_abs_mask_ps();

            __m256 dx = _mm256_and_ps(_mm256_sub_ps(qx, bx), abs_mask);
            __m256 dy = _mm256_and_ps(_mm256_sub_ps(qy, by), abs_mask);
            __m256 dz = _mm256_and_ps(_mm256_sub_ps(qz, bz), abs_mask);

            // Result: max(|dx|, |dy|, |dz|)
            __m256 max_val = _mm256_max_ps(_mm256_max_ps(dx, dy), dz);
            _mm256_storeu_ps(out_distances, max_val);
        }
    };

    /**
     * @brief High-level batch dispatcher.
     * * @tparam Metric A type that satisfies the SIMDMetric concept.
     * @param query The target point for distance calculation.
     * @param dataset The AoSoA dataset containing target points.
     * @param out_results Destination vector for computed distances.
     */
    template <SIMDMetric Metric>
    void compute_batch_distances(
        const std::array<float, 3>& query,
        const DatasetAoSoA<float, 3, 8>& dataset,
        std::vector<float>& out_results)
    {
        const std::size_t num_blocks = dataset.block_count();
        out_results.resize(num_blocks * 8);

        for (std::size_t i = 0; i < num_blocks; ++i) {
            // The compiler inlines the specific 'evaluate' code here, 
            // resolving the static polymorphism at compile-time.
            Metric::evaluate(query, dataset.get_block(i), &out_results[i * 8]);
        }
    }

} // namespace fc::metrics