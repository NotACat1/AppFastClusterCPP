#pragma once

#include <immintrin.h>
#include <array>
#include <vector>
#include <concepts>
#include "dataset_aosoa.hpp"

namespace fc::metrics {

    /**
     * @brief Constraint for SIMD-accelerated distance metrics using 256-bit YMM registers.
     * * Requirements for a valid Metric implementation:
     * 1. `evaluate`: Static method calculating distances for 8 points simultaneously.
     * 2. Inputs: A single query point (AoS) and a SIMD block of 8 points (SoA).
     * 3. Output: Writes 8 results into a provided float buffer.
     * * This layout facilitates high-throughput data processing by allowing linear
     * memory loads directly into YMM registers.
     */
    template <typename Metric>
    concept MetricAoSoA = requires(const std::array<float, 3>&q, const SimdBlock<float, 3, 8>&b, float* out) {
        { Metric::evaluate(q, b, out) } -> std::same_as<void>;
    };

    /**
     * @brief Low-level hardware utilities for SIMD bit-level operations.
     */
    namespace detail {
        /**
         * @brief Generates an ABS mask for 32-bit single-precision floats.
         * * By performing a bitwise AND with 0x7FFFFFFF, we clear the sign bit (the MSB
         * of a 32-bit IEEE 754 float), effectively computing the absolute value
         * without a branching instruction or specialized hardware logic.
         */
        inline __m256 get_abs_mask_ps() {
            return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        }
    }

    // --- Static Polymorphism Implementations (Metric Policies) ---

    /**
     * @brief Squared Euclidean Distance (L2 Squared) using AVX2 FMA instructions.
     * * @details Optimization Strategy:
     * * **Broadcasting**: Query coordinates are broadcast to all 8 SIMD lanes.
     * * **Parallel Loading**: Uses `_mm256_load_ps` for aligned 32-byte memory access.
     * * **FMA (Fused Multiply-Add)**: Leverages `_mm256_fmadd_ps` to compute (a*b + c)
     * in a single instruction cycle, reducing rounding errors and increasing throughput.
     */
    struct SquaredEuclideanAoSoA {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            // Broadcast query coordinates to fill 8-lane YMM registers
            __m256 qx = _mm256_set1_ps(query[0]);
            __m256 qy = _mm256_set1_ps(query[1]);
            __m256 qz = _mm256_set1_ps(query[2]);

            // Load 8-point coordinates from aligned memory block (SoA layout)
            __m256 bx = _mm256_load_ps(block.lanes[0]);
            __m256 by = _mm256_load_ps(block.lanes[1]);
            __m256 bz = _mm256_load_ps(block.lanes[2]);

            // Vectorized subtraction
            __m256 dx = _mm256_sub_ps(qx, bx);
            __m256 dy = _mm256_sub_ps(qy, by);
            __m256 dz = _mm256_sub_ps(qz, bz);

            // Accumulate squared distances: result = (dx^2 + dy^2 + dz^2)
            __m256 distSq = _mm256_mul_ps(dx, dx);
            distSq = _mm256_fmadd_ps(dy, dy, distSq);
            distSq = _mm256_fmadd_ps(dz, dz, distSq);

            // Store results to output (unaligned store used to ensure compatibility)
            _mm256_storeu_ps(out_distances, distSq);
        }
    };

    /**
     * @brief Standard Euclidean Distance (L2 Norm).
     * * @note Micro-architectural Impact: This metric incurs a performance penalty
     * due to the high latency of the `_mm256_sqrt_ps` instruction compared to
     * simple arithmetic.
     */
    struct EuclideanAoSoA {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            SquaredEuclideanAoSoA::evaluate(query, block, out_distances);
            __m256 dist = _mm256_loadu_ps(out_distances);
            dist = _mm256_sqrt_ps(dist);
            _mm256_storeu_ps(out_distances, dist);
        }
    };

    /**
     * @brief Manhattan Distance (L1 Norm).
     * * @details Implementation: Computes absolute differences via bitwise logic
     * and sums them. This metric is computationally lighter than L2 and avoids
     * the precision loss of squaring large values.
     */
    struct ManhattanAoSoA {
        static inline void evaluate(const std::array<float, 3>& query, const SimdBlock<float, 3, 8>& block, float out_distances[8]) {
            __m256 qx = _mm256_set1_ps(query[0]);
            __m256 qy = _mm256_set1_ps(query[1]);
            __m256 qz = _mm256_set1_ps(query[2]);

            __m256 bx = _mm256_load_ps(block.lanes[0]);
            __m256 by = _mm256_load_ps(block.lanes[1]);
            __m256 bz = _mm256_load_ps(block.lanes[2]);

            __m256 abs_mask = detail::get_abs_mask_ps();

            // Compute bitwise Absolute Difference: |q - b|
            __m256 dx = _mm256_and_ps(_mm256_sub_ps(qx, bx), abs_mask);
            __m256 dy = _mm256_and_ps(_mm256_sub_ps(qy, by), abs_mask);
            __m256 dz = _mm256_and_ps(_mm256_sub_ps(qz, bz), abs_mask);

            // L1 = |dx| + |dy| + |dz|
            __m256 dist = _mm256_add_ps(_mm256_add_ps(dx, dy), dz);
            _mm256_storeu_ps(out_distances, dist);
        }
    };

    /**
     * @brief Chebyshev Distance (L-infinity Norm).
     * * @details Logic: Identifies the maximum axial difference across 8 points.
     * Leverages the hardware `_mm256_max_ps` instruction, which maps directly
     * to high-speed comparison logic in the execution unit.
     */
    struct ChebyshevAoSoA {
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

            // Find max across dimensions for each lane: max(|dx|, |dy|, |dz|)
            __m256 max_val = _mm256_max_ps(_mm256_max_ps(dx, dy), dz);
            _mm256_storeu_ps(out_distances, max_val);
        }
    };

    /**
     * @brief High-performance batch dispatcher for distance calculations.
     * * @tparam Metric A policy type satisfying the MetricAoSoA concept.
     * @param query The reference point from which distances are measured.
     * @param dataset The AoSoA-formatted container optimized for SIMD access.
     * @param out_results Buffer for the computed distances (resized to match num_points).
     * * @note Static Dispatch: By using the Metric as a template parameter, the
     * compiler performs full inlining of the 'evaluate' method into the loop body,
     * enabling inter-procedural optimizations and eliminating call overhead.
     */
    template <MetricAoSoA Metric>
    void compute_distances_aosoa(
        const std::array<float, 3>& query,
        const DatasetAoSoA<float, 3, 8>& dataset,
        std::vector<float>& out_results)
    {
        const std::size_t num_blocks = dataset.block_count();
        out_results.resize(num_blocks * 8);

        for (std::size_t i = 0; i < num_blocks; ++i) {
            // Process 8 points per iteration via vectorized instructions
            Metric::evaluate(query, dataset.get_block(i), &out_results[i * 8]);
        }
    }
} // namespace fc::metrics