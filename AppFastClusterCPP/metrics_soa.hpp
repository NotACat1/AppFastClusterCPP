#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <concepts>
#include <cmath>
#include "dataset_soa.hpp"

namespace fc::metrics {

    /**
     * @brief Interface constraint for Structure of Arrays (SoA) distance policies.
     * * Requirements for a valid Metric implementation:
     * 1. `accumulate`: Static method for partial distance calculation per coordinate axis.
     * 2. `finalize`: Static method for post-calculation transformations (e.g., square root).
     * * This split allows the dispatcher to stream through each dimension's contiguous
     * memory buffer independently, maximizing L1/L2 cache line utilization.
     */
    template <typename Metric, typename T>
    concept MetricSoA = requires(T q_val, const T * axis_data, std::size_t n, std::vector<T>&out) {
        { Metric::accumulate(q_val, axis_data, n, out) } -> std::same_as<void>;
        { Metric::finalize(out) } -> std::same_as<void>;
    };

    // --- Concrete Metric Policies (Static Polymorphism) ---

    /** * @brief Squared Euclidean Distance (Squared L2 Norm) Policy.
     * * Designed for high-throughput batch processing. The linear accumulation loop
     * is highly susceptible to compiler auto-vectorization (SSE/AVX) because
     * the `axis_data` is guaranteed to be a contiguous, unit-stride memory block.
     */
    struct SquaredEuclideanSoA {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            for (std::size_t i = 0; i < n; ++i) {
                const T diff = q_val - axis_data[i];
                out[i] += diff * diff;
            }
        }

        /** @brief No post-processing required for squared distance. */
        template <typename T>
        static inline void finalize(std::vector<T>&) {}
    };

    /** * @brief Standard Euclidean Distance (L2 Norm) Policy.
     * * Reuses the accumulation logic of `SquaredEuclideanSoA` and applies a
     * deferred square root operation during the finalization phase.
     */
    struct EuclideanSoA {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            SquaredEuclideanSoA::accumulate(q_val, axis_data, n, out);
        }

        /** @brief Performs batch square root on the accumulated results. */
        template <typename T>
        static inline void finalize(std::vector<T>& out) {
            for (auto& val : out) {
                val = std::sqrt(val);
            }
        }
    };

    /** * @brief Manhattan Distance (L1 / Taxicab) Policy.
     * * Computes the sum of absolute differences. Efficiently handled by SIMD
     * hardware through absolute value instructions (e.g., `_mm256_and_ps`
     * with a sign-bit mask).
     */
    struct ManhattanSoA {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] += std::abs(q_val - axis_data[i]);
            }
        }

        template <typename T>
        static inline void finalize(std::vector<T>&) {}
    };

    /** * @brief Chebyshev Distance (L-infinity / Maximum) Policy.
     * * Identifies the maximum axial displacement. In an SoA layout, this
     * translates to a series of `std::max` operations across the output buffer.
     */
    struct ChebyshevSoA {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = std::max(out[i], std::abs(q_val - axis_data[i]));
            }
        }

        template <typename T>
        static inline void finalize(std::vector<T>&) {}
    };

    // --- Generic Dispatcher ---

    /**
     * @brief High-performance batch distance computer for SoA datasets.
     * * **Spatial Locality**: This function processes the dataset dimension by dimension.
     * By iterating over one contiguous axis-buffer at a time, it minimizes cache
     * misses and allows the hardware prefetcher to work effectively.
     * * **Zero-Overhead Abstraction**: The `Metric` policy is resolved at compile-time,
     * ensuring that no virtual function calls or indirect branching occurs
     * during the hot loops.
     * * @tparam Metric A type satisfying the MetricSoA concept.
     * @tparam T The numeric coordinate type.
     * @tparam Dim The number of dimensions.
     */
    template <typename Metric, MLCoordinate T, std::size_t Dim>
        requires MetricSoA<Metric, T>
    void compute_distances_soa(
        const std::array<T, Dim>& query,
        const DatasetSoA<T, Dim>& dataset,
        std::vector<T>& out_distances)
    {
        const std::size_t n = dataset.size();

        // Reset output buffer; 'assign' ensures the vector is pre-sized and zeroed.
        out_distances.assign(n, T{ 0 });

        // Iterate through dimensions sequentially to exploit SoA contiguous memory
        for (std::size_t d = 0; d < Dim; ++d) {
            Metric::accumulate(query[d], dataset.axis_data(d), n, out_distances);
        }

        // Apply final metric transformations (e.g., sqrt for L2)
        Metric::finalize(out_distances);
    }
} // namespace fc::metrics