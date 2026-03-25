#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <concepts>
#include "dataset_soa.hpp"

namespace fc::metrics {

    /**
     * @brief Concept defining the requirements for an SoA distance policy.
     * * Policies must provide an 'accumulate' method for per-axis processing
     * and an optional 'finalize' method for post-processing (e.g., sqrt).
     */
    template <typename P, typename T>
    concept SoAMetricPolicy = requires(T q_val, const T * axis_data, std::size_t n, std::vector<T>&out) {
        { P::accumulate(q_val, axis_data, n, out) } -> std::same_as<void>;
        { P::finalize(out) } -> std::same_as<void>;
    };

    // --- Concrete Metric Policies (Static Polymorphism) ---

    /** @brief Policy for Squared Euclidean Distance (L2 Squared). */
    struct SquaredEuclideanPolicy {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            // Compiler can easily vectorize this linear accumulation
            for (std::size_t i = 0; i < n; ++i) {
                const T diff = q_val - axis_data[i];
                out[i] += diff * diff;
            }
        }
        template <typename T>
        static inline void finalize(std::vector<T>&) { /* No post-processing needed */ }
    };

    /** @brief Policy for Standard Euclidean Distance (L2). */
    struct EuclideanPolicy {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            SquaredEuclideanPolicy::accumulate(q_val, axis_data, n, out);
        }
        template <typename T>
        static inline void finalize(std::vector<T>& out) {
            for (auto& val : out) val = std::sqrt(val);
        }
    };

    /** @brief Policy for Manhattan Distance (L1 / Taxicab). */
    struct ManhattanPolicy {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] += std::abs(q_val - axis_data[i]);
            }
        }
        template <typename T>
        static inline void finalize(std::vector<T>&) { /* No post-processing needed */ }
    };

    /** @brief Policy for Chebyshev Distance (L-infinity). */
    struct ChebyshevPolicy {
        template <typename T>
        static inline void accumulate(T q_val, const T* axis_data, std::size_t n, std::vector<T>& out) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = std::max(out[i], std::abs(q_val - axis_data[i]));
            }
        }
        template <typename T>
        static inline void finalize(std::vector<T>&) { /* No post-processing needed */ }
    };

    // --- Generic Dispatcher ---

    /**
     * @brief High-performance batch distance computer for SoA datasets.
     * * Resolves the distance metric at compile-time via the Policy template parameter.
     * * Maximizes cache efficiency by streaming through axis-contiguous data.
     * * @tparam Policy The distance calculation policy (SquaredEuclidean, Manhattan, etc.).
     * @param query The target point for distance calculations.
     * @param dataset The SoA dataset containing target points.
     * @param out_distances Output vector to store calculated distances.
     */
    template <typename Policy, MLCoordinate T, std::size_t Dim>
        requires SoAMetricPolicy<Policy, T>
    void compute_distances_soa(
        const std::array<T, Dim>& query,
        const DatasetSoA<T, Dim>& dataset,
        std::vector<T>& out_distances)
    {
        const std::size_t n = dataset.size();
        out_distances.assign(n, T{ 0 });

        // Process dimensions sequentially to leverage spatial locality in SoA buffers
        for (std::size_t d = 0; d < Dim; ++d) {
            Policy::accumulate(query[d], dataset.axis_data(d), n, out_distances);
        }

        Policy::finalize(out_distances);
    }

} // namespace fc::metrics