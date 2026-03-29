#pragma once

#include <algorithm>
#include <concepts>
#include <cmath>
#include "dataset_aos.hpp"

namespace fc::metrics {

    /**
     * @brief Interface constraint for static distance metrics.
     * * Ensures that metric implementations provide a static `evaluate` method.
     * This follows the "Policy" design pattern, allowing for zero-overhead
     * compile-time polymorphism without the cost of virtual table lookups.
     * * @tparam Metric The implementation struct/class.
     * @tparam T The numeric coordinate type.
     * @tparam Dim The dimensionality of the points.
     */
    template <typename Metric, typename T, std::size_t Dim>
    concept MetricAoS = requires(const PointAoS<T, Dim>&p1, const PointAoS<T, Dim>&p2) {
        { Metric::evaluate(p1, p2) } -> std::same_as<T>;
    };

    // --- Concrete Metric Implementations ---

    /**
     * @brief Squared Euclidean Distance (Squared L2 Norm).
     * * @note This is the preferred metric for performance-critical pathfinding or
     * clustering (e.g., K-Means). It is monotonically increasing with Euclidean
     * distance but avoids the expensive `std::sqrt` operation.
     */
    struct SquaredEuclideanAoS {
        template <MLCoordinate T, std::size_t Dim>
        static inline T evaluate(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
            T dist = 0;
            for (std::size_t i = 0; i < Dim; ++i) {
                const T diff = p1[i] - p2[i];
                dist += diff * diff;
            }
            return dist;
        }
    };

    /**
     * @brief Standard Euclidean Distance (L2 Norm).
     * * Represents the true straight-line distance between points in Euclidean space.
     * Internally leverages `SquaredEuclideanAoS` to minimize logic duplication.
     */
    struct EuclideanAoS {
        template <MLCoordinate T, std::size_t Dim>
        static inline T evaluate(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
            return std::sqrt(SquaredEuclideanAoS::evaluate(p1, p2));
        }
    };

    /**
     * @brief Manhattan Distance (L1 Norm / Taxicab Geometry).
     * * Calculates distance as the sum of absolute differences along Cartesian axes.
     * Often more robust than L2 in high-dimensional feature spaces where
     * the "curse of dimensionality" can make L2 distances converge.
     */
    struct ManhattanAoS {
        template <MLCoordinate T, std::size_t Dim>
        static inline T evaluate(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
            T dist = 0;
            for (std::size_t i = 0; i < Dim; ++i) {
                dist += std::abs(p1[i] - p2[i]);
            }
            return dist;
        }
    };

    /**
     * @brief Chebyshev Distance (L-infinity Norm / Maximum Metric).
     * * Returns the maximum absolute difference across any single dimension.
     * Useful in grid-based movement where diagonal steps have the same cost
     * as orthogonal steps (e.g., King's moves on a chessboard).
     */
    struct ChebyshevAoS {
        template <MLCoordinate T, std::size_t Dim>
        static inline T evaluate(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
            T max_dist = 0;
            for (std::size_t i = 0; i < Dim; ++i) {
                max_dist = std::max(max_dist, std::abs(p1[i] - p2[i]));
            }
            return max_dist;
        }
    };

    /**
     * @brief Unified entry point for distance calculations.
     * * Employs C++20 constraints to ensure the Metric type provides a valid
     * static interface. By using a template parameter for the Metric, the
     * compiler can inline the specific `evaluate` logic, eliminating
     * function call overhead.
     * * @return Calculated distance of type T.
     */
    template <typename T, std::size_t Dim, typename Metric>
        requires MetricAoS<Metric, T, Dim>
    inline T compute_distance_aos(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
        return Metric::evaluate(p1, p2);
    }

} // namespace fc::metrics