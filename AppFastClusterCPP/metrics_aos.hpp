#pragma once

#include <algorithm>
#include <concepts>
#include "dataset_aos.hpp"

namespace fc::metrics {

    /**
     * @brief Concept defining the static interface for a scalar distance metric.
     * Ensures that any metric implementation provides a static 'evaluate' method
     * compatible with PointAoS structures.
     */
    template <typename M, typename T, std::size_t Dim>
    concept ScalarMetric = requires(const PointAoS<T, Dim>&p1, const PointAoS<T, Dim>&p2) {
        { M::evaluate(p1, p2) } -> std::same_as<T>;
    };

    // --- Concrete Metric Implementations (Static Polymorphism Tags) ---

    /**
     * @brief Squared Euclidean Distance (Squared L2 Norm).
     * @note This is the most computationally efficient metric as it avoids the
     * expensive square root operation. Ideal for comparison-based algorithms
     * like K-Means or Nearest Neighbor search.
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
     * @details Represents the shortest straight-line distance between two points.
     */
    struct EuclideanAoS {
        template <MLCoordinate T, std::size_t Dim>
        static inline T evaluate(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
            return std::sqrt(SquaredEuclideanAoS::evaluate(p1, p2));
        }
    };

    /**
     * @brief Manhattan Distance (L1 Norm / Taxicab Geometry).
     * @details Calculates the distance as the sum of absolute differences
     * along each axis. Useful in high-dimensional spaces where L2 might
     * suffer from the "curse of dimensionality".
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
     * @details Defined as the maximum absolute difference between any
     * coordinate dimension. Equivalent to the number of moves a King
     * makes on a chessboard.
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
     * @brief Generic distance calculation wrapper.
     * Demonstrates zero-overhead abstraction by resolving the metric at compile-time.
     */
    template <typename T, std::size_t Dim, typename Metric>
        requires ScalarMetric<Metric, T, Dim>
    inline T compute_distance(const PointAoS<T, Dim>& p1, const PointAoS<T, Dim>& p2) {
        return Metric::evaluate(p1, p2);
    }

} // namespace fc::metrics