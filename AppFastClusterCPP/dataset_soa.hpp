#pragma once

#include <vector>
#include <array>
#include <boost/align/aligned_allocator.hpp>
#include "ml_coordinate.hpp"

namespace fc {

    /**
     * @brief A Structure of Arrays (SoA) dataset representation.
     * * In this layout, coordinates for each dimension are stored in separate contiguous
     * memory buffers. This is highly efficient for algorithms that perform operations
     * on specific dimensions (e.g., computing a bounding box) or for achieving
     * maximum throughput in SIMD-based distance calculations.
     */
    template <MLCoordinate T, std::size_t Dim>
    class DatasetSoA {
    public:
        /** * @brief Internal storage where each element of the array is a vector representing one dimension.
         * * Uses a 64-byte aligned allocator to ensure memory addresses are compatible
         * with the most demanding SIMD sets (like AVX-512) and to prevent cache line splitting.
         */
        std::array<std::vector<T, boost::alignment::aligned_allocator<T, 64>>, Dim> axes;

        /**
         * @brief Decomposes a point and appends each coordinate to its respective axis vector.
         * @param point The N-dimensional point to be added.
         */
        void push_back(const std::array<T, Dim>& point) {
            for (std::size_t i = 0; i < Dim; ++i) {
                axes[i].push_back(point[i]);
            }
        }

        /** @brief Returns the total number of points stored in the dataset. */
        std::size_t size() const { return axes[0].size(); }

        /**
         * @brief Provides a raw pointer to the contiguous data of a specific axis.
         * * Essential for low-level optimizations and passing data to kernels
         * or SIMD intrinsic loops.
         * * @param axis_idx The index of the dimension (0 to Dim-1).
         * @return Const pointer to the start of the axis data.
         */
        const T* axis_data(std::size_t axis_idx) const { return axes[axis_idx].data(); }
    };

}