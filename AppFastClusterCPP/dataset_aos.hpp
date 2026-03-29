#pragma once

#include <vector>
#include <array>
#include <boost/align/aligned_allocator.hpp>
#include "ml_coordinate.hpp"

namespace fc {

    /**
     * @brief A point representation using the Array of Structures (AoS) layout.
     * * Forced 32-byte alignment ensures compatibility with AVX/AVX2 SIMD load/store
     * instructions (e.g., _mm256_load_ps).
     * * @note This alignment may introduce tail padding if (Dim * sizeof(T)) is not
     * a multiple of 32. This prioritizes vectorized execution speed over
     * absolute memory density.
     * * @tparam T Coordinate type, must satisfy the MLCoordinate concept.
     * @tparam Dim Dimensionality of the point.
     */
    template <MLCoordinate T, std::size_t Dim>
    struct alignas(32) PointAoS {
        std::array<T, Dim> coords;

        /**
         * @brief Provides read-only access to the coordinate at the specified index.
         * @param i Index of the dimension (0 to Dim-1).
         * @return Const reference to the coordinate value.
         */
        T operator[](std::size_t i) const { return coords[i]; }

        /**
         * @brief Provides read/write access to the coordinate at the specified index.
         * @param i Index of the dimension (0 to Dim-1).
         * @return Reference to the coordinate value.
         */
        T& operator[](std::size_t i) { return coords[i]; }
    };

    /**
     * @brief High-performance container for AoS points with guaranteed memory alignment.
     * * Uses boost::alignment::aligned_allocator to ensure that the heap-allocated
     * contiguous memory block starts on a 32-byte boundary. This prevents
     * performance penalties associated with unaligned SIMD memory access and
     * ensures each PointAoS element remains correctly aligned within the vector.
     */
    template <MLCoordinate T, std::size_t Dim>
    using DatasetAoS = std::vector<PointAoS<T, Dim>,
        boost::alignment::aligned_allocator<PointAoS<T, Dim>, 32>>;

} // namespace fc