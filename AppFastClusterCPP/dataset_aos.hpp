#pragma once

#include <vector>
#include <array>
#include <boost/align/aligned_allocator.hpp>
#include "ml_coordinate.hpp"

namespace fc {

    /**
     * @brief A point representation using Array of Structures (AoS) layout.
     * * Forced 32-byte alignment ensures compatibility with AVX/AVX2 load instructions.
     * Note: This may introduce padding if (Dim * sizeof(T)) < 32, prioritizing
     * memory alignment over storage density.
     */
    template <MLCoordinate T, std::size_t Dim>
    struct alignas(32) PointAoS {
        std::array<T, Dim> coords;

        /**
         * @brief Element access operator for coordinate indexing.
         */
        T operator[](std::size_t i) const { return coords[i]; }
    };

    /**
     * @brief Dataset container for AoS points.
     * * Utilizes boost::alignment::aligned_allocator to maintain 32-byte
     * boundary alignment for the entire contiguous memory block,
     * preventing performance degradation due to unaligned SIMD access.
     */
    template <MLCoordinate T, std::size_t Dim>
    using DatasetAoS = std::vector<PointAoS<T, Dim>,
        boost::alignment::aligned_allocator<PointAoS<T, Dim>, 32>>;

}