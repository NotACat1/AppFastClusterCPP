#pragma once

#include <vector>
#include <array>
#include <boost/align/aligned_allocator.hpp>
#include "ml_coordinate.hpp"

namespace fc {

    /**
     * @brief A SIMD-optimized memory block using AoSoA (Hybrid) layout.
     * * Data is organized as [Dimension][SimdWidth] to allow vertical
     * vectorization (processing multiple points' identical dimensions simultaneously).
     * Aligned to 64 bytes to match AVX-512 requirements and typical CPU cache line sizes.
     */
    template <MLCoordinate T, std::size_t Dim, std::size_t SimdWidth = 8>
    struct alignas(64) SimdBlock {
        // Layout example for 3D (Dim=3, Width=8):
        // [X0..X7][Y0..Y7][Z0..Z7]
        T lanes[Dim][SimdWidth];
    };

    /**
     * @brief Dataset container utilizing the AoSoA (Array of Structures of Arrays) pattern.
     * * This architecture provides a compromise between SoA (vectorization efficiency)
     * and AoS (cache locality). It is ideal for ML algorithms like KNN or K-Means
     * where distance calculations can be heavily vectorized using SIMD intrinsics.
     */
    template <MLCoordinate T, std::size_t Dim, std::size_t SimdWidth = 8>
    class DatasetAoSoA {
    private:
        using BlockType = SimdBlock<T, Dim, SimdWidth>;

        // Contiguous storage of SIMD blocks with 64-byte alignment
        std::vector<BlockType, boost::alignment::aligned_allocator<BlockType, 64>> blocks;
        std::size_t total_points = 0;

    public:
        /**
         * @brief Inserts a point into the next available SIMD lane.
         * * Automatically allocates a new SimdBlock if the current one is full.
         * @param point The N-dimensional point to be added.
         */
        void add_point(const std::array<T, Dim>& point) {
            const std::size_t block_idx = total_points / SimdWidth;
            const std::size_t lane_idx = total_points % SimdWidth;

            if (block_idx >= blocks.size()) {
                blocks.emplace_back();
            }

            for (std::size_t d = 0; d < Dim; ++d) {
                blocks[block_idx].lanes[d][lane_idx] = point[d];
            }
            total_points++;
        }

        /** @brief Returns a constant reference to a specific SIMD block. */
        const BlockType& get_block(std::size_t i) const { return blocks[i]; }

        /** @brief Returns the total number of allocated blocks. */
        std::size_t block_count() const { return blocks.size(); }

        /** @brief Returns the total number of points stored in the dataset. */
        std::size_t size() const { return total_points; }
    };

}