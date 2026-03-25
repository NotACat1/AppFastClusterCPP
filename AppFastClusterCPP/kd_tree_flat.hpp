#pragma once

#include <vector>
#include <cstdint>
#include <boost/align/aligned_allocator.hpp>

namespace fc {

    /**
     * @brief A flattened KD-tree node designed with Data-Oriented Design (DOD) principles.
     * * This structure is typically 20-32 bytes (depending on compiler padding). By replacing
     * raw pointers with 32-bit integer indices, we achieve high cache predictability,
     * reduced memory footprint, and trivial serialization for disk I/O or network transfer.
     */
    struct alignas(16) KDNodeFlat {
        float split_val;       ///< Threshold value of the splitting hyperplane.
        int32_t point_idx;     ///< Index of the reference point in the associated Dataset.
        int32_t left_child;    ///< Index of the left child in the nodes array (-1 if leaf).
        int32_t right_child;   ///< Index of the right child in the nodes array (-1 if leaf).
        int32_t split_dim;     ///< The dimension/axis used for splitting (e.g., 0 for X, 1 for Y).
    };

    /**
     * @brief A high-performance container for the flattened KD-tree.
     * * Unlike pointer-based trees that cause "pointer chasing" and cache misses,
     * this class stores nodes in a contiguous block of memory to maximize
     * spatial locality during recursive or iterative traversal.
     */
    class KDTreeFlat {
    public:
        /**
         * @brief Internal node storage.
         * * Utilizes 64-byte alignment to ensure nodes are cache-line friendly,
         * significantly reducing latency during high-frequency tree lookups.
         */
        std::vector<KDNodeFlat, boost::alignment::aligned_allocator<KDNodeFlat, 64>> nodes;

        int32_t root_idx = -1; ///< Entry point of the tree. Remains -1 if the tree is unbuilt.

        /**
         * @brief Checks if the tree structure is empty or uninitialized.
         */
        bool empty() const { return nodes.empty() || root_idx == -1; }
    };

} // namespace fc