#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"

namespace fc::algorithms {

    /**
     * @class KDTreeBuilder
     * @brief A static utility class for constructing a balanced, flattened KD-Tree.
     * * This builder uses a median-of-elements approach to ensure the resulting tree
     * is balanced, providing O(log N) search complexity. The tree is stored in a
     * flat vector representation for cache efficiency and easy serialization.
     * * @tparam T The coordinate scalar type (must satisfy fc::MLCoordinate).
     * @tparam Dim The spatial dimensionality of the dataset.
     */
    template <fc::MLCoordinate T, std::size_t Dim>
    class KDTreeBuilder {
    public:
        /**
         * @brief Entry point for building a KD-Tree from a DatasetAoS.
         * @param dataset The input collection of points in Array-of-Structures layout.
         * @return KDTreeFlat A flattened KD-Tree structure ready for spatial queries.
         */
        static KDTreeFlat build(const fc::DatasetAoS<T, Dim>& dataset) {
            KDTreeFlat tree;
            if (dataset.empty()) {
                return tree;
            }

            // Initialize an index buffer to avoid copying heavy point data during partitioning
            std::vector<int32_t> indices(dataset.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Pre-allocate memory to prevent reallocations during recursive construction
            tree.nodes.reserve(dataset.size());

            // Initiate recursive construction starting from the root at depth 0
            tree.root_idx = build_recursive(tree, dataset, indices, 0, indices.size(), 0);

            return tree;
        }

    private:
        /**
         * @brief Recursively partitions the dataset and populates the flat tree structure.
         * * Uses std::nth_element to perform an in-place partial sort, finding the median
         * point for the current split axis in O(N) time at each level.
         * * @param tree The tree instance being populated.
         * @param dataset Reference to the source data.
         * @param indices Buffer of point indices to be partitioned.
         * @param start Starting range in the index buffer (inclusive).
         * @param end Ending range in the index buffer (exclusive).
         * @param depth Current recursion depth, used to determine the split axis.
         * @return int32_t The index of the created node within the tree's flat vector.
         */
        static int32_t build_recursive(
            KDTreeFlat& tree,
            const fc::DatasetAoS<T, Dim>& dataset,
            std::vector<int32_t>& indices,
            std::size_t start,
            std::size_t end,
            std::size_t depth
        ) {
            // Base case: range is empty
            if (start >= end) {
                return -1;
            }

            // Select split axis based on current depth (cycling through dimensions)
            std::size_t axis = depth % Dim;
            std::size_t mid = start + (end - start) / 2;

            // Partially sort the index range to place the median element at the 'mid' position
            std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
                [&](int32_t a, int32_t b) {
                    return dataset[a][axis] < dataset[b][axis];
                });

            // Extract the pivot point and the value used for the spatial split
            int32_t point_idx = indices[mid];
            float split_val = static_cast<float>(dataset[point_idx][axis]);

            /** * Capture the current vector size as the node index.
             * Important: vector.reserve() in the entry method prevents invalidation
             * of indices during emplace_back.
             */
            int32_t node_idx = static_cast<int32_t>(tree.nodes.size());
            tree.nodes.emplace_back(); // Allocate slot for the current node

            // Recursively construct left and right subtrees
            int32_t left = build_recursive(tree, dataset, indices, start, mid, depth + 1);
            int32_t right = build_recursive(tree, dataset, indices, mid + 1, end, depth + 1);

            // Populate the allocated node with spatial metadata and child pointers
            tree.nodes[node_idx] = {
                split_val,
                point_idx,
                left,
                right,
                static_cast<int32_t>(axis)
            };

            return node_idx;
        }
    };
} // namespace fc::algorithms