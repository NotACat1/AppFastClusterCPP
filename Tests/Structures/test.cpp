#include "pch.h"
#include <gtest/gtest.h>
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/dataset_soa.hpp"
#include "../../AppFastClusterCPP/dataset_aosoa.hpp"
#include "../../AppFastClusterCPP/kd_tree_flat.hpp"

namespace fc::test {

    /**
     * @brief Unit tests for verifying data integrity across different memory layouts.
     */

     // AoS Layout: Verifies that points are stored as contiguous objects and indexed correctly.
    TEST(DatasetTest, AoSStoreAndRetrieve) {
        DatasetAoS<float, 3> ds;
        PointAoS<float, 3> p{ {1.0f, 2.0f, 3.0f} };
        ds.push_back(p);

        EXPECT_EQ(ds.size(), 1);
        EXPECT_FLOAT_EQ(ds[0][0], 1.0f);
        EXPECT_FLOAT_EQ(ds[0][2], 3.0f);
    }

    // SoA Layout: Verifies that coordinates are correctly transposed into separate axis buffers.
    TEST(DatasetTest, SoAStoreAndRetrieve) {
        DatasetSoA<float, 3> ds;
        ds.push_back({ 10.0f, 20.0f, 30.0f });

        EXPECT_EQ(ds.size(), 1);

        // Validate that data is correctly routed to specific dimension arrays
        EXPECT_FLOAT_EQ(ds.axis_data(0)[0], 10.0f);
        EXPECT_FLOAT_EQ(ds.axis_data(2)[0], 30.0f);
    }

    // AoSoA Layout: Verifies correct placement in SIMD lanes and block-level indexing.
    TEST(DatasetTest, AoSoAStoreAndRetrieve) {
        DatasetAoSoA<float, 3, 8> ds;
        ds.add_point({ 1.1f, 2.2f, 3.3f });

        // Verify first lane of the primary SIMD block
        auto& block = ds.get_block(0);
        EXPECT_FLOAT_EQ(block.lanes[0][0], 1.1f);
        EXPECT_FLOAT_EQ(block.lanes[1][0], 2.2f);
        EXPECT_FLOAT_EQ(block.lanes[2][0], 3.3f);
    }

    /**
     * @brief KD-Tree internal data layout tests.
     * These tests verify low-level properties required for performance,
     * SIMD safety, and predictable memory behavior.
     */

     // Verifies that KD-Tree node alignment and size satisfy SIMD/cache requirements.
    TEST(KDTreeDataTest, NodeAlignmentAndSize) {
        // Node must be aligned to 16 bytes (SSE-friendly alignment)
        EXPECT_EQ(alignof(KDNodeFlat), 16);

        // Node size should be a multiple of 16 to avoid cache-line fragmentation
        // and allow safe vectorized loads/stores
        EXPECT_TRUE(sizeof(KDNodeFlat) % 16 == 0);
    }

    // Verifies that dataset memory is aligned for SIMD operations used by the KD-Tree.
    TEST(KDTreeDataTest, DatasetAlignment) {
        DatasetAoS<float, 3> dataset;
        dataset.push_back({ {1.0f, 2.0f, 3.0f} });

        // Address of underlying storage
        auto address = reinterpret_cast<std::uintptr_t>(dataset.data());

        // Expect 32-byte alignment (AVX-friendly)
        // Required for fast distance computations and vector loads
        EXPECT_EQ(address % 32, 0);
    }

    // Verifies correct behavior of a newly constructed (empty) KD-Tree.
    TEST(KDTreeDataTest, TreeEmptyState) {
        KDTreeFlat tree;

        // Newly created tree must report empty
        EXPECT_TRUE(tree.empty());

        // Root index should be invalid (-1) to indicate no nodes
        EXPECT_EQ(tree.root_idx, -1);
    }

} // namespace fc::test