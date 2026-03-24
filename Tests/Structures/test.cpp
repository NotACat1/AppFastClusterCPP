#include "pch.h"
#include <gtest/gtest.h>
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/dataset_soa.hpp"
#include "../../AppFastClusterCPP/dataset_aosoa.hpp"

/**
 * @brief Unit tests for verifying data integrity across different memory layouts.
 */

 // AoS Layout: Verifies that points are stored as contiguous objects and indexed correctly.
TEST(DatasetTest, AoSStoreAndRetrieve) {
    fc::DatasetAoS<float, 3> ds;
    fc::PointAoS<float, 3> p{ {1.0f, 2.0f, 3.0f} };
    ds.push_back(p);

    EXPECT_EQ(ds.size(), 1);
    EXPECT_FLOAT_EQ(ds[0][0], 1.0f);
    EXPECT_FLOAT_EQ(ds[0][2], 3.0f);
}

// SoA Layout: Verifies that coordinates are correctly transposed into separate axis buffers.
TEST(DatasetTest, SoAStoreAndRetrieve) {
    fc::DatasetSoA<float, 3> ds;
    ds.push_back({ 10.0f, 20.0f, 30.0f });

    EXPECT_EQ(ds.size(), 1);
    // Validate that data is correctly routed to specific dimension arrays
    EXPECT_FLOAT_EQ(ds.axis_data(0)[0], 10.0f);
    EXPECT_FLOAT_EQ(ds.axis_data(2)[0], 30.0f);
}

// AoSoA Layout: Verifies correct placement in SIMD lanes and block-level indexing.
TEST(DatasetTest, AoSoAStoreAndRetrieve) {
    fc::DatasetAoSoA<float, 3, 8> ds;
    ds.add_point({ 1.1f, 2.2f, 3.3f });

    // Verify first lane of the primary SIMD block
    auto& block = ds.get_block(0);
    EXPECT_FLOAT_EQ(block.lanes[0][0], 1.1f);
    EXPECT_FLOAT_EQ(block.lanes[1][0], 2.2f);
    EXPECT_FLOAT_EQ(block.lanes[2][0], 3.3f);
}