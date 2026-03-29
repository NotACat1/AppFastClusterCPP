#include "pch.h"
#include <gtest/gtest.h>
#include "../../AppFastClusterCPP/dbscan.hpp"
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/dataset_aosoa.hpp"
#include "../../AppFastClusterCPP/dataset_soa.hpp"
#include "../../AppFastClusterCPP/metrics_aos.hpp"
#include "../../AppFastClusterCPP/metrics_aosoa.hpp"
#include "../../AppFastClusterCPP/metrics_soa.hpp"
#include "../../AppFastClusterCPP/radius_search_aos.hpp"
#include "../../AppFastClusterCPP/radius_search_aosoa.hpp"
#include "../../AppFastClusterCPP/radius_search_kdtree.hpp"
#include "../../AppFastClusterCPP/radius_search_kdtree_single.hpp"
#include "../../AppFastClusterCPP/radius_search_soa.hpp"

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::algorithms;

    /**
     * @brief Test fixture for the DBSCAN clustering algorithm.
     * Provides helper methods to initialize datasets in different memory layouts.
     */
    class DBSCANTest : public ::testing::Test {
    protected:
        // Creates a simple 3D dataset using Array of Structures (AoS) layout.
        DatasetAoS<float, 3> create_simple_dataset_aos() {
            DatasetAoS<float, 3> data;
            // Cluster 1: Centered around (0, 0, 0)
            data.push_back({ {0.0f, 0.0f, 0.0f} });
            data.push_back({ {0.1f, 0.1f, 0.1f} });
            data.push_back({ {0.2f, 0.0f, 0.1f} });

            // Cluster 2: Centered around (10, 10, 10)
            data.push_back({ {10.0f, 10.0f, 10.0f} });
            data.push_back({ {10.1f, 10.1f, 10.1f} });
            data.push_back({ {10.2f, 10.0f, 10.1f} });

            // Noise point: Isolated from both clusters
            data.push_back({ {5.0f, 5.0f, 5.0f} });
            return data;
        }

        // Creates a simple 3D dataset using Structure of Arrays (SoA) layout.
        DatasetSoA<float, 3> create_simple_dataset_soa() {
            DatasetSoA<float, 3> dataset;
            dataset.push_back({ 0.0f, 0.0f, 0.0f });
            dataset.push_back({ 0.1f, 0.0f, 0.0f });
            dataset.push_back({ 0.0f, 0.1f, 0.0f });

            dataset.push_back({ 10.0f, 10.0f, 10.0f });
            dataset.push_back({ 10.1f, 10.0f, 10.0f });
            dataset.push_back({ 10.0f, 10.1f, 10.0f });

            dataset.push_back({ 5.0f, 5.0f, 5.0f });
            return dataset;
        }

        // Creates a simple 3D dataset using SIMD-friendly AoSoA layout (8-lane width).
        DatasetAoSoA<float, 3, 8> create_simple_dataset_aosoa() {
            DatasetAoSoA<float, 3, 8> dataset;
            dataset.add_point({ 0.0f, 0.0f, 0.0f });
            dataset.add_point({ 0.1f, 0.0f, 0.0f });
            dataset.add_point({ 0.0f, 0.1f, 0.0f });

            dataset.add_point({ 10.0f, 10.0f, 10.0f });
            dataset.add_point({ 10.1f, 10.0f, 10.0f });
            dataset.add_point({ 10.0f, 10.1f, 10.0f });

            dataset.add_point({ 5.0f, 5.0f, 5.0f });
            return dataset;
        }
    };

    /**
     * @brief Test fixture for KD-Tree spatial indexing and search.
     */
    class KDTreeTest : public ::testing::Test {
    protected:
        DatasetAoS<float, 3> create_kdtree_dataset() {
            DatasetAoS<float, 3> data;
            data.push_back({ {0.0f, 0.0f, 0.0f} }); // Index 0
            data.push_back({ {1.0f, 1.0f, 1.0f} }); // Index 1
            data.push_back({ {5.0f, 5.0f, 5.0f} }); // Index 2
            return data;
        }

        // Manually constructs a balanced KD-Tree for deterministic testing.
        KDTreeFlat create_manual_kdtree() {
            KDTreeFlat tree;
            tree.nodes.resize(3);
            tree.root_idx = 0;

            // Root: Point 1 (1.0, 1.0, 1.0), split on X-axis (dim 0)
            tree.nodes[0] = { 1.0f, 1, 1, 2, 0 };

            // Left child: Point 0 (0.0, 0.0, 0.0), split on Y-axis (dim 1)
            tree.nodes[1] = { 0.0f, 0, -1, -1, 1 };

            // Right child: Point 2 (5.0, 5.0, 5.0), split on Z-axis (dim 2)
            tree.nodes[2] = { 5.0f, 2, -1, -1, 2 };

            return tree;
        }
    };

    // =========================================================
    // DBSCAN ALGORITHM TESTS
    // =========================================================

    /**
     * @test Verifies DBSCAN clustering using the AoS (Array of Structures) layout.
     */
    TEST_F(DBSCANTest, BasicClusteringAoS) {
        auto dataset = create_simple_dataset_aos();
        DBSCAN dbscan;

        // Lambda for radius search in AoS layout
        auto query_aos = [&](std::size_t idx, float r) {
            return radius_search_brute_force_aos<float, 3, metrics::EuclideanAoS>(dataset, dataset[idx], r);
            };

        auto result = dbscan.run(dataset.size(), 1.0f, 2, query_aos);

        EXPECT_EQ(result.num_clusters, 2);
        EXPECT_EQ(result.labels[0], result.labels[1]);   // Points in Cluster 1
        EXPECT_EQ(result.labels[3], result.labels[4]);   // Points in Cluster 2
        EXPECT_NE(result.labels[0], result.labels[3]);   // Clusters must be distinct
        EXPECT_EQ(result.labels[6], DBSCAN::NOISE);      // Isolated point should be noise
    }

    /**
     * @test Verifies DBSCAN clustering using the AoSoA layout with SIMD-aligned search.
     */
    TEST_F(DBSCANTest, BasicClusteringAoSoA) {
        auto dataset = create_simple_dataset_aosoa();
        DBSCAN dbscan;

        auto query_aosoa = [&](std::size_t idx, float r) {
            constexpr std::size_t SimdWidth = 8;
            const std::size_t block_idx = idx / SimdWidth;
            const std::size_t lane_idx = idx % SimdWidth;

            const auto& block = dataset.get_block(block_idx);

            // Reconstruct point from SIMD lanes for the query
            std::array<float, 3> query_point = {
                block.lanes[0][lane_idx],
                block.lanes[1][lane_idx],
                block.lanes[2][lane_idx]
            };

            return radius_search_brute_force_aosoa<metrics::EuclideanAoSoA>(dataset, query_point, r);
            };

        auto result = dbscan.run(dataset.size(), 1.0f, 2, query_aosoa);

        EXPECT_EQ(result.num_clusters, 2);
        EXPECT_EQ(result.labels[0], result.labels[1]);
        EXPECT_EQ(result.labels[3], result.labels[4]);
        EXPECT_NE(result.labels[0], result.labels[3]);
        EXPECT_EQ(result.labels[6], DBSCAN::NOISE);
    }

    /**
     * @test Verifies DBSCAN clustering using the SoA (Structure of Arrays) layout.
     */
    TEST_F(DBSCANTest, BasicClusteringSoA) {
        auto dataset = create_simple_dataset_soa();
        DBSCAN dbscan;

        auto query_soa = [&](std::size_t idx, float r) {
            std::array<float, 3> query_point = {
                dataset.axis_data(0)[idx],
                dataset.axis_data(1)[idx],
                dataset.axis_data(2)[idx]
            };

            return radius_search_brute_force_soa<float, 3, metrics::EuclideanSoA>(dataset, query_point, r);
            };

        auto result = dbscan.run(dataset.size(), 1.0f, 2, query_soa);

        EXPECT_EQ(result.num_clusters, 2);
        EXPECT_EQ(result.labels[0], result.labels[1]);
        EXPECT_EQ(result.labels[3], result.labels[4]);
        EXPECT_NE(result.labels[0], result.labels[3]);
        EXPECT_EQ(result.labels[6], DBSCAN::NOISE);
    }

    /**
     * @test Ensures DBSCAN handles empty input datasets gracefully.
     */
    TEST_F(DBSCANTest, EmptyDataset) {
        DBSCAN dbscan;
        auto query = [](std::size_t, float) { return std::vector<std::size_t>{}; };
        auto result = dbscan.run(0, 1.0f, 2, query);
        EXPECT_EQ(result.num_clusters, 0);
        EXPECT_TRUE(result.labels.empty());
    }

    // =========================================================
    // KD-TREE SEARCH TESTS
    // =========================================================

    /**
     * @test Verifies that a single KD-Tree radius search finds all expected points.
     */
    TEST_F(KDTreeTest, SingleSearch_FindsPointsInRadius) {
        auto dataset = create_kdtree_dataset();
        auto tree = create_manual_kdtree();

        PointAoS<float, 3> query_pt = { 0.0f, 0.0f, 0.0f };
        float radius = 2.0f;

        auto result = radius_search_kdtree_single<float, 3, metrics::EuclideanAoS>(tree, dataset, query_pt, radius);

        ASSERT_EQ(result.size(), 2);

        // Check for presence of indices 0 and 1
        bool found_0 = (result[0] == 0 || result[1] == 0);
        bool found_1 = (result[0] == 1 || result[1] == 1);

        EXPECT_TRUE(found_0);
        EXPECT_TRUE(found_1);
    }

    /**
     * @test Ensures radius search on an empty KD-Tree returns an empty result set.
     */
    TEST_F(KDTreeTest, SingleSearch_EmptyTree) {
        DatasetAoS<float, 3> dataset;
        KDTreeFlat tree;

        PointAoS<float, 3> query_pt = { 0.0f, 0.0f, 0.0f };

        auto result = radius_search_kdtree_single<float, 3, metrics::EuclideanAoS>(tree, dataset, query_pt, 1.0f);

        EXPECT_TRUE(result.empty());
    }

    /**
     * @test Verifies batch radius search correctly processes multiple query points.
     */
    TEST_F(KDTreeTest, BatchSearch_FindsPointsInRadiusForMultipleQueries) {
        auto dataset = create_kdtree_dataset();
        auto tree = create_manual_kdtree();

        DatasetAoS<float, 3> queries = {
            PointAoS<float, 3>{{0.0f, 0.0f, 0.0f}},
            PointAoS<float, 3>{{6.0f, 6.0f, 6.0f}}
        };

        float radius = 2.0f;

        auto results = radius_search_kdtree_batch<float, 3, metrics::EuclideanAoS>(tree, dataset, queries, radius);

        ASSERT_EQ(results.size(), 2);

        // First query results
        EXPECT_EQ(results[0].size(), 2);
        bool q0_found_0 = (results[0][0] == 0 || results[0][1] == 0);
        bool q0_found_1 = (results[0][0] == 1 || results[0][1] == 1);
        EXPECT_TRUE(q0_found_0);
        EXPECT_TRUE(q0_found_1);

        // Second query results
        ASSERT_EQ(results[1].size(), 1);
        EXPECT_EQ(results[1][0], 2);
    }

} // namespace fc::algorithms::test