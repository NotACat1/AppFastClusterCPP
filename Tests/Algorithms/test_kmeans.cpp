#include "pch.h"
#include <gtest/gtest.h>
#include <vector>

#include "../../AppFastClusterCPP/kmeans.hpp" 

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

    using namespace fc::algorithms;

    /**
     * @class KMeansTest
     * @brief Unit test fixture for validating the KMeans clustering algorithm.
     * * Provides utility methods for synthetic dataset generation and manages
     * shared state for KMeans functional testing.
     */
    class KMeansTest : public ::testing::Test {
    protected:
        /**
         * @brief Generates a synthetic 2D dataset with two well-separated clusters.
         * * Layout:
         * - Cluster 1: Points localized around the origin (0, 0).
         * - Cluster 2: Points localized around (10, 10).
         * @return Flat vector of float values (interleaved X, Y coordinates).
         */
        std::vector<float> create_simple_2d_dataset() {
            return {
                0.0f, 0.0f,   // Point 0 (Cluster 1 candidate)
                0.1f, 0.0f,   // Point 1
                0.0f, 0.1f,   // Point 2
                10.0f, 10.0f, // Point 3 (Cluster 2 candidate)
                10.1f, 10.0f, // Point 4
                10.0f, 10.1f  // Point 5
            };
        }

        L2SquaredDistance l2_dist;
    };

    // =========================================================
    // FUNCTIONAL & BOUNDARY TESTS
    // =========================================================

    /**
     * @test Verifies that the algorithm correctly identifies and partitions
     * clearly separated clusters in a 2D space.
     */
    TEST_F(KMeansTest, BasicClustering) {
        auto data = create_simple_2d_dataset();
        KMeans kmeans;

        const std::size_t num_points = 6;
        const std::size_t dim = 2;
        const std::size_t k = 2;
        const std::size_t max_iters = 100;

        auto result = kmeans.run(data.data(), num_points, dim, k, max_iters, l2_dist, 42);

        // Validate output structure dimensions
        ASSERT_EQ(result.labels.size(), num_points);
        ASSERT_EQ(result.centroids.size(), k * dim);
        EXPECT_GT(result.num_iterations, 0);

        // Verification: Ensure spatial locality for the first cluster
        EXPECT_EQ(result.labels[0], result.labels[1]);
        EXPECT_EQ(result.labels[0], result.labels[2]);

        // Verification: Ensure spatial locality for the second cluster
        EXPECT_EQ(result.labels[3], result.labels[4]);
        EXPECT_EQ(result.labels[3], result.labels[5]);

        // Verification: Ensure clusters are distinct
        EXPECT_NE(result.labels[0], result.labels[3]);
    }

    /**
     * @test Ensures the algorithm handles invalid cluster counts (K=0) gracefully
     * without causing crashes or undefined behavior.
     */
    TEST_F(KMeansTest, ZeroK) {
        auto data = create_simple_2d_dataset();
        KMeans kmeans;

        auto result = kmeans.run(data.data(), 6, 2, 0, 100, l2_dist);

        EXPECT_TRUE(result.labels.empty());
        EXPECT_TRUE(result.centroids.empty());
    }

    /**
     * @test Validates the internal safeguard that caps the number of clusters (K)
     * to the total number of available data points (N).
     */
    TEST_F(KMeansTest, KGreaterThanNumPoints) {
        std::vector<float> data = {
            1.0f, 2.0f, // Point 0
            3.0f, 4.0f  // Point 1
        };
        KMeans kmeans;

        const std::size_t num_points = 2;
        const std::size_t dim = 2;
        const std::size_t requested_k = 5; // Intentionally out-of-bounds

        auto result = kmeans.run(data.data(), num_points, dim, requested_k, 100, l2_dist, 42);

        // The implementation must clamp K to num_points
        EXPECT_EQ(result.centroids.size(), num_points * dim);
        ASSERT_EQ(result.labels.size(), num_points);

        // With K=N, each point should effectively become its own centroid
        EXPECT_NE(result.labels[0], result.labels[1]);
    }

    /**
     * @test Confirms the deterministic behavior of the algorithm when a fixed
     * random seed is provided. Crucial for regression testing and debugging.
     */
    TEST_F(KMeansTest, ReproducibilityWithSeed) {
        auto data = create_simple_2d_dataset();
        KMeans kmeans;

        const std::size_t num_points = 6;
        const std::size_t dim = 2;
        const std::size_t k = 2;
        const uint32_t seed = 12345;

        // Perform two independent runs with identical parameters
        auto result1 = kmeans.run(data.data(), num_points, dim, k, 100, l2_dist, seed);
        auto result2 = kmeans.run(data.data(), num_points, dim, k, 100, l2_dist, seed);

        // Verify that results are bit-identical
        EXPECT_EQ(result1.labels, result2.labels);
        EXPECT_EQ(result1.centroids, result2.centroids);
        EXPECT_EQ(result1.num_iterations, result2.num_iterations);
    }

} // namespace fc::algorithms::test