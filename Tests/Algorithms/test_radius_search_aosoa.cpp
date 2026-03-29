#include "pch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "../../AppFastClusterCPP/radius_search_aosoa.hpp"
#include "../../AppFastClusterCPP/metrics_aosoa.hpp"

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::metrics;

    /**
     * @brief Helper utility to populate an AoSoA dataset from a standard vector of points.
     */
    fc::DatasetAoSoA<float, 3, 8> create_dataset(const std::vector<std::array<float, 3>>& points) {
        fc::DatasetAoSoA<float, 3, 8> ds;
        for (const auto& p : points) {
            ds.add_point(p);
        }
        return ds;
    }

    class RadiusSearchAoSoATest : public ::testing::Test {
    protected:
        std::array<float, 3> query = { 0.0f, 0.0f, 0.0f };
    };

    /**
     * 1. Basic Functionality & Block Boundary Test
     * Verifies filtering correctness, specifically ensuring points are correctly
     * identified across multiple SIMD blocks (e.g., 8 points in the first block, 2 in the second).
     */
    TEST_F(RadiusSearchAoSoATest, BasicFunctionality) {
        // Create 10 points: 8 fully occupy the first block, 2 overflow into the second block.
        std::vector<std::array<float, 3>> points = {
            {0.1f, 0.0f, 0.0f}, {0.2f, 0.0f, 0.0f}, {0.3f, 0.0f, 0.0f}, {0.4f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f}, {0.6f, 0.0f, 0.0f}, {0.7f, 0.0f, 0.0f}, {0.8f, 0.0f, 0.0f},
            {0.9f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f} // Index #9 is outside the radius
        };
        auto ds = create_dataset(points);

        // Search within radius 0.55
        float radius = 0.55f;
        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::SquaredEuclideanAoSoA>(ds, query, radius * radius);

        std::sort(indices.begin(), indices.end());
        std::vector<std::size_t> expected = { 0, 1, 2, 3, 4 };
        EXPECT_EQ(indices, expected);
    }

    /**
     * 2. Tail Handling Test
     * Ensures the search does not collect "garbage" data from uninitialized
     * SIMD lanes in the final partially-filled block.
     */
    TEST_F(RadiusSearchAoSoATest, TailHandlingTest) {
        std::vector<std::array<float, 3>> points = {
            {0.0f, 0.0f, 0.0f}, // Index #0
            {0.1f, 0.1f, 0.1f}  // Index #1
        };
        auto ds = create_dataset(points);

        // The block has 8 slots; 6 are uninitialized. 
        // If 'points_in_block' is handled incorrectly, the search might return > 2 results.
        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::ManhattanAoSoA>(ds, query, 100.0f);

        EXPECT_EQ(indices.size(), 2);
    }

    /**
     * 3. Manhattan (L1) Metric Test
     * Validates SIMD mask operations for L1 distance calculation.
     */
    TEST_F(RadiusSearchAoSoATest, ManhattanSimdCorrectness) {
        std::vector<std::array<float, 3>> points = {
            {-1.0f, -1.0f, -1.0f}, // L1 = 3.0
            { 1.0f,  1.0f,  0.5f}  // L1 = 2.5
        };
        auto ds = create_dataset(points);

        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::ManhattanAoSoA>(ds, query, 2.6f);

        ASSERT_EQ(indices.size(), 1);
        EXPECT_EQ(indices[0], 1);
    }

    /**
     * 4. Chebyshev (L-infinity) Metric Test
     * Validates SIMD MAX operations for L-inf distance calculation.
     */
    TEST_F(RadiusSearchAoSoATest, ChebyshevSimdCorrectness) {
        std::vector<std::array<float, 3>> points = {
            {10.0f, 0.1f, 0.1f}, // Max diff = 10.0
            {2.0f, 2.0f, 2.0f}    // Max diff = 2.0
        };
        auto ds = create_dataset(points);

        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::ChebyshevAoSoA>(ds, query, 3.0f);

        ASSERT_EQ(indices.size(), 1);
        EXPECT_EQ(indices[0], 1);
    }

    /**
     * 5. Euclidean Optimization Test
     * Verifies that searching with the Euclidean metric correctly utilizes
     * squared radius comparisons to avoid expensive square root operations.
     */
    TEST_F(RadiusSearchAoSoATest, EuclideanOptimizationTest) {
        std::vector<std::array<float, 3>> points = {
            {1.0f, 1.0f, 1.0f} // Distance = sqrt(3) ≈ 1.732
        };
        auto ds = create_dataset(points);

        // Radius 1.8. If the optimization (comparing radius^2) is correct, the point should be found.
        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::EuclideanAoSoA>(ds, query, 1.8f);

        EXPECT_EQ(indices.size(), 1);
    }

    /**
     * 6. Parallel Execution Consistency
     * Checks for OpenMP race conditions. If the results vector is not handled
     * thread-safely, the size may fluctuate or be lower than expected.
     */
    TEST_F(RadiusSearchAoSoATest, ParallelConsistency) {
        const int count = 1000;
        std::vector<std::array<float, 3>> points;
        for (int i = 0; i < count; ++i) points.push_back({ 0.001f * i, 0.0f, 0.0f });

        auto ds = create_dataset(points);

        // This search should return exactly 101 points (0.0 to 0.1 inclusive)
        auto indices = fc::algorithms::radius_search_brute_force_aosoa<fc::metrics::SquaredEuclideanAoSoA>(ds, query, 0.1f * 0.1f);

        // Check for consistency; common race conditions would result in significant deviations.
        EXPECT_TRUE(indices.size() >= 100 && indices.size() <= 101);
    }

}