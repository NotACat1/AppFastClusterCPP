#include "pch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

// Project-specific headers for SoA (Structure of Arrays) data structures and search algorithms
#include "../../AppFastClusterCPP/radius_search_soa.hpp" 
#include "../../AppFastClusterCPP/dataset_soa.hpp"
#include "../../AppFastClusterCPP/metrics_soa.hpp"

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::metrics;

    // Type aliases for test consistency
    using T = float;
    constexpr std::size_t Dim = 3;

    /**
     * @brief Utility helper to populate a DatasetSoA from a standard vector of arrays.
     * Useful for setting up known configurations in unit tests.
     */
    DatasetSoA<T, Dim> create_test_dataset(const std::vector<std::array<T, Dim>>& points) {
        DatasetSoA<T, Dim> ds;
        for (const auto& p : points) {
            ds.push_back(p);
        }
        return ds;
    }

    /**
     * @brief Checks if a specific index exists within the result vector.
     */
    bool contains(const std::vector<std::size_t>& vec, std::size_t index) {
        return std::find(vec.begin(), vec.end(), index) != vec.end();
    }

    // --- Unit Tests ---

    // Verifies that searching an empty dataset returns an empty result set without crashing.
    TEST(RadiusSearchSoA, EmptyDataset) {
        DatasetSoA<T, Dim> empty_ds;
        std::array<T, Dim> query = { 0, 0, 0 };

        auto results = radius_search_brute_force_soa<T, Dim, SquaredEuclideanPolicy>(empty_ds, query, 1.0f);

        EXPECT_TRUE(results.empty());
    }

    // Ensures that points outside the specified radius are correctly filtered out.
    TEST(RadiusSearchSoA, NoPointsInRange) {
        auto ds = create_test_dataset({
            {10.0f, 10.0f, 10.0f},
            {20.0f, 20.0f, 20.0f}
            });
        std::array<T, Dim> query = { 0, 0, 0 };

        // Expected distance to closest point is ~17.3; search radius is 5.0.
        auto results = radius_search_brute_force_soa<T, Dim, SquaredEuclideanPolicy>(ds, query, 5.0f);

        EXPECT_TRUE(results.empty());
    }

    // Validates that all points are returned when the radius is sufficiently large.
    TEST(RadiusSearchSoA, AllPointsInRange) {
        auto ds = create_test_dataset({
            {0.1f, 0.1f, 0.1f},
            {0.2f, 0.0f, 0.1f},
            {0.0f, 0.1f, 0.0f}
            });
        std::array<T, Dim> query = { 0, 0, 0 };

        auto results = radius_search_brute_force_soa<T, Dim, SquaredEuclideanPolicy>(ds, query, 10.0f);

        EXPECT_EQ(results.size(), 3);
    }

    // Tests the logic of SquaredEuclideanPolicy (compares sum of squares directly to radius).
    TEST(RadiusSearchSoA, SquaredEuclideanPolicyCorrectness) {
        auto ds = create_test_dataset({
            {1.0f, 0.0f, 0.0f}, // dist^2 = 1
            {1.0f, 1.0f, 0.0f}, // dist^2 = 2
            {2.0f, 0.0f, 0.0f}  // dist^2 = 4
            });
        std::array<T, Dim> query = { 0, 0, 0 };

        auto results = radius_search_brute_force_soa<T, Dim, SquaredEuclideanPolicy>(ds, query, 2.5f);

        EXPECT_EQ(results.size(), 2);
        EXPECT_TRUE(contains(results, 0));
        EXPECT_TRUE(contains(results, 1));
        EXPECT_FALSE(contains(results, 2));
    }

    /**
     * @brief Tests the EuclideanPolicy optimization.
     * Verifies that the internal logic squares the input radius to perform comparisons
     * in squared space, avoiding expensive sqrt() calls.
     */
    TEST(RadiusSearchSoA, EuclideanPolicySpecialization) {
        auto ds = create_test_dataset({
            {1.0f, 1.0f, 1.0f} // Euclidean dist = sqrt(3) ≈ 1.732
            });
        std::array<T, Dim> query = { 0, 0, 0 };

        // Input radius = 2.0. Internal effective_radius should be 4.0.
        // Since 3.0 (dist^2) <= 4.0, the point must be found.
        auto results = radius_search_brute_force_soa<T, Dim, EuclideanPolicy>(ds, query, 2.0f);

        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0], 0);
    }

    // Validates correctness for Manhattan (L1) distance metric.
    TEST(RadiusSearchSoA, ManhattanPolicyCorrectness) {
        auto ds = create_test_dataset({
            {1.0f, 1.0f, 1.0f}, // L1 = 3.0
            {2.0f, 0.0f, 0.5f}  // L1 = 2.5
            });
        std::array<T, Dim> query = { 0, 0, 0 };

        auto results = radius_search_brute_force_soa<T, Dim, ManhattanPolicy>(ds, query, 2.7f);

        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0], 1);
    }

    /**
     * @brief Stress test for large datasets and OpenMP parallelization.
     * Verifies thread safety and ensures that the merging of local thread results
     * does not introduce duplicate indices or data corruption.
     */
    TEST(RadiusSearchSoA, LargeDatasetOpenMP) {
        DatasetSoA<T, Dim> ds;
        const std::size_t count = 10000;
        for (std::size_t i = 0; i < count; ++i) {
            ds.push_back({ static_cast<T>(i), 0, 0 });
        }

        std::array<T, Dim> query = { 0, 0, 0 };
        T radius = 100.0f;

        // For SquaredEuclidean, points with index 0 to 100 (101 points total) should be within range.
        auto results = radius_search_brute_force_soa<T, Dim, SquaredEuclideanPolicy>(ds, query, radius * radius);

        EXPECT_EQ(results.size(), 101);

        // Sort and check for uniqueness to catch potential race conditions in parallel result merging.
        std::sort(results.begin(), results.end());
        auto it = std::unique(results.begin(), results.end());
        EXPECT_EQ(it, results.end()) << "Duplicate indices found in parallel result! Check OpenMP reduction/critical sections.";
    }

} // namespace fc::algorithms::test