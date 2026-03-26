#include "pch.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

/* Project-specific headers for spatial indexing and metrics */
#include "../../AppFastClusterCPP/kd_tree_flat.hpp"
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/metrics_aos.hpp"
#include "../../AppFastClusterCPP/radius_search_kdtree_single.hpp"

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @class KDTreeRadiusSearchSingleTest
     * @brief Test fixture for validating single-query radius search performance and correctness.
     */
    class KDTreeRadiusSearchSingleTest : public ::testing::Test {
    protected:
        using T = float;
        static constexpr std::size_t Dim = 2;

        DatasetAoS<T, Dim> dataset;
        KDTreeFlat tree;

        /** * @brief Utility to construct a PointAoS with specified coordinates.
         * Ensures type-safety and consistency with the template parameters.
         */
        PointAoS<T, Dim> make_point(std::initializer_list<T> coords) {
            PointAoS<T, Dim> p;
            std::copy(coords.begin(), coords.end(), p.coords.begin());
            return p;
        }

        /**
         * @brief Sets up a deterministic environment for testing.
         * Manually constructs a balanced KD-tree to verify traversal logic independently of the builder.
         */
        void SetUp() override {
            // Initialize 2D dataset: 0:(0,0), 1:(1,1), 2:(2,2), 3:(10,10)
            dataset = {
                make_point({0.0f, 0.0f}),
                make_point({1.0f, 1.0f}),
                make_point({2.0f, 2.0f}),
                make_point({10.0f, 10.0f})
            };

            tree.nodes.resize(4);

            // Root node: Using point 1 as pivot on X-axis (Dim 0)
            tree.root_idx = 1;
            tree.nodes[1] = { 1.0f, 1, 0, 2, 0 };

            // Left branch: Leaf node containing point 0, split on Y-axis (Dim 1)
            tree.nodes[0] = { 0.0f, 0, -1, -1, 1 };

            // Right branch: Internal node containing point 2, split on Y-axis (Dim 1)
            tree.nodes[2] = { 2.0f, 2, -1, 3, 1 };

            // Distant leaf: Point 3
            tree.nodes[3] = { 10.0f, 3, -1, -1, 0 };
        }

        /** @brief Verifies if a specific index exists within the search result set. */
        bool contains(const std::vector<std::size_t>& vec, std::size_t val) {
            return std::find(vec.begin(), vec.end(), val) != vec.end();
        }
    };

    // --- Unit Tests ---

    /** @test Validates that searching an uninitialized/empty tree returns no results. */
    TEST_F(KDTreeRadiusSearchSingleTest, ReturnsEmptyOnEmptyTree) {
        KDTreeFlat empty_tree;
        PointAoS<T, Dim> query = make_point({ 0.0f, 0.0f });

        auto result = radius_search_kdtree_single<T, Dim, SquaredEuclideanAoS>(empty_tree, dataset, query, T(1.0));
        EXPECT_TRUE(result.empty());
    }

    /** @test Verifies that a zero-radius search correctly identifies only the identical point. */
    TEST_F(KDTreeRadiusSearchSingleTest, ZeroRadiusFindsSelf) {
        PointAoS<T, Dim> query = make_point({ 1.0f, 1.0f });
        auto result = radius_search_kdtree_single<T, Dim, SquaredEuclideanAoS>(tree, dataset, query, T(0.0));

        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result[0], 1);
    }

    /** @test Checks standard Euclidean distance search logic for multiple neighbors. */
    TEST_F(KDTreeRadiusSearchSingleTest, FindsMultiplePointsEuclidean) {
        PointAoS<T, Dim> query = make_point({ 0.5f, 0.5f });
        T radius = 1.0f;

        auto result = radius_search_kdtree_single<T, Dim, EuclideanAoS>(tree, dataset, query, radius);

        EXPECT_EQ(result.size(), 2);
        EXPECT_TRUE(contains(result, 0));
        EXPECT_TRUE(contains(result, 1));
    }

    /** @test Validates the Manhattan (L1) metric implementation. */
    TEST_F(KDTreeRadiusSearchSingleTest, ManhattanMetricSearch) {
        PointAoS<T, Dim> query = make_point({ 0.0f, 0.0f });

        // Search with radius 1.5: should only find point (0,0)
        auto res1 = radius_search_kdtree_single<T, Dim, ManhattanAoS>(tree, dataset, query, T(1.5));
        EXPECT_EQ(res1.size(), 1);
        EXPECT_EQ(res1[0], 0);

        // Search with radius 2.1: should include point (1,1) as |1-0| + |1-0| = 2.0
        auto res2 = radius_search_kdtree_single<T, Dim, ManhattanAoS>(tree, dataset, query, T(2.1));
        EXPECT_EQ(res2.size(), 2);
        EXPECT_TRUE(contains(res2, 1));
    }

    /** @test Verifies the pruning logic to ensure distant branches are effectively skipped. */
    TEST_F(KDTreeRadiusSearchSingleTest, DoesNotVisitFarBranches) {
        PointAoS<T, Dim> query = make_point({ 11.0f, 11.0f });
        T radius = 2.0f;

        // Use squared radius for SquaredEuclidean metric to avoid unnecessary sqrt calls
        auto result = radius_search_kdtree_single<T, Dim, SquaredEuclideanAoS>(tree, dataset, query, radius * radius);

        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result[0], 3); // Point (10,10)
    }

} // namespace fc::algorithms::test