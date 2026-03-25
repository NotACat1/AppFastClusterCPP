#include "pch.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

/* Project-specific headers for KD-Tree, AoS data layout, and metrics */
#include "../../AppFastClusterCPP/kd_tree_flat.hpp"
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/metrics_aos.hpp"
#include "../../AppFastClusterCPP/radius_search_kdtree.hpp" 

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Helper function to create a PointAoS instance from an initializer list.
     * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @param coords List of coordinates to initialize the point.
     * @return PointAoS<T, Dim> Initialized point structure.
     */
    template<typename T, std::size_t Dim>
    PointAoS<T, Dim> make_point(std::initializer_list<T> coords) {
        PointAoS<T, Dim> p;
        std::copy(coords.begin(), coords.end(), p.coords.begin());
        return p;
    }

    /**
     * @class KDTreeRadiusSearchTest
     * @brief Test fixture for validating the KD-Tree batch radius search algorithm.
     */
    class KDTreeRadiusSearchTest : public ::testing::Test {
    protected:
        using T = float;
        static constexpr std::size_t Dim = 2;

        DatasetAoS<T, Dim> dataset;
        KDTreeFlat tree;

        /**
         * @brief Prepares a deterministic dataset and a manually constructed KD-Tree.
         * The manual construction ensures predictable traversal paths for unit testing.
         */
        void SetUp() override {
            // Sample 2D points: 0:(0,0), 1:(1,1), 2:(2,2), 3:(10,10), 4:(1,0)
            dataset = {
                make_point<T, Dim>({0.0f, 0.0f}),
                make_point<T, Dim>({1.0f, 1.0f}),
                make_point<T, Dim>({2.0f, 2.0f}),
                make_point<T, Dim>({10.0f, 10.0f}),
                make_point<T, Dim>({1.0f, 0.0f})
            };

            /* * Manual KD-Tree Construction:
             * Root: Point 1 (1,1), SplitDim=0 (X), SplitVal=1.0
             * Left: [0(0,0), 4(1,0)] -> Node 0(0,0), SplitDim=1 (Y), SplitVal=0.0
             * Right: [2(2,2), 3(10,10)] -> Node 2(2,2), SplitDim=0 (X), SplitVal=2.0
             */
            tree.nodes.resize(5);

            // Root Node
            tree.root_idx = 1;
            tree.nodes[1] = { 1.0f, 1, 0, 2, 0 }; // Pt 1, Left->0, Right->2, Dim 0

            // Left Branch
            tree.nodes[0] = { 0.0f, 0, -1, 4, 1 }; // Pt 0, Left->None, Right->4, Dim 1
            tree.nodes[4] = { 0.0f, 4, -1, -1, 0 }; // Leaf Point 4

            // Right Branch
            tree.nodes[2] = { 2.0f, 2, -1, 3, 0 }; // Pt 2, Left->None, Right->3, Dim 0
            tree.nodes[3] = { 10.0f, 3, -1, -1, 0 }; // Leaf Point 3
        }

        /** @brief Utility to verify if a point index exists in the search result. */
        bool contains(const std::vector<std::size_t>& vec, std::size_t val) {
            return std::find(vec.begin(), vec.end(), val) != vec.end();
        }
    };

    /** @test Verifies that the algorithm handles empty trees or empty query sets gracefully. */
    TEST_F(KDTreeRadiusSearchTest, EmptyInputs) {
        KDTreeFlat empty_tree;
        DatasetAoS<T, Dim> queries = { make_point<T, Dim>({0, 0}) };

        auto results = radius_search_kdtree_batch<T, Dim, SquaredEuclideanAoS>(empty_tree, dataset, queries, 1.0f);
        EXPECT_TRUE(results.empty());

        DatasetAoS<T, Dim> empty_queries;
        auto results2 = radius_search_kdtree_batch<T, Dim, SquaredEuclideanAoS>(tree, dataset, empty_queries, 1.0f);
        EXPECT_TRUE(results2.empty());
    }

    /** @test Validates correctness for a standard query point within a specific radius. */
    TEST_F(KDTreeRadiusSearchTest, SingleQueryBasic) {
        DatasetAoS<T, Dim> queries = { make_point<T, Dim>({0.5f, 0.5f}) };
        T radius = 1.0f; // Squared distance threshold

        auto results = radius_search_kdtree_batch<T, Dim, SquaredEuclideanAoS>(tree, dataset, queries, radius);

        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].size(), 3); // Expected points: 0, 1, and 4
        EXPECT_TRUE(contains(results[0], 0));
        EXPECT_TRUE(contains(results[0], 1));
        EXPECT_TRUE(contains(results[0], 4));
    }

    /** @test Verifies behavior across different distance metrics (L2 vs Squared L2). */
    TEST_F(KDTreeRadiusSearchTest, EuclideanMetricLogic) {
        DatasetAoS<T, Dim> queries = { make_point<T, Dim>({0.0f, 0.0f}) };
        T radius = 1.1f;

        // With SquaredEuclidean, the radius is treated as dist^2
        auto res_sq = radius_search_kdtree_batch<T, Dim, SquaredEuclideanAoS>(tree, dataset, queries, radius);

        // With Euclidean, the radius is the actual distance (internally squared during pruning)
        auto res_eucl = radius_search_kdtree_batch<T, Dim, EuclideanAoS>(tree, dataset, queries, radius);

        // Point (1,0) has dist^2 = 1.0. Since 1.0 < 1.1 and 1.0 < (1.1^2), it should be included in both.
        EXPECT_TRUE(contains(res_sq[0], 4));
        EXPECT_TRUE(contains(res_eucl[0], 4));
    }

} // namespace fc::algorithms::test