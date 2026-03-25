#include "pch.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

// Project-specific headers for AoS data layout and search algorithms
#include "../../AppFastClusterCPP/dataset_aos.hpp"
#include "../../AppFastClusterCPP/metrics_aos.hpp"
#include "../../AppFastClusterCPP/radius_search_aos.hpp"

namespace fc::algorithms::test {

    using namespace fc;
    using namespace fc::metrics;

    /**
     * @brief Test fixture for Array-of-Structures (AoS) radius search validation.
     * * Sets up a standardized, reproducible 2D point cloud. The geometry is specifically
     * chosen to differentiate the behavior of various distance metrics (L1, L2, L-infinity).
     * * @tparam T Coordinate scalar type (allows testing both float and double precision).
     */
    template <typename T>
    class RadiusSearchAosTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 2;
        DatasetAoS<T, Dim> dataset;

        void SetUp() override {
            // Construct a synthetic 2D dataset with known spatial relationships:
            // Indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(2,2), 4:(10,10)
            dataset.push_back(PointAoS<T, Dim>{{T(0), T(0)}});
            dataset.push_back(PointAoS<T, Dim>{{T(1), T(0)}});
            dataset.push_back(PointAoS<T, Dim>{{T(0), T(1)}});
            dataset.push_back(PointAoS<T, Dim>{{T(2), T(2)}});
            dataset.push_back(PointAoS<T, Dim>{{T(10), T(10)}});
        }
    };

    // --- Type Parameterization ---
    // Execute the entire test suite for both standard floating-point precisions
    // to ensure no internal truncations or overflow errors exist.
    using TestTypes = ::testing::Types<float, double>;
    TYPED_TEST_CASE(RadiusSearchAosTest, TestTypes);

    /**
     * @brief Edge Case: Ensures the algorithm gracefully handles empty data structures
     * without triggering out-of-bounds memory access or undefined behavior.
     */
    TYPED_TEST(RadiusSearchAosTest, EmptyDataset) {
        DatasetAoS<TypeParam, 2> empty_ds;
        PointAoS<TypeParam, 2> query{ {0, 0} };

        auto result = radius_search_brute_force_aos<TypeParam, 2, SquaredEuclideanAoS>(empty_ds, query, 1.0);
        EXPECT_TRUE(result.empty());
    }

    /**
     * @brief Validates the Squared Euclidean (L2^2) distance policy.
     * Evaluates against raw squared distances to avoid square root overhead.
     */
    TYPED_TEST(RadiusSearchAosTest, SquaredEuclideanCorrectness) {
        PointAoS<TypeParam, 2> query{ {0, 0} };

        // Search radius: 1.5. 
        // Points (1,0) and (0,1) have a squared distance of 1.0.
        // Point (2,2) has a squared distance of 8.0 (fails condition).
        TypeParam radius = 1.5;

        auto result = radius_search_brute_force_aos<TypeParam, 2, SquaredEuclideanAoS>(this->dataset, query, radius);

        // Sort results because radius search algorithms generally do not guarantee neighbor ordering
        std::sort(result.begin(), result.end());
        std::vector<std::size_t> expected = { 0, 1, 2 };
        EXPECT_EQ(result, expected);
    }

    /**
     * @brief Validates the Manhattan (L1) distance policy.
     * Sum of absolute coordinate differences: |x1 - x2| + |y1 - y2|.
     */
    TYPED_TEST(RadiusSearchAosTest, ManhattanCorrectness) {
        PointAoS<TypeParam, 2> query{ {0, 0} };

        // Distance to (2,2) is |2| + |2| = 4.0.
        // A radius of 2.1 should capture (0,0), (1,0), and (0,1), but exclude (2,2).
        TypeParam radius = 2.1;

        auto result = radius_search_brute_force_aos<TypeParam, 2, ManhattanAoS>(this->dataset, query, radius);

        std::sort(result.begin(), result.end());
        std::vector<std::size_t> expected = { 0, 1, 2 };
        EXPECT_EQ(result, expected);
    }

    /**
     * @brief Validates the Chebyshev (L-infinity) distance policy.
     * Maximum absolute coordinate difference: max(|x1 - x2|, |y1 - y2|).
     */
    TYPED_TEST(RadiusSearchAosTest, ChebyshevCorrectness) {
        PointAoS<TypeParam, 2> query{ {0, 0} };

        // Distance to (2,2) is max(2, 2) = 2.0.
        // A radius of 2.1 should capture all points except the distant outlier (10,10).
        TypeParam radius = 2.1;

        auto result = radius_search_brute_force_aos<TypeParam, 2, ChebyshevAoS>(this->dataset, query, radius);

        std::sort(result.begin(), result.end());
        std::vector<std::size_t> expected = { 0, 1, 2, 3 };
        EXPECT_EQ(result, expected);
    }

    /**
     * @brief Internal Optimization Check: Standard Euclidean (L2) distance.
     * * Crucial test to ensure that the internal optimization (squaring the search
     * radius to compare against squared distances) does not alter the logical
     * inclusion boundary of the search sphere.
     */
    TYPED_TEST(RadiusSearchAosTest, EuclideanOptimizationCorrectness) {
        PointAoS<TypeParam, 2> query{ {0, 0} };

        // True distance to (2,2) is sqrt(8) ≈ 2.828.
        // A radius of 3.0 strictly encompasses (2,2).
        TypeParam radius = 3.0;

        auto result = radius_search_brute_force_aos<TypeParam, 2, EuclideanAoS>(this->dataset, query, radius);

        std::sort(result.begin(), result.end());
        std::vector<std::size_t> expected = { 0, 1, 2, 3 };
        EXPECT_EQ(result, expected);
    }

    /**
     * @brief Edge Case: Zero radius search.
     * Evaluates exact coordinate matching capability (identity resolution).
     */
    TYPED_TEST(RadiusSearchAosTest, ZeroRadius) {
        PointAoS<TypeParam, 2> query{ {0, 0} };
        TypeParam radius = 0.0;

        auto result = radius_search_brute_force_aos<TypeParam, 2, SquaredEuclideanAoS>(this->dataset, query, radius);

        // Should strictly return the query point itself, assuming it exists in the dataset.
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result[0], 0);
    }

} // namespace fc::algorithms::test