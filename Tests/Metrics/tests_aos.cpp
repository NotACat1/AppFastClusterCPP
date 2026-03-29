#include "pch.h"
#include <gtest/gtest.h>
#include "../../AppFastClusterCPP/metrics_aos.hpp"

namespace fc::metrics::test {

    template <typename T, std::size_t Dim>
    using Point = PointAoS<T, Dim>;

    /**
     * @brief Test fixture for Distance Metrics.
     * @details Leverages GTest's Typed Tests to validate implementations
     * across both single (float) and double precision types.
     */
    template <typename T>
    class MetricsAosTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;

        // Test vectors for standard geometric verification
        const Point<T, Dim> origin = { {0, 0, 0} };
        const Point<T, Dim> p1 = { {3, 4, 0} };
        const Point<T, Dim> p2 = { {-1, -1, -1} };
        const Point<T, Dim> p3 = { {1, 2, 3} };
    };

    using CoordinateTypes = ::testing::Types<float, double>;
    TYPED_TEST_CASE(MetricsAosTest, CoordinateTypes);

    // --- Correctness Verification ---

    TYPED_TEST(MetricsAosTest, Correctness) {
        using T = TypeParam;
        // Calculation: (3-0)^2 + (4-0)^2 + 0^2 = 9 + 16 = 25
        EXPECT_EQ(SquaredEuclideanAoS::evaluate(this->origin, this->p1), static_cast<T>(25));

        // Euclidean
        EXPECT_NEAR(EuclideanAoS::evaluate(this->origin, this->p1), static_cast<T>(5), 1e-6);

        // Manhattan
        EXPECT_EQ(ManhattanAoS::evaluate(this->origin, this->p1), static_cast<T>(7));

        // Chebyshev
        EXPECT_EQ(ChebyshevAoS::evaluate(this->origin, this->p1), static_cast<T>(4));
    }

    TYPED_TEST(MetricsAosTest, Properties) {
        // Identity
        EXPECT_EQ(SquaredEuclideanAoS::evaluate(this->p3, this->p3), 0);
        // Symmetry
        auto d12 = EuclideanAoS::evaluate(this->p1, this->p2);
        auto d21 = EuclideanAoS::evaluate(this->p2, this->p1);
        EXPECT_NEAR(d12, d21, 1e-7);
    }

    TYPED_TEST(MetricsAosTest, Dispatcher) {
        using T = TypeParam;
        auto res = compute_distance_aos<T, 3, ManhattanAoS>(this->origin, this->p1);
        EXPECT_EQ(res, static_cast<T>(7));
    }

    TEST(MetricsAosInfrastructure, AlignmentAndConcepts) {
        // Alignment
        EXPECT_EQ(alignof(PointAoS<float, 3>), 32);

        // Concepts
        static_assert(MetricAoS<SquaredEuclideanAoS, float, 3>);
        static_assert(MetricAoS<ManhattanAoS, double, 10>);
    }

} // namespace fc::metrics::test