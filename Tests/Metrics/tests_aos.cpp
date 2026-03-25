#include "pch.h"
#include <gtest/gtest.h>
#include "../../AppFastClusterCPP/metric_aos.hpp"

namespace fc::metrics::test {

    template <typename T, std::size_t Dim>
    using Point = PointAoS<T, Dim>;

    /**
     * @brief Test fixture for Distance Metrics.
     * @details Leverages GTest's Typed Tests to validate implementations
     * across both single (float) and double precision types.
     */
    template <typename T>
    class MetricsTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;

        // Test vectors for standard geometric verification
        const Point<T, Dim> origin = { {0, 0, 0} };
        const Point<T, Dim> p1 = { {3, 4, 0} };   // Classic 3-4-5 Pythagorean triple
        const Point<T, Dim> p2 = { {-1, -1, -1} };
        const Point<T, Dim> p3 = { {1, 2, 3} };
    };

    using CoordinateTypes = ::testing::Types<float, double>;
    TYPED_TEST_CASE(MetricsTest, CoordinateTypes);

    // --- Correctness Verification ---

    TYPED_TEST(MetricsTest, SquaredEuclideanCorrectness) {
        using T = TypeParam;
        // Calculation: (3-0)^2 + (4-0)^2 + 0^2 = 9 + 16 = 25
        EXPECT_EQ(SquaredEuclideanAoS::evaluate(this->origin, this->p1), static_cast<T>(25));

        // Calculation: (1 - (-1))^2 + (2 - (-1))^2 + (3 - (-1))^2 = 4 + 9 + 16 = 29
        EXPECT_EQ(SquaredEuclideanAoS::evaluate(this->p2, this->p3), static_cast<T>(29));
    }

    TYPED_TEST(MetricsTest, EuclideanCorrectness) {
        using T = TypeParam;
        // Verify L2 norm with floating-point epsilon tolerance
        EXPECT_NEAR(EuclideanAoS::evaluate(this->origin, this->p1), static_cast<T>(5), 1e-6);
    }

    TYPED_TEST(MetricsTest, ManhattanCorrectness) {
        using T = TypeParam;
        // Verify L1 norm (Taxicab distance)
        EXPECT_EQ(ManhattanAoS::evaluate(this->origin, this->p1), static_cast<T>(7));
        EXPECT_EQ(ManhattanAoS::evaluate(this->p2, this->p3), static_cast<T>(9));
    }

    TYPED_TEST(MetricsTest, ChebyshevCorrectness) {
        using T = TypeParam;
        // Verify L-infinity norm (Maximum coordinate difference)
        EXPECT_EQ(ChebyshevAoS::evaluate(this->origin, this->p1), static_cast<T>(4));
        EXPECT_EQ(ChebyshevAoS::evaluate(this->p2, this->p3), static_cast<T>(4));
    }

    // --- Fundamental Metric Properties ---

    TYPED_TEST(MetricsTest, IdentityProperty) {
        // Distance from a point to itself must be zero
        EXPECT_EQ(SquaredEuclideanAoS::evaluate(this->p3, this->p3), 0);
        EXPECT_EQ(ManhattanAoS::evaluate(this->p3, this->p3), 0);
        EXPECT_EQ(ChebyshevAoS::evaluate(this->p3, this->p3), 0);
    }

    TYPED_TEST(MetricsTest, SymmetryProperty) {
        // Ensure d(a, b) == d(b, a) to verify commutative property
        auto d12 = EuclideanAoS::evaluate(this->p1, this->p2);
        auto d21 = EuclideanAoS::evaluate(this->p2, this->p1);
        EXPECT_NEAR(d12, d21, 1e-7);
    }

    // --- Abstraction & Concept Validation ---

    TYPED_TEST(MetricsTest, ComputeDistanceWrapper) {
        using T = TypeParam;
        // Validates that the generic wrapper correctly dispatches to the static policy
        auto res = compute_distance<T, 3, ManhattanAoS>(this->origin, this->p1);
        EXPECT_EQ(res, static_cast<T>(7));
    }

    /**
     * @brief Hardwired memory alignment verification.
     * @details Confirms that PointAoS adheres to the 32-byte alignment
     * required for AVX/AVX2 load instructions.
     */
    TEST(AlignmentTest, PointAoSAlignment) {
        PointAoS<float, 3> p;
        PointAoS<double, 4> p_double;

        // Check alignment at the type level
        EXPECT_EQ(alignof(decltype(p)), 32);
        EXPECT_EQ(alignof(decltype(p_double)), 32);

        // Verify that DatasetAoS maintains heap-level alignment via its custom allocator
        DatasetAoS<float, 3> dataset;
        dataset.push_back({ {1, 2, 3} });
        dataset.push_back({ {4, 5, 6} });

        for (const auto& point : dataset) {
            auto addr = reinterpret_cast<std::uintptr_t>(&point);
            EXPECT_EQ(addr % 32, 0) << "Memory address splitting detected! Point is not 32-byte aligned.";
        }
    }

    /**
     * @brief Compile-time validation of ScalarMetric requirements.
     */
    TEST(ConceptTest, ScalarMetricCheck) {
        static_assert(ScalarMetric<SquaredEuclideanAoS, float, 3>);
        static_assert(ScalarMetric<ManhattanAoS, double, 10>);
        static_assert(ScalarMetric<ChebyshevAoS, float, 1>);
    }
} // namespace fc::metrics::test