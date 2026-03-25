#include "pch.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <cmath>
#include "../../AppFastClusterCPP/metric_soa.hpp"

namespace fc::metrics::test {

    /**
     * @brief Test fixture for Structure of Arrays (SoA) metrics.
     * @details Validates the performance-oriented SoA layout across multiple
     * coordinate types. This fixture ensures that batch distance calculations
     * remain consistent across different floating-point precisions.
     * @tparam T Floating-point type (float or double).
     */
    template <typename T>
    class SoAMetricsTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;

        DatasetSoA<T, Dim> dataset;
        std::array<T, Dim> query = { 0, 0, 0 }; // Origin point for simplified verification
        std::vector<T> results;

        void SetUp() override {
            // Point 0: Identity case (matches query exactly, expected distance: 0)
            dataset.push_back({ 0, 0, 0 });

            // Point 1: Standard 3-4-5 Pythagorean triple (Expected L2=5, L2Sq=25)
            dataset.push_back({ 3, 4, 0 });

            // Point 2: Negative coordinates to verify absolute value handling and accumulation
            dataset.push_back({ -1, -2, -2 });

            // Point 3: Uniform multi-axis values for symmetry verification
            dataset.push_back({ 2, 2, 2 });
        }
    };

    using CoordinateTypes = ::testing::Types<float, double>;
    TYPED_TEST_CASE(SoAMetricsTest, CoordinateTypes);

    // --- Batch Dispatcher Mathematical Correctness Tests ---

    TYPED_TEST(SoAMetricsTest, SquaredEuclideanBatch) {
        using T = TypeParam;
        compute_distances_soa<SquaredEuclideanPolicy>(this->query, this->dataset, this->results);

        ASSERT_EQ(this->results.size(), 4);
        EXPECT_FLOAT_EQ(this->results[0], static_cast<T>(0));
        // Verify L2 Squared: 3^2 + 4^2 + 0^2 = 25
        EXPECT_FLOAT_EQ(this->results[1], static_cast<T>(25));
        // Verify L2 Squared: (-1)^2 + (-2)^2 + (-2)^2 = 1 + 4 + 4 = 9
        EXPECT_FLOAT_EQ(this->results[2], static_cast<T>(9));
        // Verify L2 Squared: 2^2 + 2^2 + 2^2 = 12
        EXPECT_FLOAT_EQ(this->results[3], static_cast<T>(12));
    }

    TYPED_TEST(SoAMetricsTest, EuclideanBatch) {
        using T = TypeParam;
        compute_distances_soa<EuclideanPolicy>(this->query, this->dataset, this->results);

        ASSERT_EQ(this->results.size(), 4);
        EXPECT_NEAR(this->results[0], static_cast<T>(0), 1e-6);
        EXPECT_NEAR(this->results[1], static_cast<T>(5), 1e-6); // Standard L2 norm
        EXPECT_NEAR(this->results[2], static_cast<T>(3), 1e-6); // L2 norm of negative vector
        EXPECT_NEAR(this->results[3], std::sqrt(static_cast<T>(12)), 1e-6);
    }

    TYPED_TEST(SoAMetricsTest, ManhattanBatch) {
        using T = TypeParam;
        compute_distances_soa<ManhattanPolicy>(this->query, this->dataset, this->results);

        ASSERT_EQ(this->results.size(), 4);
        // Verify L1 norm: |3| + |4| + |0| = 7
        EXPECT_FLOAT_EQ(this->results[1], static_cast<T>(7));
        // Verify L1 norm: |-1| + |-2| + |-2| = 5
        EXPECT_FLOAT_EQ(this->results[2], static_cast<T>(5));
        // Verify L1 norm: |2| + |2| + |2| = 6
        EXPECT_FLOAT_EQ(this->results[3], static_cast<T>(6));
    }

    TYPED_TEST(SoAMetricsTest, ChebyshevBatch) {
        using T = TypeParam;
        compute_distances_soa<ChebyshevPolicy>(this->query, this->dataset, this->results);

        ASSERT_EQ(this->results.size(), 4);
        // Verify L-infinity norm: max(|3|, |4|, |0|) = 4
        EXPECT_FLOAT_EQ(this->results[1], static_cast<T>(4));
        // Verify L-infinity norm: max(|-1|, |-2|, |-2|) = 2
        EXPECT_FLOAT_EQ(this->results[2], static_cast<T>(2));
        EXPECT_FLOAT_EQ(this->results[3], static_cast<T>(2));
    }

    // --- Edge Case Handling ---

    /**
     * @brief Ensures the dispatcher handles null/empty datasets gracefully.
     * @details Verifies that no invalid memory access occurs and no unexpected
     * allocations are performed when the input size is zero.
     */
    TYPED_TEST(SoAMetricsTest, EmptyDataset) {
        using T = TypeParam;
        DatasetSoA<T, 3> empty_dataset;
        std::vector<T> results;

        compute_distances_soa<SquaredEuclideanPolicy>(this->query, empty_dataset, results);

        EXPECT_TRUE(results.empty());
    }

    // --- Static Analysis and Infrastructure Tests ---

    /**
     * @brief Compile-time validation of the SoAMetricPolicy concept.
     * @details Ensures that all static policies adhere to the required interface
     * for per-axis accumulation and finalization.
     */
    TEST(SoAConceptTest, StaticPolicyValidation) {
        static_assert(SoAMetricPolicy<SquaredEuclideanPolicy, float>);
        static_assert(SoAMetricPolicy<EuclideanPolicy, double>);
        static_assert(SoAMetricPolicy<ManhattanPolicy, float>);
        static_assert(SoAMetricPolicy<ChebyshevPolicy, double>);
    }

    /**
     * @brief Memory Alignment Verification.
     * @details Validates 64-byte alignment for axis-specific buffers. This is
     * critical for AVX-512 instruction compatibility and avoiding
     * "cache line split" performance penalties during streaming reads.
     */
    TEST(SoAMemoryTest, AxisDataAlignment) {
        DatasetSoA<float, 3> dataset;

        // Populate to trigger initial memory allocation in the internal containers
        dataset.push_back({ 1.0f, 2.0f, 3.0f });
        dataset.push_back({ 4.0f, 5.0f, 6.0f });

        for (std::size_t d = 0; d < 3; ++d) {
            const float* data_ptr = dataset.axis_data(d);
            auto addr = reinterpret_cast<std::uintptr_t>(data_ptr);

            // Verify the memory address is a multiple of 64 bytes
            EXPECT_EQ(addr % 64, 0) << "SoA buffer for axis " << d << " violates 64-byte alignment!";
        }
    }
} // namespace fc::metrics::test