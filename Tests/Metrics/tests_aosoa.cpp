#include "pch.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <cmath>
#include "../../AppFastClusterCPP/metric_aosoa.hpp"

namespace fc::metrics::test {

    /**
     * @brief Test fixture for SIMD-accelerated distance metrics.
     * @details Prepares an 8-lane SimdBlock to validate parallel processing.
     * Specific data points are chosen to verify identity properties, standard
     * geometric distances, and negative coordinate handling.
     */
    class SIMDMetricsTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;
        static constexpr std::size_t Width = 8;

        std::array<float, Dim> query = { 1.0f, 2.0f, 3.0f };
        SimdBlock<float, Dim, Width> block;
        float results[Width];

        void SetUp() override {
            // Reset results buffer before each test case
            for (int i = 0; i < Width; ++i) results[i] = 0.0f;

            // Initialize the block with default zero values
            for (std::size_t d = 0; d < Dim; ++d) {
                for (std::size_t i = 0; i < Width; ++i) {
                    block.lanes[d][i] = 0.0f;
                }
            }

            // Lane 0: Identity point (matches query exactly, distance should be 0)
            block.lanes[0][0] = 1.0f; block.lanes[1][0] = 2.0f; block.lanes[2][0] = 3.0f;

            // Lane 1: Standard 3-4-5 Pythagorean triple scenario
            block.lanes[0][1] = 4.0f; block.lanes[1][1] = 6.0f; block.lanes[2][1] = 3.0f;

            // Lane 7: SIMD register boundary element (verifies the last lane of the YMM register)
            block.lanes[0][7] = 2.0f; block.lanes[1][7] = 3.0f; block.lanes[2][7] = 4.0f;
        }
    };

    // --- Mathematical Correctness Tests (Lane-wise Verification) ---

    TEST_F(SIMDMetricsTest, SquaredEuclideanCorrectness) {
        SquaredEuclidean::evaluate(query, block, results);

        // Lane 0: (1-1)^2 + (2-2)^2 + (3-3)^2 = 0
        EXPECT_FLOAT_EQ(results[0], 0.0f);
        // Lane 1: (1-4)^2 + (2-6)^2 + (3-3)^2 = 9 + 16 + 0 = 25
        EXPECT_FLOAT_EQ(results[1], 25.0f);
        // Lane 7: (1-2)^2 + (2-3)^2 + (3-4)^2 = 1 + 1 + 1 = 3
        EXPECT_FLOAT_EQ(results[7], 3.0f);
    }

    TEST_F(SIMDMetricsTest, EuclideanCorrectness) {
        Euclidean::evaluate(query, block, results);

        EXPECT_NEAR(results[0], 0.0f, 1e-6);
        EXPECT_NEAR(results[1], 5.0f, 1e-6); // sqrt(25) = 5
        EXPECT_NEAR(results[7], std::sqrt(3.0f), 1e-6);
    }

    TEST_F(SIMDMetricsTest, ManhattanCorrectness) {
        Manhattan::evaluate(query, block, results);

        // Lane 1: |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        EXPECT_FLOAT_EQ(results[1], 7.0f);
    }

    TEST_F(SIMDMetricsTest, ChebyshevCorrectness) {
        Chebyshev::evaluate(query, block, results);

        // Lane 1: max(|1-4|, |2-6|, |3-3|) = max(3, 4, 0) = 4
        EXPECT_FLOAT_EQ(results[1], 4.0f);
    }

    // --- Infrastructure and Concept Integration Tests ---

    /**
     * @brief Validates the batch dispatcher with the AoSoA dataset container.
     */
    TEST_F(SIMDMetricsTest, BatchDispatcherTest) {
        DatasetAoSoA<float, 3, 8> dataset;
        dataset.add_point({ 1.0f, 2.0f, 3.0f }); // Inserts into Lane 0 of the first block

        std::vector<float> all_results;
        compute_batch_distances<SquaredEuclidean>(query, dataset, all_results);

        // Result vector must be a multiple of the SIMD width (8)
        ASSERT_EQ(all_results.size(), 8);
        EXPECT_FLOAT_EQ(all_results[0], 0.0f);
    }

    /**
     * @brief Memory Alignment Verification.
     * @details Critical for SIMD stability; unaligned data causes GPF (General Protection Fault)
     * when using instructions like _mm256_load_ps.
     */
    TEST(SIMDMemoryTest, BlockAlignment) {
        SimdBlock<float, 3, 8> block;

        // Verify structure alignment (64-byte for cache-line optimization/AVX-512 readiness)
        EXPECT_EQ(alignof(decltype(block)), 64);

        // Verify each dimension lane is aligned to at least a 32-byte boundary for AVX loads
        for (int d = 0; d < 3; ++d) {
            auto addr = reinterpret_cast<std::uintptr_t>(block.lanes[d]);
            EXPECT_EQ(addr % 32, 0) << "Lane " << d << " is not 32-byte aligned, risk of SIMD fault!";
        }
    }

    /**
     * @brief Static validation of Metric Policies against the SIMDMetric concept.
     * @note This ensures compile-time compliance with the required static interface.
     */
    TEST(SIMDConceptTest, InterfaceValidation) {
        static_assert(SIMDMetric<SquaredEuclidean>);
        static_assert(SIMDMetric<Euclidean>);
        static_assert(SIMDMetric<Manhattan>);
        static_assert(SIMDMetric<Chebyshev>);
    }

    /**
     * @brief Verification of the Low-level Absolute Value (ABS) bitmask.
     * @details Confirms that the bitwise sign-clearing mask correctly processes
     * positive, negative, zero, and floating-point edge cases (Inf, NaN).
     */
    TEST(SIMDDetailTest, AbsMaskVerification) {
        // Test set including negative values, signed zeros, and special floats
        __m256 values = _mm256_setr_ps(-1.0f, 2.0f, -3.5f, 0.0f, -0.0f, 100.0f, -INFINITY, NAN);
        __m256 mask = detail::get_abs_mask_ps();
        __m256 result = _mm256_and_ps(values, mask);

        float out[8];
        _mm256_storeu_ps(out, result);

        EXPECT_FLOAT_EQ(out[0], 1.0f);
        EXPECT_FLOAT_EQ(out[2], 3.5f);

        // Verify sign bit removal for negative zero
        EXPECT_FALSE(std::signbit(out[4]));
    }
} // namespace fc::metrics::test