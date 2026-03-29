#include "pch.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "../../AppFastClusterCPP/metrics_aosoa.hpp"

namespace fc::metrics::test {

    /**
     * @brief Test fixture for SIMD-accelerated distance metrics using AoSoA layout.
     * @details Validates distance computations across an 8-lane SIMD block (AVX/256-bit).
     * The AoSoA (Array of Structures of Arrays) format is specifically designed to
     * facilitate efficient vectorized horizontal/vertical processing.
     */
    class MetricsAosoaTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;
        static constexpr std::size_t Width = 8; // Matches AVX float throughput (256-bit / 32-bit)

        std::array<float, Dim> query = { 1.0f, 2.0f, 3.0f };
        SimdBlock<float, Dim, Width> block;
        float results[Width];

        /**
         * @brief Initializes the SIMD block with specific test cases.
         */
        void SetUp() override {
            std::fill(std::begin(results), std::end(results), 0.0f);
            for (auto& lane : block.lanes) std::fill(std::begin(lane), std::end(lane), 0.0f);

            // Lane 0: Identity case (distance to self should be zero)
            block.lanes[0][0] = 1.0f; block.lanes[1][0] = 2.0f; block.lanes[2][0] = 3.0f;

            // Lane 1: Standard 3-4-5 triangle vector configuration
            // Point: (4, 6, 3) | Query: (1, 2, 3) | Diff: (3, 4, 0)
            block.lanes[0][1] = 4.0f; block.lanes[1][1] = 6.0f; block.lanes[2][1] = 3.0f;
        }
    };

    /**
     * @brief Functional correctness test for vectorized metric implementations.
     */
    TEST_F(MetricsAosoaTest, Correctness) {
        // Squared Euclidean: sum((p_i - q_i)^2) -> 3^2 + 4^2 + 0^2 = 25
        SquaredEuclideanAoSoA::evaluate(query, block, results);
        EXPECT_FLOAT_EQ(results[0], 0.0f);
        EXPECT_FLOAT_EQ(results[1], 25.0f);

        // Euclidean: sqrt(SquaredEuclidean) -> 5.0
        EuclideanAoSoA::evaluate(query, block, results);
        EXPECT_NEAR(results[1], 5.0f, 1e-6);

        // Manhattan (L1): sum(|p_i - q_i|) -> |3| + |4| + |0| = 7
        ManhattanAoSoA::evaluate(query, block, results);
        EXPECT_FLOAT_EQ(results[1], 7.0f);

        // Chebyshev (L-infinity): max(|p_i - q_i|) -> max(3, 4, 0) = 4
        ChebyshevAoSoA::evaluate(query, block, results);
        EXPECT_FLOAT_EQ(results[1], 4.0f);
    }

    /**
     * @brief Tests high-level dispatching, memory alignment, and C++20 Concepts.
     */
    TEST_F(MetricsAosoaTest, DispatcherAndInfrastructure) {
        DatasetAoSoA<float, 3, 8> dataset;
        dataset.add_point({ 1.0f, 2.0f, 3.0f });

        std::vector<float> all_results;
        // Verify the dispatcher correctly processes the dataset through SIMD kernels
        compute_distances_aosoa<SquaredEuclideanAoSoA>(query, dataset, all_results);

        ASSERT_EQ(all_results.size(), 8); // Ensures padding/lane alignment is preserved
        EXPECT_FLOAT_EQ(all_results[0], 0.0f);

        /** * @note 64-byte alignment is required to ensure compatibility with
         * AVX-512 and to prevent cache-line splits in 256-bit AVX operations.
         */
        EXPECT_EQ(alignof(SimdBlock<float, 3, 8>), 64);

        // Static interface validation via C++20 Concepts
        static_assert(MetricAoSoA<SquaredEuclideanAoSoA>);
    }

    /**
     * @brief Unit test for the low-level bitwise Absolute Value mask.
     * @details Verifies that the sign-bit clearing mask works correctly for
     * IEEE 754 floating-point values including INF and signed zeros.
     */
    TEST(MetricsAosoaDetail, AbsMask) {
        // Prepare a vector with mixed positive/negative values and edge cases
        __m256 values = _mm256_setr_ps(-1.0f, 2.0f, -0.0f, -INFINITY, 0, 0, 0, 0);

        // Apply bitwise AND with the absolute value mask (clears the high bit)
        __m256 result = _mm256_and_ps(values, detail::get_abs_mask_ps());

        float out[8];
        _mm256_storeu_ps(out, result);

        EXPECT_FLOAT_EQ(out[0], 1.0f);      // -1.0 -> 1.0
        EXPECT_FALSE(std::signbit(out[2])); // -0.0 -> 0.0 (sign bit must be cleared)
    }

} // namespace fc::metrics::test