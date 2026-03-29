#include "pch.h"
#include <gtest/gtest.h>
#include <vector>
#include "../../AppFastClusterCPP/metrics_soa.hpp"

namespace fc::metrics::test {

    /**
     * @brief Test fixture for Structure of Arrays (SoA) metrics.
     * @details Validates the performance-oriented SoA layout across multiple
     * coordinate types. The SoA layout is preferred over AoS in high-performance
     * computing to ensure contiguous memory access for specific dimensions,
     * thereby improving SIMD utilization and cache efficiency.
     * * @tparam T Floating-point precision (float or double).
     */
    template <typename T>
    class MetricsSoaTest : public ::testing::Test {
    protected:
        static constexpr std::size_t Dim = 3;
        DatasetSoA<T, Dim> dataset;
        std::array<T, Dim> query = { 0, 0, 0 };
        std::vector<T> results;

        /**
         * @brief Pre-populates the SoA dataset with known vectors for analytical verification.
         */
        void SetUp() override {
            dataset.push_back({ 0, 0, 0 });    // Case: Identity / Origin
            dataset.push_back({ 3, 4, 0 });    // Case: Standard 3-4-5 geometric verification
            dataset.push_back({ -1, -2, -2 }); // Case: Multi-dimensional negative coordinates
        }
    };

    // Support both single and double precision for robust coverage
    using CoordinateTypes = ::testing::Types<float, double>;
    TYPED_TEST_CASE(MetricsSoaTest, CoordinateTypes);

    /**
     * @brief Verifies batch distance calculations across different distance metrics.
     * @details Ensures that the SoA-optimized kernels return identical results
     * to their analytical definitions.
     */
    TYPED_TEST(MetricsSoaTest, BatchCorrectness) {
        using T = TypeParam;

        // Squared Euclidean: sum((p_i - q_i)^2) -> 3^2 + 4^2 + 0^2 = 25
        compute_distances_soa<SquaredEuclideanSoA>(this->query, this->dataset, this->results);
        EXPECT_FLOAT_EQ(this->results[1], static_cast<T>(25));

        // Euclidean: Standard L2 norm -> sqrt(25) = 5
        compute_distances_soa<EuclideanSoA>(this->query, this->dataset, this->results);
        EXPECT_NEAR(this->results[1], static_cast<T>(5), 1e-6);

        // Manhattan (L1): sum(|p_i - q_i|) -> |-1| + |-2| + |-2| = 5
        compute_distances_soa<ManhattanSoA>(this->query, this->dataset, this->results);
        EXPECT_FLOAT_EQ(this->results[2], static_cast<T>(5));

        // Chebyshev (L-infinity): max(|p_i - q_i|) -> max(3, 4, 0) = 4
        compute_distances_soa<ChebyshevSoA>(this->query, this->dataset, this->results);
        EXPECT_FLOAT_EQ(this->results[1], static_cast<T>(4));
    }

    /**
     * @brief Validates architectural alignment requirements and C++20 Concept compliance.
     */
    TEST(MetricsSoaInfrastructure, AlignmentAndConcepts) {
        DatasetSoA<float, 3> dataset;
        dataset.push_back({ 1, 2, 3 });

        /**
         * @details Verification of 64-byte alignment for axis buffers.
         * 64-byte alignment is critical for modern CPUs to:
         * 1. Enable AVX-512 vectorized instructions.
         * 2. Align buffers with the standard CPU cache line size to prevent false sharing.
         */
        auto addr = reinterpret_cast<std::uintptr_t>(dataset.axis_data(0));
        EXPECT_EQ(addr % 64, 0);

        /**
         * @brief Static verification of MetricSoA constraints.
         * Ensures that the metric classes provide the required interface for
         * the SoA distance dispatcher at compile-time.
         */
        static_assert(MetricSoA<SquaredEuclideanSoA, float>, "SquaredEuclideanSoA fails MetricSoA concept");
        static_assert(MetricSoA<ManhattanSoA, double>, "ManhattanSoA fails MetricSoA concept");
    }

} // namespace fc::metrics::test