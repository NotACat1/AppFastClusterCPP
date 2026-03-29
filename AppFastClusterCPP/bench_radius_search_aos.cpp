#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "radius_search_aos.hpp"

namespace {
    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Factory function for generating synthetic Array-of-Structures (AoS) datasets.
     * * Uses a deterministic random number generator (fixed seed) to ensure benchmark
     * reproducibility across different hardware environments and test runs.
     * * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @param n Number of points to generate.
     * @return DatasetAoS<T, Dim> A populated dataset within the range [-100, 100].
     */
    template <typename T, std::size_t Dim>
    DatasetAoS<T, Dim> create_random_dataset(std::size_t n) {
        DatasetAoS<T, Dim> ds(n);
        // Fixed seed ensures that data distribution does not change between runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(static_cast<T>(-100.0), static_cast<T>(100.0));

        for (auto& point : ds) {
            for (std::size_t d = 0; d < Dim; ++d) {
                point[d] = dis(gen);
            }
        }
        return ds;
    }

    /**
     * @brief Benchmark fixture for brute-force radius search using AoS layout.
     * * Measures the performance of a linear scan to find neighbors within a sphere.
     * This benchmark captures the overhead of distance calculation, point iteration,
     * and potential dynamic memory allocation within the search results.
     * * @tparam Metric Distance calculation policy (e.g., EuclideanAoS).
     * @tparam Dim Spatial dimensionality of the dataset.
     */
    template <typename Metric, std::size_t Dim>
    void BM_RadiusSearch_AoS_BruteForce(benchmark::State& state) {
        const std::size_t num_points = static_cast<std::size_t>(state.range(0));
        auto dataset = create_random_dataset<float, Dim>(num_points);

        // Standardized query point positioned at the origin
        PointAoS<float, Dim> query;
        query.coords.fill(0.0f);
        float radius = 10.0f;

        // Main benchmark hot loop
        for (auto _ : state) {
            auto indices = radius_search_brute_force_aos<float, Dim, Metric>(dataset, query, radius);

            // Prevent the compiler from optimizing away the result vector (Dead Code Elimination)
            benchmark::DoNotOptimize(indices);
        }

        /**
         * Report hardware-level metrics:
         * ItemsProcessed: Total points scanned across all iterations.
         * BytesProcessed: Cumulative memory throughput for the search operation.
         */
        state.SetItemsProcessed(state.iterations() * num_points);
        state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointAoS<float, Dim>));

        // Metadata to help differentiate results in benchmark outputs
        state.counters["Dim"] = static_cast<double>(Dim);
    }
}

// --- Benchmark Registration ---

/**
 * Standard configurations for 3D spatial searches.
 * Unit(kMillisecond) is used to capture the high latency of large brute-force scans.
 */
#define AOS_COMMON_ARGS ->Arg(100'000)->Unit(benchmark::kMillisecond)

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoS_BruteForce, EuclideanAoS, 3) AOS_COMMON_ARGS;
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoS_BruteForce, ManhattanAoS, 3) AOS_COMMON_ARGS;
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoS_BruteForce, ChebyshevAoS, 3) AOS_COMMON_ARGS;

/**
 * Dimensionality Stress Tests:
 * Comparing 2D vs High-Dimensional (128D) performance to observe how
 * dimensionality affects data cache locality and calculation overhead.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoS_BruteForce, EuclideanAoS, 2)
->Arg(10'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoS_BruteForce, EuclideanAoS, 128)
->Arg(10'000)->Unit(benchmark::kMicrosecond);