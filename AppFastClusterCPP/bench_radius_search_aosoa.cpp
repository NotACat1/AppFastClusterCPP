#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <omp.h>

// Project-specific headers for AoSoA (Array-of-Structures-of-Arrays) data layout
#include "dataset_aosoa.hpp"
#include "metrics_aosoa.hpp"
#include "radius_search_aosoa.hpp"

namespace {

    /**
     * @brief Utility to generate a synthetic AoSoA dataset for benchmarking.
     * * The AoSoA layout combines the cache friendliness of AoS with the
     * SIMD vectorization efficiency of SoA.
     * * @param n Total number of points to generate.
     * @return fc::DatasetAoSoA<float, 3, 8> A dataset with 3D coordinates and 8-lane SIMD blocks.
     */
    fc::DatasetAoSoA<float, 3, 8> create_random_aosoa_dataset(std::size_t n) {
        fc::DatasetAoSoA<float, 3, 8> ds;

        // Fixed seed ensures deterministic spatial distribution across benchmark runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

        for (std::size_t i = 0; i < n; ++i) {
            ds.add_point({ dis(gen), dis(gen), dis(gen) });
        }
        return ds;
    }

    /**
     * @brief Benchmark harness for evaluating radius search on AoSoA data structures.
     * * Measures the throughput of brute-force search across various distance metrics
     * and multithreaded configurations.
     * * @tparam Metric The SIMD-enabled distance policy (e.g., SquaredEuclidean).
     */
    template <typename Metric>
    void BM_RadiusSearch_AoSoA(benchmark::State& state) {
        const std::size_t num_points = state.range(0);
        const int num_threads = state.range(1);

        // Dynamically adjust OpenMP thread pool for the current execution state
        omp_set_num_threads(num_threads);

        auto dataset = create_random_aosoa_dataset(num_points);

        // Origin-based query point for standardized distance testing
        std::array<float, 3> query = { 0.0f, 0.0f, 0.0f };
        float radius = 10.0f;

        for (auto _ : state) {
            auto indices = fc::algorithms::radius_search_brute_force_aosoa<Metric>(
                dataset, query, radius
            );
            // Prevent compiler from eliding the search operation
            benchmark::DoNotOptimize(indices);
        }

        // Throughput metrics: points processed per second
        state.SetItemsProcessed(state.iterations() * num_points);

        // Memory bandwidth metrics: total bytes loaded from AoSoA blocks
        state.SetBytesProcessed(state.iterations() * dataset.block_count() * sizeof(fc::SimdBlock<float, 3, 8>));

        // Metadata counters for performance reporting
        state.counters["Threads"] = num_threads;
    }

} // namespace

// --- Benchmark Registration ---

/**
 * 1. Metric Policy Comparison
 * Fixed workload (100k points), single-threaded execution.
 * Purpose: Isolate the raw computational efficiency of different distance metrics
 * and verify the performance gains from SIMD intrinsics (AVX2/AVX-512)
 * compared to standard AoS implementations.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::SquaredEuclidean)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::Euclidean)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::Manhattan)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::Chebyshev)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

/**
 * 2. Parallel Scalability (OpenMP)
 * Large workload (1M points), varying thread counts.
 * Purpose: Analyze the speedup factor and parallel efficiency as hardware
 * concurrency increases from 1 to 8 cores.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::SquaredEuclidean)
->Args({ 1'000'000, 1 })
->Args({ 1'000'000, 2 })
->Args({ 1'000'000, 4 })
->Args({ 1'000'000, 8 })
->Unit(benchmark::kMillisecond);

/**
 * 3. Data Volume Scalability
 * Tests range from 10k to 10M points using maximum system concurrency.
 * Purpose: Observe how the AoSoA layout maintains cache locality as the dataset
 * exceeds various CPU cache levels (L1/L2/L3).
 */
static void CustomArguments(benchmark::Benchmark* b) {
    int max_threads = omp_get_max_threads();
    for (int i = 10'000; i <= 10'000'000; i *= 10) {
        b->Args({ i, max_threads });
    }
}

BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA, fc::metrics::SquaredEuclidean)
->Apply(CustomArguments)->Unit(benchmark::kMillisecond);