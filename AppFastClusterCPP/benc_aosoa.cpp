#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <array>

#include "metric_aosoa.hpp"
#include "dataset_aosoa.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Utility to populate an AoSoA (Array of Structures of Arrays) dataset.
 * * Generates uniform random coordinates to ensure the benchmark measures
 * actual arithmetic throughput rather than compiler-optimized constant folding.
 */
template <typename T, std::size_t Dim, std::size_t SimdWidth>
DatasetAoSoA<T, Dim, SimdWidth> generate_random_aosoa(std::size_t num_points) {
    DatasetAoSoA<T, Dim, SimdWidth> dataset;
    std::mt19937 gen(42); // Fixed seed for deterministic results across runs
    std::uniform_real_distribution<T> dist(-100.0, 100.0);

    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<T, Dim> point;
        for (std::size_t d = 0; d < Dim; ++d) {
            point[d] = dist(gen);
        }
        dataset.add_point(point);
    }
    return dataset;
}

/**
 * @brief Benchmark fixture for SIMD-accelerated distance metrics.
 * * Measures the performance of batch distance computations using manual
 * SIMD intrinsics. The AoSoA layout ensures data is pre-aligned for
 * 256-bit AVX/AVX2 registers.
 * * @tparam Metric The SIMD policy (e.g., SquaredEuclidean, Manhattan).
 */
template <typename Metric>
static void BM_SIMD_Distance(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    const std::size_t Dim = 3; // Fixed dimension per SIMDMetric concept
    const std::size_t SimdWidth = 8;

    // Pre-allocate and initialize data outside the hot loop
    auto dataset = generate_random_aosoa<float, Dim, SimdWidth>(num_points);
    std::array<float, Dim> query = { 1.0f, 2.0f, 3.0f };
    std::vector<float> results;
    results.reserve(num_points);

    // Benchmark Hot Loop
    for (auto _ : state) {
        compute_batch_distances<Metric>(query, dataset, results);

        // Prevent Dead Code Elimination: ensure results are committed to memory
        benchmark::DoNotOptimize(results.data());
        // Force memory synchronization to ensure all writes are completed
        benchmark::ClobberMemory();
    }

    // Performance Metadata
    state.SetItemsProcessed(state.iterations() * num_points);
    // Track effective memory bandwidth (Dimensions * sizeof(float) per point)
    state.SetBytesProcessed(state.iterations() * num_points * Dim * sizeof(float));
}

// --- Benchmark Registration ---
// Evaluate performance across various data scales (1K to 64K points)
#define SIMD_ARGS ->RangeMultiplier(4)->Range(1024, 65536)->Unit(benchmark::kMicrosecond)

BENCHMARK_TEMPLATE(BM_SIMD_Distance, SquaredEuclidean) SIMD_ARGS;
BENCHMARK_TEMPLATE(BM_SIMD_Distance, Euclidean) SIMD_ARGS;
BENCHMARK_TEMPLATE(BM_SIMD_Distance, Manhattan) SIMD_ARGS;
BENCHMARK_TEMPLATE(BM_SIMD_Distance, Chebyshev) SIMD_ARGS;