#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <array>
#include "metrics_aosoa.hpp"
#include "dataset_aosoa.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Synthetic data generator for Array-of-Structures-of-Arrays (AoSoA) datasets.
 * * Populates the AoSoA container with uniform random coordinates. Utilizing random
 * data is critical to prevent the compiler from performing "constant folding" or
 * aggressive dead-code elimination that could artificially inflate performance results.
 * * @tparam T           The scalar coordinate type (e.g., float, double).
 * @tparam Dim         Spatial dimensionality.
 * @tparam SimdWidth   The number of elements per SIMD lane (e.g., 8 for AVX2/float).
 * @param num_points   Total number of points to ingest into the container.
 * @return A DatasetAoSoA populated with deterministic random values.
 */
template <typename T, std::size_t Dim, std::size_t SimdWidth>
DatasetAoSoA<T, Dim, SimdWidth> generate_random_aosoa(std::size_t num_points) {
    DatasetAoSoA<T, Dim, SimdWidth> dataset;

    // Use a fixed seed to ensure deterministic results across benchmark iterations.
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(static_cast<T>(-100.0), static_cast<T>(100.0));

    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<T, Dim> point;
        for (std::size_t d = 0; d < Dim; ++d) point[d] = dist(gen);
        dataset.add_point(point);
    }
    return dataset;
}

/**
 * @brief Benchmark fixture for evaluating SIMD-accelerated distance metrics.
 * * This fixture measures the execution latency and throughput of distance calculations
 * leveraging hand-vectorized kernels. The AoSoA layout ensures that data is
 * pre-aligned to match the target CPU's register width (e.g., 256-bit for AVX2),
 * minimizing unaligned load penalties and maximizing instruction-level parallelism (ILP).
 * * @tparam Metric The SIMD policy to evaluate (e.g., SquaredEuclideanAoSoA).
 */
template <typename Metric>
static void BM_AoSoA_Distance(benchmark::State& state) {
    const std::size_t num_points = static_cast<std::size_t>(state.range(0));
    const std::size_t Dim = 3;
    const std::size_t Width = 8; // Targeted at 256-bit AVX registers (8 * 32-bit float)

    auto dataset = generate_random_aosoa<float, Dim, Width>(num_points);
    std::array<float, Dim> query = { 1.0f, 1.0f, 1.0f };

    std::vector<float> results;
    results.reserve(num_points);

    /**
     * @section Benchmark Hot Loop
     * Processes the dataset in batches. The 'compute_distances_aosoa' function
     * is expected to utilize the AoSoA layout for efficient SIMD 'gather'/'load' ops.
     */
    for (auto _ : state) {
        compute_distances_aosoa<Metric>(query, dataset, results);

        /**
         * Optimization Barriers:
         * DoNotOptimize ensures 'results' are not optimized away by the compiler.
         * ClobberMemory forces a memory fence, ensuring all stores are retired before timing ends.
         */
        benchmark::DoNotOptimize(results.data());
        benchmark::ClobberMemory();
    }

    /**
     * @section Performance Metadata
     * Provides high-level hardware metrics to assess the algorithm's efficiency
     * relative to theoretical peak performance.
     */
    state.SetItemsProcessed(state.iterations() * num_points);

    // Calculate effective memory bandwidth based on data read during the search.
    state.SetBytesProcessed(state.iterations() * num_points * Dim * sizeof(float));
}

// --- Benchmark Registration ---

/**
 * Registration across a range of dataset sizes (1K to 64K points).
 * Multiplier(4) allows for analyzing cache-scaling behavior (L1 vs L2 vs L3/DRAM).
 */
#define AOSOA_ARGS ->RangeMultiplier(4)->Range(1024, 65536)->Unit(benchmark::kMicrosecond)

BENCHMARK_TEMPLATE(BM_AoSoA_Distance, SquaredEuclideanAoSoA) AOSOA_ARGS;
BENCHMARK_TEMPLATE(BM_AoSoA_Distance, EuclideanAoSoA) AOSOA_ARGS;
BENCHMARK_TEMPLATE(BM_AoSoA_Distance, ManhattanAoSoA) AOSOA_ARGS;
BENCHMARK_TEMPLATE(BM_AoSoA_Distance, ChebyshevAoSoA) AOSOA_ARGS;