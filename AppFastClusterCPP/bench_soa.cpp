#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <array>
#include "metrics_soa.hpp"
#include "dataset_soa.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Populates a Structure-of-Arrays (SoA) dataset with synthetic coordinates.
 * * This utility ensures that each coordinate axis is stored in a contiguous memory block.
 * Using a deterministic PRNG (Pseudo-Random Number Generator) ensures that benchmark
 * results remain reproducible across different execution environments.
 * * @tparam T           Scalar type for coordinates (e.g., float, double).
 * @tparam Dim         Spatial dimensionality of the dataset.
 * @param  num_points  Total number of points to ingest into the SoA container.
 * @return A DatasetSoA initialized with values in the range [-100, 100].
 */
template <typename T, std::size_t Dim>
DatasetSoA<T, Dim> generate_random_soa(std::size_t num_points) {
    DatasetSoA<T, Dim> dataset;
    // Fixed seed to maintain a consistent workload for performance analysis
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(static_cast<T>(-100.0), static_cast<T>(100.0));

    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<T, Dim> point;
        for (std::size_t d = 0; d < Dim; ++d) {
            point[d] = dist(gen);
        }
        dataset.push_back(point);
    }
    return dataset;
}

/**
 * @brief Benchmark fixture for evaluating performance of SoA-based distance policies.
 * * The SoA layout is specifically designed to maximize cache-line utilization and
 * facilitate compiler auto-vectorization. By accessing coordinates axis-by-axis,
 * the CPU can effectively leverage SIMD instructions (AVX/AVX-512/NEON) to process
 * multiple points per clock cycle.
 * * @tparam Metric  Distance calculation policy (must satisfy MetricSoA concept).
 * @tparam T       Floating-point precision (float/double).
 * @tparam Dim     Spatial dimensionality.
 */
template <typename Metric, typename T, std::size_t Dim>
static void BM_SoA_Distance(benchmark::State& state) {
    const std::size_t num_points = static_cast<std::size_t>(state.range(0));

    // Dataset preparation: Data for each dimension is stored in separate contiguous arrays
    auto dataset = generate_random_soa<T, Dim>(num_points);
    std::array<T, Dim> query;
    query.fill(static_cast<T>(1.0));

    std::vector<T> results;
    // Pre-reserve capacity to isolate kernel computational cost from vector reallocations
    results.reserve(num_points);

    // Main measurement loop
    for (auto _ : state) {
        compute_distances_soa<Metric, T, Dim>(query, dataset, results);

        /**
         * Optimization Barriers:
         * DoNotOptimize: Prevents the compiler from eliding the entire computation.
         * ClobberMemory: Forces a memory fence, ensuring all result writes are retired
         * before the next timing iteration.
         */
        benchmark::DoNotOptimize(results.data());
        benchmark::ClobberMemory();
    }

    // Set high-level performance metrics for throughput and bandwidth analysis
    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * Dim * sizeof(T));
}

// --- Benchmark Registration ---

/**
 * Configure benchmark arguments.
 * The range (1K to 131K) is chosen to observe performance transitions
 * as the data footprint scales from L1/L2 cache residency into DRAM-bound scenarios.
 */
#define SOA_ARGS ->RangeMultiplier(8)->Range(1024, 131072)->Unit(benchmark::kMicrosecond)

BENCHMARK_TEMPLATE(BM_SoA_Distance, SquaredEuclideanSoA, float, 3) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, EuclideanSoA, float, 3) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, ManhattanSoA, float, 3) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, ChebyshevSoA, float, 3) SOA_ARGS;