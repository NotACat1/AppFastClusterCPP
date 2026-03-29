#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include "metrics_aos.hpp"
#include "dataset_aos.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Generates a synthetic dataset for Array-of-Structures (AoS) performance testing.
 * * Populates a contiguous memory block with PointAoS objects using a deterministic
 * random number generator. This ensures reproducible benchmark results across
 * different runs and environments.
 * * @tparam T      The scalar type (e.g., float, double).
 * @tparam Dim    The spatial dimensionality.
 * @param  size   Total number of points to generate.
 * @return A DatasetAoS container initialized with values in the range [-100, 100].
 */
template <typename T, std::size_t Dim>
DatasetAoS<T, Dim> generate_random_aos(std::size_t size) {
    DatasetAoS<T, Dim> dataset(size);

    // Fixed seed for deterministic workload generation
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(static_cast<T>(-100.0), static_cast<T>(100.0));

    for (auto& point : dataset) {
        for (std::size_t i = 0; i < Dim; ++i) {
            point[i] = dist(gen);
        }
    }
    return dataset;
}

/**
 * @brief Benchmark fixture for evaluating AoS-based distance metric performance.
 * * Measures the computational latency of point-to-point distance calculations.
 * This benchmark targets the overhead of AoS memory access patterns and evaluates
 * the compiler's ability to perform SIMD auto-vectorization on the metric logic.
 * * @tparam Metric  The distance computation policy (e.g., SquaredEuclideanAoS).
 * @tparam T       Floating-point precision (float or double).
 * @tparam Dim     Spatial dimensionality of the dataset.
 */
template <typename Metric, typename T, std::size_t Dim>
static void BM_AoS_Distance(benchmark::State& state) {
    const std::size_t num_points = static_cast<std::size_t>(state.range(0));

    // Data setup: Initialize dataset and a static query point
    auto dataset = generate_random_aos<T, Dim>(num_points);
    PointAoS<T, Dim> query;
    std::fill(query.coords.begin(), query.coords.end(), T{ 1 });

    // Output buffer to store distance results
    std::vector<T> results(num_points);

    // Main benchmark loop
    for (auto _ : state) {
        for (std::size_t i = 0; i < num_points; ++i) {
            // Static dispatch of the distance metric to allow full inlining
            results[i] = compute_distance_aos<T, Dim, Metric>(query, dataset[i]);
        }

        // Prevent the compiler from optimizing away the loop results
        benchmark::DoNotOptimize(results.data());
        // Force memory synchronization to ensure all writes are retired
        benchmark::ClobberMemory();
    }

    // Report hardware-level metrics for throughput and memory bandwidth analysis
    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointAoS<T, Dim>));
}

// Configuration macro for benchmark scaling and timing units
#define AOS_ARGS ->Arg(1024)->Arg(4096)->Unit(benchmark::kMicrosecond)

// Register benchmark instances for various distance policies
BENCHMARK_TEMPLATE(BM_AoS_Distance, SquaredEuclideanAoS, float, 3) AOS_ARGS;
BENCHMARK_TEMPLATE(BM_AoS_Distance, EuclideanAoS, float, 3) AOS_ARGS;
BENCHMARK_TEMPLATE(BM_AoS_Distance, ManhattanAoS, float, 3) AOS_ARGS;
BENCHMARK_TEMPLATE(BM_AoS_Distance, ChebyshevAoS, float, 3) AOS_ARGS;