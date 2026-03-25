#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "metric_aos.hpp" 
#include "dataset_aos.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Mock data generator for standard Array of Structures (AoS) datasets.
 * * Provides a contiguous block of memory containing aligned PointAoS objects.
 */
template <typename T, std::size_t Dim>
DatasetAoS<T, Dim> generate_random_dataset(std::size_t size) {
    DatasetAoS<T, Dim> dataset(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-100.0, 100.0);

    for (auto& point : dataset) {
        for (std::size_t i = 0; i < Dim; ++i) {
            point.coords[i] = dist(gen);
        }
    }
    return dataset;
}

/**
 * @brief General-purpose benchmark for scalar distance metrics.
 * * Evaluates point-to-point distance calculations. This captures the performance
 * of standard C++ loops and potential auto-vectorization by the compiler.
 * * @tparam Metric Static policy for distance calculation.
 * @tparam T Floating-point precision (float/double).
 * @tparam Dim Spatial dimensionality.
 */
template <typename Metric, typename T, std::size_t Dim>
static void BM_DistanceMetric(benchmark::State& state) {
    const std::size_t num_points = state.range(0);

    // Prepare two datasets to simulate pairwise distance computation
    auto dataset_a = generate_random_dataset<T, Dim>(num_points);
    auto dataset_b = generate_random_dataset<T, Dim>(num_points);

    // Measurement Cycle
    for (auto _ : state) {
        for (std::size_t i = 0; i < num_points; ++i) {
            // Invoke the zero-overhead abstraction wrapper
            T result = compute_distance<T, Dim, Metric>(dataset_a[i], dataset_b[i]);

            // Ensure the result is not optimized out by the compiler
            benchmark::DoNotOptimize(result);
        }
    }

    // Throughput reporting
    state.SetItemsProcessed(state.iterations() * num_points);
    // Bytes processed: Total size of two points (A and B) per iteration
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(PointAoS<T, Dim>) * 2);
}

// --- Benchmark Registration ---
// Configuration for standard dataset sizes and time units
#define ARGS ->Arg(1024)->Arg(4096)->Unit(benchmark::kMicrosecond)

// L2 Squared: Primary metric for comparison-heavy algorithms
BENCHMARK_TEMPLATE(BM_DistanceMetric, SquaredEuclideanAoS, float, 3) ARGS;
BENCHMARK_TEMPLATE(BM_DistanceMetric, SquaredEuclideanAoS, float, 16) ARGS;
BENCHMARK_TEMPLATE(BM_DistanceMetric, SquaredEuclideanAoS, double, 16) ARGS;

// Standard Metrics (L2, L1, L-inf)
BENCHMARK_TEMPLATE(BM_DistanceMetric, EuclideanAoS, float, 16) ARGS;
BENCHMARK_TEMPLATE(BM_DistanceMetric, ManhattanAoS, float, 16) ARGS;
BENCHMARK_TEMPLATE(BM_DistanceMetric, ChebyshevAoS, float, 16) ARGS;