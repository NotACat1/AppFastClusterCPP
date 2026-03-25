#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <array>

#include "metric_soa.hpp"
#include "dataset_soa.hpp"

using namespace fc;
using namespace fc::metrics;

/**
 * @brief Utility to populate an SoA (Structure of Arrays) dataset with synthetic data.
 * @details Pre-allocates memory for each axis to minimize allocation overhead during setup.
 * A fixed seed is used to ensure benchmark reproducibility across different runs.
 */
template <typename T, std::size_t Dim>
DatasetSoA<T, Dim> generate_random_soa(std::size_t num_points) {
    DatasetSoA<T, Dim> dataset;

    // Pre-reserve capacity for all axis vectors to prevent reallocations
    for (auto& axis : dataset.axes) {
        axis.reserve(num_points);
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-100.0, 100.0);

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
 * @brief Benchmark fixture for SoA-based distance policies.
 * @details Evaluates the performance of batch processing where data for each dimension
 * is contiguous. This layout is ideal for compiler auto-vectorization (SIMD).
 * @tparam Policy The distance metric policy (e.g., SquaredEuclideanPolicy).
 */
template <typename Policy, typename T, std::size_t Dim>
static void BM_SoA_Distance(benchmark::State& state) {
    const std::size_t num_points = state.range(0);

    auto dataset = generate_random_soa<T, Dim>(num_points);
    std::array<T, Dim> query;
    query.fill(1.0f); // Standardized query point for all iterations

    std::vector<T> results;
    // Pre-allocate results to isolate computational cost from memory management
    results.reserve(num_points);

    for (auto _ : state) {
        compute_distances_soa<Policy, T, Dim>(query, dataset, results);

        // Prevent the compiler from optimizing away the result vector
        benchmark::DoNotOptimize(results.data());
        // Force memory synchronization to ensure all writes are retired
        benchmark::ClobberMemory();
    }

    // Report throughput in terms of items and effective memory bandwidth
    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * (Dim + 1) * sizeof(T));
}

// --- Benchmark Registration ---
// Ranges selected to observe performance from L1 cache hits to DRAM-bound scenarios
#define SOA_ARGS ->RangeMultiplier(8)->Range(1024, 131072)->Unit(benchmark::kMicrosecond)

BENCHMARK_TEMPLATE(BM_SoA_Distance, SquaredEuclideanPolicy, float, 3) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, SquaredEuclideanPolicy, float, 16) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, EuclideanPolicy, float, 16) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, ManhattanPolicy, float, 16) SOA_ARGS;
BENCHMARK_TEMPLATE(BM_SoA_Distance, ChebyshevPolicy, float, 16) SOA_ARGS;