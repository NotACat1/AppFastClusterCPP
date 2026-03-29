#include <benchmark/benchmark.h>
#include <omp.h>
#include <random>
#include <vector>
#include "dataset_soa.hpp"
#include "metrics_soa.hpp"
#include "radius_search_soa.hpp"

namespace {
    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Generates a synthetic dataset in Structure of Arrays (SoA) format.
     * * SoA layout is utilized here to maximize SIMD vectorization potential
     * by keeping coordinate components contiguous in memory.
     * * @tparam T Coordinate scalar type.
     * @tparam Dim Dimensionality of the point space.
     * @param n Number of points to generate.
     * @return DatasetSoA containing randomized point data.
     */
    template <typename T, std::size_t Dim>
    DatasetSoA<T, Dim> create_random_dataset_soa(std::size_t n) {
        DatasetSoA<T, Dim> ds;
        for (auto& axis : ds.axes) axis.reserve(n);

        // Fixed seed ensures deterministic data distribution across benchmark runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(-100.0f, 100.0f);

        for (std::size_t i = 0; i < n; ++i) {
            std::array<T, Dim> pt;
            for (std::size_t d = 0; d < Dim; ++d) pt[d] = dis(gen);
            ds.push_back(pt);
        }
        return ds;
    }

    /**
     * @brief Benchmark fixture for Brute-Force Radius Search using SoA layout.
     * * This test evaluates the performance of exhaustive search, scaling with
     * respect to both point density and available CPU threads via OpenMP.
     * * @tparam Policy The distance metric policy (e.g., SquaredEuclideanSoA).
     * @tparam Dim Space dimensionality.
     */
    template <typename Policy, std::size_t Dim>
    void BM_RadiusSearch_SoA_BruteForce(benchmark::State& state) {
        const std::size_t num_points = state.range(0);
        const int num_threads = static_cast<int>(state.range(1));

        // Configure OpenMP thread pool for the parallel search implementation
        omp_set_num_threads(num_threads);
        auto dataset = create_random_dataset_soa<float, Dim>(num_points);

        // Define a static query point and search radius
        std::array<float, Dim> query;
        query.fill(0.0f);
        const float radius = 10.0f;

        for (auto _ : state) {
            auto indices = radius_search_brute_force_soa<float, Dim, Policy>(dataset, query, radius);
            // Prevent compiler from eliding the search operation
            benchmark::DoNotOptimize(indices);
        }

        // Calculate throughput and effective memory bandwidth usage
        state.SetItemsProcessed(state.iterations() * num_points);
        state.SetBytesProcessed(state.iterations() * num_points * Dim * sizeof(float));

        // Metadata for performance analysis
        state.counters["Threads"] = num_threads;
        state.counters["Dim"] = Dim;
    }
}

// --- Registration and Configuration ---

/**
 * @brief Helper to generate argument pairs for point count and thread scaling.
 * * Compares single-threaded baseline performance against the system's maximum hardware concurrency.
 */
static void SoAArguments(benchmark::Benchmark* b) {
    int max_threads = omp_get_max_threads();
    for (int t : {1, max_threads}) {
        b->Args({ 100'000, t });
    }
}

// Standard metric benchmarks
BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA_BruteForce, SquaredEuclideanSoA, 3)
->Apply(SoAArguments)
->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA_BruteForce, ManhattanSoA, 3)
->Apply(SoAArguments)
->Unit(benchmark::kMicrosecond);

/**
 * @section Stress Test
 * Benchmarking high-dimensional space performance (e.g., 64D) to observe
 * the impact of memory bandwidth bottlenecks on brute-force execution.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA_BruteForce, SquaredEuclideanSoA, 64)
->Args({ 100'000, 8 })
->Unit(benchmark::kMillisecond);