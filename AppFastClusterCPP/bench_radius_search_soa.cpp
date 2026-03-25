#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <omp.h>

// Project-specific headers for SoA data structures and algorithms
#include "dataset_soa.hpp"
#include "metrics_soa.hpp"
#include "radius_search_soa.hpp"

namespace {

    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Utility for generating a synthetic Structure-of-Arrays (SoA) dataset.
     * * Uses a fixed seed to ensure deterministic results across benchmark runs.
     * Pre-allocates memory for each axis to minimize allocation overhead during setup.
     */
    template <typename T, std::size_t Dim>
    DatasetSoA<T, Dim> create_random_soa_dataset(std::size_t n) {
        DatasetSoA<T, Dim> ds;

        // Ensure memory is contiguous and pre-allocated for SoA layout
        for (std::size_t d = 0; d < Dim; ++d) {
            ds.axes[d].reserve(n);
        }

        // Fixed seed for reproducibility across different hardware/runs
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
     * @brief Core benchmark template for radius search performance analysis.
     * * Evaluates the brute-force SoA search across different metrics,
     * dimensions, and thread counts.
     * * @tparam Policy The distance metric policy (e.g., SquaredEuclideanPolicy).
     * @tparam Dim The spatial dimensionality of the dataset.
     */
    template <typename Policy, std::size_t Dim>
    void BM_RadiusSearch_SoA(benchmark::State& state) {
        const std::size_t num_points = state.range(0);
        const int num_threads = state.range(1);

        // Configure OpenMP runtime for the current benchmark iteration
        omp_set_num_threads(num_threads);

        auto dataset = create_random_soa_dataset<float, Dim>(num_points);

        // Define a query point at the origin
        std::array<float, Dim> query;
        query.fill(0.0f);

        float radius = 10.0f;

        for (auto _ : state) {
            auto indices = radius_search_brute_force_soa<float, Dim, Policy>(
                dataset, query, radius
            );
            // Prevent compiler from optimizing away the search operation
            benchmark::DoNotOptimize(indices);
        }

        // Throughput reporting: total items (points) processed per second
        state.SetItemsProcessed(state.iterations() * num_points);
        // Bandwidth reporting: total bytes read from axis arrays
        state.SetBytesProcessed(state.iterations() * num_points * Dim * sizeof(float));

        // Metadata counters for post-run analysis
        state.counters["Threads"] = num_threads;
        state.counters["Dim"] = Dim;
    }

} // namespace

// --- Benchmark Registration ---

/**
 * 1. Metric Policy Comparison
 * Fixed dataset size (100k points), single-threaded execution.
 * Purpose: Measure the baseline efficiency of different distance calculations
 * and evaluate the compiler's ability to apply SIMD auto-vectorization.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 3)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, EuclideanPolicy, 3)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, ManhattanPolicy, 3)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, ChebyshevPolicy, 3)
->Args({ 100'000, 1 })->Unit(benchmark::kMicrosecond);

/**
 * 2. Multi-threaded Scalability (OpenMP)
 * Dataset size: 1M points, 3D space.
 * Purpose: Analyze parallel efficiency and speedup when scaling from 1 to 8 threads.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 3)
->Args({ 1'000'000, 1 })
->Args({ 1'000'000, 2 })
->Args({ 1'000'000, 4 })
->Args({ 1'000'000, 8 })
->Unit(benchmark::kMillisecond);

/**
 * 3. Dimensionality Scaling Impact
 * Fixed dataset size (100k points), utilizing maximum hardware concurrency.
 * Purpose: Observe how high-dimensional data affects cache locality.
 * As dimensionality increases, "jumping" between large axis arrays may lead to
 * increased cache misses and memory bandwidth saturation.
 */
static void DimArguments(benchmark::Benchmark* b) {
    int max_threads = omp_get_max_threads();
    b->Args({ 100'000, max_threads });
}

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 2)
->Apply(DimArguments)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 3)
->Apply(DimArguments)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 16)
->Apply(DimArguments)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_SoA, SquaredEuclideanPolicy, 64)
->Apply(DimArguments)->Unit(benchmark::kMicrosecond);