#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <omp.h>

// Project-specific headers for KD-Tree, AoS data layout, and batch algorithms
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "radius_search_kdtree.hpp"

namespace {

    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    // --- Utility Functions ---

    /**
     * @brief Generates a synthetic dataset using an Array-of-Structures (AoS) layout.
     * * Utilizes a fixed seed to ensure deterministic results across different benchmark runs.
     * * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @param n Number of points to generate.
     * @return DatasetAoS<T, Dim> The generated collection of points.
     */
    template <MLCoordinate T, std::size_t Dim>
    DatasetAoS<T, Dim> create_random_aos_dataset(std::size_t n) {
        DatasetAoS<T, Dim> ds;
        ds.reserve(n);

        // Deterministic seeding for reproducible spatial distributions
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(-100.0, 100.0);

        for (std::size_t i = 0; i < n; ++i) {
            PointAoS<T, Dim> pt;
            for (std::size_t d = 0; d < Dim; ++d) pt.coords[d] = dis(gen);
            ds.push_back(pt);
        }
        return ds;
    }

    /**
     * @brief Placeholder for KD-Tree construction.
     * * Integration Point: Replace the logic below with the actual tree builder
     * (e.g., build_kd_tree). Note that tree balance significantly impacts
     * the efficiency of the search phase.
     */
    template <MLCoordinate T, std::size_t Dim>
    KDTreeFlat build_tree_for_benchmark(const DatasetAoS<T, Dim>& dataset) {
        KDTreeFlat tree;
        // Placeholder: actual tree-building logic should ensure O(log N) depth.
        return tree;
    }

    // --- Performance Evaluation ---

    /**
     * @brief Benchmarks the performance of parallel batch radius searches.
     * * This test evaluates the throughput of searching for multiple query points
     * simultaneously across different thread counts and data distributions.
     * * @tparam Metric Distance calculation policy.
     * @tparam Dim Spatial dimensionality.
     */
    template <typename Metric, std::size_t Dim>
    void BM_KDTree_RadiusSearch_Batch(benchmark::State& state) {
        const std::size_t num_points = state.range(0);    // Database size (N)
        const std::size_t num_queries = state.range(1);   // Query batch size (M)
        const int num_threads = state.range(2);           // OpenMP thread count

        // Configure the OpenMP environment for the current run
        omp_set_num_threads(num_threads);

        // 1. Setup Phase: Preparation (excluded from the timing measurement)
        auto dataset = create_random_aos_dataset<float, Dim>(num_points);
        auto queries = create_random_aos_dataset<float, Dim>(num_queries);
        auto tree = build_tree_for_benchmark(dataset);

        float radius = 15.0f;

        // 2. Main Benchmarking Loop
        for (auto _ : state) {
            auto results = radius_search_kdtree_batch<float, Dim, Metric>(
                tree, dataset, queries, radius
            );
            // Ensure the compiler does not optimize away the result of the search
            benchmark::DoNotOptimize(results);
        }

        // 3. Performance Metrics and Statistics
        // Reporting items processed as queries per second
        state.SetItemsProcessed(state.iterations() * num_queries);
        // Reporting bytes processed as the raw volume of query point data
        state.SetBytesProcessed(state.iterations() * num_queries * sizeof(PointAoS<float, Dim>));

        // Metadata for report analysis
        state.counters["Threads"] = num_threads;
        state.counters["TreeSize"] = static_cast<double>(num_points);
        state.counters["BatchSize"] = static_cast<double>(num_queries);
    }

} // namespace

// --- Benchmark Registration ---

/**
 * 1. Parallel Scalability Analysis
 * Evaluates the speedup and efficiency of the algorithm as thread counts increase
 * using a standard 3D space and Squared Euclidean distance.
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, SquaredEuclideanAoS, 3)
->Args({ 100'000, 1024, 1 })
->Args({ 100'000, 1024, 2 })
->Args({ 100'000, 1024, 4 })
->Args({ 100'000, 1024, 8 })
->Args({ 100'000, 1024, 12 })
->Unit(benchmark::kMillisecond);

/**
 * 2. Metric Overhead Comparison
 * Compares the relative performance of different distance policies in 3D
 * with a fixed thread pool size (8 threads).
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, SquaredEuclideanAoS, 3)
->Args({ 100'000, 1024, 8 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, ManhattanAoS, 3)
->Args({ 100'000, 1024, 8 })->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, ChebyshevAoS, 3)
->Args({ 100'000, 1024, 8 })->Unit(benchmark::kMicrosecond);

/**
 * 3. High-Dimensionality Stress Test
 * Evaluates performance in 32D space to observe the "Curse of Dimensionality,"
 * where the pruning effectiveness of KD-Trees typically degrades toward O(N).
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, SquaredEuclideanAoS, 32)
->Args({ 100'000, 512, 8 })
->Unit(benchmark::kMillisecond);