#include <benchmark/benchmark.h>
#include <omp.h>
#include <random>
#include <vector>
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "kd_tree_flat.hpp"
#include "radius_search_kdtree.hpp"

namespace {
    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Generates a synthetic dataset for Array-of-Structures (AoS) spatial queries.
     * * Utilizes a deterministic pseudo-random number generator (PRNG) to ensure
     * uniform point distribution. This is critical for KD-Tree benchmarks, as
     * clustered or skewed data significantly alters tree balance and traversal depth.
     * * @tparam T    Coordinate scalar type (e.g., float, double).
     * @tparam Dim  Spatial dimensionality.
     * @param  n    Total number of points to generate.
     * @return A pre-allocated DatasetAoS populated with deterministic values.
     */
    template <typename T, std::size_t Dim>
    DatasetAoS<T, Dim> create_random_dataset(std::size_t n) {
        DatasetAoS<T, Dim> ds(n);
        std::mt19937 gen(42); // Fixed seed for reproducible topological structures
        std::uniform_real_distribution<T> dis(static_cast<T>(-100.0), static_cast<T>(100.0));

        for (auto& point : ds) {
            for (std::size_t d = 0; d < Dim; ++d) {
                point[d] = dis(gen);
            }
        }
        return ds;
    }

    /**
     * @brief Benchmark fixture for multi-threaded, batch KD-Tree radius searches.
     * * Evaluates the query latency of a flattened KD-Tree. By processing queries in batches,
     * the algorithm can effectively distribute the workload across multiple OpenMP threads,
     * hiding memory latency and maximizing CPU utilization.
     * * @tparam Metric  Distance calculation policy (e.g., SquaredEuclideanAoS).
     * @tparam Dim     Spatial dimensionality.
     */
    template <typename Metric, std::size_t Dim>
    void BM_KDTree_RadiusSearch_Batch(benchmark::State& state) {
        // Extract benchmark parameters
        const std::size_t num_points = static_cast<std::size_t>(state.range(0));
        const std::size_t num_queries = static_cast<std::size_t>(state.range(1));
        const int num_threads = static_cast<int>(state.range(2));

        // Configure the OpenMP thread pool for parallel batch execution
        omp_set_num_threads(num_threads);

        // Setup Phase: Generate the search space and the batch of query points
        auto dataset = create_random_dataset<float, Dim>(num_points);
        auto queries = create_random_dataset<float, Dim>(num_queries);

        /**
         * @note Spatial Index Construction
         * In a complete implementation, the KD-Tree must be built from the dataset here.
         * e.g., KDTreeFlat tree = build_kd_tree(dataset);
         * Index construction is excluded from the benchmark loop to strictly
         * isolate query execution latency ($O(\log N)$ traversal per query).
         */
        KDTreeFlat tree;
        float radius = 15.0f;

        // --- Benchmark Hot Loop ---
        for (auto _ : state) {
            // Execute the batch spatial search
            auto results = radius_search_kdtree_batch<float, Dim, Metric>(tree, dataset, queries, radius);

            // Optimization Barrier: Ensure the 2D results vector is not elided by the compiler
            benchmark::DoNotOptimize(results);
        }

        /**
         * Performance Metadata:
         * Unlike brute-force methods, KD-Tree throughput is measured by the number of
         * *queries processed* per second, rather than total dataset points scanned,
         * because the tree actively prunes the search space.
         */
        state.SetItemsProcessed(state.iterations() * num_queries);

        // Export parameters to the benchmark report for scaling and bottleneck analysis
        state.counters["Threads"] = static_cast<double>(num_threads);
        state.counters["Queries"] = static_cast<double>(num_queries);
    }
}

// --- Benchmark Registration ---

/**
 * Multi-threading and Batch Size Analysis:
 * Evaluates a dataset of 100,000 points against a batch of 1,024 queries.
 * - Args({ 100k, 1024, 1 }): Establishes the single-threaded baseline for tree traversal.
 * - Args({ 100k, 1024, 8 }): Measures multi-core scaling and identifies potential
 * thread contention or false sharing during results accumulation.
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Batch, SquaredEuclideanAoS, 3)
->Args({ 100'000, 1024, 1 })
->Args({ 100'000, 1024, 8 })
->Unit(benchmark::kMillisecond);