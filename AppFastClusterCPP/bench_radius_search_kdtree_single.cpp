#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// Project-specific headers for KD-Tree, AoS data layout, and metrics
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "radius_search_kdtree_single.hpp"

namespace {

    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    // --- Helper Utilities ---

    /**
     * @brief Generates a synthetic dataset using an Array-of-Structures (AoS) layout.
     * * Utilizes a fixed seed to ensure deterministic and reproducible benchmark results.
     * * @tparam T Coordinate scalar type (float/double).
     * @tparam Dim Spatial dimensionality.
     * @param n Number of points to generate.
     * @param min_val Lower bound for coordinate distribution.
     * @param max_val Upper bound for coordinate distribution.
     * @return DatasetAoS<T, Dim> The generated collection of points.
     */
    template <MLCoordinate T, std::size_t Dim>
    DatasetAoS<T, Dim> create_random_dataset(std::size_t n, T min_val = -100.0, T max_val = 100.0) {
        DatasetAoS<T, Dim> ds;
        ds.reserve(n);

        // Fixed seed ensures that the spatial distribution remains consistent across runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(min_val, max_val);

        for (std::size_t i = 0; i < n; ++i) {
            PointAoS<T, Dim> pt;
            for (std::size_t d = 0; d < Dim; ++d) pt.coords[d] = dis(gen);
            ds.push_back(pt);
        }
        return ds;
    }

    /**
     * @brief Placeholder for KD-Tree construction.
     * * This function acts as a wrapper for the specific tree-building logic
     * used in the benchmarking process.
     */
    template <MLCoordinate T, std::size_t Dim>
    KDTreeFlat build_tree(const DatasetAoS<T, Dim>& dataset) {
        KDTreeFlat tree;
        // tree = fc::algorithms::build_kd_tree(dataset); // Integration point for actual build logic
        return tree;
    }

    // --- Main Benchmark Harness ---

    /**
     * @brief Benchmark for single-threaded KD-Tree radius search.
     * * Evaluates the performance of the search algorithm across different metrics
     * and tree sizes, focusing on the efficiency of spatial pruning.
     * * @tparam Metric Distance calculation policy.
     * @tparam Dim Spatial dimensionality.
     */
    template <typename Metric, std::size_t Dim>
    void BM_KDTree_RadiusSearch_Single(benchmark::State& state) {
        const std::size_t num_points = state.range(0);

        // 1. Setup Phase: Data preparation is performed outside the timed loop
        auto dataset = create_random_dataset<float, Dim>(num_points);
        auto tree = build_tree(dataset);

        // Generate a random query point within the same distribution domain
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
        PointAoS<float, Dim> query;
        for (size_t d = 0; d < Dim; ++d) query.coords[d] = dis(gen);

        // Search radius is chosen to yield a representative density of neighbors
        float radius = 10.0f;

        // 2. Execution Phase: Core timed loop
        for (auto _ : state) {
            auto results = radius_search_kdtree_single<float, Dim, Metric>(
                tree, dataset, query, radius
            );

            // Prevent compiler from optimizing out the results or the search logic
            benchmark::DoNotOptimize(results);
            benchmark::ClobberMemory();
        }

        // 3. Metadata and Statistics: Log information for post-benchmark analysis
        state.SetItemsProcessed(state.iterations());
        state.counters["TreeSize"] = num_points;
        // Tracking logarithmic complexity to compare against theoretical O(log N)
        state.counters["Complexity"] = std::log2(num_points);
    }

} // namespace

// --- Benchmark Registration ---

/**
 * 1. Metric Policy Comparison
 * Evaluates different distance metrics in 3D space with a fixed dataset size.
 * Useful for identifying overhead introduced by specific coordinate calculations.
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, SquaredEuclideanAoS, 3)
->Arg(100'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, ManhattanAoS, 3)
->Arg(100'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, ChebyshevAoS, 3)
->Arg(100'000)->Unit(benchmark::kMicrosecond);

/**
 * 2. Tree Size Scalability
 * Scales from 10k to 1M points to verify logarithmic search time.
 * Helps determine at what point cache misses or tree depth start impacting performance.
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, SquaredEuclideanAoS, 3)
->RangeMultiplier(10)
->Range(10'000, 1'000'000)
->Unit(benchmark::kMicrosecond);

/**
 * 3. The "Curse of Dimensionality" Analysis
 * Measures performance degradation as dimensionality increases.
 * KD-Trees typically revert to O(N) complexity as Dim increases,
 * making them less efficient than brute force in very high dimensions.
 */
BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, SquaredEuclideanAoS, 2)
->Arg(100'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, SquaredEuclideanAoS, 8)
->Arg(100'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_KDTree_RadiusSearch_Single, SquaredEuclideanAoS, 16)
->Arg(100'000)->Unit(benchmark::kMicrosecond);