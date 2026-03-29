#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <cmath>
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "radius_search_kdtree_single.hpp"

namespace {
    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Generates a synthetic dataset of points with uniform distribution.
     * @tparam T Coordinate type (e.g., float, double).
     * @tparam Dim Dimensionality of the space.
     * @param n Number of points to generate.
     * @return DatasetAoS populated with random coordinates.
     */
    template <MLCoordinate T, std::size_t Dim>
    DatasetAoS<T, Dim> create_random_dataset(std::size_t n) {
        DatasetAoS<T, Dim> ds(n);
        // Use a fixed seed to ensure benchmark reproducibility across runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(-100.0, 100.0);

        for (auto& pt : ds) {
            for (std::size_t d = 0; d < Dim; ++d) pt[d] = dis(gen);
        }
        return ds;
    }

    /**
     * @brief Benchmark fixture for single-point radius search in a KD-Tree.
     * @tparam Metric Distance metric (e.g., SquaredEuclidean, Manhattan).
     * @tparam Dim Dimensionality of the point space.
     */
    template <typename Metric, std::size_t Dim>
    void BM_RadiusSearch_KDTree_Single(benchmark::State& state) {
        const std::size_t num_points = state.range(0);

        // Prepare the reference dataset and build the spatial index
        auto dataset = create_random_dataset<float, Dim>(num_points);

        // Integration Note: Replace with actual tree construction logic if needed
        KDTreeFlat tree;

        // Generate a deterministic query point outside the dataset generation sequence
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
        PointAoS<float, Dim> query;
        for (size_t d = 0; d < Dim; ++d) query[d] = dis(gen);

        const float radius = 10.0f;

        // Main benchmark loop
        for (auto _ : state) {
            auto results = radius_search_kdtree_single<float, Dim, Metric>(tree, dataset, query, radius);

            // Prevent the compiler from optimizing away the search operation
            benchmark::DoNotOptimize(results);
            // Ensure all memory writes are completed to prevent instruction reordering
            benchmark::ClobberMemory();
        }

        // Performance counters for post-analysis
        state.SetItemsProcessed(state.iterations());
        state.counters["TreeSize"] = static_cast<double>(num_points);
        state.counters["Log2N"] = std::log2(num_points); // Theoretical search complexity reference
        state.counters["Dim"] = Dim;
    }
}

// --- Benchmark Registration ---

// Common arguments for single-point search benchmarks
#define KDTREE_SINGLE_ARGS ->Arg(100'000)->Unit(benchmark::kMicrosecond)

// Test performance across different distance metrics
BENCHMARK_TEMPLATE(BM_RadiusSearch_KDTree_Single, SquaredEuclideanAoS, 3) KDTREE_SINGLE_ARGS;
BENCHMARK_TEMPLATE(BM_RadiusSearch_KDTree_Single, ManhattanAoS, 3) KDTREE_SINGLE_ARGS;
BENCHMARK_TEMPLATE(BM_RadiusSearch_KDTree_Single, ChebyshevAoS, 3) KDTREE_SINGLE_ARGS;

/**
 * @section Scalability
 * Measures how search time scales with the number of points (N).
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_KDTree_Single, SquaredEuclideanAoS, 3)
->RangeMultiplier(10)
->Range(10'000, 1'000'000)
->Unit(benchmark::kMicrosecond);

/**
 * @section Curse of Dimensionality
 * Measures performance degradation as the number of dimensions increases.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_KDTree_Single, SquaredEuclideanAoS, 16) KDTREE_SINGLE_ARGS;