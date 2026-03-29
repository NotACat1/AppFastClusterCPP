#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// Project-specific headers for clustering and spatial indexing
#include "dbscan.hpp"
#include "kd_tree_flat.hpp"
#include "dataset_aos.hpp"
#include "dataset_soa.hpp"
#include "dataset_aosoa.hpp"
#include "metrics_aos.hpp"
#include "metrics_aosoa.hpp"
#include "metrics_soa.hpp"
#include "radius_search_aos.hpp"
#include "radius_search_aosoa.hpp"
#include "radius_search_soa.hpp"
#include "radius_search_kdtree.hpp"
#include "radius_search_kdtree_single.hpp"

using namespace fc;
using namespace fc::algorithms;
using namespace fc::metrics;

// --- Global Configuration ---
constexpr std::size_t Dim = 3;  // Dimensionality of the point cloud
using T = float;                // Working precision

// ============================================================================
// --- Utility Functions: Data Generation & Tree Construction ---
// ============================================================================

/**
 * @brief Populates a generic dataset with uniformly distributed random points.
 * @tparam Dataset Supports AoS, SoA, or AoSoA storage layouts.
 * Uses compile-time dispatching (if constexpr) to handle different insertion APIs.
 */
template<typename Dataset>
void fill_random(Dataset& ds, std::size_t n) {
    std::mt19937 gen(42); // Fixed seed for deterministic benchmark results
    std::uniform_real_distribution<T> dis(0.0f, 100.0f);

    for (std::size_t i = 0; i < n; ++i) {
        std::array<T, Dim> pt = { dis(gen), dis(gen), dis(gen) };

        // SFINAE-like check for specific dataset insertion methods
        if constexpr (requires { ds.add_point(pt); }) {
            ds.add_point(pt);
        }
        else {
            ds.push_back({ pt });
        }
    }
}

/** @brief Factory function for AoS datasets. */
DatasetAoS<T, Dim> generate_dataset_aos(std::size_t num_points) {
    DatasetAoS<T, Dim> dataset;
    dataset.reserve(num_points);
    fill_random(dataset, num_points);
    return dataset;
}

/**
 * @brief Recursively builds a cache-friendly flat KD-Tree.
 * Uses std::nth_element to perform an O(N) median split per level.
 * @return Index of the created node in the flat array.
 */
int32_t build_kdtree_recursive(std::vector<int32_t>& indices, int depth, KDTreeFlat& tree, const DatasetAoS<T, Dim>& dataset) {
    if (indices.empty()) return -1;

    int32_t axis = depth % Dim;
    size_t mid = indices.size() / 2;

    // Partial sort to find the median along the current axis
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [&](int32_t a, int32_t b) {
            return dataset[a][axis] < dataset[b][axis];
        });

    int32_t node_idx = static_cast<int32_t>(tree.nodes.size());
    tree.nodes.push_back(KDNodeFlat{}); // Allocate node slot

    int32_t point_idx = indices[mid];

    // Splitting index space for children
    std::vector<int32_t> left_indices(indices.begin(), indices.begin() + mid);
    std::vector<int32_t> right_indices(indices.begin() + mid + 1, indices.end());

    int32_t left_child = build_kdtree_recursive(left_indices, depth + 1, tree, dataset);
    int32_t right_child = build_kdtree_recursive(right_indices, depth + 1, tree, dataset);

    // Finalize node data (Flattened structure for better spatial locality)
    tree.nodes[node_idx] = {
        dataset[point_idx][axis], // split_val
        point_idx,                // point_idx
        left_child,               // left_child
        right_child,              // right_child
        axis                      // split_dim
    };

    return node_idx;
}

/** @brief Entry point for flat KD-Tree construction. */
KDTreeFlat build_kdtree(const DatasetAoS<T, Dim>& dataset) {
    KDTreeFlat tree;
    tree.nodes.reserve(dataset.size());
    std::vector<int32_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    tree.root_idx = build_kdtree_recursive(indices, 0, tree, dataset);
    return tree;
}

// ============================================================================
// --- Benchmark Group 1: Memory Layout Comparison (Brute-force) ---
// ============================================================================

/** @test Baseline performance using Array-of-Structures (AoS). */
static void BM_DBSCAN_AoS_BruteForce(benchmark::State& state) {
    const std::size_t N = state.range(0);
    DatasetAoS<T, Dim> dataset;
    fill_random(dataset, N);
    DBSCAN dbscan;

    for (auto _ : state) {
        auto result = dbscan.run(N, 5.0f, 10, [&](std::size_t idx, float r) {
            return radius_search_brute_force_aos<T, Dim, EuclideanAoS>(dataset, dataset[idx], r);
            });
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(N);
}

/** @test Evaluates cache efficiency using Structure-of-Arrays (SoA). */
static void BM_DBSCAN_SoA_BruteForce(benchmark::State& state) {
    const std::size_t N = state.range(0);
    DatasetSoA<T, Dim> dataset;
    fill_random(dataset, N);
    DBSCAN dbscan;

    for (auto _ : state) {
        auto result = dbscan.run(N, 5.0f, 10, [&](std::size_t idx, float r) {
            std::array<T, Dim> query = {
                dataset.axis_data(0)[idx],
                dataset.axis_data(1)[idx],
                dataset.axis_data(2)[idx]
            };
            return radius_search_brute_force_soa<T, Dim, EuclideanSoA>(dataset, query, r);
            });
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(N);
}

/** @test Evaluates SIMD utilization (AVX/SSE) with AoSoA memory layout. */
static void BM_DBSCAN_AoSoA_SIMD(benchmark::State& state) {
    const std::size_t N = state.range(0);
    constexpr std::size_t SimdWidth = 8;
    DatasetAoSoA<T, Dim, SimdWidth> dataset;
    fill_random(dataset, N);
    DBSCAN dbscan;

    for (auto _ : state) {
        auto result = dbscan.run(N, 5.0f, 10, [&](std::size_t idx, float r) {
            const std::size_t b = idx / SimdWidth;
            const std::size_t l = idx % SimdWidth;
            const auto& block = dataset.get_block(b);
            std::array<T, Dim> query = { block.lanes[0][l], block.lanes[1][l], block.lanes[2][l] };

            return radius_search_brute_force_aosoa<EuclideanAoSoA>(dataset, query, r);
            });
        benchmark::DoNotOptimize(result);
    }
    state.SetComplexityN(N);
}

// ============================================================================
// --- Benchmark Group 2: KD-Tree Search Performance ---
// ============================================================================

/** @test Measures latency of isolated KD-Tree radius queries. */
static void BM_KDTree_SingleRadiusSearch(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    const T radius = 5.0f;

    auto dataset = generate_dataset_aos(num_points);
    auto tree = build_kdtree(dataset);

    std::mt19937 gen(1337);
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);

    for (auto _ : state) {
        state.PauseTiming();
        size_t query_idx = dist(gen);
        PointAoS<T, Dim> query_point = dataset[query_idx];
        state.ResumeTiming();

        auto neighbors = radius_search_kdtree_single<T, Dim, EuclideanAoS>(tree, dataset, query_point, radius);

        benchmark::DoNotOptimize(neighbors);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}

/** @test Measures throughput of batch KD-Tree radius queries. */
static void BM_KDTree_BatchRadiusSearch(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    const T radius = 5.0f;

    auto dataset = generate_dataset_aos(num_points);
    auto queries = generate_dataset_aos(num_points / 10);
    auto tree = build_kdtree(dataset);

    for (auto _ : state) {
        auto results = radius_search_kdtree_batch<T, Dim, EuclideanAoS>(tree, dataset, queries, radius);
        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * queries.size());
}

// ============================================================================
// --- Benchmark Group 3: End-to-End DBSCAN + KD-Tree Integration ---
// ============================================================================

/** @test DBSCAN with on-the-fly (Lazy) KD-Tree queries. */
static void BM_DBSCAN_KDTree_Single(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    const T eps = 5.0f;
    const std::size_t min_pts = 10;

    auto dataset = generate_dataset_aos(num_points);
    auto tree = build_kdtree(dataset);

    for (auto _ : state) {
        DBSCAN dbscan;
        auto region_query = [&](size_t index, float query_eps) {
            return radius_search_kdtree_single<T, Dim, EuclideanAoS>(tree, dataset, dataset[index], query_eps);
            };
        auto result = dbscan.run(dataset.size(), eps, min_pts, region_query);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

/** * @test DBSCAN with pre-computed (Eager) KD-Tree batch queries.
 * Evaluates the performance gains of batch processing despite higher memory pressure.
 */
static void BM_DBSCAN_KDTree_Batch(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    const T eps = 5.0f;
    const std::size_t min_pts = 10;

    auto dataset = generate_dataset_aos(num_points);
    auto tree = build_kdtree(dataset);

    for (auto _ : state) {
        DBSCAN dbscan;
        // Pre-calculating neighbor lists for all points using batch spatial query
        auto all_neighbors = radius_search_kdtree_batch<T, Dim, EuclideanAoS>(tree, dataset, dataset, eps);

        auto region_query = [&all_neighbors](std::size_t index, float /* radius */) -> const std::vector<std::size_t>&{
            return all_neighbors[index];
            };

        auto result = dbscan.run(num_points, eps, min_pts, region_query);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * num_points);
}

// ============================================================================
// --- Benchmark Registration ---
// ============================================================================

// Group 1: Layout efficiency tests (O(N^2) complexity expected)
BENCHMARK(BM_DBSCAN_AoS_BruteForce)->RangeMultiplier(2)->Range(128, 2048)->Complexity();
BENCHMARK(BM_DBSCAN_SoA_BruteForce)->RangeMultiplier(2)->Range(128, 2048)->Complexity();
BENCHMARK(BM_DBSCAN_AoSoA_SIMD)->RangeMultiplier(2)->Range(128, 2048)->Complexity();

// Group 2: Spatial indexing scalability
BENCHMARK(BM_KDTree_SingleRadiusSearch)->RangeMultiplier(2)->Range(1024, 128 * 1024)->UseRealTime();
BENCHMARK(BM_KDTree_BatchRadiusSearch)->RangeMultiplier(2)->Range(1024, 128 * 1024)->UseRealTime();

// Group 3: Full algorithm integration tests
BENCHMARK(BM_DBSCAN_KDTree_Single)->RangeMultiplier(2)->Range(1024, 128 * 1024)->UseRealTime();
BENCHMARK(BM_DBSCAN_KDTree_Batch)->RangeMultiplier(2)->Range(1024, 128 * 1024)->UseRealTime();