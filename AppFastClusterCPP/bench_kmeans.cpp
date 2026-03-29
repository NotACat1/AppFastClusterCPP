#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <cmath>

// Core KMeans implementation and algorithm definitions
#include "kmeans.hpp"

using namespace fc::algorithms;

// --- Performance Baseline Constants ---
constexpr std::size_t DefaultDim = 3;    // Standard 3D spatial data
constexpr std::size_t DefaultK = 16;     // Baseline cluster count
constexpr std::size_t MaxIters = 100;    // Hard cap for convergence benchmarking

// ============================================================================
// --- Data Generation Utilities ---
// ============================================================================

/**
 * @brief Generates a contiguous flat buffer of float values (num_points * dim).
 * Designed to minimize pointer indirection and maximize CPU cache line utilization
 * by ensuring spatial locality within the dataset.
 * @return Contiguous vector representing an interleaved point cloud.
 */
std::vector<float> generate_flat_dataset(std::size_t num_points, std::size_t dim) {
    std::vector<float> data(num_points * dim);
    std::mt19937 gen(42); // Deterministic seed for reproducible benchmarking
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);

    for (auto& val : data) {
        val = dis(gen);
    }
    return data;
}

/**
 * @struct CosineDistance
 * @brief Compute-intensive distance metric functor.
 * Used to evaluate the impact of arithmetic complexity (normalization/sqrt/division)
 * on the overall algorithm execution time compared to Euclidean distance.
 */
struct CosineDistance {
    inline float operator()(const float* p1, const float* p2, std::size_t dim) const {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            dot_product += p1[i] * p2[i];
            norm_a += p1[i] * p1[i];
            norm_b += p2[i] * p2[i];
        }
        // Handle zero vectors to prevent NaN results
        if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
        return 1.0f - (dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }
};

// ============================================================================
// --- Benchmark Suite 1: Cardinality Scalability (N) ---
// ============================================================================

/** @test Evaluates algorithmic scalability as the number of data points (N) grows. */
static void BM_KMeans_VaryPoints(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    auto data = generate_flat_dataset(num_points, DefaultDim);
    KMeans kmeans;
    L2SquaredDistance dist_func;

    for (auto _ : state) {
        auto result = kmeans.run(data.data(), num_points, DefaultDim,
            DefaultK, MaxIters, dist_func);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory(); // Prevent compiler from caching memory reads
    }

    state.SetComplexityN(num_points);
    // Track throughput: points processed per unit of time across all iterations
    state.SetItemsProcessed(state.iterations() * num_points);
}

// ============================================================================
// --- Benchmark Suite 2: Cluster Scalability (K) ---
// ============================================================================

/** @test Measures performance impact of increasing the number of centroids (K). */
static void BM_KMeans_VaryClusters(benchmark::State& state) {
    const std::size_t num_points = 10000;
    const std::size_t k = state.range(0);
    auto data = generate_flat_dataset(num_points, DefaultDim);
    KMeans kmeans;
    L2SquaredDistance dist_func;

    for (auto _ : state) {
        auto result = kmeans.run(data.data(), num_points, DefaultDim,
            k, MaxIters, dist_func);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetComplexityN(k);
}

// ============================================================================
// --- Benchmark Suite 3: Dimensionality Scalability (D) ---
// ============================================================================

/** @test Evaluates performance with high-dimensional vectors (D). */
static void BM_KMeans_VaryDim(benchmark::State& state) {
    const std::size_t num_points = 10000;
    const std::size_t dim = state.range(0);
    auto data = generate_flat_dataset(num_points, dim);
    KMeans kmeans;
    L2SquaredDistance dist_func;

    for (auto _ : state) {
        auto result = kmeans.run(data.data(), num_points, dim,
            DefaultK, MaxIters, dist_func);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetComplexityN(dim);
}

// ============================================================================
// --- Benchmark Suite 4: Metric Comparison (Arithmetic Overhead) ---
// ============================================================================

/** @test Compares the cost of Cosine Distance vs standard Euclidean metrics. */
static void BM_KMeans_CosineDistance(benchmark::State& state) {
    const std::size_t num_points = state.range(0);
    auto data = generate_flat_dataset(num_points, DefaultDim);
    KMeans kmeans;
    CosineDistance dist_func;

    for (auto _ : state) {
        auto result = kmeans.run(data.data(), num_points, DefaultDim,
            DefaultK, MaxIters, dist_func);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetComplexityN(num_points);
}

// ============================================================================
// --- Benchmark Registration & Configuration ---
// ============================================================================

// Group 1: Scale by N (Linear complexity O(N) verification)
BENCHMARK(BM_KMeans_VaryPoints)
->RangeMultiplier(2)->Range(1024, 256 * 1024)
->Complexity(benchmark::oN)
->UseRealTime();

// Group 2: Scale by K
BENCHMARK(BM_KMeans_VaryClusters)
->RangeMultiplier(2)->Range(4, 256)
->Complexity(benchmark::oN);

// Group 3: Scale by Dimension (Evaluation of cache pressure and wide distance calculation)
BENCHMARK(BM_KMeans_VaryDim)
->RangeMultiplier(2)->Range(2, 512)
->Complexity(benchmark::oN);

// Group 4: Performance differential based on distance calculation cost
BENCHMARK(BM_KMeans_CosineDistance)
->RangeMultiplier(2)->Range(1024, 64 * 1024)
->Complexity(benchmark::oN)
->UseRealTime();