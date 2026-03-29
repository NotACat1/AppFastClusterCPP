#include <benchmark/benchmark.h>
#include <omp.h>
#include <random>
#include <vector>
#include "dataset_aosoa.hpp"
#include "metrics_aosoa.hpp"
#include "radius_search_aosoa.hpp"

namespace {
    using namespace fc;
    using namespace fc::metrics;
    using namespace fc::algorithms;

    /**
     * @brief Generates a synthetic dataset optimized for SIMD execution (AoSoA layout).
     * * The Array-of-Structures-of-Arrays (AoSoA) memory model groups data into fixed-size
     * blocks corresponding to the CPU's vector register width (LaneWidth). This generator
     * populates those blocks using a deterministic PRNG to guarantee reproducible
     * benchmark results across different hardware topologies.
     * * @tparam Dim       Spatial dimensionality of the dataset.
     * @tparam LaneWidth  SIMD vector length (e.g., 8 for 256-bit AVX2 floats).
     * @param  n          Total number of points to ingest.
     * @return DatasetAoSoA structured for contiguous, vectorized memory access.
     */
    template <std::size_t Dim, std::size_t LaneWidth>
    DatasetAoSoA<float, Dim, LaneWidth> create_random_dataset_aosoa(std::size_t n) {
        DatasetAoSoA<float, Dim, LaneWidth> ds;
        // Fixed seed ensures the computational workload remains identical between runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

        for (std::size_t i = 0; i < n; ++i) {
            std::array<float, Dim> pt;
            for (std::size_t d = 0; d < Dim; ++d) {
                pt[d] = dis(gen);
            }
            // The add_point method handles the complex mapping from a scalar 
            // point representation into the internal blocked AoSoA memory layout.
            ds.add_point(pt);
        }
        return ds;
    }

    /**
     * @brief Evaluates multi-threaded, SIMD-accelerated brute-force radius search.
     * * This benchmark explicitly measures the combined performance gains of data-level
     * parallelism (SIMD via AoSoA) and thread-level parallelism (OpenMP). It allows
     * for empirical analysis of multi-core scaling efficiency and memory bandwidth limits.
     * * @tparam Metric     Distance policy (e.g., SquaredEuclideanAoSoA).
     * @tparam Dim        Spatial dimensionality.
     * @tparam LaneWidth  Hardware-specific SIMD width (defaults to 8 for AVX/AVX2).
     */
    template <typename Metric, std::size_t Dim, std::size_t LaneWidth = 8>
    void BM_RadiusSearch_AoSoA_BruteForce(benchmark::State& state) {
        // Extract benchmark parameters injected via Args()
        const std::size_t num_points = static_cast<std::size_t>(state.range(0));
        const int num_threads = static_cast<int>(state.range(1));

        // Dynamically configure the OpenMP thread pool for this specific benchmark iteration
        omp_set_num_threads(num_threads);

        // Prepare the aligned dataset and a standardized query point
        auto dataset = create_random_dataset_aosoa<Dim, LaneWidth>(num_points);
        std::array<float, Dim> query;
        query.fill(0.0f);
        float radius = 10.0f;

        // --- Benchmark Hot Loop ---
        for (auto _ : state) {
            auto indices = radius_search_brute_force_aosoa<Metric>(dataset, query, radius);

            // Optimization Barrier: Prevents the compiler from eliding the search computation
            benchmark::DoNotOptimize(indices);
        }

        /**
         * Report hardware-level throughput metrics.
         * Note: BytesProcessed is calculated using the exact 'block_count()' rather
         * than 'num_points' to account for memory padding intrinsic to the AoSoA layout.
         */
        state.SetItemsProcessed(state.iterations() * num_points);
        state.SetBytesProcessed(state.iterations() * dataset.block_count() * sizeof(SimdBlock<float, Dim, LaneWidth>));

        // Export the active thread count to the benchmark output table for scaling analysis
        state.counters["Threads"] = static_cast<double>(num_threads);
    }
}

// --- Benchmark Registration ---

/**
 * Multi-threading Scaling Analysis:
 * We test the 3D Squared Euclidean search using 100,000 points.
 * The Args({Points, Threads}) configuration isolates the effect of OpenMP:
 * - {100'000, 1}: Establishes the single-threaded SIMD baseline.
 * - {100'000, 8}: Evaluates parallel scaling efficiency across 8 logical cores.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_AoSoA_BruteForce, SquaredEuclideanAoSoA, 3)
->Args({ 100'000, 1 })
->Args({ 100'000, 8 })
->Unit(benchmark::kMicrosecond);