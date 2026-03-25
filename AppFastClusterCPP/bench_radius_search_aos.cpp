#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// Project-specific headers for Array-of-Structures (AoS) layout and distance metrics
#include "dataset_aos.hpp"
#include "metrics_aos.hpp"
#include "radius_search_aos.hpp" 

namespace {

    /**
     * @brief Utility function to generate a synthetic Array-of-Structures (AoS) dataset.
     * * Ensures contiguous memory allocation and reproducible data distribution
     * for consistent baseline benchmarking.
     * * @tparam T Coordinate scalar type (e.g., float, double).
     * @tparam Dim Spatial dimensionality.
     * @param n Total number of points to generate.
     * @return fc::DatasetAoS<T, Dim> The generated AoS dataset.
     */
    template <typename T, std::size_t Dim>
    fc::DatasetAoS<T, Dim> create_random_dataset(std::size_t n) {
        fc::DatasetAoS<T, Dim> ds;
        ds.reserve(n);

        // Deterministic seeding to guarantee reproducible spatial distributions across runs
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dis(-100.0, 100.0);

        for (std::size_t i = 0; i < n; ++i) {
            std::array<T, Dim> coords;
            for (std::size_t d = 0; d < Dim; ++d) coords[d] = dis(gen);
            ds.push_back({ coords });
        }
        return ds;
    }

    /**
     * @brief Core benchmark template evaluating brute-force radius search performance.
     * * Measures the baseline throughput of sequential, unoptimized AoS traversal.
     * * @tparam Metric Distance calculation policy.
     * @tparam Dim Spatial dimensionality.
     */
    template <typename Metric, std::size_t Dim>
    void BM_RadiusSearch_BruteForce(benchmark::State& state) {
        const std::size_t num_points = state.range(0);
        auto dataset = create_random_dataset<float, Dim>(num_points);

        // Place the query point at the origin to standardize distance calculations
        fc::PointAoS<float, Dim> query;
        query.coords.fill(0.0f);

        // Tuning the search radius to return approximately 1% of the dataset.
        // This prevents dynamic memory allocation overhead (from resizing the results vector) 
        // from skewing the actual algorithmic execution time measurements.
        float radius = 10.0f;

        for (auto _ : state) {
            auto indices = fc::algorithms::radius_search_brute_force_aos<float, Dim, Metric>(
                dataset, query, radius
            );
            // Force the compiler to materialize the result to prevent dead-code elimination
            benchmark::DoNotOptimize(indices);
        }

        // Throughput metrics: total points processed per second
        state.SetItemsProcessed(state.iterations() * num_points);

        // Memory bandwidth metrics: total bytes read from the contiguous AoS buffer
        state.SetBytesProcessed(state.iterations() * num_points * sizeof(fc::PointAoS<float, Dim>));
    }

} // namespace

// --- Benchmark Registration ---

/**
 * 1. Distance Metric Comparison
 * Fixed workload (100k points in 3D space).
 * Purpose: Isolate the computational cost of different mathematical distance policies
 * (Euclidean vs. Manhattan vs. Chebyshev) when processing standard AoS structures.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::EuclideanAoS, 3)
->Arg(100'000)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::ManhattanAoS, 3)
->Arg(100'000)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::ChebyshevAoS, 3)
->Arg(100'000)->Unit(benchmark::kMillisecond);

/**
 * 2. Dimensionality Impact Analysis
 * Fixed point count (10k points) with increasing dimensions (2D, 16D, 128D).
 * Purpose: Observe performance degradation caused by bloated point structures.
 * As dimensionality increases, fewer points fit into cache lines, turning the
 * application from compute-bound to memory-bandwidth-bound.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::EuclideanAoS, 2)
->Arg(10'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::EuclideanAoS, 16)
->Arg(10'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::EuclideanAoS, 128)
->Arg(10'000)->Unit(benchmark::kMicrosecond);

/**
 * 3. Workload Scalability
 * Scales from 1k to 1M points in standard 3D space.
 * Purpose: Verify the linear O(N) time complexity of the brute-force search algorithm.
 */
BENCHMARK_TEMPLATE(BM_RadiusSearch_BruteForce, fc::metrics::EuclideanAoS, 3)
->RangeMultiplier(10)->Range(1000, 1'000'000)
->Unit(benchmark::kMillisecond);