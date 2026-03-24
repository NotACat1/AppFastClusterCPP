#include <benchmark/benchmark.h>
#include "dataset_aos.hpp"
#include "dataset_soa.hpp"
#include "dataset_aosoa.hpp"

// Scale factor for the performance tests to ensure data exceeds L1/L2 cache thresholds
const std::size_t num_points = 100'000;

/**
 * @brief Benchmark for Array of Structures (AoS) data layout.
 * Measures the overhead of a standard iterative sum. While AoS provides
 * excellent spatial locality for individual points, it may suffer from
 * redundant padding and less efficient vectorization in wide SIMD sets.
 */
static void BM_AoS_Sum(benchmark::State& state) {
    fc::DatasetAoS<float, 3> ds(num_points, fc::PointAoS<float, 3>{{1.0f, 1.0f, 1.0f}});
    for (auto _ : state) {
        float sum = 0;
        for (const auto& p : ds) {
            sum += p[0] + p[1] + p[2];
        }
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_AoS_Sum);

/**
 * @brief Benchmark for Array of Structures of Arrays (AoSoA / Hybrid) layout.
 * Evaluates performance of "blocked" memory access. This layout is designed
 * to maximize SIMD utilization by processing data in fixed-width 'lanes'
 * while maintaining better cache-friendliness than pure SoA.
 */
static void BM_AoSoA_Sum(benchmark::State& state) {
    fc::DatasetAoSoA<float, 3, 8> ds;
    for (size_t i = 0; i < num_points; ++i) ds.add_point({ 1.0f, 1.0f, 1.0f });

    for (auto _ : state) {
        float sum = 0;
        for (std::size_t b = 0; b < ds.block_count(); ++b) {
            auto& block = ds.get_block(b);
            for (std::size_t d = 0; d < 3; ++d) {
                for (std::size_t s = 0; s < 8; ++s) {
                    sum += block.lanes[d][s];
                }
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_AoSoA_Sum);

/**
 * @brief Benchmark for Structure of Arrays (SoA) data layout.
 * Evaluates performance of per-axis processing. SoA is typically the fastest
 * for large-scale attribute scans and allows the compiler's auto-vectorizer
 * to emit highly efficient streaming instructions.
 */
static void BM_SoA_Sum(benchmark::State& state) {
    fc::DatasetSoA<float, 3> ds;
    for (size_t i = 0; i < num_points; ++i) ds.push_back({ 1.0f, 1.0f, 1.0f });

    for (auto _ : state) {
        float sum = 0;
        for (std::size_t d = 0; d < 3; ++d) {
            const float* data = ds.axis_data(d);
            for (std::size_t i = 0; i < num_points; ++i) {
                sum += data[i];
            }
        }
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_SoA_Sum);