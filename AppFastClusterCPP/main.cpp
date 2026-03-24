#include <benchmark/benchmark.h>

/**
 * @brief Entry point for the Google Benchmark suite.
 * * This macro generates a main() function that automatically discovers and
 * executes all registered benchmarks (BM_AoS_Sum, BM_SoA_Sum, etc.),
 * reporting execution time and CPU cycles.
 */
BENCHMARK_MAIN();