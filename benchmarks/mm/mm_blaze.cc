#include <blaze/Blaze.h>
#include <benchmark/benchmark.h>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using namespace blaze;

static void benchmark_mm_eigen(benchmark::State& state) {
    const int n = state.range(0);

    DynamicMatrix<double, rowMajor> a(n, n), b(n, n), c(n, n);

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            a(i, j) = rand<double>();
            b(i, j) = rand<double>();
        }
    }

    for(auto _ : state) {
        benchmark::DoNotOptimize(c = a * b);
    }
}

BENCHMARK(benchmark_mm_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
