// #define BLAZE_BLAS_MODE
// #define BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION

#include <benchmark/benchmark.h>
#include <blaze/Blaze.h>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using namespace blaze;

static void benchmark_stranspose_blaze(benchmark::State& state) {
    const int n = state.range(0);

    DynamicMatrix<double, rowMajor> a(n, n), b(n, n);

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            a(i, j) = rand<double>();
        }
    }

    for(auto _ : state) {
        benchmark::DoNotOptimize(b = trans(trans(a)));
    }
}

BENCHMARK(benchmark_stranspose_blaze)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
