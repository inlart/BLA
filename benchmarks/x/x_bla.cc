#include <bla/matrix.h>
#include <benchmark/benchmark.h>
#include <iostream>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using Matrix = bla::Matrix<double>;

static void benchmark_x_bla(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n}), b({n, n}), c({n, n});

    a.fill(1.);
    allscale::api::user::algorithm::pfor(b.size(), [&](auto p) { b[p] = (p[0] + p[1] + 1) / (double)(n * n); });

    for(auto _ : state) {
        benchmark::DoNotOptimize(c = a + 0.0001 * (b + b * b));
    }
}

BENCHMARK(benchmark_x_bla)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
