#include <allscale/api/user/data/matrix.h>
#include <benchmark/benchmark.h>
#include <iostream>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using Matrix = allscale::api::user::data::Matrix<double>;

static void benchmark_x_allscale(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n});
    Matrix b({n, n});

    a.fill(1.);
    allscale::api::user::algorithm::pfor(b.size(), [&](auto p) { b[p] = (p[0] + p[1] + 1) / (double)(n * n); });

    for(auto _ : state) {
        benchmark::DoNotOptimize((a + 0.0001 * (b + b * b)).eval());
    }
}

BENCHMARK(benchmark_x_allscale)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
