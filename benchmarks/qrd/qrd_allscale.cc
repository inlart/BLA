#include <bla/matrix.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using Matrix = bla::Matrix<double>;

static void benchmark_qrd_allscale(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n});
    Matrix lower({n, n});
    Matrix upper({n, n});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);

    for(auto _ : state) {
        benchmark::DoNotOptimize(a.QRDecomposition());
        //        benchmark::DoNotOptimize(lower = lud.lower());
        //        benchmark::DoNotOptimize(upper = lud.upper());
    }
}

BENCHMARK(benchmark_qrd_allscale)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
