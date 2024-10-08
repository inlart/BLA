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

static void benchmark_submm_bla(benchmark::State& state) {
    const int n = state.range(0);
    const int k = n / 2;

    Matrix a({n, n}), b({n, n}), c({n - k - 1, n - k - 1});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    std::uniform_int_distribution<> udis(0, n - 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);
    b.fill_seq(g);


    for(auto _ : state) {
        benchmark::DoNotOptimize(c = a.column(k).bottomRows(n - k - 1) * b.row(k).bottomColumns(n - k - 1));
    }
}

BENCHMARK(benchmark_submm_bla)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
