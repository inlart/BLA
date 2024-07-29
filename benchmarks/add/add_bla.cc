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

static void benchmark_add_bla(benchmark::State& state) {
    const int n = state.range(0);

    Matrix res({n, n});
    Matrix a({n, n});
    Matrix b({n, n});
    Matrix c({n, n});
    Matrix d({n, n});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);
    b.fill_seq(g);
    c.fill_seq(g);
    d.fill_seq(g);

    for(auto _ : state) {
        benchmark::DoNotOptimize(res = a + b + c + d);
    }
}

BENCHMARK(benchmark_add_bla)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
