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

static void benchmark_rowswap_allscale(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);

    std::uniform_int_distribution<> udis(0, n - 1);


    for(auto _ : state) {
        int i1 = udis(gen);
        int i2 = udis(gen);
        a.row(i1).swap(a.row(i2));
        benchmark::DoNotOptimize(a[{i1, 0}]);
    }
}

BENCHMARK(benchmark_rowswap_allscale)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
