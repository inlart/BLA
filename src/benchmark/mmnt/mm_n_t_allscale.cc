#include <allscale/api/user/data/matrix.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using Matrix = allscale::api::user::data::Matrix<double>;

static void benchmark_mmnt_allscale(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n});
    Matrix b({n, n});

    Matrix mult({n, n});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);
    b.fill_seq(g);

    for(auto _ : state) {
        benchmark::DoNotOptimize(mult = a * b.transpose());
    }
}

BENCHMARK(benchmark_mmnt_allscale)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
