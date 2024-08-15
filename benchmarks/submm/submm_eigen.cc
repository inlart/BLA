#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using Eigen::MatrixXd;

static void benchmark_submm_eigen(benchmark::State& state) {
    const int n = state.range(0);
    const int k = n / 2;

    MatrixXd a = MatrixXd::Random(n, n), b = MatrixXd::Random(n, n), c(n - k - 1, n - k - 1);

    for(auto _ : state) {
        benchmark::DoNotOptimize(c = a.col(k).tail(n - k - 1) * b.row(k).tail(n - k - 1));
    }
}

BENCHMARK(benchmark_submm_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
