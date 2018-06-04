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

static void benchmark_x_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd a = MatrixXd::Ones(n, n);
    MatrixXd b = MatrixXd(n, n);

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            b(i, j) = (i + j + 1) / (double)(n * n);
        }
    }

    for(auto _ : state) {
        benchmark::DoNotOptimize(b = a + 0.0001 * (b + b * b));
    }
}

BENCHMARK(benchmark_x_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
