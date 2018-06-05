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

static void benchmark_transpose_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd a = MatrixXd::Random(n, n);
    MatrixXd transpose = MatrixXd::Zero(n, n);

    for(auto _ : state) {
        benchmark::DoNotOptimize(transpose = a.transpose());
    }
}

BENCHMARK(benchmark_transpose_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
