#include <Eigen/Dense>
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

static void benchmark_subtraction_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd a = MatrixXd::Random(n, n);

    int k = n / 2;

    for(auto _ : state) {
        benchmark::DoNotOptimize(a.bottomRightCorner(a.rows() - k - 1, a.cols() - k - 1).noalias() -= a.col(k).tail(a.rows() - k - 1) * a.row(k).tail(a.cols() - k - 1));
    }
}

BENCHMARK(benchmark_subtraction_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
