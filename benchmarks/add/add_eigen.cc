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

static void benchmark_add_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd res = MatrixXd(n, n);
    MatrixXd a = MatrixXd::Random(n, n);
    MatrixXd b = MatrixXd::Random(n, n);
    MatrixXd c = MatrixXd::Random(n, n);
    MatrixXd d = MatrixXd::Random(n, n);

    for(auto _ : state) {
        benchmark::DoNotOptimize(res = a + b + c + d);
    }
}

BENCHMARK(benchmark_add_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
