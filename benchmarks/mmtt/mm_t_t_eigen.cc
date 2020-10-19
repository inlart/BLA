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

static void benchmark_mmtt_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd a = MatrixXd::Random(n, n);
    MatrixXd b = MatrixXd::Random(n, n);

    for(auto _ : state) {
        benchmark::DoNotOptimize((a.transpose() * b.transpose()).eval());
    }
}

BENCHMARK(benchmark_mmtt_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
