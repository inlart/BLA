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

    MatrixXd a = MatrixXd::Random(n, n);
    MatrixXd b = MatrixXd::Random(n, n);
    MatrixXd mult = MatrixXd::Zero(n, n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> udis(0, n - 1);

    int k = udis(gen);

    for(auto _ : state) {
        benchmark::DoNotOptimize((a.col(k).tail(n - k - 1) * b.row(k).tail(n - k - 1)).eval());
//        benchmark::DoNotOptimize(mult.bottomRightCorner(n - k - 1, n - k - 1) -=
//                a.col(k).tail(n - k - 1) * b.row(k).tail(n - k - 1));
    }
}

BENCHMARK(benchmark_submm_eigen)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
