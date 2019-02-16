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

#ifndef BENCHMARK_STEP
#define BENCHMARK_STEP 32
#endif


using Eigen::MatrixXd;

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for(int i = BENCHMARK_MIN_SIZE; i <= BENCHMARK_MAX_SIZE; i += BENCHMARK_STEP)
        b->Arg(i);
}

static void benchmark_mm_eigen(benchmark::State& state) {
    const int n = state.range(0);

    MatrixXd a = MatrixXd::Random(n, n);
    MatrixXd b = MatrixXd::Random(n, n);

    for(auto _ : state) {
        benchmark::DoNotOptimize((a * b).eval());
    }
}

BENCHMARK(benchmark_mm_eigen)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
