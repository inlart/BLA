#include <boost/numeric/ublas/matrix.hpp>
#include <benchmark/benchmark.h>
#include <random>

#ifndef BENCHMARK_MIN_SIZE
#define BENCHMARK_MIN_SIZE 128
#endif

#ifndef BENCHMARK_MAX_SIZE
#define BENCHMARK_MAX_SIZE 2048
#endif

using namespace boost::numeric::ublas;

static void benchmark_simplifytranspose_ublas(benchmark::State& state) {
    const int n = state.range(0);

    matrix<double> a(n, n), b(n, n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            a(i, j) = g();
        }
    }

    for(auto _ : state) {
        benchmark::DoNotOptimize(b = trans(trans(a)));
    }
}

BENCHMARK(benchmark_simplifytranspose_ublas)->RangeMultiplier(2)->Range(BENCHMARK_MIN_SIZE, BENCHMARK_MAX_SIZE)->UseRealTime();

BENCHMARK_MAIN();
