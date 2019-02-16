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

#ifndef BENCHMARK_STEP
#define BENCHMARK_STEP 32
#endif


using Matrix = allscale::api::user::data::Matrix<double>;

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for(int i = BENCHMARK_MIN_SIZE; i <= BENCHMARK_MAX_SIZE; i += BENCHMARK_STEP)
        b->Arg(i);
}

static void benchmark_mm_allscale_rowsplit(benchmark::State& state) {
    const int n = state.range(0);

    Matrix a({n, n});
    Matrix b({n, n});
    Matrix c({n, n});


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);
    b.fill_seq(g);

    for(auto _ : state) {
        matrix_multiplication_pblas(c, a, b);
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(benchmark_mm_allscale_rowsplit)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();
