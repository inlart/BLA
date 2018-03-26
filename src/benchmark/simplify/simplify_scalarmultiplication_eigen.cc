// sys
#include <iostream>
#include <stdlib.h>
#include <time.h>

// libs
#include <Eigen/Dense>
using namespace Eigen;

// own
#include "bench_result.h"
#include "timer.h"

BenchResult bench_eigen(int n) {
	BenchResult res(NUMBER_BENCHMARK_RUNS);

	MatrixXd a = MatrixXd::Ones(n, n);
	MatrixXd result = MatrixXd(n, n);

	auto first_elem_square_sums = 0.0; // to try and avoid optimiser pitfals

	for(int i = 0; i < NUMBER_BENCHMARK_RUNS; ++i) {
		{
			Timer t;
			result = a * 5 * 6 * 8;
			res.addMeasurement(t.elapsed());
		}
		first_elem_square_sums += result(0, 0) * result(0, 0);
	}
	if(first_elem_square_sums < 0) { // this should never happen and is just to disuade the optimiser
		return BenchResult(0);
	}
	return res;
}

int main(int argc, const char** argv) {
	srand(time(NULL));
	int matrix_size = 650;
	if(argc > 1) { matrix_size = std::stoi(argv[1]); }

	auto t = bench_eigen(matrix_size);

	if(argc > 2) { // output the raw machine processable data for benchmarking scripts
		std::cout << t.raw() << std::endl;
	} else {
		std::cout << "Eigen square matrix multiplication with size " << matrix_size << "x" << matrix_size << ": " << t.summary() << std::endl;
	}
}
