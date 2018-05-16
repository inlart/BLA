// system
#include <iostream>

#include <Eigen/Dense>
#include <allscale/api/user/data/matrix.h>

// own
#include "bench_result.h"
#include "timer.h"
#include "utility.h"

using Matrix = allscale::api::user::data::Matrix<double>;

BenchResult bench_allscale(int n) {
    BenchResult res(NUMBER_BENCHMARK_RUNS);

    Matrix a({n, n});
    Matrix b({n, n});
    Matrix c({n, n});
    Matrix d({n, n});
    Matrix mult({n, n});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill(g);
    b.fill(g);
    Eigen::MatrixXd a_eigen = a.toEigenMatrix();
    Eigen::MatrixXd b_eigen = b.toEigenMatrix();
    Eigen::MatrixXd c_eigen = c.toEigenMatrix();
    Eigen::MatrixXd d_eigen = d.toEigenMatrix();
    Matrix res_eigen = (a_eigen + b_eigen + c_eigen + d_eigen).eval();
    auto first_elem_square_sums = 0.0; // to try and avoid optimiser pitfals

    for(int i = 0; i < NUMBER_BENCHMARK_RUNS; ++i) {
        {
            Timer t;
            mult = a + b + c + d;
            res.addMeasurement(t.elapsed());
        }
        first_elem_square_sums += mult[{0, 0}] * mult[{0, 0}];
        if(!isAlmostEqual(mult, res_eigen, 0.001)) {
            std::cerr << "Matrix multiplication check failed" << std::endl;
            return BenchResult(0);
        }
    }
    if(first_elem_square_sums < 0) { // this should never happen and is just to disuade the optimiser
        return BenchResult(0);
    }
    return res;
}

int main(int argc, const char** argv) {
    srand(time(NULL));
    int matrix_size = 650;
    if(argc > 1) {
        matrix_size = std::stoi(argv[1]);
    }

    auto t = bench_allscale(matrix_size);

    if(argc > 2) { // output the raw machine processable data for benchmarking scripts
        std::cout << t.raw() << std::endl;
    } else {
        std::cout << "Allscale square matrix multiplication with size " << matrix_size << "x" << matrix_size << ": " << t.summary() << std::endl;
    }
}
