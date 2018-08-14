#include <allscale/api/user/data/matrix.h>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace allscale::api::user::data;

template <typename T>
Matrix<T> pagerank(const Matrix<T>& adj, T d, T eps) {
    assert_eq(adj.rows(), adj.columns());

    const coordinate_type N = adj.columns();

    Matrix<T> v({N, 1});
    Matrix<T> last_v({N, 1});

    v.fill(static_cast<T>(1) / static_cast<T>(N));

    Matrix<T> teleport(adj.size());

    teleport.fill((1 - d) / N);

    Matrix<T> m_hat = (d * adj) + teleport;

    do {
        std::cout << v << std::endl << std::endl;
        last_v = v;
        v = m_hat * last_v;

    } while((v - last_v).norm() > eps);

    return v;
}

int main() {
    Matrix<double> m1({5, 5});

    m1 << 0, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0.5, 0, 0, 0, 0, 0.5, 1, 0;

    auto v = pagerank(m1, 0.85, 1E-8);
    std::cout << v << std::endl;


    return EXIT_SUCCESS;
}
