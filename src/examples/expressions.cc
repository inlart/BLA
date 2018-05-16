#include <allscale/api/user/data/matrix.h>
#include <cstdlib>

using namespace allscale::api::user::data;

using data_type = double;

int main() {
    Matrix<data_type> m1({10, 15});
    Matrix<data_type> m2({15, 7});
    Matrix<data_type> m3({10, 7});
    Matrix<data_type> m4({10, 7});

    m1.fill(2);

    m2.fill(1);

    m3.fill(4);

    m4.fill(7);

    Matrix<data_type> result = 3 * m3 + m1 * m2 + m4;

    return EXIT_SUCCESS;
}
