#include <Eigen/Eigen>
#include <bla/matrix.h>
#include <allscale/api/user/algorithm/pfor.h>

#include <type_traits>

using namespace allscale::api::user;
using bla::point_type;

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toEigenMatrix(const bla::Matrix<T>& m) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(m.rows(), m.columns());
    allscale::api::user::algorithm::pfor(m.size(), [&](const point_type& p) { result(p.x, p.y) = m[p]; });
    return result;
}

template <typename Derived>
auto toAllscaleMatrix(const Eigen::MatrixBase<Derived>& m) -> bla::Matrix<std::remove_cv_t<std::remove_reference_t<decltype(m(0,0))>>> {
    bla::Matrix<std::remove_cv_t<std::remove_reference_t<decltype(m(0,0))>>> result({m.rows(), m.cols()});
    allscale::api::user::algorithm::pfor(result.size(), [&](const point_type& p) { result[p] = m(p.x, p.y); });
    return result;
}