#include <Eigen/Eigen>
#include <allscale/api/user/data/matrix.h>
#include <allscale/api/user/algorithm/pfor.h>

#include <type_traits>

using namespace allscale::api::user;
using allscale::api::user::data::point_type;

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toEigenMatrix(const allscale::api::user::data::Matrix<T>& m) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(m.rows(), m.columns());
    algorithm::pfor(m.size(), [&](const point_type& p) { result(p.x, p.y) = m[p]; });
    return result;
}

template <typename Derived>
auto toAllscaleMatrix(const Eigen::MatrixBase<Derived>& m) -> allscale::api::user::data::Matrix<std::remove_cv_t<std::remove_reference_t<decltype(m(0,0))>>> {
    allscale::api::user::data::Matrix<std::remove_cv_t<std::remove_reference_t<decltype(m(0,0))>>> result({m.rows(), m.cols()});
    algorithm::pfor(result.size(), [&](const point_type& p) { result[p] = m(p.x, p.y); });
    return result;
}