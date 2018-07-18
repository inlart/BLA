#pragma once

#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
class Matrix : public AccessBase<Matrix<T>> {
    using map_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using cmap_type = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    using typename MatrixExpression<Matrix<T>>::PacketScalar;

public:
    Matrix(const point_type& size) : m_data(size) {
    }

    template <typename E>
    Matrix(const MatrixExpression<E>& mat) : m_data(mat.size()) {
        AccessBase<Matrix<T>>::evaluate(mat);
    }

    template <typename Derived>
    Matrix(const Eigen::MatrixBase<Derived>& matrix) : m_data({matrix.rows(), matrix.cols()}) {
        algorithm::pfor(size(), [&](const point_type& p) { m_data[p] = matrix(p.x, p.y); });
    }

    Matrix(const Matrix& mat) : m_data(mat.size()) {
        AccessBase<Matrix<T>>::evaluate(mat);
    }

    Matrix(Matrix&&) = default;

    Matrix& operator=(const Matrix& mat) {
        AccessBase<Matrix<T>>::evaluate(mat);

        return *this;
    }

    template <typename E2>
    Matrix& operator=(const MatrixExpression<E2>& mat) {
        AccessBase<Matrix<T>>::evaluate(mat);

        return *this;
    }

    Matrix& operator=(Matrix&&) = default;

    Matrix& operator=(const T& value) {
        fill(value);
        return (*this);
    }

    T& operator[](const point_type& pos) {
        return m_data[pos];
    }

    const T& operator[](const point_type& pos) const {
        return m_data[pos];
    }

    point_type size() const {
        return m_data.size();
    }

    coordinate_type rows() const {
        return m_data.size()[0];
    }

    coordinate_type columns() const {
        return m_data.size()[1];
    }

    map_type eigenSub(const range_type& r) {
        return map_type(&m_data[{r.x, 0}], r.y, columns());
    }

    cmap_type eigenSub(const range_type& r) const {
        return cmap_type(&m_data[{r.x, 0}], r.y, columns());
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toEigenMatrix() {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(rows(), columns());
        algorithm::pfor(size(), [&](const point_type& p) { result(p.x, p.y) = m_data[p]; });
        return result;
    }

    map_type getEigenMap() {
        return eigenSub({0, rows()});
    }

    cmap_type getEigenMap() const {
        return eigenSub({0, rows()});
    }

    const Matrix<T>& eval() const {
        return *this;
    }
    Matrix<T>& eval() {
        return *this;
    }

    coordinate_type stride() const {
        return columns();
    }

private:
    data::Grid<T, 2> m_data;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
