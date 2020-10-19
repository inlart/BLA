#pragma once

#include "bla/impl/expressions/access.h"
#include "bla/impl/forward.h"
#include "bla/impl/types.h"
#include <allscale/api/user/data/grid.h>

namespace bla {
namespace impl {

template <typename T>
class Matrix : public AccessBase<Matrix<T>> {
    using typename MatrixExpression<Matrix<T>>::PacketScalar;

public:
    Matrix(const point_type& size) : m_data(size) {
    }

    template <typename E>
    Matrix(const MatrixExpression<E>& mat) : m_data(mat.size()) {
        AccessBase<Matrix<T>>::evaluate(mat);
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
    allscale::api::user::data::Grid<T, 2> m_data;
};

} // namespace impl
} // namespace bla
