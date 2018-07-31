#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E>
class MatrixView<E, ViewType::Lower> : public MatrixExpression<MatrixView<E, ViewType::Lower>> {
    using typename MatrixExpression<MatrixView<E, ViewType::Lower>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixView(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        if(pos.x >= pos.y)
            return expression[pos];
        return static_cast<T>(0);
    }

    point_type size() const {
        return expression.size();
    }

    coordinate_type rows() const {
        return expression.rows();
    }

    coordinate_type columns() const {
        return expression.columns();
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

template <typename E>
class MatrixView<E, ViewType::UnitLower> : public MatrixExpression<MatrixView<E, ViewType::UnitLower>> {
    using typename MatrixExpression<MatrixView<E, ViewType::UnitLower>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixView(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        if(pos.x > pos.y)
            return expression[pos];
        else if(pos.x == pos.y)
            return static_cast<T>(1);
        return static_cast<T>(0);
    }

    point_type size() const {
        return expression.size();
    }

    coordinate_type rows() const {
        return expression.rows();
    }

    coordinate_type columns() const {
        return expression.columns();
    }

    Exp getExpression() const {
        return expression;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());
        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));

        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(b.rows(), columns());
        using ct = coordinate_type;

        for(ct ii = 0; ii < x.columns(); ++ii) { // TODO: pfor
            for(ct i = 0; i < rows(); ++i) {
                for(ct j = 0; j < i; ++j) {
                    x[{i, ii}] -= x[{j, ii}] * (*this)[{i, j}];
                }
            }
        }
    }


private:
    Exp expression;
};

template <typename E>
class MatrixView<E, ViewType::Upper> : public MatrixExpression<MatrixView<E, ViewType::Upper>> {
    using typename MatrixExpression<MatrixView<E, ViewType::Upper>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixView(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        if(pos.x <= pos.y)
            return expression[pos];
        return static_cast<T>(0);
    }

    point_type size() const {
        return expression.size();
    }

    coordinate_type rows() const {
        return expression.rows();
    }

    coordinate_type columns() const {
        return expression.columns();
    }

    Exp getExpression() const {
        return expression;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());

        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));

        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(b.rows(), columns());
        using ct = coordinate_type;

        for(ct ii = 0; ii < x.columns(); ++ii) { // TODO: pfor
            for(ct i = rows() - 1; i >= 0; --i) {
                for(ct j = rows() - 1; j > i; --j) {
                    x[{i, ii}] -= x[{j, ii}] * (*this)[{i, j}];
                }
                x[{i, ii}] /= (*this)[{i, i}];
            }
        }
    }

private:
    Exp expression;
};

template <typename E>
class MatrixView<E, ViewType::UnitUpper> : public MatrixExpression<MatrixView<E, ViewType::UnitUpper>> {
    using typename MatrixExpression<MatrixView<E, ViewType::UnitUpper>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixView(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        if(pos.x < pos.y)
            return expression[pos];
        else if(pos.x == pos.y)
            return static_cast<T>(1);
        return static_cast<T>(0);
    }

    point_type size() const {
        return expression.size();
    }

    coordinate_type rows() const {
        return expression.rows();
    }

    coordinate_type columns() const {
        return expression.columns();
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
