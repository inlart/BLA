#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

namespace bla {
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

    Matrix<T> inverse() const {
        Matrix<T> inverse{IdentityMatrix<T>(size())};
        using ct = coordinate_type;

        for(ct i = 0; i < this->rows(); ++i) {
            for(ct j = 0; j <= i; ++j) {
                for(ct ii = j; ii < i; ++ii) {
                    inverse[{i, j}] -= (*this)[{i, ii}] * inverse[{ii, j}];
                }
                inverse[{i, j}] /= (*this)[{i, i}];
            }
        }

        return inverse;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());
        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));


        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(x.rows(), columns());
        using ct = coordinate_type;

        allscale::api::user::algorithm::pfor(allscale::utils::Vector<ct, 1>(x.columns()), [&](const auto& p) {
            const ct ii = p[0];
            for(ct i = 0; i < this->rows(); ++i) {
                for(ct j = 0; j < i; ++j) {
                    x[{i, ii}] -= x[{j, ii}] * (*this)[{i, j}];
                }
                x[{i, ii}] /= (*this)[{i, i}];
            }
        });
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

    Matrix<T> inverse() const {
        Matrix<T> inverse{*this};
        using ct = coordinate_type;

        for(ct i = 1; i < this->rows(); ++i) {
            for(ct j = 0; j < i; ++j) {
                inverse[{i, j}] *= -1;
                for(ct ii = j + 1; ii < i; ++ii) {
                    inverse[{i, j}] -= (*this)[{i, ii}] * inverse[{ii, j}];
                }
            }
        }

        return inverse;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());
        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));


        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(x.rows(), columns());
        using ct = coordinate_type;

        for(ct i = 0; i < this->rows(); ++i) {
            for(ct j = 0; j < i; ++j) {
                auto val = (*this)[{i, j}];
                for(ct ii = 0; ii < x.columns(); ++ii) {
                    x[{i, ii}] -= x[{j, ii}] * val;
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

    Matrix<T> inverse() const {
        Matrix<T> inverse{IdentityMatrix<T>(size())};
        using ct = coordinate_type;

        for(ct i = this->rows() - 1; i >= 0; --i) {
            for(ct j = i; j < this->rows(); ++j) {
                for(ct ii = j; ii > i; --ii) {
                    inverse[{i, j}] -= (*this)[{i, ii}] * inverse[{ii, j}];
                }
                inverse[{i, j}] /= (*this)[{i, i}];
            }
        }

        return inverse;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());

        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));

        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(x.rows(), columns());
        using ct = coordinate_type;

        allscale::api::user::algorithm::pfor(allscale::utils::Vector<ct, 1>(x.columns()), [&](const auto& p) {
            const ct ii = p[0];
            for(ct i = this->rows() - 1; i >= 0; --i) {
                for(ct j = this->rows() - 1; j > i; --j) {
                    x[{i, ii}] -= x[{j, ii}] * (*this)[{i, j}];
                }
                x[{i, ii}] /= (*this)[{i, i}];
            }
        });
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

    Matrix<T> inverse() const {
        Matrix<T> inverse{*this};
        using ct = coordinate_type;

        for(ct i = this->rows() - 2; i >= 0; --i) {
            for(ct j = i + 1; j < this->rows(); ++j) {
                inverse[{i, j}] *= -1;
                for(ct ii = j - 1; ii > i; --ii) {
                    inverse[{i, j}] -= (*this)[{i, ii}] * inverse[{ii, j}];
                }
            }
        }

        return inverse;
    }

    Matrix<T> solve(SubMatrix<Matrix<T>> b) const {
        assert_eq(b.rows(), columns());

        Matrix<T> x(b);

        solveInPlace(SubMatrix<Matrix<T>>(x));

        return x;
    }

    void solveInPlace(SubMatrix<Matrix<T>> x) const {
        assert_eq(x.rows(), columns());
        using ct = coordinate_type;

        for(ct i = this->rows() - 1; i >= 0; --i) {
            for(ct j = this->rows() - 1; j > i; --j) {
                auto val = (*this)[{i, j}];
                for(ct ii = 0; ii < x.columns(); ++ii) {
                    x[{i, ii}] -= x[{j, ii}] * val;
                }
            }
        }
    }

private:
    Exp expression;
};

} // namespace impl
} // namespace bla
