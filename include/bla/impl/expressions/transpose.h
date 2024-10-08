#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

namespace bla {
namespace impl {

template <typename E>
class MatrixTranspose : public MatrixExpression<MatrixTranspose<E>> {
    using typename MatrixExpression<MatrixTranspose<E>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixTranspose(Exp u) : expression(u) {
    }

    T operator[](const point_type& pos) const {
        return expression[{pos.y, pos.x}];
    }

    void evaluation(Matrix<T>& tmp) {
        using ct = coordinate_type;

        using block_type = SimdBlock<decltype(expression.template packet<point_type>({0, 0}))>;

        allscale::api::user::algorithm::pfor(point_type{rows() / block_type::size()[0], columns() / block_type::size()[1]}, [&](const auto& pos) {
            coordinate_type i = pos.x * block_type::size()[0];
            coordinate_type j = pos.y * block_type::size()[1];
            block_type b(expression, {j, i});

            b.transpose();
            b.load_to(tmp, {i, j});
        });


        // transpose the rest that can't be done with a full block
        // right side
        for(ct i = 0; i < rows() - rows() % block_type::size()[0]; ++i) {
            for(ct j = columns() - columns() % block_type::size()[1]; j < columns(); ++j) {
                tmp[{i, j}] = expression[{j, i}];
            }
        }

        // bottom
        for(ct i = rows() - rows() % block_type::size()[0]; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                tmp[{i, j}] = expression[{j, i}];
            }
        }
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return expression.columns();
    }

    coordinate_type columns() const {
        return expression.rows();
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

} // namespace impl
} // namespace bla
