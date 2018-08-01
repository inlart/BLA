#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/transpositions.h"
#include "allscale/api/user/data/impl/types.h"
#include "allscale/utils/vector.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
class PermutationMatrix : public MatrixExpression<PermutationMatrix<T>> {
public:
    PermutationMatrix(coordinate_type c) : values(utils::Vector<coordinate_type, 1>{c}), swaps(0) {
        algorithm::pfor(utils::Vector<coordinate_type, 1>{c}, [&](const auto& p) { values[p] = p[0]; });
    }

    // TODO: remove this
    PermutationMatrix(const PermutationMatrix<T>& mat)
        : MatrixExpression<PermutationMatrix<T>>(), values(utils::Vector<coordinate_type, 1>{mat.rows()}), swaps(0) {
        algorithm::pfor(utils::Vector<coordinate_type, 1>{rows()}, [&](const auto& p) { values[p] = mat.values[p]; });
    }

    PermutationMatrix(PermutationMatrix<T>&&) = default;

    PermutationMatrix(const Transpositions& t) : swaps(0) {
        for(coordinate_type i = 0; i < t.length(); ++i) {
            swap(i, t[i]);
        }
    }

    PermutationMatrix& operator=(const Transpositions& t) {
        for(coordinate_type i = 0; i < t.length(); ++i) {
            swap(i, t[i]);
        }

        return *this;
    }

    T operator[](const point_type& pos) const {
        assert_lt(pos, size());
        return values[{pos.x}] == pos.y ? static_cast<T>(1) : static_cast<T>(0);
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return values.size()[0];
    }

    coordinate_type columns() const {
        return values.size()[0];
    }

    void swap(coordinate_type i, coordinate_type j) {
        if(i == j)
            return;

        coordinate_type old = values[{i}];
        values[{i}] = values[{j}];
        values[{j}] = old;
        swaps++;
    }

    coordinate_type permutation(coordinate_type i) const {
        assert_lt(i, rows());
        return values[i];
    }

    int numSwaps() const {
        return swaps;
    }

private:
    Grid<coordinate_type, 1> values;
    int swaps;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
