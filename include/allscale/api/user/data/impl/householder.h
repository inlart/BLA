#pragma once

#include "allscale/api/user/data/impl/expressions/matrix.h"

#include <cmath>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
struct Householder {
    template <bool V = false>
    Householder(SubMatrix<const Matrix<T>, V> m, point_type size) : P(size), v(m.size()) {
        P.identity();
        compute(m);
    }

    void applyLeft(Matrix<T>& m) const {
        assert_eq(m.rows(), m.columns());
        assert_eq(m.size(), P.size());
        m = (P * m).eval();
    }


    void applyRight(Matrix<T>& m) const {
        assert_eq(m.rows(), m.columns());
        assert_eq(m.size(), P.size());
        m *= P;
    }

    const Matrix<T>& getP() const {
        return P;
    }

private:
    // Householder Matrix
    Matrix<T> P;
    Matrix<T> v;


    template <bool V = false>
    void compute(SubMatrix<const Matrix<T>, V> m) {
        Matrix<T> v(m.size());

        v = m;
        T mag = v.product(v).accumulate();

        mag -= v[{0, 0}] * v[{0, 0}];

        v[{0, 0}] += v[{0, 0}] < 0 ? std::sqrt(mag) : -std::sqrt(mag);

        mag += v[{0, 0}] * v[{0, 0}];
        mag = std::sqrt(mag);

        // TODO: set fail state
        if(mag < 1E-10)
            return;

        v /= mag;

        P.bottomRows(m.rows()).bottomColumns(m.rows()) -= (v * v.transpose()) * 2.0;
    }
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
