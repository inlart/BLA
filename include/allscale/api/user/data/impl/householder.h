#pragma once

#include "allscale/api/user/data/impl/expressions/matrix.h"

#include <cmath>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

// TODO: there might be better ways to apply a householder matrix / consider complex values

template <typename T>
struct Householder {
    template <bool V = false>
    Householder(SubMatrix<const Matrix<T>, V> m, point_type size) : P(size), v(m), beta(0) {
        P.identity();
        compute(m);
    }

    // apply the Householder matrix on the left
    void applyLeft(Matrix<T>& m) const {
        assert_eq(m.rows(), m.columns());
        assert_eq(m.size(), P.size());
        m = (P * m).eval();
    }

    // apply the Householder matrix on the right
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
    T beta;


    template <bool V = false>
    void compute(SubMatrix<const Matrix<T>, V> m) {
        T norm = v.norm();
        int rho = (v[{0, 0}] < static_cast<T>(0)) - (static_cast<T>(0) < v[{0, 0}]); // -sign(v[0])
        T u1 = v[{0, 0}] - rho * norm;
        v /= u1;
        v[{0, 0}] = static_cast<T>(1);
        beta = -rho * u1 / norm;


        P.bottomRows(m.rows()).bottomColumns(m.rows()) -= (v * v.transpose()) * beta;
    }
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
