#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

#include <Vc/Vc>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E>
class AccessBase : public MatrixExpression<E> {
    using T = scalar_type_t<E>;
    using PacketScalar = typename MatrixExpression<E>::PacketScalar;

protected:
private:
    E& impl() {
        return static_cast<E&>(*this);
    }

    const E& impl() const {
        return static_cast<const E&>(*this);
    }

public:
    point_type size() const {
        return impl().size();
    }

    coordinate_type rows() const {
        return impl().rows();
    }

    coordinate_type columns() const {
        return impl().columns();
    }

    T& operator[](const point_type& pos) {
        return impl()[pos];
    }

    const T& operator[](const point_type& pos) const {
        return impl()[pos];
    }

    void fill(const T& value) {
        detail::set_value(value, static_cast<E&>(*this));
    }

    void fill(std::function<T(point_type)> f) {
        algorithm::pfor(size(), [&](const point_type& p) { (*this)[p] = f(p); });
    }

    void fill(std::function<T()> f) {
        algorithm::pfor(size(), [&](const point_type& p) { (*this)[p] = f(); });
    }

    void fill_seq(const T& value) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = value;
            }
        }
    }

    void fill_seq(std::function<T(point_type)> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f(point_type{i, j});
            }
        }
    }

    void fill_seq(std::function<T()> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f();
            }
        }
    }

    void zero() {
        fill(static_cast<T>(0));
    }

    void eye() {
        fill([](const auto& pos) { return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0); });
    }

    void identity() {
        assert_eq(rows(), columns());
        eye();
    }

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p));
    }

protected:
    template <typename E2>
    void evaluate(const MatrixExpression<E2>&);
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
