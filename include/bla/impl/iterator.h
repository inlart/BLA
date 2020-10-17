#pragma once

#include "bla/impl/traits.h"
#include "bla/impl/types.h"

#include "bla/impl/forward.h"

#include <allscale/utils/assert.h>
#include <allscale/utils/optional.h>
#include <functional>
#include <iterator>
#include <memory>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E>
struct Iterator : public std::iterator<std::random_access_iterator_tag, scalar_type<E>> {
    using T = scalar_type_t<E>;
    using difference_type = typename std::iterator<std::random_access_iterator_tag, scalar_type_t<E>>::difference_type;

    // TODO: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator


    utils::optional<std::reference_wrapper<const MatrixExpression<E>>> back_ref;
    coordinate_type pos;

    Iterator(const MatrixExpression<E>& m, coordinate_type pos) : back_ref(m), pos(pos) {
    }

    Iterator() : back_ref(), pos(0) {
    }

    Iterator(const Iterator&) = default;
    Iterator& operator=(const Iterator&) = default;

    T operator*() const {
        assert_true(back_ref);
        return expr()[pointPos()];
    }

    Iterator& operator++() {
        ++pos;
        return *this;
    }

    Iterator& operator--() {
        --pos;
        return *this;
    }


    template <typename E2>
    auto operator-(const Iterator<E2>& other) const {
        return pos - other.pos;
    }

    Iterator& operator+=(difference_type v) {
        pos += v;
        return *this;
    }

    Iterator& operator-=(difference_type v) {
        pos -= v;
        return *this;
    }

    Iterator operator+(difference_type v) const {
        Iterator it = *this;
        return it += v;
    }


    bool operator<(const Iterator& other) const {
        return (*this - other) > 0;
    }

    bool operator>(const Iterator& other) const {
        return other < *this;
    }

    bool operator<=(const Iterator& other) const {
        return !(*this > other);
    }

    bool operator>=(const Iterator& other) const {
        return !(*this < other);
    }

    bool operator==(const Iterator& other) const {
        if(!(bool)back_ref || !(bool)other.back_ref)
            return false;
        return std::addressof((*back_ref).get()) == std::addressof((*other.back_ref).get()) && pos == other.pos;
    }

    bool operator!=(const Iterator& other) const {
        return !(*this == other);
    }

    point_type pointPos() const {
        return {pos / expr().columns(), pos % expr().columns()};
    }

private:
    const MatrixExpression<E>& expr() const {
        return (*back_ref).get();
    }
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
