#pragma once

#include <iterator>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename Iterator>
struct IteratorWrapper : public std::iterator<std::random_access_iterator_tag, typename Iterator::value_type> {
    Iterator it;

    IteratorWrapper(const Iterator& it) : it(it) {
    }

    IteratorWrapper(const IteratorWrapper&) = default;
    IteratorWrapper& operator=(const IteratorWrapper&) = default;

    const Iterator& operator*() const {
        return it;
    }

    Iterator& operator*() {
        return it;
    }

    IteratorWrapper& operator++() {
        ++it;
        return *this;
    }

    IteratorWrapper& operator--() {
        ++it;
        return *this;
    }

    auto operator-(const IteratorWrapper& other) const {
        return it - other.it;
    }

    IteratorWrapper& operator+=(typename Iterator::difference_type v) {
        it += v;
        return *this;
    }

    IteratorWrapper& operator-=(typename Iterator::difference_type v) {
        it -= v;
        return *this;
    }

    IteratorWrapper operator+(typename Iterator::difference_type v) const {
        IteratorWrapper it = *this;
        return it += v;
    }


    bool operator<(const IteratorWrapper& other) const {
        return it < other.it;
    }

    bool operator>(const IteratorWrapper& other) const {
        return it > other.it;
    }

    bool operator<=(const IteratorWrapper& other) const {
        return it <= other.it;
    }

    bool operator>=(const IteratorWrapper& other) const {
        return it >= other.it;
    }

    bool operator==(const IteratorWrapper& other) {
        return it == other.it;
    }

    bool operator!=(const IteratorWrapper& other) {
        return it != other.it;
    }
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
