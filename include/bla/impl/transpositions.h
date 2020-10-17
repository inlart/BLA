#pragma once


#include <type_traits>

#include <allscale/api/user/data/grid.h>
#include "expressions.h"
#include "traits.h"
#include "types.h"


namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

class Transpositions {
public:
    Transpositions(coordinate_type size) : transpositions(size) {
    }

    coordinate_type& operator[](const coordinate_type& pos) {
        return transpositions[pos];
    }

    const coordinate_type& operator[](const coordinate_type& pos) const {
        return transpositions[pos];
    }

    auto length() const {
        return transpositions.size()[0];
    }

private:
    Grid<coordinate_type, 1> transpositions;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
