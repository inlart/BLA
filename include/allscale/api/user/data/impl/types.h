#pragma once

#include <allscale/api/user/data/grid.h>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

using coordinate_type = allscale::api::user::data::coordinate_type;

using point_type = GridPoint<2>;
using range_type = GridPoint<2>;
using triple_type = GridPoint<3>;

struct BlockRange {
    BlockRange() : start({0, 0}), size({0, 0}) {
    }
    BlockRange(point_type start, point_type size) : start(start), size(size) {
    }

    point_type start;
    point_type size;

    point_type range() const {
        return size;
    }

    coordinate_type area() const {
        auto x = range();
        return x.x * x.y;
    }
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
