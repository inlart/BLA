#pragma once

#include <allscale/api/user/data/grid.h>

namespace bla {
namespace impl {

using coordinate_type = allscale::api::user::data::coordinate_type;

using point_type = allscale::api::user::data::GridPoint<2>;
using range_type = allscale::api::user::data::GridPoint<2>;
using triple_type = allscale::api::user::data::GridPoint<3>;

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

enum class ViewType { Lower, UnitLower, Upper, UnitUpper };

} // end namespace impl
} // namespace bla
