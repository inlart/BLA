#pragma once

#include <allscale/api/user/data/grid.h>
#include <iostream>

#include "expressions.h"
#include "forward.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {


using point_type = GridPoint<2>;
using triple_type = GridPoint<3>;

struct RowRange {
	coordinate_type start;
	coordinate_type end;
};

struct BlockRange {
	BlockRange(point_type start, point_type size) : start(start), size(size) {}

	point_type start;
	point_type size;

	point_type range() const { return size; }

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
