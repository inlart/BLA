#pragma once

#include <allscale/api/user/data/grid.h>

#include "impl/expressions.h"
#include "impl/matrix_multiplication.h"
#include "impl/operators.h"
#include "impl/traits.h"
#include "impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {

template <typename T>
using Matrix = impl::Matrix<T>;

template <typename E>
using MatrixExpression = impl::MatrixExpression<E>;

using RowRange = impl::RowRange;

using BlockRange = impl::BlockRange;

using point_type = impl::point_type;

// using coordinate_type as defined in allscale/api/user/data/grid.h

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
