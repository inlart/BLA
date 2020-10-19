#pragma once

#include "bla/impl/decomposition.h"
#include "bla/impl/eigen.h"
#include "bla/impl/evaluate.h"
#include "bla/impl/expressions.h"
#include "bla/impl/matrix_multiplication.h"
#include "bla/impl/operators.h"
#include "bla/impl/simplify.h"
#include "bla/impl/traits.h"
#include "bla/impl/transpose.h"
#include "bla/impl/types.h"

namespace bla {

template <typename T>
using Matrix = impl::Matrix<T>;

template <typename E>
using MatrixExpression = impl::MatrixExpression<E>;

using BlockRange = impl::BlockRange;

using point_type = impl::point_type;

using range_type = impl::range_type;

using coordinate_type = impl::coordinate_type;

} // namespace bla
