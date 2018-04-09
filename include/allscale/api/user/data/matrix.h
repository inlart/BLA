#pragma once

#include "impl/expressions.h"
#include "impl/matrix_multiplication.h"
#include "impl/operators.h"
#include "impl/traits.h"

namespace allscale {
namespace api {
namespace user {
namespace data {

template <typename T>
using Matrix = impl::Matrix<T>;

template <typename E>
using MatrixExpression = impl::MatrixExpression<E>;

using RowRange = impl::RowRange;

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
