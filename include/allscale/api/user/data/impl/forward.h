#pragma once

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

/*
 * The base class for all matrix expressions
 * Elements are not modifiable
 */
template <typename E>
class MatrixExpression;

/*
 * Represents the sum of two MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class MatrixAddition;

/*
 * Represents the subtraction of MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class MatrixSubtraction;

/*
 * Represents the element wise multiplication of MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class ElementMatrixMultiplication;

/*
 * Represents the multiplication of MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class MatrixMultiplication;

/*
 * Represents the negation of the MatrixExpression E
 */
template <typename E>
class MatrixNegation;

/*
 * Represents the transposed MatrixExpression E
 */
template <typename E>
class MatrixTranspose;

/*
 * Represents the absolute values of MatrixExpression E
 */
template <typename E>
class MatrixAbs;

/*
 * Represents the multiplication of matrix * scalar
 */
template <typename E, typename U>
class MatrixScalarMultiplication;

/*
 * Represents the multiplication of scalar * matrix
 */
template <typename E, typename U>
class ScalarMatrixMultiplication;


/*
 * Represents the Matrix
 * Elements are modifiable
 * Guarantees contiguous memory
 */
template <typename T = double>
class Matrix;

/*
 * Represents a part of a Matrix
 */
template <typename E>
class SubMatrix;

/*
 * Represents an identity matrix
 */
template <typename T>
class IdentityMatrix;


template <typename Expr>
struct scalar_type;

template <typename Expr>
struct vectorizable;

template <typename E>
struct expression_member;

template <typename T>
struct is_associative;

template <typename F, typename T>
struct type_consistent;

template <typename Functor, typename T1, typename T2>
struct operation_result;


struct RowRange;

struct BlockRange;


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
