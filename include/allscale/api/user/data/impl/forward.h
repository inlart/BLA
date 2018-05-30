#pragma once

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

/*
 * Expressions
 * Defined in expressions.h
 */

// -- The base class for all matrix expressions
template <typename E>
class MatrixExpression;

// -- Represents the sum of two MatrixExpressions E1 and E2
template <typename E1, typename E2>
class MatrixAddition;

// -- Represents the subtraction of MatrixExpressions E1 and E2
template <typename E1, typename E2>
class MatrixSubtraction;

// -- Represents the element wise multiplication of MatrixExpressions E1 and E2
template <typename E1, typename E2>
class ElementMatrixMultiplication;

// -- Represents the multiplication of MatrixExpressions E1 and E2
template <typename E1, typename E2>
class MatrixMultiplication;

// -- Represents an evaluated matrix multiplication with resulting type T
template <typename T>
class EvaluatedMatrixMultiplication;

// -- Represents the negation of the MatrixExpression E
template <typename E>
class MatrixNegation;

// -- Represents the transposed MatrixExpression E
template <typename E>
class MatrixTranspose;

// -- Represents the conjugate MatrixExpression E
template <typename E>
class MatrixConjugate;

// -- Represents the absolute values of MatrixExpression E
template <typename E>
class MatrixAbs;

// -- Represents the multiplication of matrix * scalar
template <typename E, typename U>
class MatrixScalarMultiplication;

// -- Represents the multiplication of scalar * matrix
template <typename E, typename U>
class ScalarMatrixMultiplication;


// -- Represents the Matrix - contiguous memory
template <typename T = double>
class Matrix;

// -- Represents a part of a MatrixExpression
template <typename E>
class SubMatrix;

// -- Represents a part of a Matrix
template <typename T, bool Contiguous = false>
class RefSubMatrix;

// -- Represents an identity matrix
template <typename T>
class IdentityMatrix;

// -- A permutation matrix
template <typename T>
class PermutationMatrix;

/*
 * Decompositions
 * Defined in decomposition.h
 */

// -- Lower Upper Decomposition
template <typename T>
struct LUD;

// -- QR Decomposition
template <typename T>
struct QRD;

// -- Singular Value Decomposition
template <typename T>
struct SVD;

/*
 * Traits
 * Defined in traits.h
 */

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

/*
 * Types
 * Defined in types.h
 */

struct BlockRange;


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
