#pragma once

#include "bla/impl/types.h"

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

// -- Represents an evaluated expression
template <typename T>
class EvaluatedExpression;

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

// -- Represents an identity matrix
template <typename T>
class IdentityMatrix;

// -- A permutation matrix
template <typename T>
class PermutationMatrix;

// -- A view on a Matrix
template <typename E, ViewType View>
class MatrixView;

template <typename T>
class SimdBlock;

/*
 * Decompositions
 * Defined in decomposition.h
 */

// -- Lower Upper Decomposition with partial pivoting
template <typename T>
struct LUD;

// -- Lower Upper Decomposition with full pivoting
template <typename T>
struct FPLUD;

// -- QR Decomposition
template <typename T>
struct QRD;

// -- Singular Value Decomposition
template <typename T>
struct SVD;

/*
 * Eigen Solver
 * Defined in eigen.h
 */

// -- Lower Upper Decomposition
template <typename T>
struct EigenSolver;

/*
 * Traits
 * Defined in traits.h
 */

// -- extract the scalar type of a MatrixExpression
template <typename Expr>
struct scalar_type;

// -- trait to check SIMD vectorizability of a MatrixExpression
template <typename Expr>
struct vectorizable;

// -- type that is saved inside a MatrixExpression class
template <typename E>
struct expression_member;

// -- checks if T is associative
template <typename T>
struct is_associative;

// checks if F(T, T) returns a T
template <typename F, typename T>
struct type_consistent;

// result of operation Functor(T1, T2)
template <typename Functor, typename T1, typename T2>
struct operation_result;

// -- checks if E has direct access to the data
template <typename E>
struct direct_access;

// -- checks if the expression is a transpose of a matrix
template <typename E>
struct is_transpose;

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
