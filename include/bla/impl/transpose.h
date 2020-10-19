#pragma once

#include <Vc/Vc>
#include <type_traits>

#ifdef Vc_HAVE_SSE
#include <xmmintrin.h>
#endif

#ifdef Vc_HAVE_AVX
#include <immintrin.h>
#endif

#include "bla/impl/expressions.h"

namespace bla {
namespace impl {

namespace detail {

// fallback function
template <typename data_type>
void transpose(std::array<Vc::Vector<data_type, Vc::VectorAbi::Scalar>, 1>&) {
    // Nothing to do in scalar case
}

#ifdef Vc_IMPL_SSE4_2

#ifdef _MM_TRANSPOSE4_PS

// -- AVX float 4x4
void transpose(std::array<Vc::Vector<float, Vc::VectorAbi::Sse>, 4>& rows) {
    // TODO: check if this is valid
    _MM_TRANSPOSE4_PS(reinterpret_cast<__m128&>(rows[0]), reinterpret_cast<__m128&>(rows[1]), reinterpret_cast<__m128&>(rows[2]),
                      reinterpret_cast<__m128&>(rows[3]));
}

#endif // _MM_TRANSPOSE4_PS

#endif // Vc_IMPL_SSE4_2

#ifdef Vc_IMPL_AVX2

// -- AVX float 8x8
void transpose(std::array<Vc::Vector<float, Vc::VectorAbi::Avx>, 8>& rows) {
    // TODO: check if this is valid
    __m256& r1 = reinterpret_cast<__m256&>(rows[0]);
    __m256& r2 = reinterpret_cast<__m256&>(rows[1]);
    __m256& r3 = reinterpret_cast<__m256&>(rows[2]);
    __m256& r4 = reinterpret_cast<__m256&>(rows[3]);
    __m256& r5 = reinterpret_cast<__m256&>(rows[4]);
    __m256& r6 = reinterpret_cast<__m256&>(rows[5]);
    __m256& r7 = reinterpret_cast<__m256&>(rows[6]);
    __m256& r8 = reinterpret_cast<__m256&>(rows[7]);

    __m256 t1, t2, t3, t4, t5, t6, t7, t8;
    __m256 u1, u2, u3, u4, u5, u6, u7, u8;


    t1 = _mm256_unpacklo_ps(r1, r2);
    t2 = _mm256_unpackhi_ps(r1, r2);
    t3 = _mm256_unpacklo_ps(r3, r4);
    t4 = _mm256_unpackhi_ps(r3, r4);
    t5 = _mm256_unpacklo_ps(r5, r6);
    t6 = _mm256_unpackhi_ps(r5, r6);
    t7 = _mm256_unpacklo_ps(r7, r8);
    t8 = _mm256_unpackhi_ps(r7, r8);

    u1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    u3 = _mm256_shuffle_ps(t2, t4, _MM_SHUFFLE(1, 0, 1, 0));
    u4 = _mm256_shuffle_ps(t2, t4, _MM_SHUFFLE(3, 2, 3, 2));
    u5 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    u7 = _mm256_shuffle_ps(t6, t8, _MM_SHUFFLE(1, 0, 1, 0));
    u8 = _mm256_shuffle_ps(t6, t8, _MM_SHUFFLE(3, 2, 3, 2));


    r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    r2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    r4 = _mm256_permute2f128_ps(u4, u8, 0x20);
    r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    r6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    r7 = _mm256_permute2f128_ps(u3, u7, 0x31);
    r8 = _mm256_permute2f128_ps(u4, u8, 0x31);
}

// -- AVX double 4x4
void transpose(std::array<Vc::Vector<double, Vc::VectorAbi::Avx>, 4>& rows) {
    // TODO: check if this is valid
    __m256d& r1 = reinterpret_cast<__m256d&>(rows[0]);
    __m256d& r2 = reinterpret_cast<__m256d&>(rows[1]);
    __m256d& r3 = reinterpret_cast<__m256d&>(rows[2]);
    __m256d& r4 = reinterpret_cast<__m256d&>(rows[3]);

    __m256d t1, t2, t3, t4;

    t1 = _mm256_shuffle_pd(r1, r2, 0x0);
    t2 = _mm256_shuffle_pd(r3, r4, 0x0);
    t3 = _mm256_shuffle_pd(r1, r2, 0xF);
    t4 = _mm256_shuffle_pd(r3, r4, 0xF);

    r1 = _mm256_permute2f128_pd(t1, t2, 0x20);
    r2 = _mm256_permute2f128_pd(t3, t4, 0x20);
    r3 = _mm256_permute2f128_pd(t1, t2, 0x31);
    r4 = _mm256_permute2f128_pd(t3, t4, 0x31);
}


// -- AVX int 8x8
// -- TODO
// void transpose(std::array<Vc::Vector<int, Vc::Vector_abi::avx>, 8>& rows) {
// }

#endif // Vc_IMPL_AVX2


// TODO: move
template <typename Arg, typename _ = void>
struct transpose_exists : std::false_type {};

template <typename Arg>
struct transpose_exists<Arg, decltype(transpose(std::declval<Arg&>()))> : std::true_type {};

template <typename Arg>
constexpr bool transpose_exists_v = transpose_exists<Arg>::value;

} // namespace detail


template <typename simd_type>
class SimdBlock {
    using T = typename simd_type::value_type;
    using abi_type = std::conditional_t<detail::transpose_exists_v<std::array<simd_type, simd_type::size()>>, typename simd_type::abi, Vc::VectorAbi::Scalar>;
    using simd_t = Vc::Vector<T, abi_type>;

public:
    template <typename E>
    SimdBlock(const MatrixExpression<E>& exp, point_type pos) {
        for(coordinate_type i = 0; i < (coordinate_type)simd_t::size(); ++i) {
            rows[i] = exp.template packet<simd_t>({pos.x + i, pos.y});
        }
    }

    static point_type size() {
        return {simd_t::size(), simd_t::size()};
    }

    void transpose() {
        detail::transpose(rows);
    }

    void load_to(Matrix<T>& matrix, point_type pos) {
        for(coordinate_type i = 0; i < (coordinate_type)simd_t::size(); ++i) {
            rows[i].store(&matrix[{pos.x + i, pos.y}]);
        }
    }

private:
    std::array<simd_t, simd_t::size()> rows;
};


template <typename E>
MatrixTranspose<expression_tree_t<const E>> MatrixExpression<E>::transpose() const {
    return MatrixTranspose<expression_tree_t<const E>>(static_cast<const E&>(*this));
}


} // namespace impl
} // namespace bla
