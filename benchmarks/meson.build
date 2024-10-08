if get_option('b_ndebug') == 'false'
    warning('Benchmarks are built with b_ndebug == false')
endif

if get_option('buildtype') != 'release'
    warning('Benchmarks are built with buildtype != release')
endif

if get_option('optimization') != '3'
    warning('Benchmarks are built with optimization != 3')
endif

additional_args = []

if get_option('benchmarks-min-size') > 0
    additional_args += '-DBENCHMARK_MIN_SIZE=' + get_option('benchmarks-min-size').to_string()
endif

if get_option('benchmarks-max-size') > 0
    additional_args += '-DBENCHMARK_MAX_SIZE=' + get_option('benchmarks-max-size').to_string()
endif

benchmark_dep = dependency('benchmark', required: true)

benchmark_allscale_dep = [thread_dep, allscale_dep, blas_dep, vc_dep, benchmark_dep]

openmp_dep = dependency('openmp', required: true)

# BLA Benchmarks
e_add_bla = executable('benchmark_add_bla', 'add/add_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_copy_bla = executable('benchmark_copy_bla', 'copy/copy_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_lud_bla = executable('benchmark_lud_bla', 'lud/lud_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_mm_bla = executable('benchmark_mm_bla', 'mm/mm_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_mmflops_bla = executable('benchmark_mmflops_bla', 'mmflops/mmflops_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_mmnt_bla = executable('benchmark_mmnt_bla', 'mmnt/mmnt_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_mmtn_bla = executable('benchmark_mmtn_bla', 'mmtn/mmtn_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_mmtt_bla = executable('benchmark_mmtt_bla', 'mmtt/mmtt_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_qrd_bla = executable('benchmark_qrd_bla', 'qrd/qrd_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_rowswap_bla = executable('benchmark_rowswap_bla', 'rowswap/rowswap_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_simplifyscalarmultiplication_bla = executable('benchmark_simplifyscalarmultiplication_bla', 'simplifyscalarmultiplication/simplifyscalarmultiplication_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_simplifysubmatrixmultiplication_bla = executable('benchmark_simplifysubmatrixmultiplication_bla', 'simplifysubmatrixmultiplication/simplifysubmatrixmultiplication_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_simplifytranspose_bla = executable('benchmark_simplifytranspose_bla', 'simplifytranspose/simplifytranspose_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_submm_bla = executable('benchmark_submm_bla', 'submm/submm_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_subtraction_bla = executable('benchmark_subtraction_bla', 'subtraction/subtraction_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_transpose_bla = executable('benchmark_transpose_bla', 'transpose/transpose_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)
e_x_bla = executable('benchmark_x_bla', 'x/x_bla.cc', cpp_args: additional_args, include_directories: bla_inc, dependencies: benchmark_allscale_dep)

foreach i : [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    e_cutoff_block = executable('benchmark_mmcutoff_block_' + i.to_string() + '_bla', 'mmcutoff/mm_bla.cc', cpp_args: additional_args + ['-DBLA_MM_PARALLEL_CUTOFF=' + i.to_string()], include_directories: bla_inc, dependencies: benchmark_allscale_dep)
    e_cutoff_row = executable('benchmark_mmcutoff_row_' + i.to_string() + '_bla', 'mmcutoff/mm_bla_rowsplit.cc', cpp_args: additional_args + ['-DBLA_MM_PARALLEL_CUTOFF=' + i.to_string()], include_directories: bla_inc, dependencies: benchmark_allscale_dep)
endforeach

benchmark('Add BLA', e_add_bla)
benchmark('Copy BLA', e_copy_bla)
benchmark('Lower-Upper Decomposition BLA', e_lud_bla)
benchmark('Matrix Multiplication BLA', e_mm_bla)
benchmark('Matrix Multiplication FLOPS BLA', e_mmflops_bla)
benchmark('Matrix Multiplication NT BLA', e_mmnt_bla)
benchmark('Matrix Multiplication TN BLA', e_mmtn_bla)
benchmark('Matrix Multiplication TT BLA', e_mmtt_bla)
benchmark('QR Decomposition BLA', e_qrd_bla)


eigen_dep = dependency('eigen3', required: false)

# Eigen Benchmarks
if eigen_dep.found()
    benchmark_eigen_dep = [thread_dep, benchmark_dep, eigen_dep, openmp_dep]
    benchmark_eigen_blas_dep = [thread_dep, blas_dep, benchmark_dep, eigen_dep, openmp_dep]

    e_add_eigen = executable('benchmark_add_eigen', 'add/add_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_copy_eigen = executable('benchmark_copy_eigen', 'copy/copy_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_lud_eigen = executable('benchmark_lud_eigen', 'lud/lud_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_mm_eigen = executable('benchmark_mm_eigen', 'mm/mm_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_mmflops_eigen = executable('benchmark_mmflops_eigen', 'mmflops/mmflops_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_mmnt_eigen = executable('benchmark_mmnt_eigen', 'mmnt/mmnt_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_mmtn_eigen = executable('benchmark_mmtn_eigen', 'mmtn/mmtn_eigen.cc', dependencies: benchmark_eigen_dep)
    e_mmtt_eigen = executable('benchmark_mmtt_eigen', 'mmtt/mmtt_eigen.cc', dependencies: benchmark_eigen_dep)
    e_qrd_eigen = executable('benchmark_qrd_eigen', 'qrd/qrd_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_rowswap_eigen = executable('benchmark_rowswap_eigen', 'rowswap/rowswap_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_simplifyscalarmultiplication_eigen = executable('benchmark_simplifyscalarmultiplication_eigen', 'simplifyscalarmultiplication/simplifyscalarmultiplication_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_simplifysubmatrixmultiplication_eigen = executable('benchmark_simplifysubmatrixmultiplication_eigen', 'simplifysubmatrixmultiplication/simplifysubmatrixmultiplication_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_simplifytranspose_eigen = executable('benchmark_simplifytranspose_eigen', 'simplifytranspose/simplifytranspose_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_submm_eigen = executable('benchmark_submm_eigen', 'submm/submm_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_subtraction_eigen = executable('benchmark_subtraction_eigen', 'subtraction/subtraction_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_transpose_eigen = executable('benchmark_transpose_eigen', 'transpose/transpose_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)
    e_x_eigen = executable('benchmark_x_eigen', 'x/x_eigen.cc', cpp_args: additional_args, dependencies: benchmark_eigen_dep)

    benchmark('Add Eigen', e_add_eigen)
    benchmark('Copy Eigen', e_copy_eigen)
    benchmark('Lower-Upper Decomposition Eigen', e_lud_eigen)
    benchmark('Matrix Multiplication Eigen', e_mm_eigen)
    benchmark('Matrix Multiplication FLOPS Eigen', e_mmflops_eigen)
    benchmark('Matrix Multiplication NT Eigen', e_mmnt_eigen)
    benchmark('Matrix Multiplication TN Eigen', e_mmtn_eigen)
    benchmark('Matrix Multiplication TT Eigen', e_mmtt_eigen)
    benchmark('QR Decomposition Eigen', e_qrd_eigen)
endif

blaze_dep = dependency('blaze', required: false)

# Blaze Benchmarks
if blaze_dep.found()
    benchmark_blaze_dep = [thread_dep, benchmark_dep, blaze_dep, openmp_dep]
    benchmark_blaze_blas_dep = benchmark_blaze_dep + [blas_dep]

    e_add_blaze = executable('benchmark_add_blaze', 'add/add_blaze.cc', cpp_args: additional_args, dependencies: benchmark_blaze_dep)
    e_mm_blaze = executable('benchmark_mm_blaze', 'mm/mm_blaze.cc', cpp_args: ['-DBLAZE_BLAS_MODE=0'] + additional_args, dependencies: benchmark_blaze_dep)
    e_mm_blaze_blas = executable('benchmark_mm_blaze_blas', 'mm/mm_blaze.cc', cpp_args: ['-DBLAZE_BLAS_MODE=1', '-DBLAZE_BLAS_IS_PARALLEL=0', '-DBLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION=1'] + additional_args, dependencies: benchmark_blaze_blas_dep)
    e_simplifytranspose_blaze = executable('benchmark_simplifytranspose_blaze', 'simplifytranspose/simplifytranspose_blaze.cc', cpp_args: additional_args, dependencies: benchmark_blaze_dep)
    e_simplifysubmatrixmultiplication_blaze = executable('benchmark_simplifysubmatrixmultiplication_blaze', 'simplifysubmatrixmultiplication/simplifysubmatrixmultiplication_blaze.cc', cpp_args: additional_args, dependencies: benchmark_blaze_dep)
    e_simplifysubmatrixmultiplication_blaze_blas = executable('benchmark_simplifysubmatrixmultiplication_blaze_blas', 'simplifysubmatrixmultiplication/simplifysubmatrixmultiplication_blaze.cc', cpp_args: ['-DBLAZE_BLAS_MODE=1', '-DBLAZE_BLAS_IS_PARALLEL=0', '-DBLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION=1'] + additional_args, dependencies: benchmark_blaze_blas_dep)
    e_transpose_blaze = executable('benchmark_transpose_blaze', 'transpose/transpose_blaze.cc', cpp_args: additional_args, dependencies: benchmark_blaze_dep)


    benchmark('Add Blaze', e_add_blaze)
    benchmark('Matrix Multiplication Blaze', e_mm_blaze)
    benchmark('Simplify Transpose Blaze', e_simplifytranspose_blaze)
endif

boost_dep = dependency('boost', required: false)

# Boost uBLAS Benchmarks
if boost_dep.found()
    benchmark_ublas_dep = [thread_dep, benchmark_dep, boost_dep, openmp_dep]


    e_add_ublas = executable('benchmark_add_ublas', 'add/add_ublas.cc', cpp_args: additional_args, dependencies: benchmark_ublas_dep)
    e_mm_ublas = executable('benchmark_mm_ublas', 'mm/mm_ublas.cc', cpp_args: additional_args, dependencies: benchmark_ublas_dep)
    e_simplifytranspose_ublas = executable('benchmark_simplifytranspose_ublas', 'simplifytranspose/simplifytranspose_ublas.cc', cpp_args: additional_args, dependencies: benchmark_ublas_dep)
    e_simplifysubmatrixmultiplication_ublas = executable('benchmark_simplifysubmatrixmultiplication_ublas', 'simplifysubmatrixmultiplication/simplifysubmatrixmultiplication_ublas.cc', cpp_args: additional_args, dependencies: benchmark_ublas_dep)
    e_transpose_ublas = executable('benchmark_transpose_ublas', 'transpose/transpose_ublas.cc', cpp_args: additional_args, dependencies: benchmark_ublas_dep)
endif
