d_gmp = dependency('gmpxx', required: true)

e_complex = executable('example_complex', 'complex.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc])

e_expressions = executable('example_expressions', 'expressions.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc])

e_functions = executable('example_functions', 'functions.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc])

if d_gmp.found()
    e_gmp = executable('example_gmp', 'gmp.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc, d_gmp])
endif

e_matrix_multiplication = executable('example_matrix_multiplication', 'matrix_multiplication.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc])

e_pagerank = executable('example_pagerank', 'pagerank.cc', include_directories: i_bla, dependencies: [d_thread, d_allscale, d_blas, d_vc])
