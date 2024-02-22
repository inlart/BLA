d_gmp = dependency('gmpxx', required: true)
example_dep = [thread_dep, allscale_dep, blas_dep, vc_dep]

e_complex = executable('example_complex', 'complex.cc', include_directories: bla_inc, dependencies: example_dep)
e_expressions = executable('example_expressions', 'expressions.cc', include_directories: bla_inc, dependencies: example_dep)
e_functions = executable('example_functions', 'functions.cc', include_directories: bla_inc, dependencies: example_dep)
e_matrix_multiplication = executable('example_matrix_multiplication', 'matrix_multiplication.cc', include_directories: bla_inc, dependencies: example_dep)
e_pagerank = executable('example_pagerank', 'pagerank.cc', include_directories: bla_inc, dependencies: example_dep)

if d_gmp.found()
    e_gmp = executable('example_gmp', 'gmp.cc', include_directories: bla_inc, dependencies: example_dep + d_gmp)
endif