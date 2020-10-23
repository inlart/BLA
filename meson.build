project('BLA', 'cpp', default_options : ['cpp_std=c++14'])

i_bla = include_directories('include')


d_thread = dependency('threads')

d_allscale = dependency('allscale-api', required: true)

d_vc = dependency('Vc', required: true)

d_eigen = dependency('eigen3', required: false)

d_blas = dependency('openblas', required: true)

subdir('include')

# if get_option('enable-examples').enabled()
subdir('examples')
# endif
# if get_option('enable-tests').enabled()
subdir('tests')
# endif
# if get_option('enable-benchmarks').enabled()
subdir('benchmarks')
# endif
