project('BLA', 'cpp', version: '1.0.0-dev', default_options : ['cpp_std=c++14', 'b_ndebug=if-release'])

add_project_arguments('-march=native', language: 'cpp')

thread_dep = dependency('threads')
vc_dep = dependency('Vc', required: true)
eigen_dep = dependency('eigen3', required: false)
blas_dep = dependency(get_option('cblas-library'), required: true)

# include dir
bla_inc = include_directories('include')

subdir('include')

subdir('thirdparty')

if get_option('enable-examples')
    subdir('examples')
endif
if get_option('enable-tests')
    subdir('tests')
endif
if get_option('enable-benchmarks')
    subdir('benchmarks')
endif
