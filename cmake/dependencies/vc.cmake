include(ExternalProject)

set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/install/Vc)
file(MAKE_DIRECTORY ${install_dir})

ExternalProject_Add(
  VectorClasses
  GIT_REPOSITORY "https://github.com/VcDevel/Vc.git"
  GIT_TAG "1.3"
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CMAKE_ARGS ${CMAKE_EXTERNALPROJECT_FORWARDS} "-DBUILD_TESTING=OFF" "-DCMAKE_INSTALL_PREFIX=${install_dir}"
  TEST_COMMAND ""
)

ExternalProject_Get_Property(VectorClasses source_dir binary_dir)

# workaround https://itk.org/Bug/view.php?id=15052
file(MAKE_DIRECTORY ${source_dir}/include)

add_library(Vc STATIC IMPORTED)

set(_prefix ${CMAKE_STATIC_LIBRARY_PREFIX})
set(_suffix ${CMAKE_STATIC_LIBRARY_SUFFIX})

set(Vc ${_prefix}Vc${_suffix})
set_target_properties(Vc PROPERTIES
  IMPORTED_LOCATION ${binary_dir}/${Vc}

  # attach include path
  INTERFACE_INCLUDE_DIRECTORIES ${source_dir}/include
)
add_dependencies(Vc VectorClasses)

set(Vc_LIBRARIES ${install_dir}/lib)
set(Vc_INCLUDE_DIR ${install_dir}/include)

set(ALLSCALE_MATRIX_DEPENDENCIES Vc ${ALLSCALE_MATRIX_DEPENDENCIES})
