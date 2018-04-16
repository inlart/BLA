include(ExternalProject)

set(Eigen3_VERSION "3.3.4")

ExternalProject_Add(Eigen3
  URL "http://bitbucket.org/eigen/eigen/get/${Eigen3_VERSION}.tar.gz"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(Eigen3 source_dir binary_dir)

set(EIGEN3_INCLUDE_DIR ${source_dir})
