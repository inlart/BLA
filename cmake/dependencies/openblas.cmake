if(MSVC)
    message(FATAL_ERROR "OpenBLAS can't be installed using MSVC (yet).")
else()
    ExternalProject_Add(
        OpenBLAS
        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
        GIT_TAG "release-0.3.0"
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} NO_SHARED=1 NOFORTRAN=1 USE_THREAD=0 -C <SOURCE_DIR> libs
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} NO_SHARED=1 NOFORTRAN=1 USE_THREAD=0 -C <SOURCE_DIR> install PREFIX=<INSTALL_DIR> libs
        TEST_COMMAND ""
    )

    ExternalProject_Get_Property(OpenBLAS source_dir install_dir)

    set(ALLSCALE_MATRIX_DEPENDENCIES OpenBLAS ${ALLSCALE_MATRIX_DEPENDENCIES})

    message(STATUS "install dir: ${install_dir}")

    set(OpenBLAS_INCLUDE_DIRS ${install_dir}/include)
    set(OpenBLAS_LIBRARIES ${install_dir}/lib/libopenblas.a)
endif()
