macro(add_module_executable_folder folder extension prefix postfix includes always_allscale dependencies)
    # -- Get all files
    file(GLOB_RECURSE sources "${folder}/*${extension}")

    # MESSAGE(STATUS "## Adding benchmark targets")
    foreach(file ${sources})
        get_filename_component(filename ${file} NAME)
        string(REPLACE "${extension}" "" filename ${filename})
        set(filename ${prefix}${filename}${postfix})


        set(uses_gmp FALSE)
        if(${filename} MATCHES "(.*)gmp(.*)")
            set(uses_gmp TRUE)
        endif()

        if((NOT uses_gmp)  OR (GMP_FOUND AND GMPXX_FOUND))
            # -- Add Executable
            add_executable(${filename} ${file})

            # -- Dependencies
            add_dependencies(${filename} ${dependencies})

            # -- Default Includes
            target_include_directories(${filename} PUBLIC ${includes})

            # -- AllScale Definitions
            if((${filename} MATCHES "(.*)allscale(.*)") OR ${always_allscale})
                target_compile_definitions(${filename} PRIVATE EIGEN_DONT_PARALLELIZE=1)
                target_include_directories(${filename} PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

                target_link_libraries(${filename} ${CMAKE_THREAD_LIBS_INIT})
                target_link_libraries(${filename} ${Vc_LIBRARIES})
                target_link_libraries(${filename} ${OpenBLAS_LIBRARIES})
            endif()

            # -- GMP
            if(uses_gmp)
                target_include_directories(${filename} PUBLIC ${GMPXX_INCLUDE_DIR})
                target_include_directories(${filename} PUBLIC ${GMP_INCLUDE_DIR})

                target_link_libraries(${filename} ${GMPXX_LIBRARIES})
                target_link_libraries(${filename} ${GMP_LIBRARIES})
            endif()
        else()
            message(WARNING "${filename} uses gmp but gmp was not found")
        endif()
    endforeach()
endmacro()

macro(add_test_folder folder extension prefix postfix includes always_allscale dependencies)
    # -- Get all files
    file(GLOB_RECURSE sources "${folder}/*${extension}")

    # MESSAGE(STATUS "## Adding benchmark targets")
    foreach(file ${sources})
        get_filename_component(filename ${file} NAME)
        string(REPLACE "${extension}" "" filename ${filename})
        set(filename ${prefix}${filename}${postfix})

        # -- Add Executable
        add_executable(${filename} ${file})

        # -- Dependencies
        add_dependencies(${filename} ${dependencies})

        # -- Default Includes
        target_include_directories(${filename} PUBLIC ${includes})

        # -- AllScale Definitions
        if((${filename} MATCHES "(.*)allscale(.*)") OR ${always_allscale})
            target_compile_definitions(${filename} PRIVATE EIGEN_DONT_PARALLELIZE=1)
            target_include_directories(${filename} PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

            target_link_libraries(${filename} ${CMAKE_THREAD_LIBS_INIT})
            target_link_libraries(${filename} ${Vc_LIBRARIES})
            target_link_libraries(${filename} ${OpenBLAS_LIBRARIES})
        endif()

        target_link_libraries(${filename} ${CMAKE_THREAD_LIBS_INIT})
        target_link_libraries(${filename} ${GTEST_LIBRARIES})
        target_link_libraries(${filename} ${GTEST_MAIN_LIBRARIES})

        add_test(NAME ${filename} COMMAND ${filename})
    endforeach()
endmacro()
