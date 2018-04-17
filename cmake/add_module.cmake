macro(add_module_executable_folder folder extension prefix includes always_allscale dependencies)
    # -- Get all files
    file(GLOB_RECURSE sources "${folder}/*${extension}")

    # MESSAGE(STATUS "## Adding benchmark targets")
    foreach(file ${sources})
        get_filename_component(filename ${file} NAME)
        string(REPLACE "${extension}" "" filename ${filename})
        set(filename ${prefix}${filename})

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
    endforeach()
endmacro()
