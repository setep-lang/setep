cmake_minimum_required(VERSION 3.16)
project(SETEP VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Front end (generated parser, syntax tree, semantic IR) and the backends.
add_library(setep_core STATIC
    src/JimJamLexer.cpp
    src/SetepParser.cpp
    src/SetepSyntax.cpp
    src/SetepSemantic.cpp
    src/SetepFrontend.cpp
    src/SetepPython.cpp
    src/SetepGraph.cpp
    src/SetepDistribution.cpp
    src/SetepChakra.cpp
    src/SetepOnnx.cpp
)
target_include_directories(setep_core PUBLIC src)
target_compile_definitions(setep_core PUBLIC SETEP_PYTHON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/python")
if(MSVC)
    target_compile_options(setep_core PRIVATE /W3 /permissive- /Zc:__cplusplus)
endif()

add_executable(setep src/main.cpp)
target_link_libraries(setep PRIVATE setep_core)

# Tests: ctest --test-dir build. The Python tests need an interpreter; the
# PyTorch and ONNX ones are skipped (exit 77) when torch / onnx are missing.
enable_testing()
add_executable(setep_graph_test tests/GraphTest.cpp)
target_link_libraries(setep_graph_test PRIVATE setep_core)
add_test(NAME graph COMMAND setep_graph_test ${CMAKE_CURRENT_SOURCE_DIR})

set(SETEP_TEST_PYTHON "" CACHE FILEPATH "Python interpreter for the tests (torch and onnx enable more of them)")
if(NOT SETEP_TEST_PYTHON)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        set(SETEP_TEST_PYTHON "${Python3_EXECUTABLE}")
    endif()
endif()
if(SETEP_TEST_PYTHON)
    set(SETEP_TEST_WORK "${CMAKE_CURRENT_BINARY_DIR}/test-work")
    foreach(TEST_NAME errors python_models python_numerical chakra onnx)
        add_test(NAME ${TEST_NAME} COMMAND "${SETEP_TEST_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/${TEST_NAME}_test.py"
            --setep $<TARGET_FILE:setep> --work "${SETEP_TEST_WORK}/${TEST_NAME}")
        set_tests_properties(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
    endforeach()
    set_tests_properties(onnx PROPERTIES ENVIRONMENT "SETEP_TEST_INTERPRETER=${SETEP_TEST_PYTHON}")
else()
    message(STATUS "No Python interpreter: only the C++ graph test is registered")
endif()
