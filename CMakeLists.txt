cmake_minimum_required(VERSION 3.20)
project(winnex_nano VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Reuse the winnex-madhava engine WITHOUT duplicating code.
# winnex-nano compiles the Madhava sources directly (shared kernels: QK^T
# matmul, top-k, selective-top-K attention, quantization, AVX2/OpenMP/OpenCL).
# The Madhava sources are referenced (not copied) — same files, one build.
# ============================================================================
find_package(OpenMP REQUIRED)

if(NOT DEFINED MADHAVA_ROOT)
    set(MADHAVA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../winnex-madhava")
endif()

# If the winnex-madhava sources are not present locally (e.g. building from an
# sdist), fetch them from the GitHub repo (the dependency is winnex-madhava on
# PyPI; its C++ sources live in winnex-ai/winnex-madhava).
if(NOT EXISTS "${MADHAVA_ROOT}/CMakeLists.txt")
    message(STATUS "winnex-nano: winnex-madhava sources not found locally — fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        winnex_madhava_src
        GIT_REPOSITORY https://github.com/winnex-ai/winnex-madhava.git
        GIT_TAG v1.8.1
    )
    FetchContent_Populate(winnex_madhava_src)
    set(MADHAVA_ROOT "${winnex_madhava_src_SOURCE_DIR}")
endif()
message(STATUS "winnex-nano: reusing winnex-madhava at ${MADHAVA_ROOT}")

set(MADHAVA_SRC
    ${MADHAVA_ROOT}/src/winnex_madhava.cpp
    ${MADHAVA_ROOT}/src/speed_cpu.cpp
    ${MADHAVA_ROOT}/src/speed_opencl.cpp
)
add_library(madhava_reused STATIC ${MADHAVA_SRC})
target_include_directories(madhava_reused PUBLIC ${MADHAVA_ROOT}/include)
target_include_directories(madhava_reused PRIVATE ${MADHAVA_ROOT}/vendor)
target_compile_definitions(madhava_reused PRIVATE MADHAVA_HAS_OPENCL)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    target_compile_options(madhava_reused PRIVATE -O3 -mavx2 -mfma -funroll-loops)
endif()
target_link_libraries(madhava_reused PUBLIC OpenMP::OpenMP_CXX)

# ============================================================================
# winnex-nano core (Winnex spectral layer + weight balancer + tensor engine)
# ============================================================================
add_library(winnex_nano
    src/spectral_tokenizer.cpp
    src/weight_balancer.cpp
    src/tensor_ops.cpp
    src/stream_engine.cpp
)
target_include_directories(winnex_nano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(winnex_nano PUBLIC madhava_reused)

# --- Python bindings (pybind11) --------------------------------------------
# scikit-build-core sets up Python3_ROOT_DIR / Python_EXECUTABLE in the build
# environment; pybind11's pybind11_add_module resolves the interpreter without
# us requiring Development/Development.Embed (which manylinux lacks).
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_winnex_nano bindings/py_module.cpp)
target_link_libraries(_winnex_nano PRIVATE winnex_nano)
install(TARGETS _winnex_nano DESTINATION ${SKBUILD_PLATLIB_DIR})

# --- Tests -----------------------------------------------------------------
enable_testing()
add_executable(test_spectral_tokenizer tests/test_spectral_tokenizer.cpp)
target_link_libraries(test_spectral_tokenizer PRIVATE winnex_nano)
add_test(NAME spectral_tokenizer COMMAND test_spectral_tokenizer)

add_executable(test_stream tests/test_stream.cpp)
target_link_libraries(test_stream PRIVATE winnex_nano)
add_test(NAME stream COMMAND test_stream)
