# Standalone build of the Bonsai-Oryon kernel tests against the (patched) upstream ggml sources.
#   cmake -S tests/kernels -B build-ktest -DLLAMA_SRC=<patched llama.cpp> -DKTEST_ARCH=armv8.2-a+dotprod
cmake_minimum_required(VERSION 3.16)
project(bonsai_kernel_tests C CXX)

set(LLAMA_SRC "" CACHE PATH "patched llama.cpp source tree")
set(KTEST_ARCH "" CACHE STRING "-march value for this test build (empty = toolchain default)")
if (NOT LLAMA_SRC)
    message(FATAL_ERROR "set -DLLAMA_SRC")
endif()

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
if (KTEST_ARCH)
    set(GGML_CPU_ARM_ARCH ${KTEST_ARCH} CACHE STRING "" FORCE)
endif()
add_subdirectory(${LLAMA_SRC}/ggml ggml)

add_executable(test-pq2-kernels test-pq2-kernels.cpp)
target_compile_features(test-pq2-kernels PRIVATE cxx_std_17)
target_include_directories(test-pq2-kernels PRIVATE ${LLAMA_SRC}/ggml/src)
if (KTEST_ARCH)
    target_compile_options(test-pq2-kernels PRIVATE -march=${KTEST_ARCH})
endif()
target_link_libraries(test-pq2-kernels PRIVATE ggml ggml-cpu ggml-base)

# upstream ggml-level tests that cannot be built in the dynamic-backend (packaged) configuration
foreach (t test-quantize-fns test-pq2-row-shapes)
    add_executable(${t} ${LLAMA_SRC}/tests/${t}.cpp)
    target_compile_features(${t} PRIVATE cxx_std_17)
    target_link_libraries(${t} PRIVATE ggml ggml-cpu ggml-base)
endforeach()
