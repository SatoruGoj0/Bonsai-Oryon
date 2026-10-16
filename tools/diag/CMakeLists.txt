# bonsai-diag, built against an existing Bonsai-Oryon build tree (shared, dynamically loaded backends):
#   cmake -S tools/diag -B build/diag -DLLAMA_SRC=build/src -DLLAMA_BUILD_DIR=build/release
cmake_minimum_required(VERSION 3.16)
project(bonsai_diag CXX)
set(LLAMA_SRC "" CACHE PATH "patched llama.cpp source tree")
set(LLAMA_BUILD_DIR "" CACHE PATH "build tree containing bin/libggml*.so")
find_package(Threads REQUIRED)
add_executable(bonsai-diag bonsai-diag.cpp)
target_compile_features(bonsai-diag PRIVATE cxx_std_17)
target_include_directories(bonsai-diag PRIVATE ${LLAMA_SRC}/ggml/include)
target_link_libraries(bonsai-diag PRIVATE ${LLAMA_BUILD_DIR}/bin/libggml.so ${LLAMA_BUILD_DIR}/bin/libggml-base.so Threads::Threads)
set_target_properties(bonsai-diag PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")
