cmake_minimum_required(VERSION 3.15)
project(lib-bmm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

find_package(BLAS REQUIRED)

set(SOURCE_FILES src/bmm.cpp src/kernels.cpp src/wrapper.cpp)
set(HEADER_FILES include/bmm.h include/aligned_allocator.h include/kernels.h include/wrapper.h)

add_library(bmm SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(bmm PUBLIC include)
target_link_libraries(bmm PUBLIC ${BLAS_LIBRARIES})
install(TARGETS bmm DESTINATION fast_einsum)

if(NOT SKBUILD)
    # Not needed for the Python wheel build; skip to save build time.
    add_executable(performance src/performance.cpp)
    target_include_directories(performance PUBLIC include)
    target_link_libraries(performance PUBLIC bmm)

    add_subdirectory(tests)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(py_bmm src/pybind11_wrapper.cpp)
target_link_libraries(py_bmm PRIVATE bmm)
set_target_properties(py_bmm PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS py_bmm DESTINATION fast_einsum)

# add OMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bmm PUBLIC OpenMP::OpenMP_CXX)
endif()


