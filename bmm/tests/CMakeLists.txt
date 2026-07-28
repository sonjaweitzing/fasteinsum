cmake_minimum_required(VERSION 3.10)
project(catch_tests_bmm LANGUAGES CXX)

find_package(BLAS REQUIRED)

#set standards compiler independent
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#create an executable with the test cases
add_executable(catch_tests_bmm catch_main.cpp test_bmm_trivial.cpp test_bmm.cpp)

# link libraries bmm and catch to the executable
target_link_libraries(catch_tests_bmm PUBLIC bmm catch)

# executable with bmm opt tests
add_executable(catch_tests_bmm_opt catch_main.cpp)

# link libraries bmm and catch to the executable
target_link_libraries(catch_tests_bmm_opt PUBLIC bmm catch)

# add the test cases to the test suite
add_test(NAME CATCH_BMM_TESTS COMMAND catch_tests_bmm) # COMMAND is the executable to run
add_test(NAME CATCH_BMM_OPT_TESTS COMMAND catch_tests_bmm_opt) # COMMAND is the executable to run
