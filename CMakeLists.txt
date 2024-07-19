# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(DeepLearningC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")

option(USE_CUDA "Use CUDA for GPU acceleration" OFF)

include_directories(include)

file(GLOB SOURCES "src/*.c")

add_library(deeplearningc STATIC ${SOURCES})

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(deeplearningc PUBLIC USE_CUDA)
    target_include_directories(deeplearningc PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(deeplearningc PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

# Examples
add_executable(mnist_example examples/mnist_classification.c)
target_link_libraries(mnist_example deeplearningc m)

# Tests
enable_testing()
add_executable(test_tensor tests/test_tensor.c)
target_link_libraries(test_tensor deeplearningc)
add_test(NAME TensorTests COMMAND test_tensor)

# Installation
install(TARGETS deeplearningc DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)