cmake_minimum_required(VERSION 3.16)

project(attention C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED "Build shared library" ON)

set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -Wextra")

set(BLIS_ROOT "${CMAKE_SOURCE_DIR}/../blis")

include_directories(
    include
    ${BLIS_ROOT}/include/haswell
)

link_directories(
    ${BLIS_ROOT}/lib
)

# Source files
set(SOURCES
    src/kernel/attention.c
    src/kernel/softmax.c
)

if(BUILD_SHARED)
    add_library(attention SHARED ${SOURCES})

    target_link_libraries(attention
        PRIVATE
        blis
        m
    )
endif()

