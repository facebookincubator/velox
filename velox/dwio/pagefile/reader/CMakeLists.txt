

# Specify the installation prefix of Arrow, pointing to where the ArrowConfig.cmake resides
set(CMAKE_PREFIX_PATH "/path/to/arrow/install" ${CMAKE_PREFIX_PATH})

# Find Arrow package using ArrowConfig.cmake
find_package(Arrow REQUIRED CONFIG)

# Optionally, print the version of Arrow
message(STATUS "The value of Arrow version is: ${ARROW_VERSION}")

velox_add_library(
        velox_dwio_pagefile_native_reader
       PageFileReader.cpp)

velox_link_libraries(
        velox_dwio_pagefile_native_reader
        velox_dwio_common
        velox_caching
        velox_test_util
        Arrow::arrow_shared
        velox_dwio_common
        velox_arrow_bridge
        fmt::fmt)
