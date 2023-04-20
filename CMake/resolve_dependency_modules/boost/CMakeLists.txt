# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(Boost)
cmake_minimum_required(VERSION 3.14)

# Supress all warnings (in this subdir)
add_compile_options(-w)
set(FETCHCONTENT_QUIET OFF)
if(DEFINED ENV{VELOX_BOOST_URL})
  set(BOOST_SOURCE_URL "$ENV{VELOX_BOOST_URL}")
else()
  # We need to use boost > 1.70 to build it with CMake 1.81 was the first to be
  # released as a github release INCLUDING the cmake files (which are not in the
  # officale releases for some reason)
  set(VELOX_BOOST_BUILD_VERSION 1.81.0)
  string(
    CONCAT BOOST_SOURCE_URL
           "https://github.com/boostorg/boost/releases/download/"
           "boost-${VELOX_BOOST_BUILD_VERSION}/"
           "boost-${VELOX_BOOST_BUILD_VERSION}.tar.gz")
  set(VELOX_BOOST_BUILD_SHA256_CHECKSUM
      121da556b718fd7bd700b5f2e734f8004f1cfa78b7d30145471c526ba75a151c)
endif()

message(STATUS "Building boost from source")

# required for Boost::thread
if(NOT TARGET Threads::Threads)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
endif()

FetchContent_Declare(
  Boost
  URL ${BOOST_SOURCE_URL}
  URL_HASH SHA256=${VELOX_BOOST_BUILD_SHA256_CHECKSUM})

# header libs are not a valid component and fail find_package random is missing
# from follys components
list(
  APPEND
  BOOST_INCLUDE_LIBRARIES
  crc
  circular_buffer
  variant
  multi_index
  random
  uuid)
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(Boost)

configure_file(${CMAKE_CURRENT_LIST_DIR}/FindBoost.cmake.in
               ${CMAKE_CURRENT_LIST_DIR}/FindBoost.cmake @ONLY)
