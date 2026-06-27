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

velox_add_library(
  velox_process
  ProcessBase.cpp
  StackTrace.cpp
  ThreadDebugInfo.cpp
  TraceContext.cpp
  TraceHistory.cpp
  HEADERS
  ProcessBase.h
  StackTrace.h
  ThreadDebugInfo.h
  ThreadLocalRegistry.h
  TraceContext.h
  TraceHistory.h
)

velox_link_libraries(
  velox_process
  PUBLIC velox_file velox_flag_definitions Folly::folly
  PRIVATE fmt::fmt gflags::gflags
)

# Profiler need not be part of the core Velox library
if(WIN32)
  add_library(velox_profiler STATIC Profiler.cpp)
else()
  add_library(velox_profiler OBJECT Profiler.cpp)
  velox_add_test_headers(velox_profiler Profiler.h)
endif()
target_link_libraries(
  velox_profiler
  PUBLIC velox_flag_definitions Folly::folly
  PRIVATE fmt::fmt gflags::gflags glog::glog
)

if(MSVC)
  velox_set_target_properties(velox_process PROPERTIES CXX_STANDARD 20)
  # Force local vcpkg includes to be used before global vcpkg
  velox_include_directories(velox_process BEFORE PRIVATE 
    "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include")
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
