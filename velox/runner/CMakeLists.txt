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

add_subdirectory(if)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

find_package(axiom CONFIG REQUIRED)

# LocalRunnerService library
velox_add_library(
  velox_local_runner_service_lib
  LocalRunnerService.h
)

velox_link_libraries(
  velox_local_runner_service_lib
  PUBLIC
    velox_expression
    velox_functions_prestosql
    velox_local_runner_service_thrift
    axiom_runner_local_runner
    Folly::folly
    gflags::gflags
    glog::glog
)

# LocalRunnerService binary
add_executable(
  velox_local_runner_service
  LocalRunnerService.cpp
)

target_link_libraries(
  velox_local_runner_service
  PRIVATE
    velox_local_runner_service_lib
    velox_common_file
    velox_hive_connector
    velox_expression
    velox_functions_prestosql
    velox_local_runner_service_thrift
    axiom_runner_local_runner
    Folly::folly
    gflags::gflags
    glog::glog
)
