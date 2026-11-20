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

add_subdirectory(s3fs)
add_subdirectory(hdfs)
add_subdirectory(gcs)
add_subdirectory(abfs)

velox_add_library(
  velox_hive_storage_adapters_test_common
  INTERFACE
  HEADERS
  test_common/InsertTest.h
)

if(${VELOX_BUILD_TESTING})
  add_library(velox_hive_storage_adapters_server_startup test_common/ServerStartup.cpp)
  velox_add_test_headers(velox_hive_storage_adapters_server_startup test_common/ServerStartup.h)
  target_link_libraries(velox_hive_storage_adapters_server_startup velox_exception)

  add_executable(velox_hive_storage_adapters_server_startup_test test_common/ServerStartupTest.cpp)
  add_test(
    velox_hive_storage_adapters_server_startup_test
    velox_hive_storage_adapters_server_startup_test
  )
  target_link_libraries(
    velox_hive_storage_adapters_server_startup_test
    velox_hive_storage_adapters_server_startup
    GTest::gtest
    GTest::gtest_main
  )
endif()
