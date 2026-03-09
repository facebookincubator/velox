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

add_executable(simple_lru_cache_test SimpleLRUCacheTest.cpp)
add_test(simple_lru_cache_test simple_lru_cache_test)
target_link_libraries(
  simple_lru_cache_test
  PRIVATE velox_common_base Folly::folly velox_time glog::glog GTest::gtest GTest::gtest_main
)

# Split velox_cache_test into individual test binaries for parallel execution.
set(
  VELOX_CACHE_TEST_SOURCES
  AsyncDataCacheTest.cpp
  CacheTTLControllerTest.cpp
  SsdFileTest.cpp
  SsdFileTrackerTest.cpp
  StringIdMapTest.cpp
)

set(
  VELOX_CACHE_TEST_DEPS
  velox_caching
  velox_file
  velox_file_test_utils
  velox_memory
  velox_test_util
  velox_flag_definitions
  Folly::folly
  glog::glog
  GTest::gtest
  GTest::gtest_main
)

foreach(TEST_SOURCE IN LISTS VELOX_CACHE_TEST_SOURCES)
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
  set(TEST_TARGET "velox_cache_${TEST_NAME}")
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  add_test(${TEST_TARGET} ${TEST_TARGET})
  target_link_libraries(${TEST_TARGET} PRIVATE ${VELOX_CACHE_TEST_DEPS})
endforeach()

add_executable(cached_factory_test CachedFactoryTest.cpp)
add_test(cached_factory_test cached_factory_test)
target_link_libraries(
  cached_factory_test
  PRIVATE velox_common_base Folly::folly velox_time glog::glog GTest::gtest GTest::gtest_main
)
