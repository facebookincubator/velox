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

add_executable(velox_experimental_cxl_test CxlMemoryAllocatorTest.cpp CxlMemoryResourceTest.cpp)

add_test(NAME velox_experimental_cxl_test COMMAND velox_experimental_cxl_test)

# The test calls libnuma directly (numa_available, numa_max_node, ...), so it
# links NUMA_LIBRARY itself rather than relying on velox_experimental_cxl's
# private link. The external deps are listed explicitly (rather than left to
# propagate through the velox mono library) so the link does not reference
# bundled dependency archives by a path with no build rule.
target_link_libraries(
  velox_experimental_cxl_test
  PRIVATE
    velox_experimental_cxl
    velox_memory
    ${NUMA_LIBRARY}
    Folly::folly
    fmt::fmt
    gflags::gflags
    glog::glog
    GTest::gmock
    GTest::gtest
    GTest::gtest_main
)

# End-to-end seam test: runs a real aggregation query through the exec stack,
# so it pulls in the exec test harness and prestosql aggregate functions on top
# of the CXL operator library. CxlHashTableTest drives the table directly to
# cover the relocate-to-CXL primitive.
add_executable(
  velox_experimental_cxl_hash_aggregation_test CxlHashAggregationTest.cpp
                                               CxlHashTableTest.cpp
)

add_test(
  NAME velox_experimental_cxl_hash_aggregation_test
  COMMAND velox_experimental_cxl_hash_aggregation_test
)

target_link_libraries(
  velox_experimental_cxl_hash_aggregation_test
  PRIVATE
    velox_experimental_cxl_exec
    velox_experimental_cxl
    velox_aggregates
    velox_exec
    velox_exec_test_lib
    velox_functions_prestosql
    velox_hive_connector
    velox_memory
    velox_vector
    velox_type
    ${NUMA_LIBRARY}
    Boost::filesystem
    Boost::program_options
    Folly::folly
    fmt::fmt
    gflags::gflags
    glog::glog
    GTest::gmock
    GTest::gtest
)
