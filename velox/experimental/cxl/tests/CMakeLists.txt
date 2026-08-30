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
# links Numa::numa itself rather than relying on velox_experimental_cxl's link.
# The external deps are listed explicitly (rather than left to propagate through
# the velox mono library) so the link does not reference bundled dependency
# archives by a path with no build rule.
target_link_libraries(
  velox_experimental_cxl_test
  PRIVATE
    velox_experimental_cxl
    velox_memory
    Numa::numa
    Folly::folly
    fmt::fmt
    gflags::gflags
    glog::glog
    GTest::gmock
    GTest::gtest
    GTest::gtest_main
)
