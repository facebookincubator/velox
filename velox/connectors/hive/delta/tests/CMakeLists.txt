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

if(NOT VELOX_DISABLE_GOOGLETEST)
  add_executable(
    velox_hive_delta_test
    HiveDeltaSplitTest.cpp
    PartitionValueConversionTest.cpp
  )

  add_test(velox_hive_delta_test velox_hive_delta_test)

  target_link_libraries(
    velox_hive_delta_test
    velox_hive_delta_splitreader
    velox_hive_connector
    velox_connector_registry
    velox_dwio_common
    velox_type
    velox_vector
    velox_memory
    velox_core
    GTest::gtest
    GTest::gtest_main
    ${TEST_LINK_LIBS}
  )
endif()