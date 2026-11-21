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

add_executable(velox_lance_connector_test LanceConnectorTest.cpp Main.cpp)

target_link_libraries(
  velox_lance_connector_test
  velox_lance_connector
  velox_dwio_common_test_utils
  velox_exec_test_lib
  velox_exec
  GTest::gtest
  GTest::gmock
)

add_test(NAME velox_lance_connector_test COMMAND velox_lance_connector_test)

# Locate liblance_c.so at run time without requiring LD_LIBRARY_PATH.
set_target_properties(velox_lance_connector_test PROPERTIES BUILD_RPATH "${LANCE_C_LIBRARY_DIR}")

# Make the .lance fixtures available next to the test binary.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
