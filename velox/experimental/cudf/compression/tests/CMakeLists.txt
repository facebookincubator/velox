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

add_executable(
  cudf_compression_test
  CompressionTest.cpp
)

add_test(
  NAME cudf_compression_test
  COMMAND cudf_compression_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(cudf_compression_test PROPERTIES LABELS cuda_driver TIMEOUT 3000)

target_link_libraries(
  cudf_compression_test
  velox_cudf_compression
  gtest
  gtest_main
)
