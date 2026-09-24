# Copyright (c) Meta Platforms, Inc. and affiliates.
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

add_executable(nimble_ml_id_access_structure_test AccessStructureTest.cpp)

add_test(nimble_ml_id_access_structure_test nimble_ml_id_access_structure_test)

target_link_libraries(
  nimble_ml_id_access_structure_test
  # For the gflag definitions the benchmark headers declare.
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  nimble_tools_common
  openzl_cpp
  velox_common_base
  gmock
  gtest
  gflags
  fmt::fmt
  Folly::folly
)

add_executable(nimble_ml_id_block_codec_test BlockCodecTargetTest.cpp)

add_test(nimble_ml_id_block_codec_test nimble_ml_id_block_codec_test)

target_link_libraries(
  nimble_ml_id_block_codec_test
  # For the gflag definitions the benchmark headers declare.
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  nimble_tools_common
  openzl_cpp
  velox_common_base
  gmock
  gtest
  gflags
  fmt::fmt
  Folly::folly
)

add_executable(nimble_ml_id_block_lazy_test BlockLazyTargetTest.cpp)

add_test(nimble_ml_id_block_lazy_test nimble_ml_id_block_lazy_test)

target_link_libraries(
  nimble_ml_id_block_lazy_test
  # For the gflag definitions the benchmark headers declare.
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  nimble_tools_common
  openzl_cpp
  velox_common_base
  gmock
  gtest
  gflags
  fmt::fmt
  Folly::folly
)
