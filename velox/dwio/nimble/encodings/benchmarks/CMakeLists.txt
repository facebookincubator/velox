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
add_executable(nimble_fixed_bit_width_benchmark FixedBitWidthBenchmark.cpp)

target_link_libraries(
  nimble_fixed_bit_width_benchmark
  nimble_encodings_tests_utils
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_memory
)

add_executable(
  nimble_block_bit_packing_encoding_benchmark
  BlockBitPackingEncodingBenchmark.cpp
)

target_link_libraries(
  nimble_block_bit_packing_encoding_benchmark
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_memory
)

add_executable(
  nimble_exact_bit_width_storage_benchmark
  ExactBitWidthStorageBenchmark.cpp
)

target_link_libraries(
  nimble_exact_bit_width_storage_benchmark
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  velox_memory
)

add_executable(nimble_fsst_benchmark FsstBenchmark.cpp)

target_link_libraries(
  nimble_fsst_benchmark
  nimble_encodings
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_memory
  fsst
)

add_executable(nimble_huffman_benchmark HuffmanBenchmark.cpp)

target_link_libraries(
  nimble_huffman_benchmark
  nimble_encodings
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_memory
)

add_executable(nimble_encoding_view_benchmark EncodingViewBenchmark.cpp)

target_link_libraries(
  nimble_encoding_view_benchmark
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_common_base
  velox_memory
  fmt::fmt
)

add_executable(nimble_simd_for_bitpack_benchmark SimdForBitpackBenchmark.cpp)

target_link_libraries(
  nimble_simd_for_bitpack_benchmark
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  Folly::follybenchmark
  Folly::folly
  velox_common_base
  velox_memory
)
