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

# These drivers link Folly::follybenchmark and are Velox benchmark targets, not
# part of the ml_id_compression suite. The directory is entered under
# NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS as well (see ../CMakeLists.txt), so they
# need the benchmark switch guarded explicitly rather than relying on the parent.
if(VELOX_ENABLE_BENCHMARKS)
  add_executable(
    nimble_fixed_bit_width_benchmark
    FixedBitWidthBenchmark.cpp
    BenchmarkUtils.h
    DeltaBenchmarkData.h
    FixedBitWidthBenchmarkData.h
    NimbleEncodingRunner.h
    SparseBoolBenchmarkData.h
    VarintBenchmarkData.h
  )

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
    BlockBitPackingBenchmarkData.h
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

  add_executable(nimble_exact_bit_width_storage_benchmark ExactBitWidthStorageBenchmark.cpp)

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

  add_executable(nimble_huffman_benchmark HuffmanBenchmark.cpp HuffmanBenchmarkData.h)

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

  add_executable(
    nimble_simd_for_bitpack_benchmark
    SimdForBitpackBenchmark.cpp
    SimdForBitpackBenchmarkData.h
  )

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

  add_executable(
    nimble_pfor_encoding_benchmark
    PFOREncodingBenchmark.cpp
    PFOREncodingBenchmarkData.h
  )

  target_link_libraries(
    nimble_pfor_encoding_benchmark
    nimble_encodings
    nimble_common
    Folly::follybenchmark
    Folly::folly
    velox_memory
  )

  add_executable(
    nimble_fsst_encoding_benchmark
    FsstEncodingBenchmark.cpp
    FsstBenchmarkData.h
    StringBenchmarkCorpus.h
  )

  target_link_libraries(
    nimble_fsst_encoding_benchmark
    nimble_encodings
    nimble_common
    Folly::follybenchmark
    Folly::folly
    velox_memory
    fsst
    fmt::fmt
  )

  add_executable(
    nimble_prefix_encoding_benchmark
    PrefixEncodingBenchmark.cpp
    PrefixBenchmarkData.h
    StringBenchmarkCorpus.h
  )

  target_link_libraries(
    nimble_prefix_encoding_benchmark
    nimble_encodings
    nimble_common
    Folly::follybenchmark
    Folly::folly
    velox_memory
  )

  # A/B benchmark: count-based vs size-based (exponent, factor) selection
  # for ALP encoding. Not a folly-benchmark; runs once and prints a comparison
  # table.
  add_executable(nimble_alp_selection_ab_benchmark ALPSelectionABBenchmark.cpp)

  target_link_libraries(
    nimble_alp_selection_ab_benchmark
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    velox_memory
    Folly::folly
  )
endif()

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_subdirectory(ml_id_compression)
endif()
