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
if(NOT TARGET gflags::gflags)
  find_package(gflags CONFIG QUIET)
endif()

if(NOT TARGET gflags::gflags)
  find_package(gflags QUIET)
endif()

if(NOT TARGET gflags::gflags)
  foreach(
    _nimble_gflags_target
    gflags
    gflags_static
    gflags_nothreads_static
    gflags_shared
    gflags_nothreads_shared
  )
    if(TARGET ${_nimble_gflags_target})
      add_library(nimble_gflags INTERFACE)
      target_link_libraries(nimble_gflags INTERFACE ${_nimble_gflags_target})
      add_library(gflags::gflags ALIAS nimble_gflags)
      break()
    endif()
  endforeach()
endif()

add_executable(nimble_encoding_fuzzer_tests EncodingFuzzerTest.cpp)

target_compile_definitions(
  nimble_encoding_fuzzer_tests
  PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
)

target_link_libraries(
  nimble_encoding_fuzzer_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  gtest
  gtest_main
  gflags::gflags
  glog::glog
  Folly::folly
  velox_memory
)

add_test(
  nimble_encoding_fuzzer_tests
  nimble_encoding_fuzzer_tests
  --fuzzer_iterations=2
  --fuzzer_max_rows=128
  --nofuzzer_compression
)

add_executable(nimble_encoding_view_fuzzer_tests EncodingViewFuzzerTest.cpp)

target_link_libraries(
  nimble_encoding_view_fuzzer_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  gtest
  gtest_main
  gflags::gflags
  glog::glog
  Folly::folly
  velox_memory
)

add_test(
  nimble_encoding_view_fuzzer_tests
  nimble_encoding_view_fuzzer_tests
  --view_fuzzer_iterations=2
  --view_fuzzer_max_rows=128
)
