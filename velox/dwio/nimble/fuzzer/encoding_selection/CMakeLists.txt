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

# The fuzzing targets below are built but deliberately not registered with
# add_test. They keep the fuzzer sources compiling; deciding what to run in CI,
# and with which iteration counts, is a separate call.
add_library(
  nimble_writer_config_randomizer
  WriterConfigRandomizer.cpp
  WriterConfigRandomizer.h
)

target_link_libraries(
  nimble_writer_config_randomizer
  nimble_writer
  fmt::fmt
  Folly::folly
)

add_library(
  nimble_writer_fuzzer_runner
  NimbleWriterFuzzer.cpp
  NimbleWriterFuzzer.h
  NimbleWriterFuzzerRunner.cpp
  NimbleWriterFuzzerRunner.h
)

target_link_libraries(
  nimble_writer_fuzzer_runner
  nimble_common
  nimble_common_file_writer
  nimble_encodings
  nimble_index_test_utils
  nimble_random_encoding_selection_policy
  nimble_tablet_reader
  nimble_velox_schema_serialization
  nimble_batch_reader
  nimble_velox_selective
  nimble_writer
  nimble_writer_config_randomizer
  velox_memory
  velox_vector_fuzzer
  gflags::gflags
  glog::glog
  Folly::folly
)

add_executable(writer_config_randomizer_test WriterConfigRandomizerTest.cpp)

target_link_libraries(
  writer_config_randomizer_test
  nimble_random_encoding_selection_policy
  nimble_writer_config_randomizer
  nimble_writer
  gtest_main
  Folly::folly
)
add_test(writer_config_randomizer_test writer_config_randomizer_test)

# Each of these defines its own main() through folly::Init, so they link gtest
# without gtest_main.
add_executable(nimble_writer_fuzzer_test NimbleWriterFuzzerTest.cpp)

target_link_libraries(
  nimble_writer_fuzzer_test
  nimble_writer_fuzzer_runner
  velox_memory
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
)

add_executable(encoding_dispatch_consistency_test EncodingDispatchConsistencyTest.cpp)

target_link_libraries(
  encoding_dispatch_consistency_test
  nimble_writer_fuzzer_runner
  nimble_common
  velox_memory
  velox_vector_test_lib
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
)

add_executable(nimble_writer_fuzzer NimbleWriterFuzzerMain.cpp)

target_link_libraries(
  nimble_writer_fuzzer
  nimble_writer_fuzzer_runner
  velox_memory
  gflags::gflags
  glog::glog
  Folly::folly
)
