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
if(VELOX_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
add_subdirectory(legacy)
if(NIMBLE_BUILD_TESTING)
  add_subdirectory(selection/tests)
endif()

add_library(
  nimble_encodings
  ConstantEncoding.cpp
  EncodingSliceFactory.cpp
  FsstEncoding.cpp
  HuffmanEncoding.h
  MainlyConstantEncoding.cpp
  PrefixEncoding.cpp
  RLEEncoding.cpp
  SharedDictionaryTypes.cpp
  SparseBoolEncoding.cpp
  TrivialEncoding.cpp
  common/Encoding.cpp
  common/EncodingFactory.cpp
  common/EncodingLayout.cpp
  common/EncodingTypeDispatch.h
  selection/EncodingSelectionPolicy.cpp
  selection/Statistics.cpp
  views/DeltaBlockEncodingView.h
  views/EncodingViewFactory.cpp
  views/HuffmanEncodingView.h
)

# FSST is vendored at velox/external/fsst and added by velox/CMakeLists.txt
# before this directory, so the target always exists here. Nimble renames FSST's
# exported symbols to avoid clashing with any other copy linked into the same
# binary; PRIVATE keeps the renames on FSST's own translation units.
target_compile_definitions(
  fsst
  PRIVATE
    fsst_create=nimble_fsst_create
    fsst_duplicate=nimble_fsst_duplicate
    fsst_export=nimble_fsst_export
    fsst_destroy=nimble_fsst_destroy
    fsst_import=nimble_fsst_import
    fsst_decoder=nimble_fsst_decoder
    fsst_compress=nimble_fsst_compress
    fsst_decompress=nimble_fsst_decompress
)

target_link_libraries(
  nimble_encodings
  nimble_compression
  nimble_common
  velox_common_base
  velox_fastpforlib
  fsst
  Folly::folly
  absl::flat_hash_map
  protobuf::libprotobuf
)
