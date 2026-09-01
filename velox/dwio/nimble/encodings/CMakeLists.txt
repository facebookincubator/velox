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
# The ml_id_compression drivers under benchmarks/ are gated on the experimental
# encodings they exercise, not on Velox's benchmark switch, so this directory is
# entered for either. benchmarks/CMakeLists.txt guards its Folly-benchmark
# targets on VELOX_ENABLE_BENCHMARKS separately.
if(VELOX_ENABLE_BENCHMARKS OR NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
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
  SharedDictionaryCatalog.cpp
  SharedDictionaryCatalog.h
  SharedDictionaryEncoding.cpp
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
  views/TrivialEncodingView.h
  views/SparseBoolEncodingView.h
  views/SimdForBitpackEncodingView.h
  views/RLEEncodingView.h
  views/PFOREncodingView.h
  views/MainlyConstantEncodingView.h
  views/FixedBitWidthEncodingView.h
  views/FOREncodingView.h
  views/EncodingViewFactory.h
  views/EncodingView.h
  views/DictionaryEncodingView.h
  views/ConstantEncodingView.h
  views/BlockBitPackingEncodingView.h
  views/ALPEncodingView.h
  selection/Statistics.h
  selection/NestedAlpSizeEstimation.h
  selection/EncodingSizeEstimation.h
  selection/EncodingSelectionPolicy.h
  selection/EncodingSelection.h
  selection/EncodingIdentifier.h
  common/EncodingUtils.h
  common/EncodingType.h
  common/EncodingPrimitives.h
  common/EncodingPrefix.h
  common/EncodingLayout.h
  common/EncodingFactory.h
  common/Encoding.h
  common/BufferedEncoding.h
  VarintEncoding.h
  TrivialEncoding.h
  SubIntSplitSelector.h
  SubIntSplitSampler.h
  SubIntSplitMetrics.h
  SubIntSplitEncoding.h
  SubIntSplitCostModels.h
  SubIntSplitConfig.h
  SparseBoolEncoding.h
  SimdForBitpackEncoding.h
  SharedDictionaryEncoding.h
  SharedDictionaryBuilder.h
  SentinelEncoding.h
  RLEEncoding.h
  PrefixEncoding.h
  PFOREncoding.h
  NullableEncoding.h
  MainlyConstantEncoding.h
  MainlyConstantV2Encoding.h
  FsstEncoding.h
  FrequencyPartitionEncoding.h
  ForEncoding.h
  FixedBitWidthEncoding.h
  EncodingSliceFactory.h
  DictionaryEncoding.h
  DeltaEncoding.h
  DeltaBlockEncoding.h
  ConstantEncoding.h
  BlockBitPackingEncoding.h
  ALPEncoding.h
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

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/SharedDictionary.fbs"
  ""
  nimble_shared_dictionary_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_shared_dictionary_fb INTERFACE)
target_include_directories(
  nimble_shared_dictionary_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_shared_dictionary_fb nimble_shared_dictionary_schema_fb)

target_link_libraries(
  nimble_encodings
  nimble_compression
  nimble_common
  nimble_shared_dictionary_fb
  velox_common_base
  velox_fastpforlib
  fsst
  Folly::folly
  absl::flat_hash_map
  protobuf::libprotobuf
)
