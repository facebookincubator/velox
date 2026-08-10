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
add_subdirectory(benchmarks)
add_subdirectory(tests)
add_subdirectory(legacy)
add_subdirectory(selection/tests)

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

set(FSST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/fsst")
if(NOT TARGET fsst AND EXISTS "${FSST_SOURCE_DIR}/CMakeLists.txt")
  add_subdirectory("${FSST_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/fsst")
endif()
if(NOT TARGET fsst)
  include(FetchContent)
  FetchContent_Declare(
    fsst
    GIT_REPOSITORY https://github.com/cwida/fsst.git
    GIT_TAG 50dd308befd5121bd1121428a833f42cce6e1f9d
  )
  FetchContent_MakeAvailable(fsst)
  set(FSST_SOURCE_DIR "${fsst_SOURCE_DIR}")
endif()

if(TARGET fsst)
  target_include_directories(fsst PUBLIC "${FSST_SOURCE_DIR}")
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
  if(TARGET binary)
    set_target_properties(binary PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
  if(TARGET binary12)
    set_target_properties(binary12 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
else()
  target_include_directories(nimble_encodings PUBLIC "${FSST_SOURCE_DIR}")
endif()

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
