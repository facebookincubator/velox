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
add_library(nimble_tools_common EncodingUtilities.cpp EncodingUtilities.h)

target_link_libraries(nimble_tools_common nimble_common nimble_tablet_reader nimble_column_stats)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/ExternalDictionary.fbs"
  ""
  nimble_external_dictionary_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_external_dictionary_fb INTERFACE)
target_include_directories(
  nimble_external_dictionary_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_external_dictionary_fb nimble_external_dictionary_schema_fb)

add_library(nimble_dump_lib NimbleDumpLib.cpp NimbleDumpLib.h)
target_link_libraries(
  nimble_dump_lib
  nimble_tools_common
  nimble_common
  nimble_encodings
  nimble_tablet_common
  nimble_tablet_reader
  nimble_velox_reader
  nimble_writer
  nimble_velox_stats_fb
  velox_file
  Folly::folly
)

add_library(nimble_serializer_dump_lib SerializerDumpLib.cpp SerializerDumpLib.h)
target_link_libraries(
  nimble_serializer_dump_lib
  nimble_common
  nimble_serialization_header
  nimble_stream_data_parser
  nimble_legacy
  Folly::folly
)

add_library(nimble_dsl_lib NimbleDslLib.cpp NimbleDslVm.cpp NimbleDslLib.h NimbleDslVm.h)
target_link_libraries(
  nimble_dsl_lib
  nimble_dump_lib
  nimble_tools_common
  nimble_common
  nimble_tablet_reader
  nimble_velox_reader
  nimble_velox_schema_reader
  nimble_column_stats
  velox_file
  velox_memory
)

add_library(
  nimble_external_dictionary_builder
  ExternalDictionaryBuilder.cpp
  ExternalDictionaryBuilder.h
)
target_link_libraries(
  nimble_external_dictionary_builder
  nimble_external_dictionary_fb
  nimble_common
  nimble_encodings
  velox_file
  velox_memory
  Folly::folly
)

# Only the libraries are built. The command line front-ends (NimbleDump.cpp,
# NimbleDsl.cpp) and the EncodingBench and EncodingSelectionLogger
# instrumentation include Meta-internal headers that are not part of the OSS
# export, so they have no OSS build.

add_subdirectory(encoding)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
