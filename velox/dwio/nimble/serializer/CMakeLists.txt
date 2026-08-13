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

add_library(nimble_serializer_options Options.cpp)
target_link_libraries(
  nimble_serializer_options
  nimble_common
  nimble_encodings
  nimble_writer
  Folly::folly
)

# Must come after nimble_serializer_options is declared since nimble_legacy
# links against it.
add_subdirectory(legacy)

add_library(nimble_serialization_header SerializationHeader.cpp)
target_link_libraries(
  nimble_serialization_header
  nimble_serializer_options
  nimble_common
  nimble_velox_common
  Folly::folly
)

add_library(nimble_stream_data_writer StreamDataWriter.cpp)
target_link_libraries(
  nimble_stream_data_writer
  nimble_serialization_header
  nimble_serializer_options
  nimble_common
  nimble_encodings
  nimble_legacy
  nimble_velox_stream_data
  velox_memory
)

add_library(nimble_stream_data StreamData.cpp)
target_link_libraries(
  nimble_stream_data
  nimble_serializer_options
  nimble_encodings
  nimble_velox_schema
  nimble_common
  velox_memory
)

add_library(nimble_stream_data_parser StreamDataParser.cpp)
target_link_libraries(
  nimble_stream_data_parser
  nimble_serialization_header
  nimble_serializer_options
  nimble_encodings
  nimble_legacy
  nimble_velox_common
  nimble_common
  velox_memory
  Folly::folly
)

add_library(nimble_batched_stream_decoder BatchedStreamDecoder.cpp)
target_link_libraries(
  nimble_batched_stream_decoder
  nimble_stream_data
  nimble_serializer_options
  nimble_velox_common
  nimble_velox_schema_reader
  nimble_common
  velox_memory
  Folly::folly
)

add_library(nimble_deserializer Deserializer.cpp)
target_link_libraries(
  nimble_deserializer
  nimble_stream_data_parser
  nimble_batched_stream_decoder
  nimble_serializer_options
  nimble_velox_common
  nimble_velox_field_reader
  nimble_common
  velox_vector
  velox_memory
  Folly::folly
)

add_library(nimble_serializer Serializer.cpp)
target_link_libraries(
  nimble_serializer
  nimble_serializer_options
  nimble_stream_data_writer
  nimble_common
  nimble_velox_field_writer
  velox_vector
  Folly::folly
)

add_library(nimble_projector Projector.cpp)
target_link_libraries(
  nimble_projector
  nimble_serialization_header
  nimble_serializer_options
  nimble_stream_data_writer
  nimble_stream_data_parser
  nimble_legacy
  nimble_velox_schema_reader
  nimble_common
  velox_type
  Folly::folly
)

add_library(nimble_stream_slicer StreamSlicer.cpp)
target_link_libraries(
  nimble_stream_slicer
  nimble_stream_data_parser
  nimble_stream_data_writer
  nimble_serialization_header
  nimble_serializer_options
  nimble_velox_schema_reader
  nimble_common
  nimble_encodings
  velox_memory
  Folly::folly
)
