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

# BatchReaderOSS.cpp stands in for fb/NimbleReader.cpp, which the Buck build
# compiles instead: the batch reader depends on internal-only code.
add_library(
  nimble_velox_reader_factory
  NimbleReaderFactory.cpp
  BatchReaderOSS.cpp
  NimbleReaderFactory.h
)
target_link_libraries(nimble_velox_reader_factory nimble_velox_selective velox_dwio_common)

# Moved from velox/CMakeLists.txt when the nimble/velox/ folder was flattened.

add_library(nimble_velox_field_reader FieldReader.cpp FieldReader.h)
target_link_libraries(
  nimble_velox_field_reader
  nimble_encodings_legacy
  nimble_velox_schema_reader
  nimble_common
  Folly::folly
  absl::flat_hash_map
  protobuf::libprotobuf
)

add_library(
  nimble_batch_reader
  ../common/ChunkedStream.cpp
  ChunkedStreamDecoder.cpp
  ../common/StreamLabels.cpp
  BatchReader.cpp
  BatchReader.h
  ../common/StreamLabels.h
  ChunkedStreamDecoder.h
  ../common/ChunkedStream.h
)
target_link_libraries(
  nimble_batch_reader
  nimble_velox_common
  nimble_velox_schema
  nimble_velox_schema_serialization
  nimble_velox_schema_fb
  nimble_velox_metadata_fb
  nimble_velox_field_reader
  nimble_tablet_reader
  nimble_common
  nimble_encodings_legacy
  velox_common_base
  Folly::folly
)
