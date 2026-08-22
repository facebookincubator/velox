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

# ChunkedStreamWriter still lives next to its reader-side counterparts in
# velox/dwio/nimble/velox, so it is picked up by path rather than moved here.
add_library(
  nimble_writer
  ${CMAKE_CURRENT_SOURCE_DIR}/../velox/ChunkedStreamWriter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../velox/SharedDictionaryConfig.cpp
  BufferPolicy.h
  EncodingLayoutTree.cpp
  EncodingSelectionPolicyFactory.cpp
  FlushPolicy.cpp
  FlushPolicyFactory.cpp
  StreamChunker.cpp
  Writer.cpp
  WriterDefaultMetadataOSS.cpp
  WriterMemoryReclaimerOSS.cpp
  WriterOptions.h
  WriterFactory.h
  Writer.h
  StreamChunker.h
  NimbleFileMetadata.h
  FlushPolicyFactory.h
  FlushPolicy.h
  EncodingSelectionPolicyFactory.h
  EncodingLayoutTree.h
)
target_link_libraries(
  nimble_writer
  nimble_column_stats
  nimble_column_stats_utils
  nimble_common
  nimble_encodings
  nimble_index
  nimble_tablet_writer
  nimble_velox_field_writer
  nimble_velox_layout_planner
  nimble_velox_metadata_fb
  nimble_velox_schema_serialization
  nimble_velox_stats_fb
  raw_size_utils
  velox_common_io
  velox_dwio_common
  velox_key_encoder
  velox_type
  Folly::folly
)
