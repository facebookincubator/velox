# Copyright (c) Facebook, Inc. and its affiliates.
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

velox_add_library(
  velox_dwio_native_parquet_reader
  Metadata.cpp
  NestedStructureDecoder.cpp
  ParquetReader.cpp
  ParquetTypeWithId.cpp
  PageReader.cpp
  ParquetColumnReader.cpp
  ParquetData.cpp
  RepeatedColumnReader.cpp
  RleBpDecoder.cpp
  StructColumnReader.cpp
  StringColumnReader.cpp
  SemanticVersion.cpp
  HEADERS
  BooleanColumnReader.h
  BooleanDecoder.h
  DeltaBpDecoder.h
  DeltaByteArrayDecoder.h
  FloatingPointColumnReader.h
  IntegerColumnReader.h
  Metadata.h
  NestedStructureDecoder.h
  PageReader.h
  ParquetColumnReader.h
  ParquetData.h
  ParquetReader.h
  ParquetStatsContext.h
  ParquetTypeWithId.h
  RepeatedColumnReader.h
  RleBpDataDecoder.h
  RleBpDecoder.h
  SemanticVersion.h
  StringColumnReader.h
  StringDecoder.h
  StructColumnReader.h
  TimeColumnReader.h
  TimestampColumnReader.h
)

velox_link_libraries(
  velox_dwio_native_parquet_reader
  velox_dwio_parquet_common
  velox_type
  velox_dwio_common
  velox_dwio_common_compression
  fmt::fmt
  arrow
  Snappy::snappy
  zstd::zstd
)
