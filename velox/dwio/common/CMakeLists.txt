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
add_subdirectory(compression)
add_subdirectory(encryption)
add_subdirectory(exception)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
elseif(${VELOX_BUILD_TEST_UTILS})
  add_subdirectory(tests/utils)
endif()

add_library(
  velox_dwio_common_base
  BufferedInput.cpp
  CachedBufferedInput.cpp
  CacheInputStream.cpp
  DataBufferHolder.cpp
  FileSink.cpp
  InputStream.cpp
  IoStatistics.cpp
  OutputStream.cpp
  SeekableInputStream.cpp)
target_link_libraries(
  velox_dwio_common_base
  velox_dwio_common_exception
  velox_buffer
  velox_caching
  velox_exception
  velox_memory
  Folly::folly)

add_library(
  velox_dwio_common
  BitConcatenation.cpp
  BitPackDecoder.cpp
  ColumnSelector.cpp
  DecoderUtil.cpp
  DirectDecoder.cpp
  DwioMetricsLog.cpp
  FlatMapHelper.cpp
  IntDecoder.cpp
  MetadataFilter.cpp
  Options.cpp
  Range.cpp
  Reader.cpp
  ReaderFactory.cpp
  ScanSpec.cpp
  ColumnLoader.cpp
  SelectiveByteRleColumnReader.cpp
  SelectiveColumnReader.cpp
  SelectiveRepeatedColumnReader.cpp
  SelectiveStructColumnReader.cpp
  TypeUtils.cpp
  TypeWithId.cpp
  WriterFactory.cpp
  SortingWriter.cpp
  SortingWriter.h)

target_include_directories(velox_dwio_common PRIVATE ${Protobuf_INCLUDE_DIRS})

target_link_libraries(
  velox_dwio_common
  velox_common_compression
  velox_dwio_common_base
  velox_dwio_common_compression
  velox_dwio_common_encryption
  velox_expression
  velox_exec
  Boost::regex
  glog::glog)
