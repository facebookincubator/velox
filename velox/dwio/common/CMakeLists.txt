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
add_subdirectory(wrap)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
elseif(${VELOX_BUILD_TEST_UTILS})
  add_subdirectory(tests/utils)
endif()

velox_add_library(
  velox_dwio_common
  BitConcatenation.cpp
  BitPackDecoder.cpp
  BufferedInput.cpp
  CacheInputStream.cpp
  CachedBufferedInput.cpp
  ColumnLoader.cpp
  ColumnSelector.cpp
  DataBufferHolder.cpp
  DecoderUtil.cpp
  DirectBufferedInput.cpp
  DirectDecoder.cpp
  DirectInputStream.cpp
  MetricsLog.cpp
  ExecutorBarrier.cpp
  FileSink.cpp
  FlatMapHelper.cpp
  OnDemandUnitLoader.cpp
  InputStream.cpp
  IntDecoder.cpp
  MetadataFilter.cpp
  Options.cpp
  OutputStream.cpp
  ParallelUnitLoader.cpp
  ParallelFor.cpp
  Range.cpp
  Reader.cpp
  ReaderFactory.cpp
  ScanSpec.cpp
  SeekableInputStream.cpp
  SelectiveByteRleColumnReader.cpp
  SelectiveColumnReader.cpp
  SelectiveFlatMapColumnReader.cpp
  SelectiveRepeatedColumnReader.cpp
  SelectiveStructColumnReader.cpp
  SortingWriter.cpp
  SortingWriter.h
  StatisticsBuilder.cpp
  Throttler.cpp
  TypeUtils.cpp
  TypeWithId.cpp
  Writer.cpp
  WriterFactory.cpp
  HEADERS
  Adaptor.h
  Arena.h
  BitConcatenation.h
  BitPackDecoder.h
  BufferUtil.h
  BufferedInput.h
  CacheInputStream.h
  CachedBufferedInput.h
  ChainedBuffer.h
  Closeable.h
  ColumnLoader.h
  ColumnSelector.h
  ColumnVisitors.h
  DataBuffer.h
  DataBufferHolder.h
  DecoderUtil.h
  DirectBufferedInput.h
  DirectDecoder.h
  DirectInputStream.h
  ErrorTolerance.h
  ExecutorBarrier.h
  FileMetadata.h
  FileSink.h
  FilterNode.h
  FlatMapHelper.h
  FlushPolicy.h
  FormatData.h
  InputStream.h
  IntCodecCommon.h
  IntDecoder.h
  MeasureTime.h
  MetadataFilter.h
  MetricsLog.h
  Mutation.h
  OnDemandUnitLoader.h
  Options.h
  OutputStream.h
  ParallelFor.h
  ParallelUnitLoader.h
  PositionProvider.h
  RandGen.h
  Range.h
  Reader.h
  ReaderFactory.h
  Retry.h
  ScanSpec.h
  SeekableInputStream.h
  SelectiveByteRleColumnReader.h
  SelectiveColumnReader.h
  SelectiveColumnReaderInternal.h
  SelectiveFlatMapColumnReader.h
  SelectiveFloatingPointColumnReader.h
  SelectiveIntegerColumnReader.h
  SelectiveRepeatedColumnReader.h
  SelectiveStructColumnReader.h
  Statistics.h
  StatisticsBuilder.h
  StreamIdentifier.h
  StreamUtil.h
  Throttler.h
  TypeUtil.h
  TypeUtils.h
  TypeWithId.h
  UnitLoader.h
  UnitLoaderTools.h
  Writer.h
  WriterFactory.h
)

velox_include_directories(velox_dwio_common PRIVATE ${Protobuf_INCLUDE_DIRS})

velox_link_libraries(
  velox_dwio_common
  velox_buffer
  velox_caching
  velox_common_io
  velox_common_compression
  velox_common_config
  velox_common_hyperloglog
  velox_dwio_common_encryption
  velox_dwio_common_exception
  velox_exception
  velox_expression
  velox_memory
  velox_type_tz
  Folly::folly
  glog::glog
  protobuf::libprotobuf
)
