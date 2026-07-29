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
set(
  ICEBERG_SOURCES
  DeletionVectorReader.cpp
  DeletionVectorWriter.cpp
  EqualityDeleteFileReader.cpp
  IcebergColumnHandle.cpp
  IcebergConfig.cpp
  IcebergConnector.cpp
  IcebergDataFileStatistics.cpp
  IcebergDataSink.cpp
  IcebergDataSource.cpp
  IcebergDeletionVectorSink.cpp
  IcebergDwrfStatsCollector.cpp
  IcebergMergeProcessor.cpp
  IcebergMergeSink.cpp
  IcebergPartitionName.cpp
  IcebergSessionCredentials.cpp
  IcebergSplit.cpp
  IcebergSplitReader.cpp
  IcebergStatsCollector.cpp
  PartitionSpec.cpp
  PositionalDeleteFileReader.cpp
  TransformEvaluator.cpp
  TransformExprBuilder.cpp
  WriterOptionsAdapter.cpp
)

if(VELOX_ENABLE_PARQUET)
  list(APPEND ICEBERG_SOURCES IcebergParquetStatsCollector.cpp)
endif()

# Iceberg geometry reads need GEOS and the geospatial serde. They are confined to this
# table-format-specific target and gated on VELOX_ENABLE_GEO so that a VELOX_ENABLE_GEO=OFF build
# still compiles and links: the converter translation unit is dropped and IcebergSplitReader fails
# the read with a clear message instead. The format-generic Parquet/DWRF readers stay free of any
# geospatial dependency.
if(VELOX_ENABLE_GEO)
  list(APPEND ICEBERG_SOURCES IcebergGeometryConverter.cpp)
endif()

velox_add_library(
  velox_hive_iceberg_splitreader
  ${ICEBERG_SOURCES}
  HEADERS
  DeletionVectorFormat.h
  DeletionVectorReader.h
  DeletionVectorWriter.h
  EqualityDeleteFileReader.h
  IcebergColumnHandle.h
  IcebergConfig.h
  IcebergConnector.h
  IcebergDataFileStatistics.h
  IcebergDataSink.h
  IcebergDataSource.h
  IcebergDeleteFile.h
  IcebergDeletionVectorSink.h
  IcebergDwrfStatsCollector.h
  IcebergMergeProcessor.h
  IcebergMergeSink.h
  IcebergMetadataColumns.h
  IcebergParquetStatsCollector.h
  IcebergPartitionName.h
  IcebergSessionCredentials.h
  IcebergSplit.h
  IcebergSplitReader.h
  IcebergStatsCollector.h
  PartitionSpec.h
  PositionalDeleteFileReader.h
  TransformEvaluator.h
  TransformExprBuilder.h
  WriterOptionsAdapter.h
  IcebergFieldId.h
  IcebergFieldMetadata.h
  IcebergGeometryConverter.h
)

velox_link_libraries(
  velox_hive_iceberg_splitreader
  velox_connector
  velox_functions_iceberg
  velox_dwio_dwrf_reader
  velox_dwio_dwrf_writer
  Folly::folly
)

if(VELOX_ENABLE_PARQUET)
  velox_link_libraries(velox_hive_iceberg_splitreader velox_dwio_arrow_parquet_writer)
endif()

if(VELOX_ENABLE_GEO)
  velox_compile_definitions(velox_hive_iceberg_splitreader PRIVATE VELOX_ENABLE_GEO)
  velox_link_libraries(
    velox_hive_iceberg_splitreader velox_common_geospatial_serde GEOS::geos
  )
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
