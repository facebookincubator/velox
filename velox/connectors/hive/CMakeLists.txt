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
  velox_file_config
  STATIC
  FileConfig.cpp
  FileConfigProvider.cpp
  HEADERS
  FileConfig.h
  HiveConfigMacrosDefine.h
  HiveConfigMacrosUndef.h
  FileConfigProvider.h
)
velox_link_libraries(velox_file_config velox_core velox_dwio_dwrf_common velox_exception)

velox_add_library(
  velox_file_connector
  STATIC
  BufferedInputBuilder.cpp
  ExtractionUtils.cpp
  FileColumnHandle.cpp
  FileConnector.cpp
  FileConnectorUtil.cpp
  FileDataSource.cpp
  FileSplitReader.cpp
  FileTableHandle.cpp
  PartitionValue.cpp
  HEADERS
  BufferedInputBuilder.h
  ExtractionUtils.h
  FileColumnHandle.h
  FileConnector.h
  FileConnectorSplit.h
  FileConnectorUtil.h
  FileDataSource.h
  FileScanState.h
  FileSplitReader.h
  FileTableHandle.h
  PartitionValue.h
)

velox_link_libraries(
  velox_file_connector
  PUBLIC velox_file_config velox_connector velox_caching velox_dwio_common
  PRIVATE velox_common_io velox_dwio_dwrf_common velox_exec
)

if(VELOX_ENABLE_PARQUET)
  velox_link_libraries(velox_file_config velox_dwio_parquet_common)
  velox_link_libraries(velox_file_connector PRIVATE velox_dwio_parquet_common)
endif()

velox_add_library(velox_hive_config OBJECT HiveConfig.cpp)
velox_link_libraries(
  velox_hive_config
  velox_file_config
  velox_core
  velox_dwio_dwrf_common
  velox_exception
)

if(VELOX_ENABLE_PARQUET)
  velox_link_libraries(velox_hive_config velox_dwio_parquet_common)
endif()

add_subdirectory(iceberg)
add_subdirectory(paimon)

velox_add_library(
  velox_hive_connector
  OBJECT
  FileDataSink.cpp
  FileIndexReader.cpp
  HiveConfig.cpp
  HiveConfigProvider.cpp
  HiveConnector.cpp
  HiveConnectorSplit.cpp
  HiveConnectorUtil.cpp
  HiveDataSink.cpp
  HiveDataSource.cpp
  HiveIndexSource.cpp
  HiveIndexSourceUtils.cpp
  HivePartitionName.cpp
  HiveSplitReader.cpp
  PartitionIdGenerator.cpp
  SortingWriter.cpp
  TableHandle.cpp
  HEADERS
  FileDataSink.h
  FileHandle.h
  FileIndexReader.h
  FileProperties.h
  HiveConfig.h
  HiveConfigProvider.h
  HiveConnector.h
  HiveConnectorSplit.h
  HiveConnectorUtil.h
  HiveDataSink.h
  HiveDataSource.h
  HiveIndexSource.h
  HiveIndexSourceUtils.h
  HivePartitionFunction.h
  HivePartitionName.h
  HiveSplitReader.h
  IndexReader.h
  PartitionIdGenerator.h
  SortingWriter.h
  TableHandle.h
)

velox_link_libraries(
  velox_hive_connector
  PUBLIC velox_file_connector
  PRIVATE
    velox_common_io
    velox_connector
    velox_dwio_catalog_fbhive
    velox_dwio_dwrf_common
    velox_exec
    velox_hive_partition_function
    velox_key_encoder
)

if(VELOX_ENABLE_PARQUET)
  velox_link_libraries(velox_hive_connector PRIVATE velox_dwio_parquet_common)
endif()

velox_add_library(velox_hive_partition_function HivePartitionFunction.cpp)

velox_link_libraries(velox_hive_partition_function velox_core velox_exec velox_hive_hash)

add_subdirectory(storage_adapters)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()
