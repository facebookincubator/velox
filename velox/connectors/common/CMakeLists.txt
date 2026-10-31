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
  OBJECT
  FileConfig.cpp
  FileConfigProvider.cpp
  HEADERS
  FileConfig.h
  FileConfigMacrosDefine.h
  FileConfigMacrosUndef.h
  FileConfigProvider.h
)
velox_link_libraries(velox_file_config velox_core velox_dwio_dwrf_common velox_exception)

velox_add_library(
  velox_file_connector
  OBJECT
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
  PUBLIC velox_file_config velox_connector
  PRIVATE velox_common_io velox_dwio_dwrf_common velox_exec
)

if(VELOX_ENABLE_PARQUET)
  velox_link_libraries(velox_file_config velox_dwio_parquet_common)
  velox_link_libraries(velox_file_connector PRIVATE velox_dwio_parquet_common)
endif()

if(VELOX_BUILD_TESTING AND NOT VELOX_DISABLE_GOOGLETEST)
  add_subdirectory(tests)
endif()
