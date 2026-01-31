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
  velox_hive_iceberg_reader
  IcebergSplit.cpp
  IcebergSplitReader.cpp
  PositionalDeleteFileReader.cpp
)

velox_link_libraries(velox_hive_iceberg_reader velox_connector velox_functions_iceberg Folly::folly)

if(VELOX_ENABLE_PARQUET)
  velox_add_library(
    velox_hive_iceberg_writer
    IcebergColumnHandle.cpp
    IcebergConnector.cpp
    IcebergDataSink.cpp
    IcebergPartitionName.cpp
    IcebergSplit.cpp
    PartitionSpec.cpp
    TransformEvaluator.cpp
    TransformExprBuilder.cpp
  )
  velox_link_libraries(
    velox_hive_iceberg_writer
    velox_connector
    velox_functions_iceberg
    Folly::folly
    velox_dwio_parquet_field_id
  )
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
