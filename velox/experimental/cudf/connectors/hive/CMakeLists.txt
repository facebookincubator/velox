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

add_library(
  velox_cudf_hive_connector
  BufferedInputDataSource.cpp
  CudfDynamicFilter.h
  CudfHiveConfig.cpp
  CudfHiveConnector.cpp
  CudfHiveConnectorSplit.cpp
  CudfHiveDataSink.cpp
  CudfHiveDataSource.cpp
  CudfHiveTableHandle.cpp
  CudfIntegerMembership.cu
  CudfIntegerMembership.h
  CudfSplitReader.cpp
  CudfSplitReaderByteFetch.cpp
)

target_link_libraries(
  velox_cudf_hive_connector
  PRIVATE velox_common_io velox_dwio_common velox_hive_connector cuco::cuco
  PUBLIC cudf::cudf nvtx3::nvtx3-cpp velox_cudf_expression
)

if(VELOX_ENABLE_ABFS)
  target_link_libraries(velox_cudf_hive_connector PRIVATE velox_abfs)
endif()

add_subdirectory(iceberg)

target_compile_options(
  velox_cudf_hive_connector
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)
