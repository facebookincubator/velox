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
  velox_cudf_hive_connector
  CudfHiveConfig.cpp
  CudfHiveConnector.cpp
  CudfHiveConnectorSplit.cpp
  CudfHiveDataSource.cpp
  CudfHiveDataSourceHelpers.cpp
  CudfHiveDataSink.cpp
  CudfHiveTableHandle.cpp
)

if(VELOX_MONO_LIBRARY)
  set_target_properties(velox PROPERTIES CUDA_ARCHITECTURES native)
else()
  set_target_properties(velox_cudf_hive_connector PROPERTIES CUDA_ARCHITECTURES native)
endif()

velox_link_libraries(
  velox_cudf_hive_connector
  PRIVATE velox_hive_connector
  PUBLIC cudf::cudf ucxx::ucxx velox_cudf_expression
)
