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
  velox_cudf_iceberg_connector
  CudfDeletionVectorReader.cpp
  CudfDeletionVectorReader.cu
  CudfIcebergConnector.cpp
  CudfIcebergDataSource.cpp
  CudfIcebergSplitReader.cpp
)

set_target_properties(velox_cudf_iceberg_connector PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(velox_cudf_iceberg_connector PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

target_link_libraries(
  velox_cudf_iceberg_connector
  PRIVATE velox_hive_iceberg_splitreader velox_hive_connector velox_functions_iceberg cuco::cuco
  PUBLIC cudf::cudf velox_cudf_hive_connector
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
