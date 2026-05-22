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

add_library(velox_cudf_abfs_io_source RegisterCudfAbfsIoSource.cpp)

target_link_libraries(
  velox_cudf_abfs_io_source
  PUBLIC velox_cudf_io_source_registry
)

if(VELOX_ENABLE_ABFS)
  target_sources(
    velox_cudf_abfs_io_source
    PRIVATE CudfAbfsIoSource.cpp CudfAbfsPinnedBufferPool.cpp
  )

  target_link_libraries(
    velox_cudf_abfs_io_source
    PUBLIC velox_abfs cudf::cudf
    PRIVATE Azure::azure-storage-blobs Azure::azure-storage-files-datalake
  )
endif()
