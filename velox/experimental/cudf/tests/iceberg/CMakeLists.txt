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

# Tests for the cudf iceberg connector defined under
# velox/experimental/cudf/connectors/hive/iceberg/. They live here (rather
# than next to the connector sources) so the executables land alongside
# the other velox_cudf_* tests via velox_add_cudf_test() — which is what
# the cudf-tests CI artifact glob and ctest invocation in
# .github/workflows/linux-build-base.yml pick up.

add_library(velox_cudf_iceberg_test_lib CudfIcebergTestBase.cpp)

target_link_libraries(
  velox_cudf_iceberg_test_lib
  velox_cudf_exec_test_lib
  velox_cudf_iceberg_connector
  velox_hive_iceberg_splitreader
  velox_dwio_parquet_writer
  velox_exec_test_lib
  velox_test_util
)

velox_add_cudf_test(
  NAME velox_cudf_iceberg_read_test
  SOURCES
    Main.cpp
    CudfIcebergReadTest.cpp
    CudfEqualityDeleteFileReaderTest.cpp
    CudfIcebergGapTests.cpp
  LIBS
    velox_cudf_iceberg_test_lib
    velox_cudf_iceberg_connector
    velox_cudf_exec_test_lib
    velox_exec
    velox_exec_test_lib
    velox_test_util
    fmt::fmt
  TIMEOUT 3000
)

velox_add_cudf_test(
  NAME velox_cudf_iceberg_deletion_vector_test
  SOURCES CudfDeletionVectorReaderTest.cpp
  LIBS
    velox_cudf_iceberg_connector
    velox_cudf_exec_test_lib
    velox_exec
    velox_exec_test_lib
    velox_test_util
    fmt::fmt
  TIMEOUT 3000
)
