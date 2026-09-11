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

add_executable(
  velox_abfs_test
  AbfsFileSystemTest.cpp
  DynamicSasTokenClientProviderTest.cpp
  AzuriteServer.cpp
  MockDataLakeFileClient.cpp
)
velox_add_test_headers(velox_abfs_test AzuriteServer.h MockDataLakeFileClient.h)

add_test(velox_abfs_test velox_abfs_test)
target_link_libraries(
  velox_abfs_test
  PRIVATE velox_abfs velox_exec_test_lib GTest::gtest GTest::gtest_main
)

target_compile_options(velox_abfs_test PRIVATE -Wno-deprecated-declarations)

add_executable(
  velox_abfs_registration_test
  AbfsFileSystemRegistrationTest.cpp
  AbfsUtilTest.cpp
  AzureClientProvidersTest.cpp
  AzureClientProviderFactoriesTest.cpp
)

add_test(velox_abfs_registration_test velox_abfs_registration_test)
target_link_libraries(
  velox_abfs_registration_test
  PRIVATE velox_abfs velox_exec_test_lib GTest::gtest GTest::gtest_main
)

target_compile_options(velox_abfs_registration_test PRIVATE -Wno-deprecated-declarations)

add_executable(
  velox_abfs_stage1_transport_test
  ../FollyHttpTransport.cpp
  ../FollyResponseBodyStream.cpp
  ../EventSocketChannelFactory.cpp
  ../FollyHttpConnection.cpp
  FollyHttpTransportTest.cpp
  FollyHttpTransportC1Test.cpp
)

add_test(velox_abfs_stage1_transport_test velox_abfs_stage1_transport_test)
target_link_libraries(
  velox_abfs_stage1_transport_test
  PRIVATE
    Folly::folly
    Boost::headers
    Azure::azure-core
    Azure::azure-storage-blobs
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)
target_compile_definitions(
  velox_abfs_stage1_transport_test
  PRIVATE VELOX_ABFS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

add_executable(
  velox_abfs_stage2_runtime_test
  ../AbfsAsyncRuntime.cpp
  ../FollyHttpTransport.cpp
  ../FollyResponseBodyStream.cpp
  ../EventSocketChannelFactory.cpp
  ../FollyHttpConnection.cpp
  AbfsAsyncRuntimeTest.cpp
)

add_test(velox_abfs_stage2_runtime_test velox_abfs_stage2_runtime_test)
target_link_libraries(
  velox_abfs_stage2_runtime_test
  PRIVATE
    Folly::folly
    Boost::headers
    Azure::azure-core
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(
  VELOX_ENABLE_BENCHMARKS
  AND VELOX_ENABLE_PARQUET
  AND VELOX_ENABLE_HIVE_CONNECTOR)
  add_executable(
    velox_abfs_parquet_benchmark
    AbfsParquetBenchmark.cpp
    AzuriteServer.cpp)
  target_link_libraries(
    velox_abfs_parquet_benchmark
    PRIVATE
      velox_abfs
      velox_hive_connector
      velox_dwio_parquet_reader
      velox_dwio_parquet_writer
      velox_exec
      velox_exec_test_lib
      GTest::gtest
      Folly::folly
  )
  target_compile_options(
    velox_abfs_parquet_benchmark PRIVATE -Wno-deprecated-declarations)
endif()
