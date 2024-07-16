add_library(velox_cudf tocudf.cpp)
target_link_libraries(velox_cudf velox_exception velox_common_base velox_exec)

add_executable(velox_customdriver_test CustomDriverTest.cpp Main.cpp)
add_test(
  NAME velox_customdriver_test
  COMMAND velox_customdriver_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(velox_customdriver_test PROPERTIES TIMEOUT 3000)

target_link_libraries(
  velox_customdriver_test
  velox_cudf
  velox_aggregates
  velox_dwio_common
  velox_dwio_common_exception
  velox_dwio_common_test_utils
  velox_dwio_parquet_reader
  velox_dwio_parquet_writer
  velox_exec
  velox_exec_test_lib
  velox_functions_json
  velox_functions_lib
  velox_functions_prestosql
  velox_functions_test_lib
  velox_hive_connector
  velox_memory
  velox_serialization
  velox_test_util
  velox_type
  velox_vector
  velox_vector_fuzzer
  velox_window
  Boost::atomic
  Boost::context
  Boost::date_time
  Boost::filesystem
  Boost::program_options
  Boost::regex
  Boost::thread
  Boost::system
  gtest
  gtest_main
  gmock
  Folly::folly
  gflags::gflags
  glog::glog
  fmt::fmt
  ${FILESYSTEM})

