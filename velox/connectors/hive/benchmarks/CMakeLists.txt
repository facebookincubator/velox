add_executable(velox_hive_partition_function_benchmark HivePartitionFunctionBenchmark.cpp)

target_link_libraries(velox_hive_partition_function_benchmark
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_vector_fuzzer 
  velox_expression
  velox_parse_parser
  velox_parse_utils
  velox_parse_expression
  velox_exec
  velox_exec_test_util
  velox_hive_connector
  velox_hive_partition_function
  velox_memory
  ${FOLLY_WITH_DEPENDENCIES}
  ${FOLLY_BENCHMARK}
  ${FMT})