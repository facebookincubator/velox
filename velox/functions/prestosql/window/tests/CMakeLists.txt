add_executable(velox_windows_test RowNumberTest.cpp)

add_test(
  NAME velox_windows_test
  COMMAND velox_windows_test
  WORKING_DIRECTORY .)

target_link_libraries(
  velox_windows_test
  velox_core
  velox_dwio_common_test_utils
  velox_exec
  velox_exec_test_util
  velox_vector_test_lib
  velox_window
  ${gflags_LIBRARIES}
  gtest
  gtest_main)
