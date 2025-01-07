# To test functions being added by dynamically linked libraries, we compile
# `MyDynamicFunction.cpp` as a small .so library, and use the
# VELOX_TEST_DYNAMIC_LIBRARY_PATH macro to locate the .so binary.

add_library(velox_function_my_dynamic SHARED MyDynamicFunction.cpp)
  target_link_libraries(
    velox_function_my_dynamic xsimd fmt)

add_library(velox_int_function_my_dynamic SHARED MyDynamicIntFunction.cpp)
target_link_libraries(
  velox_int_function_my_dynamic xsimd fmt)

add_library(velox_str_function_my_dynamic SHARED MyDynamicStrFunction.cpp)
target_link_libraries(
  velox_str_function_my_dynamic xsimd fmt)

add_library(velox_function_same_twice_my_dynamic SHARED MyDynamicSameTwiceFunction.cpp)
  target_link_libraries(
    velox_function_same_twice_my_dynamic xsimd fmt)

add_library(velox_function_err_my_dynamic SHARED MyDynamicErrFunction.cpp)
    target_link_libraries(
      velox_function_err_my_dynamic xsimd fmt)

# Here's the actual test which will dynamically load the library defined above.
add_executable(velox_function_dynamic_link_test DynamicLinkTest.cpp)

target_compile_definitions(velox_function_dynamic_link_test PRIVATE
  VELOX_TEST_DYNAMIC_LIBRARY_PATH="${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(velox_function_dynamic_link_test PRIVATE
  VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_test(NAME velox_function_dynamic_link_test
         COMMAND velox_function_dynamic_link_test)

target_link_libraries(
  velox_function_dynamic_link_test
  velox_functions_test_lib
  velox_dynamic_function_loader
  velox_function_registry
  xsimd
  GTest::gmock
  GTest::gtest
  GTest::gtest_main)
