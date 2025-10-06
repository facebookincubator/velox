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

# To test functions being added by dynamically linked libraries, we compile
# `DynamicFunction.cpp` as a small .so library, and use the
# VELOX_TEST_DYNAMIC_LIBRARY_PATH macro to locate the .so binary.

add_library(velox_function_dynamic SHARED DynamicFunction.cpp)
add_library(velox_overwrite_int_function_dynamic SHARED DynamicIntFunctionOverwrite.cpp)
add_library(velox_overwrite_varchar_function_dynamic SHARED DynamicVarcharFunctionOverwrite.cpp)
add_library(velox_function_err_dynamic SHARED DynamicErrFunction.cpp)
add_library(velox_overload_int_function_dynamic SHARED DynamicIntFunctionOverload.cpp)
add_library(velox_overload_varchar_function_dynamic SHARED DynamicVarcharFunctionOverload.cpp)
add_library(velox_function_non_default_dynamic SHARED DynamicFunctionNonDefault.cpp)

add_library(velox_function_dynamic_link_test_common INTERFACE)
target_link_libraries(velox_function_dynamic_link_test_common INTERFACE fmt::fmt glog::glog xsimd)
if(folly_SOURCE STREQUAL BUNDLED)
  # We can prevent errors related to 'is being linked both statically
  # and dynamically into this executable,' particularly for
  # folly::hazptr_use_executor(), by "-Wl,--exclude-libs,ALL". But it
  # works only for bundled Folly.
  target_link_libraries(velox_function_dynamic_link_test_common INTERFACE Folly::Folly)
else()
  # We use only for Folly headers for system Folly. If we link Folly
  # (static library) to shared libraries, multiple Folly exist in the
  # same process.
  target_include_directories(
    velox_function_dynamic_link_test_common
    INTERFACE $<TARGET_PROPERTY:Folly::folly_deps,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()
if(APPLE)
  target_link_options(
    velox_function_dynamic_link_test_common
    INTERFACE "-Wl,-undefined,dynamic_lookup"
  )
else()
  # This ensures compatibility during Linux compilation by preventing errors
  # related to 'is being linked both statically and dynamically into this
  # executable,' particularly for folly::hazptr_use_executor().
  #
  # This works for bundled Folly but not work with system Folly.
  target_link_options(velox_function_dynamic_link_test_common INTERFACE "-Wl,--exclude-libs,ALL")
endif()

target_link_libraries(velox_function_dynamic PRIVATE velox_function_dynamic_link_test_common)
target_link_libraries(
  velox_overwrite_int_function_dynamic
  PRIVATE velox_function_dynamic_link_test_common
)
target_link_libraries(
  velox_overwrite_varchar_function_dynamic
  PRIVATE velox_function_dynamic_link_test_common
)
target_link_libraries(velox_function_err_dynamic PRIVATE velox_function_dynamic_link_test_common)
target_link_libraries(
  velox_overload_int_function_dynamic
  PRIVATE velox_function_dynamic_link_test_common
)
target_link_libraries(
  velox_overload_varchar_function_dynamic
  PRIVATE velox_function_dynamic_link_test_common
)
target_link_libraries(
  velox_function_non_default_dynamic
  PRIVATE velox_function_dynamic_link_test_common
)

# Here's the actual test which will dynamically load the library defined above.
add_executable(velox_function_dynamic_link_test DynamicLinkTest.cpp)

target_link_libraries(
  velox_function_dynamic_link_test
  velox_functions_test_lib
  velox_dynamic_library_loader
  velox_function_registry
  xsimd
  GTest::gmock
  GTest::gtest
  GTest::gtest_main
)

target_compile_definitions(
  velox_function_dynamic_link_test
  PRIVATE VELOX_TEST_DYNAMIC_LIBRARY_PATH="${CMAKE_CURRENT_BINARY_DIR}"
)
target_compile_definitions(
  velox_function_dynamic_link_test
  PRIVATE VELOX_TEST_DYNAMIC_LIBRARY_PATH_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

add_test(NAME velox_function_dynamic_link_test COMMAND velox_function_dynamic_link_test)
