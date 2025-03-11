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

find_package(Java COMPONENTS Development)

if(Java_Development_FOUND)
  add_custom_command(
    OUTPUT FloatGenerator.class
    COMMAND ${Java_JAVAC_EXECUTABLE} -d ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/FloatGenerator.java
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/FloatGenerator.java)
  add_custom_target(java_float_generator DEPENDS FloatGenerator.class)

  add_executable(velox_float_to_string_fuzzer FloatToStringFuzzer.cpp)
  add_test(float_to_string_fuzzer float_to_string_fuzzer)

  target_link_libraries(
    velox_float_to_string_fuzzer
    fmt::fmt
    velox_type
    GTest::gtest
    GTest::gtest_main
    gflags::gflags
    glog::glog)

  add_dependencies(velox_float_to_string_fuzzer java_float_generator)
endif()
