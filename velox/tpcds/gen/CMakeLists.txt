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

cmake_policy(SET CMP0079 NEW)

project(TPCDS)

# Add subdirectories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dsdgen build)

# Generate tpcds_idx.hpp
set(INPUT_FILE tpcds.idx)
set(OUTPUT_FILE tpcds_idx.hpp)
set(WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(
  OUTPUT ${WORKING_DIR}/${OUTPUT_FILE}
  COMMAND xxd -i ${INPUT_FILE} > ${OUTPUT_FILE}
  DEPENDS ${WORKING_DIR}/${INPUT_FILE}
  COMMENT "Generating ${OUTPUT_FILE} from ${INPUT_FILE}"
  WORKING_DIRECTORY ${WORKING_DIR})

add_custom_target(generate_tpcds_idx ALL DEPENDS ${WORKING_DIR}/${OUTPUT_FILE})
add_dependencies(dsdgen generate_tpcds_idx)

add_library(append_info OBJECT utils/append_info-c.cpp)
target_compile_options(append_info PRIVATE -w)
target_link_libraries(
  append_info velox_vector_test_lib Folly::folly xsimd)

target_link_libraries(
  dsdgen append_info)

add_library(velox_tpcds_gen TpcdsGen.cpp DSDGenIterator.cpp)
target_include_directories(velox_tpcds_gen PUBLIC dsdgen/include)
target_compile_options(velox_tpcds_gen PRIVATE -w)
target_link_libraries(
  velox_tpcds_gen
  velox_memory
  velox_vector
  dsdgen
  append_info
  fmt::fmt)
