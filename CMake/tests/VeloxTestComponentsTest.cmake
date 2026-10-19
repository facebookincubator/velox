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
cmake_minimum_required(VERSION 3.28)

if(DEFINED TEST_CASE)
  include("${CMAKE_CURRENT_LIST_DIR}/../VeloxTestComponents.cmake")
  set(VELOX_BUILD_TESTING OFF)
  set(VELOX_MONO_LIBRARY OFF)
  set(available_components status vector)
  set(expected_components "")

  if(TEST_CASE STREQUAL "full")
    set(VELOX_BUILD_TESTING ON)
    set(VELOX_MONO_LIBRARY ON)
  elseif(TEST_CASE STREQUAL "disabled")
    set(VELOX_MONO_LIBRARY ON)
  elseif(TEST_CASE STREQUAL "status")
    set(VELOX_TEST_COMPONENTS status)
    set(expected_components status)
  elseif(TEST_CASE STREQUAL "multiple")
    set(VELOX_TEST_COMPONENTS "status;vector")
    set(expected_components "status;vector")
  elseif(TEST_CASE STREQUAL "duplicate")
    set(VELOX_TEST_COMPONENTS "status;status")
    set(expected_components status)
  elseif(TEST_CASE STREQUAL "unknown")
    set(VELOX_TEST_COMPONENTS statuz)
  elseif(TEST_CASE STREQUAL "unregistered")
    set(available_components status)
    set(VELOX_TEST_COMPONENTS "status;vector")
  elseif(TEST_CASE STREQUAL "mixed_all")
    set(VELOX_TEST_COMPONENTS "ALL;status")
  elseif(TEST_CASE STREQUAL "empty")
    set(VELOX_TEST_COMPONENTS "")
  elseif(TEST_CASE STREQUAL "full_conflict")
    set(VELOX_TEST_COMPONENTS status)
    set(VELOX_BUILD_TESTING ON)
  elseif(TEST_CASE STREQUAL "mono")
    set(VELOX_TEST_COMPONENTS status)
    set(VELOX_MONO_LIBRARY ON)
    set(expected_components status)
  else()
    message(FATAL_ERROR "Unknown test case: ${TEST_CASE}")
  endif()

  set(original_build_testing "${VELOX_BUILD_TESTING}")
  set(original_mono_library "${VELOX_MONO_LIBRARY}")
  velox_configure_test_components(${available_components})
  if(NOT "${VELOX_SELECTED_TEST_COMPONENTS}" STREQUAL "${expected_components}")
    message(
      FATAL_ERROR
      "Expected '${expected_components}', got '${VELOX_SELECTED_TEST_COMPONENTS}'."
    )
  endif()
  if(NOT "${VELOX_BUILD_TESTING}" STREQUAL "${original_build_testing}")
    message(FATAL_ERROR "Component selection changed the full-suite option.")
  endif()
  if(NOT "${VELOX_MONO_LIBRARY}" STREQUAL "${original_mono_library}")
    message(FATAL_ERROR "Component selection changed the library layout.")
  endif()
  return()
endif()

function(check_case name expected_error)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DTEST_CASE=${name}" -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(expected_error STREQUAL "")
    if(NOT result STREQUAL "0")
      message(FATAL_ERROR "${name} unexpectedly failed:\n${output}\n${error}")
    endif()
  elseif(result STREQUAL "0" OR NOT error MATCHES "${expected_error}")
    message(FATAL_ERROR "${name} did not report '${expected_error}':\n${output}\n${error}")
  endif()
endfunction()

foreach(test_case full disabled status multiple duplicate mono)
  check_case("${test_case}" "")
endforeach()
check_case(unknown "Unknown standalone test component 'statuz'")
check_case(unregistered "Unknown standalone test component 'vector'")
check_case(mixed_all "Unknown standalone test component 'ALL'")
check_case(empty "must be ALL or a non-empty component list")
check_case(full_conflict "requires VELOX_BUILD_TESTING=OFF")

message(STATUS "Test component selection: 11 cases passed.")
