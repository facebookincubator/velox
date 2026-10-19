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
include_guard(GLOBAL)

# ARGN lists the components with standalone test targets. The legacy
# VELOX_BUILD_TESTING option continues to control the full test suite.
function(velox_configure_test_components)
  set(
    VELOX_TEST_COMPONENTS
    "ALL"
    CACHE STRING
    "Standalone test components (semicolon-separated), or ALL for VELOX_BUILD_TESTING behavior."
  )
  set_property(CACHE VELOX_TEST_COMPONENTS PROPERTY STRINGS ALL ${ARGN})
  set(VELOX_SELECTED_TEST_COMPONENTS "" PARENT_SCOPE)

  if(VELOX_TEST_COMPONENTS STREQUAL "ALL")
    return()
  endif()

  if(VELOX_TEST_COMPONENTS STREQUAL "")
    message(FATAL_ERROR "VELOX_TEST_COMPONENTS must be ALL or a non-empty component list.")
  endif()
  if(VELOX_BUILD_TESTING)
    message(
      FATAL_ERROR
      "Explicit VELOX_TEST_COMPONENTS requires VELOX_BUILD_TESTING=OFF. "
      "Use VELOX_TEST_COMPONENTS=ALL for the full test suite."
    )
  endif()
  foreach(component IN LISTS VELOX_TEST_COMPONENTS)
    if(NOT component IN_LIST ARGN)
      message(
        FATAL_ERROR
        "Unknown standalone test component '${component}'. Available components: ${ARGN}."
      )
    endif()
  endforeach()

  set(selected_components "${VELOX_TEST_COMPONENTS}")
  list(REMOVE_DUPLICATES selected_components)
  set(VELOX_SELECTED_TEST_COMPONENTS "${selected_components}" PARENT_SCOPE)
  message(STATUS "Standalone test components: ${selected_components}")
endfunction()
