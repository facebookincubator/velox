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

FetchContent_Declare(
  clp
  GIT_REPOSITORY https://github.com/y-scope/clp.git
  GIT_TAG 7b1b169a89abdfe44c159d6200b168391b697877
  GIT_SUBMODULES "" GIT_SUBMODULES_RECURSE TRUE)

FetchContent_MakeAvailable(clp)

if(clp_POPULATED)
  message(STATUS "Updating submodules for clp...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E chdir "${clp_SOURCE_DIR}" git submodule update
            --init --recursive
    RESULT_VARIABLE submodule_update_result
    OUTPUT_VARIABLE submodule_update_output
    ERROR_VARIABLE submodule_update_error)
  if(NOT ${submodule_update_result} EQUAL 0)
    message(ERROR
            "Failed to update submodules for clp:\n${submodule_update_error}")
  else()
    message(STATUS "Submodules for clp updated successfully.")
  endif()
endif()
