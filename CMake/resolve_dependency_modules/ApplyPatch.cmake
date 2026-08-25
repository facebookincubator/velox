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

if(NOT IS_DIRECTORY "${SOURCE_DIR}")
  message(FATAL_ERROR "Patch source directory does not exist: ${SOURCE_DIR}")
endif()

if(NOT EXISTS "${PATCH_FILE}")
  message(FATAL_ERROR "Patch file does not exist: ${PATCH_FILE}")
endif()

find_program(GIT_EXECUTABLE NAMES git REQUIRED)

# Keep patch paths rooted at SOURCE_DIR instead of an enclosing Git worktree.
get_filename_component(source_dir "${SOURCE_DIR}" REALPATH)
get_filename_component(source_parent_dir "${source_dir}" DIRECTORY)

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env "--unset=GIT_DIR" "--unset=GIT_WORK_TREE"
    "GIT_CEILING_DIRECTORIES=${source_parent_dir}" -- "${GIT_EXECUTABLE}" apply --reverse --check --
    "${PATCH_FILE}"
  WORKING_DIRECTORY "${source_dir}"
  RESULT_VARIABLE reverse_check_result
  OUTPUT_VARIABLE reverse_check_output
  ERROR_VARIABLE reverse_check_error
)
if(reverse_check_result EQUAL 0)
  message(STATUS "Patch already applied: ${PATCH_FILE}")
  return()
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env "--unset=GIT_DIR" "--unset=GIT_WORK_TREE"
    "GIT_CEILING_DIRECTORIES=${source_parent_dir}" -- "${GIT_EXECUTABLE}" apply --check --
    "${PATCH_FILE}"
  WORKING_DIRECTORY "${source_dir}"
  RESULT_VARIABLE forward_check_result
  OUTPUT_VARIABLE forward_check_output
  ERROR_VARIABLE forward_check_error
)
if(NOT forward_check_result EQUAL 0)
  message(
    FATAL_ERROR
    "Patch cannot be applied: ${PATCH_FILE}\n"
    "${forward_check_output}${forward_check_error}"
  )
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env "--unset=GIT_DIR" "--unset=GIT_WORK_TREE"
    "GIT_CEILING_DIRECTORIES=${source_parent_dir}" -- "${GIT_EXECUTABLE}" apply -- "${PATCH_FILE}"
  WORKING_DIRECTORY "${source_dir}"
  RESULT_VARIABLE apply_result
  OUTPUT_VARIABLE apply_output
  ERROR_VARIABLE apply_error
)
if(NOT apply_result EQUAL 0)
  message(FATAL_ERROR "Failed to apply patch: ${PATCH_FILE}\n${apply_output}${apply_error}")
endif()
