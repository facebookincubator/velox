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

include(CMakePackageConfigHelpers)

function(velox_get_rpath_origin VAR)
  if(APPLE)
    set(_origin @loader_path)
  else()
    set(_origin "\$ORIGIN")
  endif()
  set(${VAR} ${_origin} PARENT_SCOPE)
endfunction()

function(pyvelox_add_module TARGET)
  pybind11_add_module(${TARGET} ${ARGN})

  if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
    target_compile_definitions(${TARGET} PRIVATE PYVELOX_VERSION=${SKBUILD_PROJECT_VERSION_FULL})
  else()
    target_compile_definitions(${TARGET} PRIVATE PYVELOX_VERSION=dev)
  endif()

  # Set the rpath so linker looks within pyvelox package for libs
  velox_get_rpath_origin(_origin)
  set_target_properties(
    ${TARGET}
    PROPERTIES INSTALL_RPATH "${_origin}/;${CMAKE_BINARY_DIR}/lib" INSTALL_RPATH_USE_LINK_PATH TRUE
  )
  install(TARGETS ${TARGET} LIBRARY DESTINATION pyvelox COMPONENT pyvelox_libraries)
endfunction()

# Glob-based header install fallback. Kept for directories that have not yet
# migrated to the HEADERS keyword in velox_add_library (which uses FILE_SET).
function(velox_install_library_headers)
  # Find any headers and install them relative to the source tree in include.
  file(GLOB _hdrs "*.h")
  if(NOT "${_hdrs}" STREQUAL "")
    cmake_path(
      RELATIVE_PATH
      CMAKE_CURRENT_SOURCE_DIR
      BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE _hdr_dir
    )
    install(FILES ${_hdrs} DESTINATION include/${_hdr_dir})
  endif()
endfunction()

# Associate headers with test/benchmark/fuzzer targets via FILE_SET for CMake
# File API discoverability. For production libraries use the HEADERS keyword
# in velox_add_library() instead.
function(velox_add_test_headers TARGET)
  get_target_property(_type ${TARGET} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    set(_scope INTERFACE)
  else()
    set(_scope PUBLIC)
  endif()

  target_sources(
    ${TARGET}
    ${_scope}
    FILE_SET HEADERS
    BASE_DIRS
    ${PROJECT_SOURCE_DIR}
    FILES
    ${ARGN}
  )
endfunction()

# Base add velox library call to add a library and install it.
function(velox_base_add_library TARGET)
  add_library(${TARGET} ${ARGN})
  install(TARGETS ${TARGET} DESTINATION lib/velox)
  velox_install_library_headers()
endfunction()

# This is extremely hackish but presents an easy path to installation.
function(velox_add_library TARGET)
  set(options OBJECT STATIC SHARED INTERFACE)
  set(oneValueArgs)
  set(multiValueArgs HEADERS)
  cmake_parse_arguments(VELOX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(library_type)
  if(VELOX_OBJECT)
    set(library_type OBJECT)
  elseif(VELOX_STATIC)
    set(library_type STATIC)
  elseif(VELOX_SHARED)
    set(library_type SHARED)
  elseif(VELOX_INTERFACE)
    set(library_type INTERFACE)
  endif()

  set(_sources ${VELOX_UNPARSED_ARGUMENTS})

  # Propagate to the underlying add_library and then install the target.
  if(VELOX_MONO_LIBRARY)
    if(TARGET velox)
      # Target already exists, append sources to it.
      target_sources(velox PRIVATE ${_sources})
      if(VELOX_BUILD_PYTHON_PACKAGE)
        install(TARGETS velox LIBRARY DESTINATION pyvelox COMPONENT pyvelox_libraries)
      endif()
    else()
      set(_type STATIC)
      if(VELOX_BUILD_SHARED)
        set(_type SHARED)
      endif()
      # Create the target if this is the first invocation.
      add_library(velox ${_type} ${_sources})
      set_target_properties(velox PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
      set_target_properties(velox PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
      install(TARGETS velox DESTINATION lib/velox EXPORT velox_targets)
      if(VELOX_BUILD_CMAKE_PACKAGE)
        set(package_cmake_dir "lib/cmake/Velox")
        set(config_cmake_in "${PROJECT_SOURCE_DIR}/CMake/VeloxConfig.cmake.in")
        set(config_cmake "${PROJECT_BINARY_DIR}/CMake/VeloxConfig.cmake")
        configure_package_config_file(
          "${config_cmake_in}"
          "${config_cmake}"
          INSTALL_DESTINATION "${package_cmake_dir}"
        )
        install(FILES "${config_cmake}" DESTINATION "${package_cmake_dir}")
        set(system_dependencies)
        if(Arrow_SOURCE STREQUAL "SYSTEM")
          list(APPEND system_dependencies Arrow)
        endif()
        if(glog_SOURCE STREQUAL "SYSTEM")
          list(APPEND system_dependencies glog)
        endif()
        if(VELOX_ENABLE_COMPRESSION_LZ4)
          list(APPEND system_dependencies lz4)
        endif()
        if(re2_SOURCE STREQUAL "SYSTEM")
          list(APPEND system_dependencies re2)
        endif()
        if(stemmer_SOURCE STREQUAL "SYSTEM")
          list(APPEND system_dependencies stemmer)
        endif()
        if(VELOX_BUILD_MINIMAL_WITH_DWIO OR VELOX_ENABLE_HIVE_CONNECTOR)
          list(APPEND system_dependencies Snappy zstd)
        endif()
        foreach(system_dependency ${system_dependencies})
          set(velox_find_module "${PROJECT_SOURCE_DIR}/CMake/Find${system_dependency}.cmake")
          if(EXISTS "${velox_find_module}")
            install(FILES "${velox_find_module}" DESTINATION "${package_cmake_dir}")
          endif()
        endforeach()
        # TODO: We can enable this once we add version to Velox.
        # set(version_cmake "${PROJECT_BINARY_DIR}/CMake/VeloxConfigVersion.cmake")
        # write_basic_package_version_file("${version_cmake}"
        #                                  COMPATIBILITY SameMajorVersion)
        # install(FILES "${version_cmake}" DESTINATION "${package_cmake_dir}")
        install(
          EXPORT velox_targets
          DESTINATION "${package_cmake_dir}"
          NAMESPACE "Velox::"
          FILE "VeloxTargets.cmake"
        )
      endif()
    endif()
    # create alias for compatability
    if(NOT TARGET ${TARGET})
      add_library(${TARGET} ALIAS velox)
    endif()
  else()
    # Create a library for each invocation.
    velox_base_add_library(${TARGET} ${library_type} ${_sources})
  endif()

  # Associate headers with the target via FILE_SET for tracking and IDE
  # integration. The glob-based velox_install_library_headers() remains as
  # fallback for directories that have not yet listed their headers explicitly.
  if(VELOX_HEADERS)
    if(VELOX_MONO_LIBRARY)
      set(_header_target velox)
      # The velox target is a real (non-INTERFACE) library, so always use PUBLIC.
      set(_header_scope PUBLIC)
    else()
      set(_header_target ${TARGET})
      if(VELOX_INTERFACE)
        set(_header_scope INTERFACE)
      else()
        set(_header_scope PUBLIC)
      endif()
    endif()
    target_sources(
      ${_header_target}
      ${_header_scope}
      FILE_SET HEADERS
      BASE_DIRS
      ${PROJECT_SOURCE_DIR}
      FILES
      ${VELOX_HEADERS}
    )
  endif()

  velox_install_library_headers()
endfunction()

function(velox_link_libraries TARGET)
  # TODO(assignUser): Handle scope keywords (they currently are empty calls ala
  # target_link_libraries(target PRIVATE))
  if(VELOX_MONO_LIBRARY)
    # These targets follow the velox_* name for consistency but are NOT actually
    # aliases to velox when building the mono lib and need to be linked
    # explicitly (this is a hack)
    set(
      explicit_targets
      velox_exec_test_lib
      # see velox/experimental/wave/README.md
      velox_wave_common
      velox_wave_decode
      velox_wave_dwio
      velox_wave_exec
      velox_wave_stream
      velox_wave_vector
    )

    foreach(_arg ${ARGN})
      list(FIND explicit_targets ${_arg} _explicit)
      if(_explicit EQUAL -1 AND "${_arg}" MATCHES "^velox_*")
        message(DEBUG "\t\tDROP: ${_arg}")
      else()
        message(DEBUG "\t\tADDING: ${_arg}")
        target_link_libraries(velox ${_arg})
      endif()
    endforeach()
  else()
    target_link_libraries(${TARGET} ${ARGN})
  endif()
endfunction()

function(velox_include_directories TARGET)
  if(VELOX_MONO_LIBRARY)
    target_include_directories(velox ${ARGN})
  else()
    target_include_directories(${TARGET} ${ARGN})
  endif()
endfunction()

function(velox_compile_definitions TARGET)
  if(VELOX_MONO_LIBRARY)
    target_compile_definitions(velox ${ARGN})
  else()
    target_compile_definitions(${TARGET} ${ARGN})
  endif()
endfunction()

function(velox_sources TARGET)
  if(VELOX_MONO_LIBRARY)
    target_sources(velox ${ARGN})
  else()
    target_sources(${TARGET} ${ARGN})
  endif()
endfunction()

# Group test sources into batched binaries to reduce link target count on CI.
# On macOS, defaults to OFF so each test source gets its own binary and
# individual tests are discoverable via 'ctest -R <TestName>'.
if(APPLE)
  option(VELOX_ENABLE_GROUPED_TESTS "Group test sources into batched binaries" OFF)
else()
  option(VELOX_ENABLE_GROUPED_TESTS "Group test sources into batched binaries" ON)
endif()

# Number of test source files per grouped test binary. Controls the trade-off
# between link time (fewer groups = faster linking) and ctest parallelism
# (more groups = more parallel test processes). Ignored when
# VELOX_ENABLE_GROUPED_TESTS is OFF.
set(VELOX_TESTS_PER_GROUP 10 CACHE STRING "Number of test source files per grouped test binary")

# Creates grouped test binaries from a list of test sources. Groups tests into
# batches of VELOX_TESTS_PER_GROUP to reduce link target count while
# maintaining ctest parallelism. When VELOX_ENABLE_GROUPED_TESTS is OFF, each
# source file becomes its own binary named after the source file (without
# extension), making individual tests discoverable via 'ctest -R <TestName>'.
#
# Usage:
#   velox_add_grouped_tests(
#     PREFIX velox_exec
#     SOURCES ${MY_SOURCES}
#     DEPS ${MY_DEPS}
#     [EXTRA_SOURCES Main.cpp]
#     [WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}]
#   )
function(velox_add_grouped_tests)
  cmake_parse_arguments(ARG "" "PREFIX;WORKING_DIRECTORY" "SOURCES;DEPS;EXTRA_SOURCES" ${ARGN})

  if(NOT ARG_WORKING_DIRECTORY)
    set(ARG_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(NOT VELOX_ENABLE_GROUPED_TESTS)
    # Create one binary per source file, named after the source file.
    foreach(_source IN LISTS ARG_SOURCES)
      get_filename_component(_name ${_source} NAME_WE)
      set(_target "${ARG_PREFIX}_${_name}")
      add_executable(${_target} ${_source} ${ARG_EXTRA_SOURCES})
      add_test(NAME ${_target} COMMAND ${_target} WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
      target_link_libraries(${_target} ${ARG_DEPS})
    endforeach()
    return()
  endif()

  list(LENGTH ARG_SOURCES _num_sources)
  math(
    EXPR
    _num_groups
    "(${_num_sources} + ${VELOX_TESTS_PER_GROUP} - 1) / ${VELOX_TESTS_PER_GROUP}"
  )

  set(_idx 0)
  set(_group 0)
  set(_current_sources "")

  foreach(_source IN LISTS ARG_SOURCES)
    list(APPEND _current_sources ${_source})
    math(EXPR _idx "${_idx} + 1")
    math(EXPR _group_end "(${_group} + 1) * ${VELOX_TESTS_PER_GROUP}")

    if(_idx GREATER_EQUAL _group_end OR _idx EQUAL _num_sources)
      set(_target "${ARG_PREFIX}_group${_group}")
      add_executable(${_target} ${_current_sources} ${ARG_EXTRA_SOURCES})
      add_test(NAME ${_target} COMMAND ${_target} WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
      target_link_libraries(${_target} ${ARG_DEPS})
      set(_current_sources "")
      math(EXPR _group "${_group} + 1")
    endif()
  endforeach()
endfunction()
