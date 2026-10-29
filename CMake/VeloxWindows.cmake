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

# Windows headers define ERROR, min, and max macros that conflict with C++ names.
add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES NOMINMAX)

if(MSVC)
  # The pinned Folly's FixedString UDL macro needs a spacing fix for MSVC.
  set(
    VELOX_WINDOWS_FOLLY_PATCH
    ${CMAKE_CURRENT_LIST_DIR}/resolve_dependency_modules/folly/folly-windows.patch
  )

  set(VELOX_WINDOWS_ALL_WARNINGS_FLAG /W4)
  set(VELOX_WARNINGS_AS_ERRORS_FLAG /WX)
  # Velox needs a conforming preprocessor, __cplusplus, and C++ name lookup.
  set(
    VELOX_WINDOWS_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
  )

  function(velox_configure_windows_boost_context)
    if(TARGET boost_context AND BOOST_CONTEXT_ASSEMBLER STREQUAL "masm")
      # Generic ASM can claim .asm before Boost enables MASM, so VS omits the files.
      get_target_property(context_sources boost_context SOURCES)
      get_target_property(context_source_dir boost_context SOURCE_DIR)
      foreach(source IN LISTS context_sources)
        if(source MATCHES "\\.asm$")
          cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${context_source_dir}" NORMALIZE)
          set_source_files_properties(
            "${source}"
            DIRECTORY "${context_source_dir}"
            PROPERTIES LANGUAGE ASM_MASM
          )
        endif()
      endforeach()
    endif()
  endfunction()

  # Boost targets are available after dependency resolution.
  cmake_language(DEFER CALL velox_configure_windows_boost_context)
endif()
