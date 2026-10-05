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

# Macros keep CMake's compiler flag variables in the calling directory's scope.
macro(velox_configure_compiler_flags)
  # These minimum versions provide the C++20 features used by Velox.
  if(
    NOT
      (
        (
          CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
          AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11
        )
        OR
          (
            CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15
          )
        OR
          (
            CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.30
          )
      )
  )
    message(
      FATAL_ERROR
      "Unsupported compiler ${CMAKE_CXX_COMPILER_ID} with version ${CMAKE_CXX_COMPILER_VERSION} found."
    )
  endif()

  add_compile_definitions(USE_VELOX_COMMON_BASE HAS_UNCAUGHT_EXCEPTIONS)

  # WIN32 describes the target OS, including 64-bit Windows. MSVC describes
  # the compiler's command-line interface, including clang-cl.
  if(WIN32)
    add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES NOMINMAX)
  endif()

  if(NOT MSVC)
    execute_process(
      COMMAND
        bash -c
        "source \"${PROJECT_SOURCE_DIR}/scripts/setup-helper-functions.sh\" && get_cxx_flags \"$ENV{CPU_TARGET}\""
      OUTPUT_VARIABLE SCRIPT_CXX_FLAGS
      RESULT_VARIABLE COMMAND_STATUS
    )
    if(NOT COMMAND_STATUS STREQUAL "0")
      message(FATAL_ERROR "Unable to determine compiler flags: ${COMMAND_STATUS}")
    endif()
    message("Setting CMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCRIPT_CXX_FLAGS}")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
    endif()

    if(VELOX_FORCE_COLORED_OUTPUT)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fcolor-diagnostics)
      endif()
    endif()

    if(ENABLE_ALL_WARNINGS)
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(
          KNOWN_COMPILER_SPECIFIC_WARNINGS
          "-Wno-range-loop-analysis \
             -Wno-mismatched-tags \
             -Wno-nullability-completeness"
        )
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(
          KNOWN_COMPILER_SPECIFIC_WARNINGS
          "-Wno-implicit-fallthrough \
             -Wno-class-memaccess \
             -Wno-comment \
             -Wno-int-in-bool-context \
             -Wno-redundant-move \
             -Wno-array-bounds \
             -Wno-maybe-uninitialized \
             -Wno-unused-result \
             -Wno-format-overflow \
             -Wno-strict-aliasing"
        )
        # Avoid compiler bug for GCC 12.2.1.
        # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
        if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "12.2.1")
          string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-restrict")
        endif()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0.0")
          string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=template-id-cdtor")
          string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-overloaded-virtual")
          string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=tautological-compare")
        endif()
      endif()
      set(
        KNOWN_WARNINGS
        "-Wno-unused \
           -Wno-unused-parameter \
           -Wno-sign-compare \
           -Wno-ignored-qualifiers \
           ${KNOWN_COMPILER_SPECIFIC_WARNINGS}"
      )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ${KNOWN_WARNINGS}")
    endif()
  endif()

  # Enable type_info deduplication across shared libraries on macOS.
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_link_options("-Wl,-flat_namespace")
  endif()
  if(UNIX AND NOT APPLE)
    add_link_options("-Wl,-export-dynamic")
  endif()

  # Allow Velox code to be linked into a dynamic library.
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endmacro()

# Apply after dependency resolution so bundled libraries do not inherit Velox's
# warnings-as-errors or sanitizer policy.
macro(velox_configure_velox_flags)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)
    add_compile_definitions(FOLLY_HAS_COROUTINES=1)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # Remove once the pinned FBOS version handles Apple's toolchain change.
    add_compile_options(-D_LIBCPP_HAS_NO_ASAN)
  endif()

  if(TREAT_WARNINGS_AS_ERRORS)
    if(MSVC)
      add_compile_options(/WX)
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
  endif()

  # Enable the ASAN and UBSAN sanitizers for the GNU-style Clang driver.
  if(
    VELOX_ENABLE_ASAN_UBSAN_SANITIZERS
    AND NOT MSVC
    AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20
  )
    set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -O1 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DVELOX_ENABLE_ASAN_UBSAN_SANITIZERS"
    )
    # Skip alignment errors and member access checks for now.
    # https://github.com/facebookincubator/velox/issues/15811
    # https://github.com/facebookincubator/velox/issues/15810
    set(
      CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined -fno-sanitize=alignment -fno-sanitize=vptr"
    )
  endif()

  message("FINAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
  message("FINAL CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
endmacro()
