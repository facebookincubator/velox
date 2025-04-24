##############################################################################
# Part of CMake configuration for GEOS
#
# Copyright (C) 2018-2019 Mateusz Loskot <mateusz@loskot.net>
#
# This is free software; you can redistribute and/or modify it under
# the terms of the GNU Lesser General Public Licence as published
# by the Free Software Foundation.
# See the COPYING file for more information.
##############################################################################

# Require CMake 3.13+ with support for meta-features that request compiler modes
# for specific C/C++ language standard levels, and object libraries.
cmake_minimum_required(VERSION 3.13)

# Default to release build so packagers don't release debug builds
set(DEFAULT_BUILD_TYPE Release)

# Require CMake 3.13+ with VS generator for complete support of VS versions and
# support by AppVeyor.
if(${CMAKE_GENERATOR} MATCHES "Visual Studio")
  cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# TODO: Follow CMake detection of git and version tagging
# https://gitlab.kitware.com/cmake/cmake/blob/master/Source/CMakeVersionSource.cmake
if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
  set(GEOS_BUILD_FROM_GIT ON)
endif()

# Make sure we know our build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
  message(STATUS "GEOS: Using default build type: ${CMAKE_BUILD_TYPE}")
else()
  message(STATUS "GEOS: Build type: ${CMAKE_BUILD_TYPE}")
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

# CMake global variables
option(BUILD_SHARED_LIBS "Build GEOS with shared libraries" ON)
set(CMAKE_CXX_STANDARD
    11
    CACHE STRING "C++ standard version to use (default is 11)")

# GEOS custom variables
option(BUILD_BENCHMARKS "Build GEOS benchmarks" OFF)
cmake_dependent_option(
  GEOS_BUILD_DEVELOPER
  "Build with compilation flags useful for development"
  ON
  "GEOS_BUILD_FROM_GIT"
  OFF)
mark_as_advanced(GEOS_BUILD_DEVELOPER)

if(POLICY CMP0092)
  # dont set /W3 warning flags by default, we already set /W4 anyway
  cmake_policy(SET CMP0092 NEW)
endif()

# -----------------------------------------------------------------------------
# Setup build directories
# -----------------------------------------------------------------------------
# Place executables and shared libraries in the same location for convenience of
# direct execution from common spot and for convenience in environments without
# RPATH support.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
message(STATUS "GEOS: Run-time output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "GEOS: Archives output: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
file(READ Version.txt _version_txt)

string(REGEX MATCH "GEOS_VERSION_MAJOR=([0-9]+)" _ ${_version_txt})
set(_version_major ${CMAKE_MATCH_1})
string(REGEX MATCH "GEOS_VERSION_MINOR=([0-9]+)" _ ${_version_txt})
set(_version_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "GEOS_VERSION_PATCH=([0-9]+)" _ ${_version_txt})
set(_version_patch ${CMAKE_MATCH_1})
# OPTIONS: "", "dev", "rc1" etc.
string(REGEX MATCH "GEOS_PATCH_WORD=([a-zA-Z0-9]+)" _ ${_version_txt})
set(_version_patch_word ${CMAKE_MATCH_1})

# Version of JTS this release is bound to
string(REGEX MATCH "JTS_PORT=([0-9a-zA-Z\.]+)" _ ${_version_txt})
set(JTS_PORT ${CMAKE_MATCH_1})

# Version of public C API
string(REGEX MATCH "CAPI_INTERFACE_CURRENT=([0-9]+)" _ ${_version_txt})
set(_version_capi_current ${CMAKE_MATCH_1})
string(REGEX MATCH "CAPI_INTERFACE_REVISION=([0-9]+)" _ ${_version_txt})
set(_version_capi_revision ${CMAKE_MATCH_1})
string(REGEX MATCH "CAPI_INTERFACE_AGE=([0-9]+)" _ ${_version_txt})
set(_version_capi_age ${CMAKE_MATCH_1})

unset(_version_txt)

math(EXPR _version_capi_major "${_version_capi_current} - ${_version_capi_age}")
set(CAPI_VERSION_MAJOR ${_version_capi_major})
set(CAPI_VERSION_MINOR ${_version_capi_age})
set(CAPI_VERSION_PATCH ${_version_capi_revision})
set(CAPI_VERSION
    "${_version_capi_major}.${_version_capi_age}.${_version_capi_revision}")

unset(_version_capi_current)
unset(_version_capi_major)
unset(_version_capi_age)
unset(_version_capi_revision)

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
  list(APPEND _project_info DESCRIPTION
       "GEOS - C++ port of the Java Topology Suite (JTS)")
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  list(APPEND _project_info HOMEPAGE_URL "http://geos.osgeo.org")
endif()

project(
  GEOS
  VERSION "${_version_major}.${_version_minor}.${_version_patch}"
  LANGUAGES C CXX ${_project_info})

if(NOT "${_version_patch_word}" STREQUAL "")
  # Re-write VERSION variables after project()
  set(GEOS_VERSION "${GEOS_VERSION}${_version_patch_word}")
  set(GEOS_VERSION_PATCH "${_version_patch}${_version_patch_word}")
endif()
set(GEOS_VERSION_NOPATCH
    "${_version_major}.${_version_minor}.${_version_patch}")

unset(_version_major)
unset(_version_minor)
unset(_version_patch)
unset(_version_patch_word)

message(STATUS "GEOS: Version ${GEOS_VERSION}")
message(STATUS "GEOS: C API Version ${CAPI_VERSION}")
message(STATUS "GEOS: JTS port ${JTS_PORT}")

# -----------------------------------------------------------------------------
# Install directories
# -----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# -----------------------------------------------------------------------------
# C++ language version and compilation flags
# -----------------------------------------------------------------------------
message(STATUS "GEOS: Require C++${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Target geos_cxx_flags: common compilation flags
# -----------------------------------------------------------------------------
add_library(geos_cxx_flags INTERFACE)
target_compile_features(geos_cxx_flags INTERFACE cxx_std_11)

# -----------------------------------------------------------------------------
# Add flags to prevent 'fused multiply-add' operations on targets (ARM64) that
# allow it, as it breaks calculations in DD.cpp. TODO: Replace DD calculations
# with 'long float' where target supports true long float, and remove other
# cases where FMA causes regression failures.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98207
# -----------------------------------------------------------------------------

target_compile_options(
  geos_cxx_flags
  INTERFACE
    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-ffp-contract=off>"
    "$<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>"
    "$<$<CXX_COMPILER_ID:MSVC>:/fp:precise>")

# Use -ffloat-store for 32-bit builds (needed to make some tests pass)
target_compile_options(
  geos_cxx_flags
  INTERFACE
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<EQUAL:4,${CMAKE_SIZEOF_VOID_P}>>:-ffloat-store>
)

# -----------------------------------------------------------------------------
# Target geos_cxx_flags: common compilation flags
# -----------------------------------------------------------------------------
option(DISABLE_GEOS_INLINE "Disable inlining" OFF)
if(NOT DISABLE_GEOS_INLINE)
  target_compile_definitions(geos_cxx_flags INTERFACE GEOS_INLINE)
  message(STATUS "GEOS: Function inlining ENABLED")
else()
  message(STATUS "GEOS: Function inlining DISABLED")
endif()

# Make sure NDEBUG is defined so that assert() is disabled for any non-debug
# build. Use a generator expression so that this works with multi-configuration
# generators.
target_compile_definitions(geos_cxx_flags
                           INTERFACE $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)

# -----------------------------------------------------------------------------
# Target geos_cxx_flags: overlayng code
# -----------------------------------------------------------------------------
# option(DISABLE_OVERLAYNG "Disable overlayng algorithms" OFF)
# if(DISABLE_OVERLAYNG) target_compile_definitions(geos_cxx_flags INTERFACE
# DISABLE_OVERLAYNG) message(STATUS "GEOS: OverlayNG DISABLED") else()
# message(STATUS "GEOS: OverlayNG ENABLED") endif()

# -----------------------------------------------------------------------------
# Target geos_developer_cxx_flags: developer mode compilation flags
# -----------------------------------------------------------------------------
# Do NOT install this target for end-users!
add_library(geos_developer_cxx_flags INTERFACE)

if(GEOS_BUILD_DEVELOPER)
  message(STATUS "GEOS: Developer mode ENABLED")
endif()

# geos_cxx_flags inherits properties from geos_developer_cxx_flags when building
# as part of the GEOS repository or on explicit request for developer
# compilation mode, as GEOS contributor. The flags are intended only for GEOS
# itself and are not part of usage requirements needed by GEOS consumers.
if(GEOS_BUILD_DEVELOPER)
  target_link_libraries(geos_cxx_flags
                        INTERFACE $<BUILD_INTERFACE:geos_developer_cxx_flags>)
endif()

target_compile_definitions(geos_cxx_flags INTERFACE USE_UNSTABLE_GEOS_CPP_API)

target_compile_definitions(
  geos_developer_cxx_flags
  INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_DEPRECATE>
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_DEPRECATE>
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>)

target_compile_options(
  geos_developer_cxx_flags
  INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:-W4>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Werror
    -pedantic
    -Wall
    -Wextra
    -Wno-long-long
    -Wcast-align
    -Wconversion
    -Wchar-subscripts
    -Wdouble-promotion
    -Wpointer-arith
    -Wformat
    -Wformat-security
    -Wshadow
    -Wuninitialized
    -Wunused-parameter
    -fno-common>
    $<$<CXX_COMPILER_ID:GNU>:-fno-implicit-inline-templates
    -Wno-psabi>
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-unknown-warning-option>
)

# -----------------------------------------------------------------------------
# Define a coverage build
# -----------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS_COVERAGE "-fprofile-arcs -ftest-coverage")

# -----------------------------------------------------------------------------
# Extra libraries
# -----------------------------------------------------------------------------
include(CheckLibraryExists)
check_library_exists(m pow "" HAVE_LIBM)

# -----------------------------------------------------------------------------
# Target geos: C++ API library
# -----------------------------------------------------------------------------
add_library(geos "")
target_link_libraries(
  geos
  PUBLIC geos_cxx_flags
  PRIVATE $<BUILD_INTERFACE:ryu>)
# ryu is an object library, nothing is actually being linked here. The
# BUILD_INTERFACE switch was necessary to build on AppVeyor (CMake 3.16.2) but
# not locally (CMake 3.16.3)
add_subdirectory(include)
add_subdirectory(src)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(
    geos PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,GEOS_DLL_EXPORT,DLL_EXPORT>)

  set_target_properties(geos PROPERTIES VERSION ${GEOS_VERSION_NOPATCH})
  set_target_properties(geos PROPERTIES SOVERSION ${GEOS_VERSION_NOPATCH})
endif()

# -----------------------------------------------------------------------------
# Target geos_c: C API library
# -----------------------------------------------------------------------------
add_library(geos_c "")
target_link_libraries(
  geos_c
  PRIVATE geos)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(
    geos_c PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,GEOS_DLL_EXPORT,DLL_EXPORT>)

  set_target_properties(geos_c PROPERTIES VERSION ${CAPI_VERSION})
  if(NOT WIN32)
    set_target_properties(geos_c PROPERTIES SOVERSION ${CAPI_VERSION_MAJOR})
  endif()
endif()

add_subdirectory(capi)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
add_subdirectory(util)

# -----------------------------------------------------------------------------
# Documentation/Examples
# -----------------------------------------------------------------------------
add_subdirectory(doc)

# -----------------------------------------------------------------------------
# Install and export targets - support 'make install' or equivalent
# -----------------------------------------------------------------------------

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/geos-config-version.cmake"
  VERSION ${GEOS_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_file(cmake/geos-config.cmake
               "${CMAKE_CURRENT_BINARY_DIR}/geos-config.cmake" COPYONLY)

install(
  TARGETS geos geos_cxx_flags
  EXPORT geos-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  TARGETS geos_c
  EXPORT geos-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT geos-targets
  FILE geos-targets.cmake
  NAMESPACE GEOS::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GEOS)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/geos-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/geos-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GEOS)
install(
  DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/geos"
            "${CMAKE_CURRENT_BINARY_DIR}/include/geos"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h")
install(
  DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/geos"
            "${CMAKE_CURRENT_BINARY_DIR}/include/geos"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp")
if(NOT DISABLE_GEOS_INLINE)
  install(
    DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/geos"
              "${CMAKE_CURRENT_BINARY_DIR}/include/geos"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.inl")
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/capi/geos_c.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tools)

# -----------------------------------------------------------------------------
# Uninstall
# -----------------------------------------------------------------------------

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${PROJECT_BINARY_DIR}/cmake/cmake_uninstall.cmake" IMMEDIATE @ONLY)

add_custom_target(uninstall "${CMAKE_COMMAND}" -P
                            "${PROJECT_BINARY_DIR}/cmake/cmake_uninstall.cmake")

# -----------------------------------------------------------------------------
# "make dist" workalike
# -----------------------------------------------------------------------------
get_property(_is_multi_config_generator GLOBAL
             PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config_generator)
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GEOS Computational Geometry Library")
  set(CPACK_PACKAGE_VENDOR "OSGeo")
  set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
  set(CPACK_SOURCE_GENERATOR "TBZ2")
  set(CPACK_PACKAGE_VERSION_MAJOR ${GEOS_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${GEOS_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${GEOS_VERSION_PATCH})
  set(CPACK_SOURCE_PACKAGE_FILE_NAME "geos-${GEOS_VERSION}")

  set(CPACK_SOURCE_IGNORE_FILES
      "/\\\\.git"
      "/autogen\\\\.sh"
      "/tools/ci"
      "/HOWTO_RELEASE"
      "/autom4te\\\\.cache"
      "\\\\.yml\$"
      "\\\\.deps"
      "/debian/"
      "/php/"
      "/.*build-.*/"
      "cmake_install\\\\.cmake"
      "/include/geos/version\\\\.h\$"
      "/tools/geos-config\$"
      "/tools/geos\\\\.pc\$"
      ${PROJECT_BINARY_DIR})

  # message(STATUS "GEOS: CPACK_SOURCE_PACKAGE_FILE_NAME:
  # ${CPACK_SOURCE_PACKAGE_FILE_NAME}") message(STATUS "GEOS:
  # CPACK_SOURCE_IGNORE_FILES: ${CPACK_SOURCE_IGNORE_FILES}") message(STATUS
  # "GEOS: CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
  include(CPack)
  add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)

  message(STATUS "GEOS: Configured 'dist' target")
endif()

unset(_is_multi_config_generator)
