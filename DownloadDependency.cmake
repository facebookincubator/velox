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

# MODULE:   DownloadDependency
#
# PROVIDES:
#   download_dependency( PROJ projectName
#                    [PREFIX prefixDir]
#                    [DOWNLOAD_DIR downloadDir]
#                    [SOURCE_DIR srcDir]
#                    [BINARY_DIR binDir]
#                    [QUIET]
#                    ...
#   )
#
#       Provides the ability to download and unpack a tarball, zip file, git repository,
#       etc. at configure time (i.e. when the cmake command is run). How the downloaded
#       and unpacked contents are used is up to the caller, but the motivating case is
#       to download source code which can then be included directly in the build with
#       add_subdirectory() after the call to download_dependency(). Source and build
#       directories are set up with this in mind.
#
#       The PROJ argument is required. The projectName value will be used to construct
#       the following variables upon exit (obviously replace projectName with its actual
#       value):
#
#           projectName_SOURCE_DIR
#           projectName_BINARY_DIR
#
#       The SOURCE_DIR and BINARY_DIR arguments are optional and would not typically
#       need to be provided. They can be specified if you want the downloaded source
#       and build directories to be located in a specific place. The contents of
#       projectName_SOURCE_DIR and projectName_BINARY_DIR will be populated with the
#       locations used whether you provide SOURCE_DIR/BINARY_DIR or not.
#
#       The DOWNLOAD_DIR argument does not normally need to be set. It controls the
#       location of the temporary CMake build used to perform the download.
#
#       The PREFIX argument can be provided to change the base location of the default
#       values of DOWNLOAD_DIR, SOURCE_DIR and BINARY_DIR. If all of those three arguments
#       are provided, then PREFIX will have no effect. The default value for PREFIX is
#       CMAKE_BINARY_DIR.
#
#       In addition to the above, any other options are passed through unmodified to
#       ExternalProject_Add() to perform the actual download, patch and update steps.
#       The following ExternalProject_Add() options are explicitly prohibited (they
#       are reserved for use by the download_dependency() command):
#
#           CONFIGURE_COMMAND
#           BUILD_COMMAND
#           INSTALL_COMMAND
#           TEST_COMMAND
#
#       Only those ExternalProject_Add() arguments which relate to downloading, patching
#       and updating of the project sources are intended to be used. Also note that at
#       least one set of download-related arguments are required.
#
# EXAMPLE USAGE:
#
#   include(DownloadDependency)
#   download_dependency(PROJ                googletest
#                    GIT_REPOSITORY      https://github.com/google/googletest.git
#                    GIT_TAG             master
#   )
#
#   add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
#
#========================================================================================


set(_DownloadDependencyDir "${CMAKE_CURRENT_LIST_DIR}")

include(CMakeParseArguments)

function(download_dependency)

    set(options USE_CMAKE_BUILD)
    set(oneValueArgs
            PROJ
            PREFIX
            DOWNLOAD_DIR
            SOURCE_DIR
            BINARY_DIR
            # Prevent the following from being passed through
            TEST_COMMAND
            )
    set(multiValueArgs "")

    cmake_parse_arguments(DD_ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Downloading/updating ${DD_ARGS_PROJ}")

    # Set up where we will put our temporary CMakeLists.txt file and also
    # the base point below which the default source and binary dirs will be.
    # The prefix must always be an absolute path.
    if (NOT DD_ARGS_PREFIX)
        set(DD_ARGS_PREFIX "${CMAKE_BINARY_DIR}")
    else()
        get_filename_component(DD_ARGS_PREFIX "${DD_ARGS_PREFIX}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    if (NOT DD_ARGS_DOWNLOAD_DIR)
        set(DD_ARGS_DOWNLOAD_DIR "${DD_ARGS_PREFIX}/${DD_ARGS_PROJ}-download")
    endif()

    # Ensure the caller can know where to find the source and build directories
    if (NOT DD_ARGS_SOURCE_DIR)
        set(DD_ARGS_SOURCE_DIR "${DD_ARGS_PREFIX}/${DD_ARGS_PROJ}-src")
    endif()
    if (NOT DD_ARGS_BINARY_DIR)
        set(DD_ARGS_BINARY_DIR "${DD_ARGS_PREFIX}/${DD_ARGS_PROJ}-build")
    endif()
    set(${DD_ARGS_PROJ}_SOURCE_DIR "${DD_ARGS_SOURCE_DIR}" PARENT_SCOPE)
    set(${DD_ARGS_PROJ}_BINARY_DIR "${DD_ARGS_BINARY_DIR}" PARENT_SCOPE)

    # The way that CLion manages multiple configurations, it causes a copy of
    # the CMakeCache.txt to be copied across due to it not expecting there to
    # be a project within a project.  This causes the hard-coded paths in the
    # cache to be copied and builds to fail.  To mitigate this, we simply
    # remove the cache if it exists before we configure the new project.  It
    # is safe to do so because it will be re-generated.  Since this is only
    # executed at the configure step, it should not cause additional builds or
    # downloads.
    file(REMOVE "${DD_ARGS_DOWNLOAD_DIR}/CMakeCache.txt")

    # Create and build a separate CMake project to carry out the download.
    # If we've already previously done these steps, they will not cause
    # anything to be updated, so extra rebuilds of the project won't occur.
    # Make sure to pass through CMAKE_MAKE_PROGRAM in case the main project
    # has this set to something not findable on the PATH.
    if (DD_ARGS_USE_CMAKE_BUILD)
        message("Got use cmake build")
     if (DD_ARGS_BUILD_COMMAND OR DD_ARGS_CONFIGURE_COMMAND)
         message(FATAL "Cant have build/configure command with USE_CMAKE_BUILD set!")
     endif()
        # Unset build/configure for external
        set(CONFIGURE_COMMAND "")
        set(BUILD_COMMAND "")
        unset(USE_CMAKE_BUILD)
    endif()

    configure_file("${_DownloadDependencyDir}/DownloadDependency.CMakeLists.cmake.in"
            "${DD_ARGS_DOWNLOAD_DIR}/CMakeLists.txt")
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"
            -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}"
            .
            RESULT_VARIABLE result
            WORKING_DIRECTORY "${DD_ARGS_DOWNLOAD_DIR}"
            )
    if(result)
        message(FATAL_ERROR "CMake step for ${DD_ARGS_PROJ} failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY "${DD_ARGS_DOWNLOAD_DIR}"
            )
    if(result)
        message(FATAL_ERROR "Build step for ${DD_ARGS_PROJ} failed: ${result}")
    endif()

endfunction()