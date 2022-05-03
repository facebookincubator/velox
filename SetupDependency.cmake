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

# MODULE:   SetupDependency
#
# PROVIDES:
#   setup_dependency( PROJ projectName
#                    [EXTERNAL_PATH path/to/installed/project]
#                    [MINIMUM_VERSION minimum required version]
#                    [PREFIX prefixDir]
#                    ...
#   )
#
#       Provides the ability to setup any dependency. The users are allowed to pass
#       in an external path to check if the dependency is already available in the system,
#       failing which this function will get and setup the dependency. We support
#       download and unpacking a tarball, zip file, git repository,
#       etc. at configure time (i.e. when the cmake command is run). How the downloaded
#       and unpacked contents are used is up to the caller, but the motivating case is
#       to download source code, build and install it in the build directory, and then
#       use find_package() on the installed dependency to add it to the project.
#
#       The PROJ argument is required. The projectName value will be used to search for
#       the installed dependencies Config file and thus this name should match find_package()
#       standards.
#       EXTERNAL_PATH is an optional argument in which the path of the already installed dependency
#       is present.
#       EXTERNAL_PATH along with MINIMUM_VERSION is used to ensure that the installed dependency meets
#       the minimum requirements. If a dependency is not present or doesnt meet the MINIMUM_VERSION the
#       function fails.
#       If neither EXTERNAL_PATH or MINIMUM_VERSION is set, then we install the required dependency in the
#       build path. The following variables upon exit are set (obviously replace projectName with its actual
#       value):
#
#           projectName_SOURCE_DIR
#           projectName_BINARY_DIR
#           projectName_INSTALL_DIR
#
#       After installation, We will use find_package() using PROJ and install location
#       to load project details.
#
#       The PREFIX argument can be provided to change the base location.
#       The default value for PREFIX is CMAKE_BINARY_DIR.
#
#       In addition to the above, any other options are passed through unmodified to
#       ExternalProject_Add() to perform the actual download, patch and update steps.
#       TEST_COMMAND from ExternalProject_Add() options are is prohibited.
#       Only those ExternalProject_Add() arguments which relate to downloading, patching
#       and updating of the project sources are intended to be used. Also note that at
#       least one set of download-related arguments are required if EXTERNAL_PATH is unset.
#       See ExternalProject_Add() documentation for these arguments.
#
# EXAMPLE USAGE:
#   # Download and setup.
#   include(SetupDependency)
#   setup_dependency(PROJ                GTest
#                    GIT_REPOSITORY      https://github.com/google/googletest.git
#                    GIT_TAG             master
#   )
#
#  # Use an already installed dependency.
#  setup_dependency(PROJ                folly
#                   EXTERNAL_PATH       /usr/local/include
#                   MINIMUM_VERSION    2022.05.02.00
#  )
#
#
#========================================================================================


set(_SetupDependencyDir "${CMAKE_CURRENT_LIST_DIR}")

include(CMakeParseArguments)

function(setup_dependency)

    set(oneValueArgs
            PROJ
            EXTERNAL_PATH
            MINIMUM_VERSION
            PREFIX
            # Prevent the following from being passed through
            TEST_COMMAND
            )
    set(multiValueArgs "")

    cmake_parse_arguments(SD_ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (SD_ARGS_EXTERNAL_PATH)
        message("Finding ${SD_ARGS_PROJ}:${SD_ARGS_MINIMUM_VERSION} in provided path ${SD_ARGS_EXTERNAL_PATH}")
        find_package(${SD_ARGS_PROJ} $SD_ARGS_MINIMUM_VERSION REQUIRED PATHS "${SD_ARGS_EXTERNAL_PATH}" NO_DEFAULT_PATH)
        return()
    endif()


    message(STATUS "Downloading/updating ${SD_ARGS_PROJ}")

    # Set up where we will put our temporary CMakeLists.txt file and also
    # the base point below which the default source and binary dirs will be.
    # The prefix must always be an absolute path.
    if (NOT SD_ARGS_PREFIX)
        set(SD_ARGS_PREFIX "${CMAKE_BINARY_DIR}")
    else()
        get_filename_component(SD_ARGS_PREFIX "${SD_ARGS_PREFIX}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    if (NOT SD_ARGS_DOWNLOAD_DIR)
        set(SD_ARGS_DOWNLOAD_DIR "${SD_ARGS_PREFIX}/${SD_ARGS_PROJ}/download")
    endif()

    # Ensure the caller can know where to find the source and build directories
    if (NOT SD_ARGS_SOURCE_DIR)
        set(SD_ARGS_SOURCE_DIR "${SD_ARGS_PREFIX}/${SD_ARGS_PROJ}/src")
    endif()

    if (NOT SD_ARGS_BINARY_DIR)
        set(SD_ARGS_BINARY_DIR "${SD_ARGS_PREFIX}/${SD_ARGS_PROJ}/build")
    endif()

    if (NOT SD_ARGS_INSTALL_DIR)
        set(SD_ARGS_INSTALL_DIR "${SD_ARGS_BINARY_DIR}/install")
    endif()

    set(${SD_ARGS_PROJ}_SOURCE_DIR "${SD_ARGS_SOURCE_DIR}" PARENT_SCOPE)
    set(${SD_ARGS_PROJ}_BINARY_DIR "${SD_ARGS_BINARY_DIR}" PARENT_SCOPE)
    set(${SD_ARGS_PROJ}_INSTALL_DIR "${SD_ARGS_INSTALL_DIR}" PARENT_SCOPE)

    file(MAKE_DIRECTORY "${SD_ARGS_INSTALL_DIR}")

    # The way that CLion manages multiple configurations, it causes a copy of
    # the CMakeCache.txt to be copied across due to it not expecting there to
    # be a project within a project.  This causes the hard-coded paths in the
    # cache to be copied and builds to fail.  To mitigate this, we simply
    # remove the cache if it exists before we configure the new project.  It
    # is safe to do so because it will be re-generated.  Since this is only
    # executed at the configure step, it should not cause additional builds or
    # downloads.
    file(REMOVE "${SD_ARGS_DOWNLOAD_DIR}/CMakeCache.txt")

    # Create and build a separate CMake project to carry out the download.
    # If we've already previously done these steps, they will not cause
    # anything to be updated, so extra rebuilds of the project won't occur.
    # Make sure to pass through CMAKE_MAKE_PROGRAM in case the main project
    # has this set to something not findable on the PATH.

    configure_file("${_SetupDependencyDir}/SetupDependency.CMakeLists.cmake.in"
            "${SD_ARGS_DOWNLOAD_DIR}/CMakeLists.txt")
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"
            -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}"
            .
            RESULT_VARIABLE result
            WORKING_DIRECTORY "${SD_ARGS_DOWNLOAD_DIR}"
            )
    if(result)
        message(FATAL_ERROR "CMake step for ${SD_ARGS_PROJ} failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY "${SD_ARGS_DOWNLOAD_DIR}"
            )
    if(result)
        message(FATAL_ERROR "Build step for ${SD_ARGS_PROJ} failed: ${result}")
    endif()

    message("Looking for ${SD_ARGS_PROJ} at " ${SD_ARGS_INSTALL_DIR})
    find_package(${SD_ARGS_PROJ} REQUIRED PATHS ${SD_ARGS_INSTALL_DIR} NO_DEFAULT_PATH)

endfunction()