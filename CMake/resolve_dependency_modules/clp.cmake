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
  GIT_TAG 0798100389bd5231b520ec48ab186275795e3790)

set(CLP_BUILD_CLP_REGEX_UTILS
    OFF
    CACHE BOOL "Build CLP regex utils")
set(CLP_BUILD_CLP_S_JSONCONSTRUCTOR
    OFF
    CACHE BOOL "Build CLP-S JSON constructor")
set(CLP_BUILD_CLP_S_REDUCER_DEPENDENCIES
    OFF
    CACHE BOOL "Build CLP-S reducer dependencies")
set(CLP_BUILD_CLP_S_SEARCH_SQL
    OFF
    CACHE BOOL "Build CLP-S search SQL")
set(CLP_BUILD_EXECUTABLES
    OFF
    CACHE BOOL "Build CLP executables")
set(CLP_BUILD_TESTING
    OFF
    CACHE BOOL "Build CLP tests")

FetchContent_Populate(clp)

list(APPEND CMAKE_MODULE_PATH "${clp_SOURCE_DIR}/components/core/cmake/Modules")
add_subdirectory(${clp_SOURCE_DIR}/components/core
                 ${clp_BINARY_DIR}/components/core)
