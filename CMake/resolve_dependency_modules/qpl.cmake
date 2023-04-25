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

if(DEFINED ENV{VELOX_QPL_URL})
  set(VELOX_QPL_SOURCE_URL "$ENV{VELOX_QPL_URL}")
else()
  set(VELOX_QPL_VERSION 1.1.0)
  string(CONCAT VELOX_QPL_SOURCE_GIT "https://github.com/intel/qpl.git")
  set(GIT_QPL_TAG "develop")
endif()

message(STATUS "Building qpl from source")
FetchContent_Declare(
  qpl
  GIT_REPOSITORY ${VELOX_QPL_SOURCE_GIT}
  GIT_TAG ${GIT_QPL_TAG})
set(QPL_INSTALL ON)
set(QPL_BUILD_TESTS OFF)
set(QPL_BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(qpl)
