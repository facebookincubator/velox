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
add_library(velox_flag_definitions OBJECT flags.cpp)
target_link_libraries(velox_flag_definitions PRIVATE gflags::gflags)

# Find include directories for gflags
set(GFLAGS_INCLUDE_DIRS "C:/Program Files (x86)/gflags/include")
#message(FATAL_ERROR "${GFLAGS_INCLUDE_DIRS}")
# Check if GFLAGS_INCLUDE_DIRS is found


if(GFLAGS_INCLUDE_DIRS)
    target_include_directories(velox_encode PRIVATE ${GFLAGS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "gflags include directories not found. Set GFLAGS_INCLUDE_DIRS manually...")
endif()
