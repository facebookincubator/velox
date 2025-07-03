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

add_library(velox_functions_aggregates_noisy_sketch BitMap.cpp)

target_link_libraries(
  velox_functions_aggregates_noisy_sketch
  velox_common_base
  velox_memory)

target_include_directories(
  velox_functions_aggregates_noisy_sketch
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
