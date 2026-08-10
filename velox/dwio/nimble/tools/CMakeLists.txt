# Copyright (c) Meta Platforms, Inc. and affiliates.
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
add_library(nimble_tools_common EncodingUtilities.cpp)

target_link_libraries(
  nimble_tools_common
  nimble_common
  nimble_tablet_reader
  nimble_column_stats
)

add_library(nimble_dump_lib NimbleDumpLib.cpp)
target_link_libraries(
  nimble_dump_lib
  nimble_tools_common
  nimble_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_tablet_common
  nimble_tablet_reader
  nimble_velox_reader
  nimble_writer
  nimble_velox_stats_fb
  velox_file
  Folly::folly
)

add_library(nimble_dsl_lib NimbleDslLib.cpp)
target_link_libraries(
  nimble_dsl_lib
  nimble_dump_lib
  nimble_tools_common
  nimble_common
  nimble_tablet_reader
  nimble_velox_reader
  nimble_velox_schema_reader
  nimble_column_stats
  velox_file
  velox_memory
)

add_subdirectory(tests)
