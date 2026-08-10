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

add_library(nimble_column_stats ColumnStatistics.cpp VectorizedStatistics.cpp)
target_link_libraries(
  nimble_column_stats
  nimble_common
  nimble_encodings
  velox_dwio_common
)

add_library(nimble_column_stats_utils ColumnStatsUtils.cpp)
target_link_libraries(
  nimble_column_stats_utils
  nimble_velox_schema
  nimble_velox_schema_builder
  raw_size_utils
  velox_vector
  velox_dwio_common
)

add_subdirectory(tests)
