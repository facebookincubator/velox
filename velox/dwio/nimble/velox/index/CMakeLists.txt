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
add_library(nimble_index_projector NimbleIndexProjector.cpp)
target_link_libraries(
  nimble_index_projector
  nimble_data_input
  nimble_index
  nimble_stream_data_writer
  nimble_tablet_reader
  nimble_tablet_reader_cache
  nimble_velox_common
  velox_caching
  velox_time
  velox_key_encoder
  velox_type
  Folly::folly
)

add_subdirectory(tests)
