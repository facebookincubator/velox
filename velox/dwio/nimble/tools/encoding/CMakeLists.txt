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
add_library(nimble_encoding_evaluation EncodingEvaluation.cpp EncodingEvaluation.h)

target_link_libraries(
  nimble_encoding_evaluation
  nimble_common
  nimble_encodings
  nimble_writer
  nimble_serializer
  nimble_deserializer
  nimble_serializer_options
  nimble_velox_schema_serialization
  velox_memory
  velox_time
  velox_vector
  Folly::folly
  fmt::fmt
)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
