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

file(GLOB PROTO_FILES substrait/*.proto)

protobuf_generate_cpp(VELOX_SUBSTRAIT_PROTO_SRCS VELOX_SUBSTRAIT_PROTO_HDRS
                      ${PROTO_FILES})

message(${VELOX_SUBSTRAIT_PROTO_SRCS})

add_library(velox_substrait_proto ${VELOX_SUBSTRAIT_PROTO_SRCS}
                                  ${VELOX_SUBSTRAIT_PROTO_HDRS})

target_link_libraries(velox_substrait_proto protobuf::libprotobuf)
target_include_directories(velox_substrait_proto
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../../../)
