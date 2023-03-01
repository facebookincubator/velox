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

# Download and update substrait proto files
FetchContent_Declare(
  substrait
  GIT_REPOSITORY https://github.com/substrait-io/substrait.git
  GIT_TAG 6a74f40f0d835a447a7175b20dd430c7c39c5d11
  PATCH_COMMAND git apply
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/substrait/substrait.patch)
FetchContent_MakeAvailable(substrait)

file(COPY ${substrait_SOURCE_DIR}/proto/substrait
     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/proto/)

# Set up build vars
set(PROTO_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto/substrait)
set(PROTO_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto/substrait")
file(GLOB PROTO_FILES ${PROTO_INPUT_DIR}/*.proto
     ${PROTO_INPUT_DIR}/extensions/*.proto)

foreach(PROTO ${PROTO_FILES})
  file(RELATIVE_PATH REL_PROTO ${PROTO_INPUT_DIR} ${PROTO})
  string(REGEX REPLACE "\\.proto" "" PROTO_NAME ${REL_PROTO})
  list(APPEND PROTO_SRCS "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.cc")
  list(APPEND PROTO_HDRS "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.h")
endforeach()
set(PROTO_OUTPUT_FILES ${PROTO_HDRS} ${PROTO_SRCS})
set_source_files_properties(${PROTO_OUTPUT_FILES} PROPERTIES GENERATED TRUE)
get_filename_component(PROTO_DIR ${PROTO_INPUT_DIR}/, DIRECTORY)

# Generate substrait cpp headers
add_custom_command(
  OUTPUT ${PROTO_OUTPUT_FILES}
  COMMAND
    ${Protobuf_PROTOC_EXECUTABLE} --proto_path ${PROJECT_SOURCE_DIR}/
    --proto_path ${Protobuf_INCLUDE_DIRS} --cpp_out ${PROJECT_SOURCE_DIR}
    ${PROTO_FILES}
  DEPENDS ${PROTO_DIR}
  COMMENT "Running PROTO compiler"
  VERBATIM)
add_custom_target(substrait_proto ALL DEPENDS ${PROTO_OUTPUT_FILES})
add_dependencies(substrait_proto protobuf::libprotobuf)

set(SRCS
    ${PROTO_SRCS}
    SubstraitParser.cpp
    SubstraitToVeloxExpr.cpp
    SubstraitToVeloxPlan.cpp
    TypeUtils.cpp
    SubstraitExtensionCollector.cpp
    VeloxToSubstraitExpr.cpp
    VeloxToSubstraitPlan.cpp
    VeloxToSubstraitType.cpp
    VeloxSubstraitSignature.cpp
    VariantToVectorConverter.cpp)

add_library(velox_substrait_plan_converter ${SRCS})
target_include_directories(velox_substrait_plan_converter
                           PUBLIC ${PROTO_OUTPUT_DIR})
target_link_libraries(velox_substrait_plan_converter velox_connector
                      velox_dwio_dwrf_common)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
