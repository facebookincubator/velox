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

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

find_package(BISON 3.0.4 QUIET)
find_package(FLEX 2.5.13 QUIET)

if(BISON_FOUND AND FLEX_FOUND)
  bison_target(
    SignatureParser
    SignatureParser.yy
    ${CMAKE_CURRENT_BINARY_DIR}/SignatureParser.yy.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/SignatureParser.yy.h
    COMPILE_FLAGS "-Werror -Wno-deprecated"
  )

  flex_target(
    SignatureParserScanner
    SignatureParser.ll
    ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp
    COMPILE_FLAGS "--prefix=veloxsp"
  )

  add_flex_bison_dependency(SignatureParserScanner SignatureParser)

  set(_PARSER_SRCS ${BISON_SignatureParser_OUTPUTS} ${FLEX_SignatureParserScanner_OUTPUTS})

  if(VELOX_MONO_LIBRARY)
    add_custom_target(
      velox_signature_gen_src
      DEPENDS ${BISON_SignatureParser_OUTPUTS} ${FLEX_SignatureParserScanner_OUTPUTS}
    )
    add_dependencies(velox velox_signature_gen_src)
  endif()
else()
  message(STATUS "Bison/Flex not found - using pre-generated parser files for signature_parser")
  set(_PARSER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/SignatureParser.yy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/Scanner.cpp
  )
  # Make generated headers available
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/generated)
endif()

include_directories(${PROJECT_BINARY_DIR})
if(FLEX_FOUND)
  include_directories(${FLEX_INCLUDE_DIRS})
endif()
velox_add_library(
  velox_signature_parser
  ${_PARSER_SRCS}
  ParseUtil.cpp
  HEADERS
  ParseUtil.h
  Scanner.h
  SignatureParser.h
)

# Windows fix: flex-generated scanner tries to include unistd.h which doesn't exist on Windows
if(WIN32)
  velox_compile_definitions(velox_signature_parser PRIVATE YY_NO_UNISTD_H)
endif()

velox_link_libraries(velox_signature_parser velox_common_base velox_type)
