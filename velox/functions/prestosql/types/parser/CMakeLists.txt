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

if(BISON_FOUND AND FLEX_FOUND)
  bison_target(
    TypeParser
    TypeParser.yy
    ${CMAKE_CURRENT_BINARY_DIR}/TypeParser.yy.cc
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/TypeParser.yy.h
    COMPILE_FLAGS "-Werror -Wno-deprecated"
  )

  flex_target(
    TypeParserScanner
    TypeParser.ll
    ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp
    COMPILE_FLAGS "--prefix=veloxprestotp"
  )

  add_flex_bison_dependency(TypeParserScanner TypeParser)
  set(_PARSER_SRCS ${BISON_TypeParser_OUTPUTS} ${FLEX_TypeParserScanner_OUTPUTS})

  if(VELOX_MONO_LIBRARY)
    add_custom_target(
      velox_presto_type_parser_gen_src
      DEPENDS ${BISON_TypeParser_OUTPUTS} ${FLEX_TypeParserScanner_OUTPUTS}
    )
    add_dependencies(velox velox_presto_type_parser_gen_src)
  endif()
else()
  message(STATUS "Bison/Flex not found - using pre-generated parser files for presto_type_parser")
  set(_PARSER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/TypeParser.yy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/Scanner.cpp
  )
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/generated)
endif()

velox_add_library(
  velox_presto_type_parser
  ${_PARSER_SRCS}
  ParserUtil.cpp
  HEADERS
  ParserUtil.h
  Scanner.h
  TypeParser.h
)
velox_include_directories(velox_presto_type_parser PRIVATE ${PROJECT_BINARY_DIR})
if(FLEX_FOUND)
  velox_include_directories(velox_presto_type_parser PRIVATE ${FLEX_INCLUDE_DIRS})
endif()
velox_link_libraries(velox_presto_type_parser velox_common_base)

if(WIN32)
  velox_compile_definitions(velox_presto_type_parser PRIVATE YY_NO_UNISTD_H)
endif()
