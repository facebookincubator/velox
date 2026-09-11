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
velox_add_library(velox_duckdb_conversion DuckConversion.cpp HEADERS DuckConversion.h)

velox_link_libraries(velox_duckdb_conversion velox_core velox_vector duckdb_static)

velox_add_library(velox_duckdb_parser DuckParser.cpp HEADERS DuckParser.h)

velox_link_libraries(
  velox_duckdb_parser
  velox_duckdb_conversion
  velox_parse_expression
  duckdb_static
)

# The vendored DuckDB headers do not compile as C++23 (their inline
# SelectStatement constructor instantiates unique_ptr<QueryNode> with an
# incomplete type, which C++23's constexpr ~unique_ptr rejects). These sources
# include <duckdb.hpp>, so build them as C++20. The trailing -std overrides the
# C++23 flag. velox_add_library may alias these targets to the single velox
# mono-library defined in another directory, so the source property must be set
# in that target's directory scope (TARGET_DIRECTORY).
get_target_property(_velox_duckdb_conversion_tgt velox_duckdb_conversion ALIASED_TARGET)
if(NOT _velox_duckdb_conversion_tgt)
  set(_velox_duckdb_conversion_tgt velox_duckdb_conversion)
endif()
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/DuckConversion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DuckParser.cpp
  TARGET_DIRECTORY ${_velox_duckdb_conversion_tgt}
  PROPERTIES COMPILE_OPTIONS -std=gnu++20
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
