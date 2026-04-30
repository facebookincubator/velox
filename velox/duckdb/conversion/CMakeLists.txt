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

set(VELOX_DUCKDB_LIBRARIES duckdb_static)
if(TARGET duckdb_generated_extension_loader)
  list(APPEND VELOX_DUCKDB_LIBRARIES duckdb_generated_extension_loader)
  if(TARGET core_functions_extension)
    list(APPEND VELOX_DUCKDB_LIBRARIES core_functions_extension)
  endif()
  if(TARGET parquet_extension)
    list(APPEND VELOX_DUCKDB_LIBRARIES parquet_extension)
  endif()
elseif(TARGET dummy_static_extension_loader)
  list(APPEND VELOX_DUCKDB_LIBRARIES dummy_static_extension_loader)
endif()
list(APPEND VELOX_DUCKDB_LIBRARIES duckdb_static)

velox_link_libraries(velox_duckdb_conversion velox_core velox_vector ${VELOX_DUCKDB_LIBRARIES})

velox_add_library(velox_duckdb_parser DuckParser.cpp HEADERS DuckParser.h)

velox_link_libraries(
  velox_duckdb_parser
  velox_duckdb_conversion
  velox_parse_expression
  ${VELOX_DUCKDB_LIBRARIES}
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
