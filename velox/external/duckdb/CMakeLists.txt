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
add_compile_definitions(DISABLE_DUCKDB_REMOTE_INSTALL)

add_library(
  duckdb
  duckdb-1.cpp
  duckdb-2.cpp
  duckdb-3.cpp
  duckdb-4.cpp
  duckdb-5.cpp
  duckdb-6.cpp
  duckdb-7.cpp
  duckdb-8.cpp
  duckdb-fastpforlib.cpp
  duckdb-fmt.cpp
  duckdb-hyperloglog.cpp
  duckdb-libpg_query.cpp
  duckdb-miniz.cpp
  duckdb-re2.cpp
  duckdb-utf8proc.cpp)

install(TARGETS duckdb  EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
set_property(TARGET duckdb PROPERTY JOB_POOL_COMPILE high_memory_pool)

# Suppress warnings when compiling duckdb.
target_compile_options(duckdb PRIVATE -w)
# 安装头文件
install(DIRECTORY ${CMAKE_SOURCE_DIR}/velox/external/duckdb
        DESTINATION include/velox/external/
        FILES_MATCHING PATTERN "*.hpp")
