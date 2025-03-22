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

add_library(velox_tpcds_connector OBJECT TpcdsConnector.cpp)
target_compile_options(velox_tpcds_connector PRIVATE -w)
target_link_libraries(
  velox_tpcds_connector velox_connector velox_tpcds_gen fmt::fmt)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-deprecated-declarations -Wno-writable-strings
                      -Wno-missing-field-initializers)
endif()

# This stringop-overflow warning seems to have lots of false positives and has
# been the source of a lot of compiler bug reports (e.g.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99578), which causes
# parquet-amalgamation.cpp to fail to compile. For now, we disable this warning
# on the affected compiler (GCC).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wno-stringop-overflow -Wno-write-strings)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
