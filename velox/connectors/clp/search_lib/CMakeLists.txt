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
velox_add_library(
  clp-s-search
  STATIC
  ClpCursor.cpp
  ClpCursor.h
  ClpQueryRunner.cpp
  ClpQueryRunner.h
  ClpVectorLoader.cpp
  ClpVectorLoader.h)

velox_link_libraries(
  clp-s-search
  PUBLIC clp_s::archive_reader
  PRIVATE
    clp_s::archive_writer
    clp_s::clp_dependencies
    clp_s::io
    clp_s::search
    clp_s::search::kql
    velox_vector)
target_compile_features(clp-s-search PRIVATE cxx_std_20)
