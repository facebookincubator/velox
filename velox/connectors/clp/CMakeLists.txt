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
#set(CLP_SRC_DIR ${clp_SOURCE_DIR}/components/core/src)
add_subdirectory(search_lib)

velox_add_library(
  velox_clp_connector
  OBJECT
  ClpConfig.cpp
  ClpConnector.cpp
  ClpDataSource.cpp
  ClpTableHandle.cpp)

velox_link_libraries(velox_clp_connector
                     PRIVATE clp-s-search simdjson::simdjson velox_connector)
target_compile_features(velox_clp_connector PRIVATE cxx_std_20)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
