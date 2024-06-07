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

add_library(velox_functions_remote_server RemoteFunctionService.cpp)

target_link_libraries(
  velox_functions_remote_server
  PUBLIC remote_function_thrift velox_functions_remote_get_serde
         velox_type_fbhive velox_memory)

add_executable(velox_functions_remote_server_main RemoteFunctionServiceMain.cpp)

target_link_libraries(
  velox_functions_remote_server_main velox_functions_remote_server
  velox_functions_prestosql)

add_library(velox_functions_remote_server_rest RemoteFunctionRestService.cpp)
target_link_libraries(
  velox_functions_remote_server_rest
  ${PROXYGEN_LIBRARIES}
  velox_functions_remote_get_serde
  velox_type_fbhive
  velox_memory
  velox_functions_prestosql)

add_executable(velox_functions_remote_server_rest_main
               RemoteFunctionServiceRestMain.cpp)

target_link_libraries(
  velox_functions_remote_server_rest_main velox_functions_remote_server_rest
  velox_functions_prestosql)
