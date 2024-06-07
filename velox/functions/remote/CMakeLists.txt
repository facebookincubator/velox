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

if(NOT DEFINED PROXYGEN_LIBRARIES)
  find_package(Sodium REQUIRED)

  find_library(PROXYGEN proxygen)
  find_library(PROXYGEN_HTTP_SERVER proxygenhttpserver)
  find_library(FIZZ fizz)
  find_library(WANGLE wangle)

  if(NOT PROXYGEN
     OR NOT PROXYGEN_HTTP_SERVER
     OR NOT FIZZ
     OR NOT WANGLE)
    message(
      FATAL_ERROR
        "One or more proxygen libraries were not found. Please ensure proxygen, proxygenhttpserver, fizz, and wangle are installed."
    )
  endif()

  set(PROXYGEN_LIBRARIES ${PROXYGEN_HTTP_SERVER} ${PROXYGEN} ${WANGLE} ${FIZZ})
endif()

add_subdirectory(if)
add_subdirectory(client)
add_subdirectory(server)
