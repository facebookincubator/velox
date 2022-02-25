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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
set(SRCS consul/base64.cpp consul/bridge.cpp consul/discovery.cpp
         consul/register.cpp consul/requests.cpp consul/easy_curl.cpp)

add_library(consul ${SRCS})
target_include_directories(consul PUBLIC consul)
target_include_directories(consul PRIVATE minizip)

# CURL
find_package(CURL REQUIRED)
target_include_directories(consul PRIVATE ${CURL_INCLUDE_DIRS})
target_link_libraries(consul ${CURL_LIBRARIES})
