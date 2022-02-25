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

set(SRCS ../infsec/client/InfSecClient.cpp ../infsec/client/DaemonClient.cpp
         ../infsec/client/util.cpp)

add_library(infsec STATIC ${SRCS})
target_include_directories(infsec PRIVATE ../infsec/client)

find_package(RapidJSON REQUIRED)
target_include_directories(infsec PRIVATE ${RapidJSON_INCLUDE_DIR})
target_link_libraries(infsec ${RapidJSON_LIBRARY})
