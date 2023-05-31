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
add_library(velox_functions_spark_windows
            RowNumber.cpp Register.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_compile_options(-Wno-stringop-overflow)
endif()

target_link_libraries(velox_functions_spark_windows velox_buffer velox_exec
                      ${FOLLY_WITH_DEPENDENCIES})