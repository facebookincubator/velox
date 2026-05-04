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
add_subdirectory(base)
add_subdirectory(caching)
add_subdirectory(compression)
add_subdirectory(config)
add_subdirectory(dynamic_registry)
add_subdirectory(encode)
add_subdirectory(file)
add_subdirectory(future)
add_subdirectory(geospatial)
add_subdirectory(hyperloglog)
add_subdirectory(io)
add_subdirectory(memory)
add_subdirectory(process)
add_subdirectory(rpc)
add_subdirectory(serialization)
add_subdirectory(time)
add_subdirectory(testutil)
add_subdirectory(fuzzer)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

velox_add_library(velox_enum_declare INTERFACE HEADERS EnumDeclare.h)

velox_add_library(velox_enum_define INTERFACE HEADERS EnumDefine.h)

velox_add_library(velox_casts INTERFACE HEADERS Casts.h)

velox_add_library(velox_scoped_registry INTERFACE HEADERS ScopedRegistry.h)
velox_link_libraries(velox_scoped_registry INTERFACE velox_exception Folly::folly)

velox_install_library_headers()
