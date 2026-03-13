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

velox_add_library(velox_lance_connector OBJECT LanceConnector.cpp)

velox_link_libraries(
  velox_lance_connector
  velox_connector
  velox_vector
  velox_arrow_bridge
  fmt::fmt
)

# Link the pre-built Lance FFI static library and platform deps.
# These go through velox_link_libraries to work with VELOX_MONO_LIBRARY mode.
if(LANCE_FFI_LIB_PATH)
  velox_link_libraries(velox_lance_connector ${LANCE_FFI_LIB_PATH})
else()
  message(
    WARNING
    "LANCE_FFI_LIB_PATH not set. You must provide the path to liblance_duckdb_ffi.a "
    "via -DLANCE_FFI_LIB_PATH=<path> for the Lance connector to link successfully."
  )
endif()

if(APPLE)
  find_library(COREFOUNDATION_LIBRARY CoreFoundation)
  find_library(SECURITY_LIBRARY Security)
  find_library(SYSTEMCONFIGURATION_LIBRARY SystemConfiguration)
  velox_link_libraries(
    velox_lance_connector
    ${COREFOUNDATION_LIBRARY}
    ${SECURITY_LIBRARY}
    ${SYSTEMCONFIGURATION_LIBRARY}
  )
else()
  velox_link_libraries(velox_lance_connector pthread dl m)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
