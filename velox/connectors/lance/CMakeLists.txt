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

# Consume a prebuilt lance-c distribution, which ships headers, libraries and
# a CMake package config. scripts/setup-ubuntu.sh installs it under /usr/local
# when BUILD_LANCE=true; set LANCE_C_ROOT to point at an install elsewhere.
set(LANCE_C_ROOT "/usr/local" CACHE PATH "Root of an installed lance-c distribution.")

# Link the shared library. The static liblance_c.a statically links all of
# DataFusion and is several hundred MB, which makes every dependent binary slow
# to link and large on disk. Must be set before find_package.
set(LanceC_LINK "shared")

find_package(LanceC QUIET PATHS ${LANCE_C_ROOT} PATH_SUFFIXES lib/cmake/LanceC)

if(NOT LanceC_FOUND)
  message(
    FATAL_ERROR
    "VELOX_ENABLE_LANCE_CONNECTOR is ON but the lance-c package was not found "
    "under ${LANCE_C_ROOT}. Install it with BUILD_LANCE=true "
    "scripts/setup-ubuntu.sh, or pass -DLANCE_C_ROOT=<dir> to use an install "
    "elsewhere."
  )
endif()

get_target_property(LANCE_C_LIBRARY LanceC::lance_c_shared IMPORTED_LOCATION)
get_filename_component(LANCE_C_LIBRARY_DIR ${LANCE_C_LIBRARY} DIRECTORY)
message(STATUS "Found lance-c: ${LANCE_C_LIBRARY}")

velox_add_library(
  velox_lance_connector
  OBJECT
  LanceConnector.cpp
  HEADERS
  LanceColumnHandle.h
  LanceConnector.h
  LanceConnectorSplit.h
  LanceTableHandle.h
)

velox_link_libraries(velox_lance_connector velox_connector velox_vector velox_arrow_bridge fmt::fmt)

# Goes through velox_link_libraries to work with VELOX_MONO_LIBRARY mode. The
# imported target carries the include directory and the platform link deps
# (pthread/dl/m on Linux, the CoreFoundation/Security/SystemConfiguration
# frameworks on macOS), so none of that is repeated here.
velox_link_libraries(velox_lance_connector LanceC::lance_c)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
