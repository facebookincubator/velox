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

# Check if roaring target exists (e.g., built by cudf)
if(TARGET roaring)
  message(STATUS "Target roaring was already found.")
  return()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(Roaring IMPORTED_TARGET roaring)

if(Roaring_FOUND)
  add_library(roaring ALIAS PkgConfig::Roaring)
  message(STATUS "Found roaring via pkg-config.")
  return()
endif()

if(Roaring_FIND_REQUIRED)
  message(FATAL_ERROR "Failed to find roaring.")
elseif(NOT Roaring_FIND_QUIETLY)
  message(WARNING "Failed to find roaring.")
endif()
