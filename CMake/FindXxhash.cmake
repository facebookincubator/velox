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

#
# Try to find Facebook xxhash library This will define Xxhash_FOUND
# Xxhash_INCLUDE_DIR Xxhash_LIBRARY
#

find_path(Xxhash_INCLUDE_DIR NAMES xxhash.h)

find_library(Xxhash_LIBRARY_RELEASE NAMES xxhash)

include(SelectLibraryConfigurations)
select_library_configurations(Xxhash)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Xxhash DEFAULT_MSG Xxhash_LIBRARY
                                  Xxhash_INCLUDE_DIR)

if(Xxhash_FOUND)
  message(STATUS "Found xxhash: ${Xxhash_LIBRARY}")
endif()

mark_as_advanced(Xxhash_INCLUDE_DIR Xxhash_LIBRARY)
