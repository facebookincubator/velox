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

# This creates a separate scope so any changed variables don't affect
# the rest of the build.
block(PROPAGATE Arrow_FOUND Arrow_VERSION)
  set(find_package_args)
  if(Arrow_FIND_VERSION)
    list(APPEND find_package_args ${Arrow_FIND_VERSION})
  endif()
  if(Arrow_FIND_QUIETLY)
    list(APPEND find_package_args QUIET)
  endif()
  if(Arrow_FIND_REQUIRED)
    list(APPEND find_package_args REQUIRED)
  endif()
  find_package(Arrow ${find_package_args} CONFIG)
  if(Arrow_FOUND)
    # ArrowTestingConfig.cmake in Arrow 21 or earlier calls
    # find_dependency(Arrow) without CONFIG. It calls this file again and causes
    # an infinite loop: https://github.com/apache/arrow/issues/46386
    if(Arrow_VERSION VERSION_LESS 22.0.0)
      set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
    endif()
    find_package(ArrowTesting ${find_package_args} CONFIG)
    if(NOT ArrowTesting_FOUND)
      set(Arrow_FOUND FALSE)
    endif()
  endif()
endblock()

# Only add the libraries once.
if(Arrow_FOUND AND NOT TARGET arrow)
  # Make the targets visible outside this directory like the bundled ones.
  set_target_properties(
    Arrow::arrow_static
    ArrowTesting::arrow_testing_static
    PROPERTIES IMPORTED_GLOBAL TRUE
  )
  add_library(arrow ALIAS Arrow::arrow_static)
  add_library(arrow_testing ALIAS ArrowTesting::arrow_testing_static)
endif()
