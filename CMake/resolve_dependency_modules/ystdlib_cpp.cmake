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
include_guard(GLOBAL)

FetchContent_Declare(
  ystdlib_cpp
  GIT_REPOSITORY https://github.com/y-scope/ystdlib-cpp.git
  GIT_TAG 0ae886c6a7ee706a3c6e1950262b63d72f71fe63)

FetchContent_Populate(ystdlib_cpp)

set(CLP_YSTDLIB_SOURCE_DIRECTORY "${ystdlib_cpp_SOURCE_DIR}")
