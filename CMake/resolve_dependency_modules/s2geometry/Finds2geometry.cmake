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

# Find the s2geometry library installed on the system.
# s2geometry installs its CMake config as "s2Config.cmake" with the s2::s2
# target. This shim bridges the package name difference so that
# find_package(s2geometry) works.

find_package(s2 CONFIG QUIET)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(s2geometry DEFAULT_MSG s2_FOUND)
