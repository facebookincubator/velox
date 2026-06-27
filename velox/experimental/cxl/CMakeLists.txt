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

# CXL experimental memory subsystem.
# This validates that libnuma is discoverable when VELOX_ENABLE_CXL is enabled.
find_package(Numa REQUIRED)

velox_add_library(velox_experimental_cxl INTERFACE HEADERS CxlMemoryResource.h)
