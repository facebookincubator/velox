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

# GEOS Configuration
set(VELOX_GEOS_BUILD_VERSION 3.11.1)
set(VELOX_GEOS_BUILD_SHA256_CHECKSUM
        6d0eb3cfa9f92d947731cc75f1750356b3bdfc07ea020553daf6af1c768e0be2)
string(
        CONCAT
        VELOX_GEOS_SOURCE_URL "https://download.osgeo.org/geos/"
        "geos-${VELOX_GEOS_BUILD_VERSION}.tar.bz2")

velox_resolve_dependency_url(GEOS)

FetchContent_Declare(
        geos
        URL ${VELOX_GEOS_SOURCE_URL}
        URL_HASH ${VELOX_GEOS_BUILD_SHA256_CHECKSUM}
)
FetchContent_MakeAvailable(geos)
