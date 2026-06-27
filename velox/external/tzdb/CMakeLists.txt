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


if(WIN32)
  # On Windows, the timezone layer uses TimeZoneMapStubs.cpp (UTC-only)
  # and does NOT link to velox_external_tzdb. Build a minimal stub library
  # with just chrono_exception.cpp so the CMake target still exists.
  velox_add_library(velox_external_tzdb chrono_exception.cpp)
else()
  velox_add_library(velox_external_tzdb chrono_exception.cpp time_zone.cpp
  tzdb_list.cpp tzdb.cpp)
endif()

velox_link_libraries(
  velox_external_tzdb
  velox_external_date
  Folly::folly
  fmt::fmt)
