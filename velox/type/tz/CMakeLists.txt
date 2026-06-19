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
if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
velox_add_library(
  velox_type_tz
  TimeZoneMap.h
  TimeZoneDatabase.cpp
  TimeZoneLinks.cpp
  TimeZoneNames.h
  TimeZoneNames.cpp
)

# On Windows, use TimeZoneMapStubs.cpp which builds the timezone database from
# the static data in TimeZoneDatabase.cpp without any velox_external_tzdb
# dependency (named zones fall back to UTC). TimeZoneMapStubsTzdbList.cpp
# provides stub implementations for tzdb symbols used by other modules
# (DateTimeFormatter, TimeWithTimezoneRegistration) that directly call into tzdb.
if(WIN32)
  if(VELOX_MONO_LIBRARY)
    target_sources(velox PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/TimeZoneMapStubs.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/TimeZoneMapStubsTzdbList.cpp)
  else()
    target_sources(velox_type_tz PRIVATE
      TimeZoneMapStubs.cpp
      TimeZoneMapStubsTzdbList.cpp)
  endif()
else()
  target_sources(velox_type_tz PRIVATE TimeZoneMap.cpp)
endif()

velox_link_libraries(velox_type_tz velox_exception fmt::fmt Folly::folly)

# Link to velox_external_tzdb only on non-Windows platforms. Windows uses the
# TimeZoneMapStubs.cpp path which has no tzdb dependency.
if(NOT WIN32)
  velox_link_libraries(velox_type_tz velox_external_tzdb)
endif()

# Add Windows-specific glog defines
if(WIN32)
  velox_compile_definitions(velox_type_tz PRIVATE GLOG_USE_GLOG_EXPORT)
endif()
