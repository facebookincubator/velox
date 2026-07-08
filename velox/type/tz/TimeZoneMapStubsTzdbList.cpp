/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Windows implementation for tzdb_list and time_zone wrapper functions.
// Split from TimeZoneMapStubs.cpp to work around MSVC Internal Compiler Error.

#ifdef _WIN32

#include "velox/external/tzdb/time_zone.h"
#include "velox/external/tzdb/tzdb_list.h"
#include "velox/external/tzdb/tzdb.h"
#include "velox/external/date/tz.h"
#include <forward_list>

namespace facebook::velox::tzdb {

// Minimal __impl definition for Windows stubs
class time_zone::__impl {
 public:
  __impl() = default;
};

std::string_view time_zone::__name() const noexcept {
  return "UTC";
}

sys_info time_zone::__get_info(
    std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>)
    const {
  return sys_info{
      date::sys_seconds::min(),
      date::sys_seconds::max(),
      std::chrono::seconds{0},
      std::chrono::minutes{0},
      "UTC"};
}

local_info time_zone::__get_info(date::local_seconds) const {
  local_info li;
  li.result = local_info::unique;
  li.first = sys_info{
      date::sys_seconds::min(),
      date::sys_seconds::max(),
      std::chrono::seconds{0},
      std::chrono::minutes{0},
      "UTC"};
  li.second = {};
  return li;
}

sys_info time_zone::__get_info_to_populate_transition(
    date::sys_seconds) const {
  return sys_info{
      date::sys_seconds::min(),
      date::sys_seconds::max(),
      std::chrono::seconds{0},
      std::chrono::minutes{0},
      "UTC"};
}

time_zone time_zone::__create(std::unique_ptr<__impl>&& /*p*/) {
  return time_zone{};
}

time_zone::~time_zone() = default;

// tzdb_list stubs
class tzdb_list::__impl {
 public:
  std::forward_list<tzdb> tzdb_list_;
};

static tzdb_list::__impl* getGlobalTzdbImpl() {
  static auto* impl = new tzdb_list::__impl();
  if (impl->tzdb_list_.empty()) {
    tzdb db;
    db.version = "stub";
    impl->tzdb_list_.push_front(std::move(db));
  }
  return impl;
}

const tzdb& tzdb_list::__front() const noexcept {
  return __impl_->tzdb_list_.front();
}

tzdb_list::const_iterator tzdb_list::__erase_after(const_iterator p) {
  return __impl_->tzdb_list_.erase_after(p);
}

tzdb_list::const_iterator tzdb_list::__begin() const noexcept {
  return __impl_->tzdb_list_.cbegin();
}

tzdb_list::const_iterator tzdb_list::__end() const noexcept {
  return __impl_->tzdb_list_.cend();
}

tzdb_list::const_iterator tzdb_list::__cbegin() const noexcept {
  return __impl_->tzdb_list_.cbegin();
}

tzdb_list::const_iterator tzdb_list::__cend() const noexcept {
  return __impl_->tzdb_list_.cend();
}

tzdb_list::~tzdb_list() = default;

tzdb_list& get_tzdb_list() {
  static tzdb_list list(getGlobalTzdbImpl());
  return list;
}

const tzdb& reload_tzdb() {
  return get_tzdb_list().front();
}

std::string remote_version() {
  return "stub";
}

const time_zone* tzdb::__current_zone() const {
  return nullptr;
}

} // namespace facebook::velox::tzdb

#endif // _WIN32
