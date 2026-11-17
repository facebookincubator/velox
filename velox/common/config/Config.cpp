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

#include <folly/String.h>
#include <re2/re2.h>
#include <algorithm>

#include "velox/common/EnumDefine.h"
#include "velox/common/config/Config.h"

namespace facebook::velox::config {
namespace {

const auto& capacityUnitNames() {
  static const folly::F14FastMap<CapacityUnit, std::string_view> kNames = {
      {CapacityUnit::BYTE, "b"},
      {CapacityUnit::KILOBYTE, "kb"},
      {CapacityUnit::MEGABYTE, "mb"},
      {CapacityUnit::GIGABYTE, "gb"},
      {CapacityUnit::TERABYTE, "tb"},
      {CapacityUnit::PETABYTE, "pb"},
  };
  return kNames;
}

double bytesPerCapacityUnit(CapacityUnit unit) {
  switch (unit) {
    case CapacityUnit::BYTE:
      return 1;
    case CapacityUnit::KILOBYTE:
      return exp2(10);
    case CapacityUnit::MEGABYTE:
      return exp2(20);
    case CapacityUnit::GIGABYTE:
      return exp2(30);
    case CapacityUnit::TERABYTE:
      return exp2(40);
    case CapacityUnit::PETABYTE:
      return exp2(50);
  }
  VELOX_UNREACHABLE("Invalid capacity unit: {}", static_cast<int>(unit));
}

} // namespace

VELOX_DEFINE_ENUM_NAME(CapacityUnit, capacityUnitNames);

uint64_t
toCapacity(const std::string& from, CapacityUnit to, std::string_view key) {
  static const RE2 kPattern(R"(^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$)");
  double value;
  std::string unit;
  if (!RE2::FullMatch(from, kPattern, &value, &unit)) {
    if (!key.empty()) {
      detail::failInvalidConfigValue(key, from, "Invalid capacity string");
    }
    VELOX_USER_FAIL("Invalid capacity string '{}'", from);
  }

  auto lowerUnit = unit;
  folly::toLowerAscii(lowerUnit);
  const auto fromUnit = CapacityUnitName::tryToCapacityUnit(lowerUnit);
  if (!fromUnit) {
    if (!key.empty()) {
      detail::failInvalidConfigValue(
          key, from, fmt::format("Invalid capacity unit '{}'", unit));
    }
    VELOX_USER_FAIL("Invalid capacity unit '{}' in '{}'", unit, from);
  }
  const double capacity =
      value * (bytesPerCapacityUnit(*fromUnit) / bytesPerCapacityUnit(to));
  // 2^64 is the first value outside the uint64_t range.
  constexpr double kCapacityLimit = 0x1p64;
  if (capacity >= kCapacityLimit) {
    if (!key.empty()) {
      detail::failInvalidConfigValue(key, from, "Capacity out of range");
    }
    VELOX_USER_FAIL("Capacity out of range: {}", from);
  }
  return static_cast<uint64_t>(capacity);
}

std::chrono::duration<double> toDuration(const std::string& str) {
  static const RE2 kPattern(R"(^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*)");

  double value;
  std::string unit;
  if (!RE2::FullMatch(str, kPattern, &value, &unit)) {
    VELOX_USER_FAIL("Invalid duration '{}'", str);
  }
  if (unit == "ns") {
    return std::chrono::duration<double, std::nano>(value);
  } else if (unit == "us") {
    return std::chrono::duration<double, std::micro>(value);
  } else if (unit == "ms") {
    return std::chrono::duration<double, std::milli>(value);
  } else if (unit == "s") {
    return std::chrono::duration<double>(value);
  } else if (unit == "m") {
    return std::chrono::duration<double, std::ratio<60>>(value);
  } else if (unit == "h") {
    return std::chrono::duration<double, std::ratio<60 * 60>>(value);
  } else if (unit == "d") {
    return std::chrono::duration<double, std::ratio<60 * 60 * 24>>(value);
  }
  VELOX_USER_FAIL("Invalid duration '{}'", str);
}

ConfigBase& ConfigBase::set(const std::string& key, const std::string& val) {
  VELOX_CHECK(mutable_, "Cannot set in immutable config");
  std::unique_lock<std::shared_mutex> l(mutex_);
  configs_[key] = val;
  return *this;
}

ConfigBase& ConfigBase::reset() {
  VELOX_CHECK(mutable_, "Cannot reset in immutable config");
  std::unique_lock<std::shared_mutex> l(mutex_);
  configs_.clear();
  return *this;
}

bool ConfigBase::valueExists(const std::string& key) const {
  std::shared_lock<std::shared_mutex> l(mutex_);
  return configs_.find(key) != configs_.end();
};

const std::unordered_map<std::string, std::string>& ConfigBase::rawConfigs()
    const {
  VELOX_CHECK(
      !mutable_,
      "Mutable config cannot return unprotected reference to raw configs.");
  return configs_;
}

std::unordered_map<std::string, std::string> ConfigBase::rawConfigsCopy()
    const {
  std::shared_lock<std::shared_mutex> l(mutex_);
  return configs_;
}

std::unordered_map<std::string, std::string> ConfigBase::rawConfigsWithPrefix(
    std::string_view prefix) const {
  std::unordered_map<std::string, std::string> filteredConfigs;
  if (prefix.empty()) {
    return filteredConfigs;
  }

  std::shared_lock<std::shared_mutex> l(mutex_);
  for (const auto& [key, value] : configs_) {
    if (key.size() >= prefix.size() &&
        key.compare(0, prefix.size(), prefix) == 0) {
      filteredConfigs.emplace(key.substr(prefix.size()), value);
    }
  }
  return filteredConfigs;
}

std::string ConfigBase::toConfigKey(std::string_view sessionKey) {
  std::string configKey{sessionKey};
  std::replace(configKey.begin(), configKey.end(), '_', '-');
  return configKey;
}

std::string ConfigBase::toSessionKey(std::string_view configKey) {
  std::string sessionKey{configKey};
  std::replace(sessionKey.begin(), sessionKey.end(), '-', '_');
  return sessionKey;
}

std::optional<std::string> ConfigBase::access(const std::string& key) const {
  std::shared_lock l{mutex_};
  if (auto it = configs_.find(key); it != configs_.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace facebook::velox::config
