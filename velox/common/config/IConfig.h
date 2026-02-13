/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#pragma once

#include <folly/Conv.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::config {

/// IConfig - Read-only config interface
/// for accessing key-value parameters.
/// Supports value retrieval by key and
/// duplication of the raw configuration data.
/// Can be used by velox::QueryConfig to access
/// externally managed system configuration.
class IConfig {
 public:
  template <typename T>
  std::optional<T> get(
      const std::string& key,
      const std::function<T(std::string, std::string)>& toT =
          [](auto /* unused */, auto value) {
            return folly::to<T>(value);
          }) const {
    if (auto val = access(key)) {
      return toT(key, *val);
    }
    return std::nullopt;
  }

  template <typename T>
  T get(
      const std::string& key,
      const T& defaultValue,
      const std::function<T(std::string, std::string)>& toT =
          [](auto /* unused */, auto value) {
            return folly::to<T>(value);
          }) const {
    if (auto val = access(key)) {
      return toT(key, *val);
    }
    return defaultValue;
  }

  virtual std::unordered_map<std::string, std::string> rawConfigsCopy()
      const = 0;

  virtual ~IConfig() = default;

 private:
  virtual std::optional<std::string> access(const std::string& key) const = 0;
};

} // namespace facebook::velox::config
