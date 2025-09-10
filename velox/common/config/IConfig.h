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

#pragma once

#include <string>
#include <optional>
#include <unordered_map>
#include <folly/Conv.h>

namespace facebook::velox::config {

class IConfig {
public:
  template <typename T>
  std::optional<T> get(
      const std::string& key,
      std::function<T(std::string, std::string)> toT = [](auto /* unused */,
                                                          auto value) {
        return folly::to<T>(value);
      }) const {
    auto val = get(key);
    if (val.has_value()) {
      return toT(key, val.value());
    }
    return std::nullopt;
  }

  // In case key doesn't exist default value will materialize
  // from defaultValue. For example:
  // 
  // std::string_view some_default_value = ...
  // std::string value = config->get(some_key, some_default_value);
  //
  // You won't create default_value in case config has the value
  template <typename T, typename U>
  T get(
      const std::string& key,
      const U& defaultValue,
      std::function<T(std::string, std::string)> toT = [](auto /* unused */,
                                                          auto value) {
        return folly::to<T>(value);
      }) const {
    auto val = get(key);
    if (val.has_value()) {
      return toT(key, val.value());
    }
    return T(defaultValue);
  }

  virtual std::unordered_map<std::string, std::string> rawConfigsCopy() const = 0;

  virtual ~IConfig() = default;

private:
  virtual std::optional<std::string> get(const std::string& key) const = 0;
};

} // namespace facebook::velox::config
