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

#include <functional>
#include <map>
#include <unordered_map>

#include "folly/Conv.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {
// The concrete config class would inherit the config base
// and then just define all the entries.
template <class ConcreteConfig>
class ConfigBase {
 public:
  template <typename T>
  class Entry {
   private:
    Entry(
        const std::string& key,
        const T& val,
        std::function<std::string(const T&)> toStr =
            [](const T& val) { return folly::to<std::string>(val); },
        std::function<T(const std::string&, const std::string&)> toT =
            [](const std::string& key, const std::string& val) {
              auto converted = folly::tryTo<T>(val);
              VELOX_CHECK(
                  converted.hasValue(),
                  fmt::format(
                      "Invalid configuration for key '{}'. Value '{}' cannot be converted to type {}.",
                      key,
                      val,
                      folly::demangle(typeid(T))));
              return converted.value();
            })
        : key_{key}, default_{val}, toStr_{toStr}, toT_{toT} {}

   public:
    const std::string& configKey() const {
      return key_;
    }

   private:
    const std::string key_;
    const T default_;
    const std::function<std::string(const T&)> toStr_;
    const std::function<T(const std::string&, const std::string&)> toT_;

    friend ConfigBase;
    friend ConcreteConfig;
  };

  template <typename T>
  ConfigBase& set(const Entry<T>& entry, const T& val) {
    configs_[entry.key_] = entry.toStr_(val);
    return *this;
  }

  template <typename T>
  ConfigBase& unset(const Entry<T>& entry) {
    configs_.erase(entry.key_);
    return *this;
  }

  ConfigBase& reset() {
    configs_.clear();
    return *this;
  }

  template <typename T>
  T get(const Entry<T>& entry) const {
    auto iter = configs_.find(entry.key_);
    return iter != configs_.end() ? entry.toT_(entry.key_, iter->second)
                                  : entry.default_;
  }

  std::map<std::string, std::string> toSerdeParams() {
    return std::map{configs_.cbegin(), configs_.cend()};
  }

 protected:
  std::unordered_map<std::string, std::string> configs_;
};

} // namespace facebook::velox::common
