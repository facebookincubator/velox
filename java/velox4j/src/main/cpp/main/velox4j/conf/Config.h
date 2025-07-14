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

#include <velox/common/serialization/Serializable.h>
#include <velox/core/QueryConfig.h>

namespace velox4j {
class ConfigArray : public facebook::velox::ISerializable {
 public:
  explicit ConfigArray(
      std::vector<std::pair<std::string, std::string>>&& values)
      : values_(std::move(values)) {}

  std::unordered_map<std::string, std::string> toMap() const;

  folly::dynamic serialize() const override;

  static std::shared_ptr<ConfigArray> empty();

  static std::shared_ptr<ConfigArray> create(const folly::dynamic& obj);

  static void registerSerDe();

 private:
  const std::vector<std::pair<std::string, std::string>> values_;
};

class ConnectorConfigArray : public facebook::velox::ISerializable {
 public:
  explicit ConnectorConfigArray(
      std::vector<std::pair<std::string, std::shared_ptr<const ConfigArray>>>&&
          values)
      : values_(std::move(values)) {}

  std::unordered_map<
      std::string,
      std::shared_ptr<facebook::velox::config::ConfigBase>>
  toMap() const;

  folly::dynamic serialize() const override;

  static std::shared_ptr<ConnectorConfigArray> create(
      const folly::dynamic& obj);

  static void registerSerDe();

 private:
  const std::vector<std::pair<std::string, std::shared_ptr<const ConfigArray>>>
      values_;
};
} // namespace velox4j
