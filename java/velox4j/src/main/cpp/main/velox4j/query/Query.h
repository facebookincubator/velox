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

#include <folly/json/dynamic.h>
#include <velox/common/serialization/Serializable.h>
#include <velox/core/PlanNode.h>
#include <memory>
#include <string>

#include "velox4j/conf/Config.h"

namespace facebook::velox4j {

/// An immutable JSON-able object that represents a query that is associated
/// with a given Velox query plan and the corresponding query configurations.
class Query : public facebook::velox::ISerializable {
 public:
  Query(
      const std::shared_ptr<const facebook::velox::core::PlanNode>& plan,
      const std::shared_ptr<const ConfigArray>& queryConfig,
      const std::shared_ptr<const ConnectorConfigArray>& connectorConfig);

  const std::shared_ptr<const facebook::velox::core::PlanNode>& plan() const;

  const std::shared_ptr<const ConfigArray>& queryConfig() const;

  const std::shared_ptr<const ConnectorConfigArray>& connectorConfig() const;

  folly::dynamic serialize() const override;

  std::string toString() const;

  static std::shared_ptr<Query> create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const std::shared_ptr<const facebook::velox::core::PlanNode> plan_;

  const std::shared_ptr<const ConfigArray> queryConfig_;

  const std::shared_ptr<const ConnectorConfigArray> connectorConfig_;
};
} // namespace facebook::velox4j
