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
#include "velox4j/eval/Evaluation.h"

#include <folly/json/dynamic-inl.h>
#include <velox/common/serialization/DeserializationRegistry.h>
#include <string>
#include <utility>

#include "velox4j/conf/Config.h"

namespace facebook::velox4j {
using namespace facebook::velox;

Evaluation::Evaluation(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<const ConfigArray>& queryConfig,
    const std::shared_ptr<const ConnectorConfigArray>& connectorConfig)
    : expr_(expr),
      queryConfig_(queryConfig),
      connectorConfig_(connectorConfig) {}

const core::TypedExprPtr& Evaluation::expr() const {
  return expr_;
}

const std::shared_ptr<const ConfigArray>& Evaluation::queryConfig() const {
  return queryConfig_;
}

const std::shared_ptr<const ConnectorConfigArray>& Evaluation::connectorConfig()
    const {
  return connectorConfig_;
}

folly::dynamic Evaluation::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "velox4j.Evaluation";
  obj["expr"] = expr_->serialize();
  obj["queryConfig"] = queryConfig_->serialize();
  obj["connectorConfig"] = connectorConfig_->serialize();
  return obj;
}

std::shared_ptr<Evaluation> Evaluation::create(
    const folly::dynamic& obj,
    void* context) {
  auto expr = std::const_pointer_cast<const core::ITypedExpr>(
      ISerializable::deserialize<core::ITypedExpr>(obj["expr"], context));
  auto queryConfig = std::const_pointer_cast<const ConfigArray>(
      ISerializable::deserialize<ConfigArray>(obj["queryConfig"], context));
  auto connectorConfig = std::const_pointer_cast<const ConnectorConfigArray>(
      ISerializable::deserialize<ConnectorConfigArray>(
          obj["connectorConfig"], context));
  return std::make_shared<Evaluation>(expr, queryConfig, connectorConfig);
}

void Evaluation::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("velox4j.Evaluation", create);
}

} // namespace facebook::velox4j
