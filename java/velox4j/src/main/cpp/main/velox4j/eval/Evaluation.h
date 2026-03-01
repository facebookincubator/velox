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
#include <velox/core/ITypedExpr.h>
#include <velox/expression/Expr.h>
#include <memory>

#include "velox4j/conf/Config.h"

namespace facebook::velox4j {

/// A JSON-able immutable struct that represents an evaluation process.
/// An evaluation is built with a Velox expression and being
/// used for creating an Evaluator that evaluates the expression
/// for input vectors.
class Evaluation : public facebook::velox::ISerializable {
 public:
  Evaluation(
      const facebook::velox::core::TypedExprPtr& expr,
      const std::shared_ptr<const ConfigArray>& queryConfig,
      const std::shared_ptr<const ConnectorConfigArray>& connectorConfig);

  const facebook::velox::core::TypedExprPtr& expr() const;

  const std::shared_ptr<const ConfigArray>& queryConfig() const;

  const std::shared_ptr<const ConnectorConfigArray>& connectorConfig() const;

  folly::dynamic serialize() const override;

  static std::shared_ptr<Evaluation> create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const facebook::velox::core::TypedExprPtr expr_;
  const std::shared_ptr<const ConfigArray> queryConfig_;
  const std::shared_ptr<const ConnectorConfigArray> connectorConfig_;
};
} // namespace facebook::velox4j
