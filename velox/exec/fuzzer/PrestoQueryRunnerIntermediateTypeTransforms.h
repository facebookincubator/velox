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

#include "velox/parse/IExpr.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec::test {
// Defines a transform for an intermediate type or a complex type that can
// contain an intermediate type.
class IntermediateTypeTransform {
 public:
  virtual ~IntermediateTypeTransform() = default;

  virtual VectorPtr transform(
      const VectorPtr& vector,
      const SelectivityVector& rows) const = 0;

  virtual core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const = 0;
};

/// Returns true if this types is an intermediate only type or contains an
/// intermediate only type.
bool isIntermediateOnlyType(const TypePtr& type);

/// Converts a Vector of an intermediate only type, or containing one, to a
/// Vector of value(s) that can be input to a projection to produce those values
/// of that type but are of types supported as input. Preserves nulls and
/// encodings.
VectorPtr transformIntermediateOnlyType(const VectorPtr& vector);

/// Converts an expression that takes in a value of an intermediate only type so
/// that it applies a transformation to convert valid input typess into values
/// of the intermediate only type.
core::ExprPtr getIntermediateOnlyTypeProjectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias);
} // namespace facebook::velox::exec::test
