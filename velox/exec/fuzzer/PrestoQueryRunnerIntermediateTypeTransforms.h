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
/// Defines a transform for a transient type (that does not exist in file
/// formats) to a target type.
class TransientTypeTransformBase {
 public:
  TransientTypeTransformBase(TypePtr transientType, TypePtr targetType)
      : transientType_(std::move(transientType)),
        targetType_(std::move(targetType)) {}
  virtual ~TransientTypeTransformBase() = default;

  /// The type of the value returned by transform().
  TypePtr targetType() const {
    return targetType_;
  };

  TypePtr transientType() const {
    return transientType_;
  };

  /// An expression tree that can convert the type of value returned by
  /// 'inputExpr' into its target type. It is assumed this has default-null
  /// behavior, i.e. a null input produces a null output.
  virtual core::ExprPtr projectToTargetType(
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const = 0;

  /// An expression tree that can convert the value returned by transform() back
  /// into its original value. It is assumed this has default-null behavior,
  /// i.e. a null input produces a null output.
  virtual core::ExprPtr projectToTransientType(
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const = 0;

 protected:
  TypePtr transientType_;
  TypePtr targetType_;
};

/// This class offers a default implementation for the
/// TransientTypeTransformBase, enabling conversion to and from target types
/// using the CAST operator. It is designed to support transient types that can
/// utilize the CAST operator for correct conversion semantics. To add support
/// for such transient types, simply create an instance of this class with the
/// appropriate parameters.
class TransientTypeTransform : public TransientTypeTransformBase {
 public:
  TransientTypeTransform(TypePtr transientType, TypePtr targetType)
      : TransientTypeTransformBase(
            std::move(transientType),
            std::move(targetType)) {}

  core::ExprPtr projectToTargetType(
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;

  core::ExprPtr projectToTransientType(
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;
};

/// Returns true if this types is an intermediate only type or contains an
/// intermediate only type.
bool isIntermediateOnlyType(const TypePtr& type);

/// Converts all the transient types in an input 'vector' to their respective
/// target types. For eg. ROW(<TRANSIENT>, MAP(<TRANSIENT>, BIGINT)) =>
/// ROW(<TARGET>, MAP(<TARGET>, BIGINT))
VectorPtr transformTransientTypes(const VectorPtr& vector);

/// Generates an expression that takes the ouput of 'inputExpr' and converts it
/// into 'ouputType'. The 'ouputType' is expected to be have same type hierarchy
/// as the type returned from 'inputExpr' with some of the leaf type replaced
/// with transient types. For eg. ouputType can be ROW(<TRANSIENT>,
/// MAP(<TRANSIENT>, BIGINT)) and type returned by 'inputExpr' can be
/// ROW(<TARGET>, MAP(<TARGET>, BIGINT)).
/// @param ouputType The expected output type of the expression, either an
/// intermediate only type, or a complex type containing one.
/// @param inputExpr The expression that will be the input to the returned
/// expression, the output of this expression will be of a type not containing
/// intermediate only types.
/// @param columnAlias The alias to give the returned expression.
core::ExprPtr getProjectionToTransformToTransientTypes(
    const TypePtr& ouputType,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias);
} // namespace facebook::velox::exec::test
