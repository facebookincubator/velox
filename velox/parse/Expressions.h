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

#include "velox/common/EnumDeclare.h"
#include "velox/common/base/Exceptions.h"
#include "velox/parse/IExpr.h"
#include "velox/type/Variant.h"

namespace facebook::velox::core {

class InputExpr : public IExpr {
 public:
  InputExpr() : IExpr(IExpr::Kind::kInput, {}) {}

  std::string toString() const override {
    return "ROW";
  }

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    return std::make_shared<InputExpr>();
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<InputExpr>();
  }

  bool operator==(const IExpr& other) const override {
    return other.is(Kind::kInput);
  }

  size_t localHash() const override {
    return 0;
  }

  VELOX_DEFINE_CLASS_NAME(InputExpr)
};

class FieldAccessExpr : public IExpr {
 public:
  FieldAccessExpr(
      std::string name,
      std::optional<std::string> alias,
      std::vector<ExprPtr> inputs)
      : IExpr{IExpr::Kind::kFieldAccess, std::move(inputs), std::move(alias)},
        name_{std::move(name)} {
    VELOX_USER_CHECK_EQ(IExpr::inputs().size(), 1);
  }

  FieldAccessExpr(std::string name, std::optional<std::string> alias)
      : IExpr{IExpr::Kind::kFieldAccess, {std::make_shared<const InputExpr>()}, std::move(alias)},
        name_{std::move(name)} {}

  const std::string& name() const {
    return name_;
  }

  bool isRootColumn() const {
    return input()->is(Kind::kInput);
  }

  std::string toString() const override;

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_USER_CHECK_EQ(newInputs.size(), 1);
    return std::make_shared<FieldAccessExpr>(
        name_, alias(), std::move(newInputs));
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<FieldAccessExpr>(name_, alias, inputs());
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<FieldAccessExpr>(name_, std::nullopt, inputs());
  }

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

  VELOX_DEFINE_CLASS_NAME(FieldAccessExpr)

 private:
  const std::string name_;
};

class CallExpr : public IExpr {
 public:
  CallExpr(
      std::string name,
      std::vector<ExprPtr> inputs,
      std::optional<std::string> alias)
      : IExpr{IExpr::Kind::kCall, std::move(inputs), std::move(alias)},
        name_{std::move(name)} {
    VELOX_USER_CHECK(!name_.empty());
  }

  const std::string& name() const {
    return name_;
  }

  std::string toString() const override;

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK_EQ(newInputs.size(), inputs().size());
    return std::make_shared<CallExpr>(
        folly::copy(name()), std::move(newInputs), alias());
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<CallExpr>(name(), inputs(), alias);
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<CallExpr>(name(), inputs(), std::nullopt);
  }

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

  VELOX_DEFINE_CLASS_NAME(CallExpr)

 protected:
  // For subclasses that use a different Kind (e.g., AggregateCallExpr).
  CallExpr(
      Kind kind,
      std::string name,
      std::vector<ExprPtr> inputs,
      std::optional<std::string> alias)
      : IExpr{kind, std::move(inputs), std::move(alias)},
        name_{std::move(name)} {
    VELOX_USER_CHECK(!name_.empty());
  }

 private:
  const std::string name_;
};

/// Sort specification for ORDER BY clauses in aggregate and window functions.
struct SortKey {
  ExprPtr expr;
  bool ascending;
  bool nullsFirst;
};

/// An aggregate function call with DISTINCT, FILTER, and ORDER BY options.
/// Extends CallExpr so it can be used anywhere a CallExpr is expected (e.g.,
/// type resolution). Carries options as part of the IExpr tree so they
/// survive expression composition.
class AggregateCallExpr : public CallExpr {
 public:
  AggregateCallExpr(
      std::string name,
      std::vector<ExprPtr> args,
      bool distinct,
      ExprPtr filter,
      std::vector<SortKey> orderBy,
      std::optional<std::string> alias = std::nullopt)
      : CallExpr(
            IExpr::Kind::kAggregate,
            std::move(name),
            std::move(args),
            std::move(alias)),
        distinct_(distinct),
        filter_(std::move(filter)),
        orderBy_(std::move(orderBy)) {}

  bool isDistinct() const {
    return distinct_;
  }

  const ExprPtr& filter() const {
    return filter_;
  }

  const std::vector<SortKey>& orderBy() const {
    return orderBy_;
  }

  std::string toString() const override;

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK_NULL(
        filter_, "Cannot replace inputs on AggregateCallExpr with filter");
    VELOX_CHECK(
        orderBy_.empty(),
        "Cannot replace inputs on AggregateCallExpr with orderBy");
    return std::make_shared<AggregateCallExpr>(
        name(), std::move(newInputs), distinct_, filter_, orderBy_, alias());
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<AggregateCallExpr>(
        name(), inputs(), distinct_, filter_, orderBy_, alias);
  }

  ExprPtr dropAlias() const final {
    return std::make_shared<AggregateCallExpr>(
        name(), inputs(), distinct_, filter_, orderBy_, std::nullopt);
  }

  bool operator==(const IExpr& other) const override;

  VELOX_DEFINE_CLASS_NAME(AggregateCallExpr)

 protected:
  size_t localHash() const override;

 private:
  const bool distinct_;
  const ExprPtr filter_;
  const std::vector<SortKey> orderBy_;
};

/// Window function call with partition keys, ordering, frame, and ignore nulls.
/// Extends CallExpr so it can be used anywhere a CallExpr is expected (e.g.,
/// type resolution). Carries window specifications as part of the IExpr tree.
class WindowCallExpr : public CallExpr {
 public:
  enum class WindowType { kRange, kRows, kGroups };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(WindowType);

  enum class BoundType {
    kUnboundedPreceding,
    kPreceding,
    kCurrentRow,
    kFollowing,
    kUnboundedFollowing,
  };

  VELOX_DECLARE_EMBEDDED_ENUM_NAME(BoundType);

  struct Frame {
    WindowType type;
    BoundType startType;
    ExprPtr startValue;
    BoundType endType;
    ExprPtr endValue;
  };

  WindowCallExpr(
      std::string name,
      std::vector<ExprPtr> args,
      std::vector<ExprPtr> partitionKeys,
      std::vector<SortKey> orderByKeys,
      std::optional<Frame> frame,
      bool ignoreNulls,
      std::optional<std::string> alias = std::nullopt)
      : CallExpr(
            IExpr::Kind::kWindow,
            std::move(name),
            std::move(args),
            std::move(alias)),
        partitionKeys_(std::move(partitionKeys)),
        orderByKeys_(std::move(orderByKeys)),
        frame_(std::move(frame)),
        ignoreNulls_(ignoreNulls) {}

  const std::vector<ExprPtr>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::vector<SortKey>& orderByKeys() const {
    return orderByKeys_;
  }

  const std::optional<Frame>& frame() const {
    return frame_;
  }

  bool isIgnoreNulls() const {
    return ignoreNulls_;
  }

  std::string toString() const override;

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK(
        partitionKeys_.empty(),
        "Cannot replace inputs on WindowCallExpr with partitionKeys");
    VELOX_CHECK(
        orderByKeys_.empty(),
        "Cannot replace inputs on WindowCallExpr with orderByKeys");
    return std::make_shared<WindowCallExpr>(
        name(),
        std::move(newInputs),
        partitionKeys_,
        orderByKeys_,
        frame_,
        ignoreNulls_,
        alias());
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<WindowCallExpr>(
        name(),
        inputs(),
        partitionKeys_,
        orderByKeys_,
        frame_,
        ignoreNulls_,
        alias);
  }

  ExprPtr dropAlias() const final {
    return std::make_shared<WindowCallExpr>(
        name(),
        inputs(),
        partitionKeys_,
        orderByKeys_,
        frame_,
        ignoreNulls_,
        std::nullopt);
  }

  bool operator==(const IExpr& other) const override;

  VELOX_DEFINE_CLASS_NAME(WindowCallExpr)

 protected:
  size_t localHash() const override;

 private:
  const std::vector<ExprPtr> partitionKeys_;
  const std::vector<SortKey> orderByKeys_;
  const std::optional<Frame> frame_;
  const bool ignoreNulls_;
};

class ConstantExpr : public IExpr,
                     public std::enable_shared_from_this<ConstantExpr> {
 public:
  ConstantExpr(TypePtr type, Variant value, std::optional<std::string> alias)
      : IExpr{IExpr::Kind::kConstant, {}, std::move(alias)},
        type_{std::move(type)},
        value_{std::move(value)} {}

  std::string toString() const override;

  const Variant& value() const {
    return value_;
  }

  const TypePtr& type() const {
    return type_;
  }

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK_EQ(newInputs.size(), 0);
    return std::make_shared<ConstantExpr>(type(), value(), alias());
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<ConstantExpr>(type(), value(), alias);
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<ConstantExpr>(type(), value(), std::nullopt);
  }

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

  VELOX_DEFINE_CLASS_NAME(ConstantExpr)

 private:
  const TypePtr type_;
  const Variant value_;
};

class CastExpr : public IExpr, public std::enable_shared_from_this<CastExpr> {
 public:
  CastExpr(
      const TypePtr& type,
      const ExprPtr& expr,
      bool isTryCast,
      std::optional<std::string> alias)
      : IExpr{IExpr::Kind::kCast, {expr}, std::move(alias)},
        type_(type),
        isTryCast_(isTryCast) {}

  std::string toString() const override;

  const TypePtr& type() const {
    return type_;
  }

  bool isTryCast() const {
    return isTryCast_;
  }

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK_EQ(newInputs.size(), 1);
    return std::make_shared<CastExpr>(
        type(), newInputs[0], isTryCast_, alias());
  }

  ExprPtr withAlias(const std::string& alias) const override {
    return std::make_shared<CastExpr>(type(), input(), isTryCast_, alias);
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<CastExpr>(
        type(), input(), isTryCast_, std::nullopt);
  }

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

  VELOX_DEFINE_CLASS_NAME(CastExpr)

 private:
  const TypePtr type_;
  const bool isTryCast_;
};

/// Represents lambda expression as a list of inputs and the body expression.
/// For example, the expression
///     (k, v) -> k + v
/// is represented using [k, v] as inputNames and k + v as body.
class LambdaExpr : public IExpr,
                   public std::enable_shared_from_this<LambdaExpr> {
 public:
  LambdaExpr(std::vector<std::string> arguments, ExprPtr body)
      : IExpr(IExpr::Kind::kLambda, {}),
        arguments_{std::move(arguments)},
        body_{std::move(body)} {
    VELOX_USER_CHECK(!arguments_.empty());
  }

  const std::vector<std::string>& arguments() const {
    return arguments_;
  }

  const ExprPtr& body() const {
    return body_;
  }

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    VELOX_CHECK_EQ(newInputs.size(), 0);
    return std::make_shared<LambdaExpr>(arguments(), body());
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<LambdaExpr>(arguments(), body());
  }

  std::string toString() const override;

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

  VELOX_DEFINE_CLASS_NAME(LambdaExpr)

 private:
  const std::vector<std::string> arguments_;
  const ExprPtr body_;
};

/// Named ROW constructor. Carries field names alongside child expressions
/// through the unresolved expression tree.
class ConcatExpr : public IExpr {
 public:
  ConcatExpr(std::vector<std::string> fieldNames, std::vector<ExprPtr> inputs)
      : IExpr(IExpr::Kind::kConcat, std::move(inputs)),
        fieldNames_(std::move(fieldNames)) {
    VELOX_CHECK_EQ(fieldNames_.size(), this->inputs().size());
  }

  const std::vector<std::string>& fieldNames() const {
    return fieldNames_;
  }

  std::string toString() const override;

  ExprPtr replaceInputs(std::vector<ExprPtr> newInputs) const override {
    return std::make_shared<ConcatExpr>(fieldNames_, std::move(newInputs));
  }

  ExprPtr dropAlias() const override {
    return std::make_shared<ConcatExpr>(fieldNames_, inputs());
  }

  bool operator==(const IExpr& other) const override;

  size_t localHash() const override;

 private:
  const std::vector<std::string> fieldNames_;
};

using AggregateCallExprPtr = std::shared_ptr<const AggregateCallExpr>;
using WindowCallExprPtr = std::shared_ptr<const WindowCallExpr>;

} // namespace facebook::velox::core
