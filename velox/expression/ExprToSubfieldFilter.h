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

#include "velox/core/ExpressionEvaluator.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"

namespace facebook::velox::exec {

inline std::unique_ptr<common::BigintRange> lessThan(
    int64_t max,
    bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(
      std::numeric_limits<int64_t>::min(), max - 1, nullAllowed);
}

inline std::unique_ptr<common::BigintRange> lessThanOrEqual(
    int64_t max,
    bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(
      std::numeric_limits<int64_t>::min(), max, nullAllowed);
}

inline std::unique_ptr<common::BigintRange> greaterThan(
    int64_t min,
    bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(
      min + 1, std::numeric_limits<int64_t>::max(), nullAllowed);
}

inline std::unique_ptr<common::BigintRange> greaterThanOrEqual(
    int64_t min,
    bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(
      min, std::numeric_limits<int64_t>::max(), nullAllowed);
}

inline std::unique_ptr<common::NegatedBigintRange> notEqual(
    int64_t value,
    bool nullAllowed = false) {
  return std::make_unique<common::NegatedBigintRange>(
      value, value, nullAllowed);
}

inline std::unique_ptr<common::NegatedBigintRange>
notBetween(int64_t lower, int64_t upper, bool nullAllowed = false) {
  return std::make_unique<common::NegatedBigintRange>(
      lower, upper, nullAllowed);
}

inline std::unique_ptr<common::DoubleRange> lessThanDouble(
    double max,
    bool nullAllowed = false) {
  return std::make_unique<common::DoubleRange>(
      std::numeric_limits<double>::lowest(),
      true,
      true,
      max,
      false,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::DoubleRange> lessThanOrEqualDouble(
    double max,
    bool nullAllowed = false) {
  return std::make_unique<common::DoubleRange>(
      std::numeric_limits<double>::lowest(),
      true,
      true,
      max,
      false,
      false,
      nullAllowed);
}

inline std::unique_ptr<common::DoubleRange> greaterThanDouble(
    double min,
    bool nullAllowed = false) {
  return std::make_unique<common::DoubleRange>(
      min,
      false,
      true,
      std::numeric_limits<double>::max(),
      true,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::DoubleRange> greaterThanOrEqualDouble(
    double min,
    bool nullAllowed = false) {
  return std::make_unique<common::DoubleRange>(
      min,
      false,
      false,
      std::numeric_limits<double>::max(),
      true,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::DoubleRange>
betweenDouble(double min, double max, bool nullAllowed = false) {
  return std::make_unique<common::DoubleRange>(
      min, false, false, max, false, false, nullAllowed);
}

inline std::unique_ptr<common::FloatRange> lessThanFloat(
    float max,
    bool nullAllowed = false) {
  return std::make_unique<common::FloatRange>(
      std::numeric_limits<float>::lowest(),
      true,
      true,
      max,
      false,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::FloatRange> lessThanOrEqualFloat(
    float max,
    bool nullAllowed = false) {
  return std::make_unique<common::FloatRange>(
      std::numeric_limits<float>::lowest(),
      true,
      true,
      max,
      false,
      false,
      nullAllowed);
}

inline std::unique_ptr<common::FloatRange> greaterThanFloat(
    float min,
    bool nullAllowed = false) {
  return std::make_unique<common::FloatRange>(
      min,
      false,
      true,
      std::numeric_limits<float>::max(),
      true,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::FloatRange> greaterThanOrEqualFloat(
    float min,
    bool nullAllowed = false) {
  return std::make_unique<common::FloatRange>(
      min,
      false,
      false,
      std::numeric_limits<float>::max(),
      true,
      true,
      nullAllowed);
}

inline std::unique_ptr<common::FloatRange>
betweenFloat(float min, float max, bool nullAllowed = false) {
  return std::make_unique<common::FloatRange>(
      min, false, false, max, false, false, nullAllowed);
}

inline std::unique_ptr<common::BigintRange>
between(int64_t min, int64_t max, bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(min, max, nullAllowed);
}

inline std::unique_ptr<common::BigintMultiRange> bigintOr(
    std::unique_ptr<common::BigintRange> a,
    std::unique_ptr<common::BigintRange> b,
    bool nullAllowed = false) {
  std::vector<std::unique_ptr<common::BigintRange>> filters;
  filters.emplace_back(std::move(a));
  filters.emplace_back(std::move(b));
  return std::make_unique<common::BigintMultiRange>(
      std::move(filters), nullAllowed);
}

inline std::unique_ptr<common::BigintMultiRange> bigintOr(
    std::unique_ptr<common::BigintRange> a,
    std::unique_ptr<common::BigintRange> b,
    std::unique_ptr<common::BigintRange> c,
    bool nullAllowed = false) {
  std::vector<std::unique_ptr<common::BigintRange>> filters;
  filters.emplace_back(std::move(a));
  filters.emplace_back(std::move(b));
  filters.emplace_back(std::move(c));
  return std::make_unique<common::BigintMultiRange>(
      std::move(filters), nullAllowed);
}

inline std::unique_ptr<common::BytesValues> equal(
    const std::string& value,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesValues>(
      std::vector<std::string>{value}, nullAllowed);
}

inline std::unique_ptr<common::BigintRange> equal(
    int64_t value,
    bool nullAllowed = false) {
  return std::make_unique<common::BigintRange>(value, value, nullAllowed);
}

inline std::unique_ptr<common::BytesRange>
between(std::string min, std::string max, bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      std::move(min), false, false, std::move(max), false, false, nullAllowed);
}

inline std::unique_ptr<common::BytesRange>
betweenExclusive(std::string min, std::string max, bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      std::move(min), false, true, std::move(max), false, true, nullAllowed);
}

inline std::unique_ptr<common::NegatedBytesRange>
notBetween(std::string min, std::string max, bool nullAllowed = false) {
  return std::make_unique<common::NegatedBytesRange>(
      std::move(min), false, false, std::move(max), false, false, nullAllowed);
}

inline std::unique_ptr<common::NegatedBytesRange> notBetweenExclusive(
    std::string min,
    std::string max,
    bool nullAllowed = false) {
  return std::make_unique<common::NegatedBytesRange>(
      std::move(min), false, true, std::move(max), false, true, nullAllowed);
}

inline std::unique_ptr<common::BytesRange> lessThanOrEqual(
    std::string max,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      "", true, false, std::move(max), false, false, nullAllowed);
}

inline std::unique_ptr<common::BytesRange> lessThan(
    std::string max,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      "", true, false, std::move(max), false, true, nullAllowed);
}

inline std::unique_ptr<common::BytesRange> greaterThanOrEqual(
    std::string min,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      std::move(min), false, false, "", true, false, nullAllowed);
}

inline std::unique_ptr<common::BytesRange> greaterThan(
    std::string min,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesRange>(
      std::move(min), false, true, "", true, false, nullAllowed);
}

inline std::unique_ptr<common::Filter> in(
    const std::vector<int64_t>& values,
    bool nullAllowed = false) {
  return common::createBigintValues(values, nullAllowed);
}

inline std::unique_ptr<common::Filter> in(
    std::initializer_list<int64_t> values,
    bool nullAllowed = false) {
  return in(std::vector<int64_t>(values), nullAllowed);
}

inline std::unique_ptr<common::Filter> notIn(
    const std::vector<int64_t>& values,
    bool nullAllowed = false) {
  return common::createNegatedBigintValues(values, nullAllowed);
}

inline std::unique_ptr<common::Filter> notIn(
    std::initializer_list<int64_t> values,
    bool nullAllowed = false) {
  return notIn(std::vector<int64_t>(values), nullAllowed);
}

inline std::unique_ptr<common::BytesValues> in(
    const std::vector<std::string>& values,
    bool nullAllowed = false) {
  return std::make_unique<common::BytesValues>(values, nullAllowed);
}

inline std::unique_ptr<common::BytesValues> in(
    std::initializer_list<std::string> values,
    bool nullAllowed = false) {
  return in(std::vector<std::string>(values), nullAllowed);
}

inline std::unique_ptr<common::NegatedBytesValues> notIn(
    const std::vector<std::string>& values,
    bool nullAllowed = false) {
  return std::make_unique<common::NegatedBytesValues>(values, nullAllowed);
}

inline std::unique_ptr<common::NegatedBytesValues> notIn(
    std::initializer_list<std::string> values,
    bool nullAllowed = false) {
  return notIn(std::vector<std::string>(values), nullAllowed);
}

inline std::unique_ptr<common::BoolValue> boolEqual(
    bool value,
    bool nullAllowed = false) {
  return std::make_unique<common::BoolValue>(value, nullAllowed);
}

inline std::unique_ptr<common::IsNull> isNull() {
  return std::make_unique<common::IsNull>();
}

inline std::unique_ptr<common::IsNotNull> isNotNull() {
  return std::make_unique<common::IsNotNull>();
}

inline std::unique_ptr<common::AlwaysTrue> alwaysTrue() {
  return std::make_unique<common::AlwaysTrue>();
}

template <typename T>
std::unique_ptr<common::MultiRange>
orFilter(std::unique_ptr<T> a, std::unique_ptr<T> b, bool nullAllowed = false) {
  std::vector<std::unique_ptr<common::Filter>> filters;
  filters.emplace_back(std::move(a));
  filters.emplace_back(std::move(b));
  return std::make_unique<common::MultiRange>(std::move(filters), nullAllowed);
}

inline std::unique_ptr<common::HugeintRange> lessThanHugeint(
    int128_t max,
    bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(
      std::numeric_limits<int128_t>::min(), max - 1, nullAllowed);
}

inline std::unique_ptr<common::HugeintRange> lessThanOrEqualHugeint(
    int128_t max,
    bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(
      std::numeric_limits<int128_t>::min(), max, nullAllowed);
}

inline std::unique_ptr<common::HugeintRange> greaterThanHugeint(
    int128_t min,
    bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(
      min + 1, std::numeric_limits<int128_t>::max(), nullAllowed);
}

inline std::unique_ptr<common::HugeintRange> greaterThanOrEqualHugeint(
    int128_t min,
    bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(
      min, std::numeric_limits<int128_t>::max(), nullAllowed);
}

inline std::unique_ptr<common::HugeintRange> equalHugeint(
    int128_t value,
    bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(value, value, nullAllowed);
}

inline std::unique_ptr<common::HugeintRange>
betweenHugeint(int128_t min, int128_t max, bool nullAllowed = false) {
  return std::make_unique<common::HugeintRange>(min, max, nullAllowed);
}

inline std::unique_ptr<common::TimestampRange> equal(
    const Timestamp& value,
    bool nullAllowed = false) {
  return std::make_unique<common::TimestampRange>(value, value, nullAllowed);
}

inline std::unique_ptr<common::TimestampRange>
between(const Timestamp& min, const Timestamp& max, bool nullAllowed = false) {
  return std::make_unique<common::TimestampRange>(min, max, nullAllowed);
}

inline std::unique_ptr<common::TimestampRange> lessThan(
    Timestamp max,
    bool nullAllowed = false) {
  --max;
  return std::make_unique<common::TimestampRange>(
      std::numeric_limits<Timestamp>::min(), max, nullAllowed);
}

inline std::unique_ptr<common::TimestampRange> lessThanOrEqual(
    const Timestamp& max,
    bool nullAllowed = false) {
  return std::make_unique<common::TimestampRange>(
      std::numeric_limits<Timestamp>::min(), max, nullAllowed);
}

inline std::unique_ptr<common::TimestampRange> greaterThan(
    Timestamp min,
    bool nullAllowed = false) {
  ++min;
  return std::make_unique<common::TimestampRange>(
      min, std::numeric_limits<Timestamp>::max(), nullAllowed);
}

inline std::unique_ptr<common::TimestampRange> greaterThanOrEqual(
    const Timestamp& min,
    bool nullAllowed = false) {
  return std::make_unique<common::TimestampRange>(
      min, std::numeric_limits<Timestamp>::max(), nullAllowed);
}

/// Language-specific translator of generic expressions into subfield filters.
/// Default language is Presto. A different language can be supported by
/// registering language-specific implementation using 'registerParser' API.
class ExprToSubfieldFilterParser {
 public:
  virtual ~ExprToSubfieldFilterParser() = default;

  /// Returns a parser provided to an earlier 'registerParser' call. Not thread
  /// safe.
  static const std::shared_ptr<ExprToSubfieldFilterParser>& getInstance() {
    VELOX_CHECK_NOT_NULL(parser_, "Parser is not registered");
    return parser_;
  }

  /// Registers a parser. Silently overwrites previously registered parser if
  /// any. Not thread safe.
  static void registerParser(
      std::shared_ptr<ExprToSubfieldFilterParser> parser) {
    VELOX_CHECK_NOT_NULL(parser);
    parser_ = std::move(parser);
  }

  /// Analyzes 'call' expression to determine if it can be expressed as a
  /// subfield filter. Returns the subfield and filter if so. Otherwise, returns
  /// std::nullopt. If 'negated' is true, considers the negation of 'call'
  /// expressions (not(call)). It is possible that 'call' expression can be
  /// represented as subfield filter, but its negation cannot.
  virtual std::optional<
      std::pair<common::Subfield, std::unique_ptr<common::Filter>>>
  leafCallToSubfieldFilter(
      const core::CallTypedExpr& call,
      core::ExpressionEvaluator* evaluator,
      bool negated = false) = 0;

  /// Combines 2 or more filters with an OR.
  /// Detects overlapping ranges of bigint and floating point values.
  /// Detects a list of single-value bigint filters and combines them into a
  /// single IN list.
  /// Returns nullptr if no combination has been made.
  static std::unique_ptr<common::Filter> makeOrFilter(
      std::vector<std::unique_ptr<common::Filter>> disjuncts);

  template <typename... Disjuncts>
  static std::unique_ptr<common::Filter> makeOrFilter(
      Disjuncts&&... disjuncts) {
    std::vector<std::unique_ptr<common::Filter>> filters;
    filters.reserve(sizeof...(Disjuncts));
    (filters.emplace_back(std::forward<Disjuncts>(disjuncts)), ...);
    return makeOrFilter(std::move(filters));
  }

 protected:
  // Converts an expression into a subfield. Returns false if the expression
  // is not a valid field expression.
  static bool toSubfield(
      const core::ITypedExpr* field,
      common::Subfield& subfield);

  // Creates a non-equal subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeNotEqualFilter(
      const core::TypedExprPtr& valueExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates an equal subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeEqualFilter(
      const core::TypedExprPtr& valueExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates a greater-than subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeGreaterThanFilter(
      const core::TypedExprPtr& lowerExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates a less-than subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeLessThanFilter(
      const core::TypedExprPtr& upperExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates a less-than-or-equal subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeLessThanOrEqualFilter(
      const core::TypedExprPtr& upperExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates a greater-than-or-equal subfield filter against the given constant.
  static std::unique_ptr<common::Filter> makeGreaterThanOrEqualFilter(
      const core::TypedExprPtr& lowerExpr,
      core::ExpressionEvaluator* evaluator);

  // Creates an in subfield filter against the given vector.
  static std::unique_ptr<common::Filter> makeInFilter(
      const core::TypedExprPtr& expr,
      core::ExpressionEvaluator* evaluator,
      bool negated);

  // Creates a between subfield filter against the given lower and upper
  // bounds.
  static std::unique_ptr<common::Filter> makeBetweenFilter(
      const core::TypedExprPtr& lowerExpr,
      const core::TypedExprPtr& upperExpr,
      core::ExpressionEvaluator* evaluator,
      bool negated);

 private:
  // Singleton parser instance.
  static std::shared_ptr<ExprToSubfieldFilterParser> parser_;
};

// Parser for Presto expressions.
class PrestoExprToSubfieldFilterParser : public ExprToSubfieldFilterParser {
 public:
  std::optional<std::pair<common::Subfield, std::unique_ptr<common::Filter>>>
  leafCallToSubfieldFilter(
      const core::CallTypedExpr& call,
      core::ExpressionEvaluator* evaluator,
      bool negated = false) override;
};

} // namespace facebook::velox::exec
