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

#include "velox/functions/sparksql/aggregates/HistogramNumericAggregate.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Conv.h>

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/aggregates/SparkNumericHistogram.h"
#include "velox/type/Conversions.h"
#include "velox/type/CppToType.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

constexpr std::string_view kInputTypes[] = {
    "tinyint",
    "smallint",
    "integer",
    "bigint",
    "real",
    "double",
    "date",
    "timestamp",
    "timestamp utc",
    "interval year to month",
};

// Keep both division operands exact to avoid double rounding.
constexpr int128_t kMaxExactDoubleInteger = int128_t{1} << 53;
constexpr std::array<double, 23> kExactPowersOfTen = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
};

class HistogramNumericDecimal {
 public:
  HistogramNumericDecimal(int32_t precision, int32_t scale)
      : scale_{scale}, type_{DECIMAL(precision, scale)} {}

  double decimalToDouble(int128_t unscaled) const {
    if (scale_ <= 22 && unscaled >= -kMaxExactDoubleInteger &&
        unscaled <= kMaxExactDoubleInteger) {
      return static_cast<double>(unscaled) / kExactPowersOfTen[scale_];
    }
    return folly::to<double>(DecimalUtil::toString(unscaled, *type_));
  }

 private:
  int32_t scale_;
  TypePtr type_;
};

double timestampToDouble(const Timestamp& value) {
  VELOX_USER_CHECK_EQ(
      value.getNanos() % 1'000,
      0,
      "histogram_numeric timestamp requires microsecond precision");
  return static_cast<double>(value.toMicros());
}

template <typename T>
T narrowCenter(double value) {
  return util::Converter<CppToType<T>::typeKind, void, util::SparkCastPolicy>::
      tryCast(value)
          .value();
}

Timestamp timestampFromDouble(double value) {
  const auto micros = narrowCenter<int64_t>(value);
  // Avoid multiplying a negative seconds value near INT64_MIN.
  auto seconds = micros / 1'000'000;
  auto remainder = micros % 1'000'000;
  if (remainder < 0) {
    --seconds;
    remainder += 1'000'000;
  }
  return Timestamp(seconds, static_cast<uint64_t>(remainder) * 1'000);
}

int32_t readRequiredConstant(
    const std::vector<VectorPtr>& constantInputs,
    size_t index,
    std::string_view name) {
  VELOX_USER_CHECK_GT(
      constantInputs.size(),
      index,
      "histogram_numeric requires a constant {} argument",
      name);
  VELOX_USER_CHECK_NOT_NULL(
      constantInputs[index],
      "histogram_numeric requires a constant {} argument",
      name);
  VELOX_USER_CHECK(
      constantInputs[index]->isConstantEncoding(),
      "histogram_numeric {} argument must be constant",
      name);
  VELOX_USER_CHECK(
      !constantInputs[index]->isNullAt(0),
      "histogram_numeric {} argument must not be null",
      name);
  return constantInputs[index]->as<ConstantVector<int32_t>>()->valueAt(0);
}

TypePtr resultCenterType(const TypePtr& resultType) {
  VELOX_USER_CHECK(
      resultType->isArray(),
      "histogram_numeric result must be ARRAY(ROW(x, y)): {}",
      resultType->toString());
  const auto& rowType = resultType->childAt(0);
  VELOX_USER_CHECK(
      rowType->isRow() && rowType->size() == 2,
      "histogram_numeric result must be ARRAY(ROW(x, y)): {}",
      resultType->toString());
  const auto& resultRowType = rowType->asRow();
  VELOX_USER_CHECK_EQ(
      resultRowType.nameOf(0),
      "x",
      "histogram_numeric result field 0 must be x");
  VELOX_USER_CHECK_EQ(
      resultRowType.nameOf(1),
      "y",
      "histogram_numeric result field 1 must be y");
  VELOX_USER_CHECK(
      rowType->childAt(1)->equivalent(*DOUBLE()),
      "histogram_numeric y result must be DOUBLE: {}",
      resultType->toString());
  return rowType->childAt(0);
}

void validateResultCenterType(
    const TypePtr& resultType,
    const TypePtr& expectedCenterType) {
  if (!resultType->isArray()) {
    return;
  }
  const auto centerType = resultCenterType(resultType);
  VELOX_USER_CHECK(
      centerType->equivalent(*expectedCenterType),
      "histogram_numeric result center type mismatch: expected {}, got {}",
      expectedCenterType->toString(),
      centerType->toString());
}

template <typename TInput, typename TOutput>
class HistogramNumericAggregate {
 public:
  using InputType = Row<TInput, int32_t>;
  using IntermediateType = Varbinary;
  using OutputType = Array<Row<Field<"x", TOutput>, Field<"y", double>>>;

  static constexpr bool default_null_behavior_ = false;
  static constexpr bool is_reducing_ = false;

  // Binds logical type metadata used independently of the erased wire state.
  void initialize(
      core::AggregationNode::Step /*step*/,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& /*resultType*/) {
    if constexpr (
        std::is_same_v<TInput, ShortDecimal<P1, S1>> ||
        std::is_same_v<TInput, LongDecimal<P1, S1>>) {
      for (const auto& type : argTypes) {
        if (type->isDecimal()) {
          setDecimalType(type);
          break;
        }
      }
    }
  }

  // Binds raw-input numBins even when the input is empty, masked, or all null.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    const auto numBins = readRequiredConstant(constantInputs, 1, "numBins");
    SparkNumericHistogram::validateNumBins(numBins);
    rawNumBins_ = numBins;
  }

  struct AccumulatorType {
    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;
    static constexpr bool is_aligned_ = true;

    AccumulatorType(
        HashStringAllocator* allocator,
        HistogramNumericAggregate* function)
        : histogram_{allocator}, function_{function} {
      if (function_->rawNumBins_.has_value()) {
        histogram_.initialize(*function_->rawNumBins_);
      }
    }

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInput> value,
        exec::optional_arg_type<int32_t> /*numBins*/) {
      if (!value.has_value()) {
        return false;
      }
      histogram_.add(function_->normalize(value.value()));
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<IntermediateType> state) {
      if (!state.has_value()) {
        return false;
      }
      const auto value = state.value();
      histogram_.mergeSerialized(std::string_view(value.data(), value.size()));
      return !histogram_.bins().empty();
    }

    bool writeIntermediateResult(
        bool /*nonNullGroup*/,
        exec::out_type<IntermediateType>& out) {
      if (histogram_.numBins() == 0) {
        return false;
      }
      out.resize(histogram_.serializedSize());
      histogram_.serialize(out.data());
      return true;
    }

    bool writeFinalResult(
        bool /*nonNullGroup*/,
        exec::out_type<OutputType>& out) {
      if (histogram_.bins().empty()) {
        return false;
      }
      out.reserve(histogram_.bins().size());
      for (const auto& bin : histogram_.bins()) {
        auto& row = out.add_item();
        function_->writeCenter(row, bin.x);
        row.template get_writer_at<1>() = bin.y;
      }
      return true;
    }

   private:
    SparkNumericHistogram histogram_;
    HistogramNumericAggregate* function_;
  };

 private:
  void setDecimalType(const TypePtr& type) {
    const auto [precision, scale] = getDecimalPrecisionScale(*type);
    if (decimal_.has_value()) {
      VELOX_USER_CHECK(
          decimalType_->equivalent(*type),
          "histogram_numeric Decimal metadata must be consistent: {} vs {}",
          decimalType_->toString(),
          type->toString());
      return;
    }
    decimal_.emplace(precision, scale);
    decimalType_ = type;
  }

  template <typename TValue>
  double normalize(TValue value) const {
    if constexpr (
        std::is_same_v<TInput, Timestamp> ||
        std::is_same_v<TInput, TimestampUtc>) {
      return timestampToDouble(value);
    } else if constexpr (
        std::is_same_v<TInput, ShortDecimal<P1, S1>> ||
        std::is_same_v<TInput, LongDecimal<P1, S1>>) {
      VELOX_CHECK(
          decimal_.has_value(),
          "histogram_numeric Decimal metadata is unavailable");
      return decimal_->decimalToDouble(static_cast<int128_t>(value));
    } else {
      return static_cast<double>(value);
    }
  }

  template <typename TRow>
  void writeCenter(TRow& row, double center) const {
    if constexpr (std::is_floating_point_v<TOutput>) {
      row.template get_writer_at<0>() = static_cast<TOutput>(center);
    } else if constexpr (
        std::is_same_v<TOutput, Timestamp> ||
        std::is_same_v<TOutput, TimestampUtc>) {
      row.template get_writer_at<0>() = timestampFromDouble(center);
    } else if constexpr (
        std::is_same_v<TOutput, Date> ||
        std::is_same_v<TOutput, IntervalYearMonth>) {
      row.template get_writer_at<0>() = narrowCenter<int32_t>(center);
    } else {
      row.template get_writer_at<0>() = narrowCenter<TOutput>(center);
    }
  }

  // Keeps only raw-input capacity; merged capacities remain per accumulator.
  std::optional<int32_t> rawNumBins_;
  std::optional<HistogramNumericDecimal> decimal_;
  TypePtr decimalType_;
};

template <typename TInput, typename TOutput>
std::unique_ptr<exec::Aggregate> makeAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const core::QueryConfig& config) {
  return std::make_unique<
      exec::SimpleAggregateAdapter<HistogramNumericAggregate<TInput, TOutput>>>(
      step, argTypes, resultType, &config);
}

template <typename TOutput>
std::unique_ptr<exec::Aggregate> makeByInputType(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const core::QueryConfig& config) {
  const auto inputType =
      !argTypes.empty() && !argTypes[0]->isVarbinary() ? argTypes[0] : DOUBLE();
  switch (inputType->kind()) {
    case TypeKind::TINYINT:
      return makeAggregate<int8_t, TOutput>(step, argTypes, resultType, config);
    case TypeKind::SMALLINT:
      return makeAggregate<int16_t, TOutput>(
          step, argTypes, resultType, config);
    case TypeKind::INTEGER:
      if (inputType->isDate()) {
        return makeAggregate<Date, TOutput>(step, argTypes, resultType, config);
      }
      if (inputType->isIntervalYearMonth()) {
        return makeAggregate<IntervalYearMonth, TOutput>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<int32_t, TOutput>(
          step, argTypes, resultType, config);
    case TypeKind::BIGINT:
      if (inputType->isShortDecimal()) {
        return makeAggregate<ShortDecimal<P1, S1>, TOutput>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<int64_t, TOutput>(
          step, argTypes, resultType, config);
    case TypeKind::HUGEINT:
      VELOX_CHECK(
          inputType->isLongDecimal(),
          "histogram_numeric supports HUGEINT only as Decimal");
      return makeAggregate<LongDecimal<P1, S1>, TOutput>(
          step, argTypes, resultType, config);
    case TypeKind::REAL:
      return makeAggregate<float, TOutput>(step, argTypes, resultType, config);
    case TypeKind::DOUBLE:
      return makeAggregate<double, TOutput>(step, argTypes, resultType, config);
    case TypeKind::TIMESTAMP:
      if (inputType->equivalent(*TIMESTAMP_UTC())) {
        return makeAggregate<TimestampUtc, TOutput>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<Timestamp, TOutput>(
          step, argTypes, resultType, config);
    default:
      VELOX_USER_FAIL(
          "Unsupported histogram_numeric input type: {}",
          inputType->toString());
  }
}

exec::AggregateFunctionSignaturePtr fixedSignature(
    std::string_view argument,
    std::string_view result) {
  return exec::AggregateFunctionSignatureBuilder()
      .argumentType(std::string(argument))
      .constantArgumentType("integer")
      .intermediateType("varbinary")
      .returnType(fmt::format("array(row(x {},y double))", result))
      .build();
}

exec::AggregateFunctionSignaturePtr decimalSignature(std::string_view result) {
  return exec::AggregateFunctionSignatureBuilder()
      .integerVariable("p")
      .integerVariable("s")
      .argumentType("decimal(p,s)")
      .constantArgumentType("integer")
      .intermediateType("varbinary")
      .returnType(fmt::format("array(row(x {},y double))", result))
      .build();
}

TypePtr canonicalType(
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  if (!argTypes.empty() && !argTypes[0]->isVarbinary()) {
    return argTypes[0];
  }
  if (resultType->isArray()) {
    return resultCenterType(resultType);
  }
  return DOUBLE();
}

std::unique_ptr<exec::Aggregate> makeCanonical(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const core::QueryConfig& config) {
  const auto type = canonicalType(argTypes, resultType);
  VELOX_USER_CHECK(
      !type->isDecimal(),
      "histogram_numeric does not support DECIMAL centers; use "
      "histogram_numeric_legacy for DOUBLE centers");
  validateResultCenterType(resultType, type);
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return makeAggregate<int8_t, int8_t>(step, argTypes, resultType, config);
    case TypeKind::SMALLINT:
      return makeAggregate<int16_t, int16_t>(
          step, argTypes, resultType, config);
    case TypeKind::INTEGER:
      if (type->isDate()) {
        return makeAggregate<Date, Date>(step, argTypes, resultType, config);
      }
      if (type->isIntervalYearMonth()) {
        return makeAggregate<IntervalYearMonth, IntervalYearMonth>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<int32_t, int32_t>(
          step, argTypes, resultType, config);
    case TypeKind::BIGINT:
      return makeAggregate<int64_t, int64_t>(
          step, argTypes, resultType, config);
    case TypeKind::REAL:
      return makeAggregate<float, float>(step, argTypes, resultType, config);
    case TypeKind::DOUBLE:
      return makeAggregate<double, double>(step, argTypes, resultType, config);
    case TypeKind::TIMESTAMP:
      if (type->equivalent(*TIMESTAMP_UTC())) {
        return makeAggregate<TimestampUtc, TimestampUtc>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<Timestamp, Timestamp>(
          step, argTypes, resultType, config);
    default:
      VELOX_USER_FAIL(
          "Unsupported histogram_numeric input type: {}", type->toString());
  }
}

std::vector<exec::AggregateFunctionSignaturePtr> canonicalSignatures() {
  std::vector<exec::AggregateFunctionSignaturePtr> signatures;
  for (const auto type : kInputTypes) {
    signatures.push_back(fixedSignature(type, type));
  }
  return signatures;
}

void registerCanonical(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  exec::registerAggregateFunction(
      name,
      canonicalSignatures(),
      makeCanonical,
      {},
      withCompanionFunctions,
      overwrite);
}

void registerLegacy(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<exec::AggregateFunctionSignaturePtr> signatures;
  for (const auto argument : kInputTypes) {
    signatures.push_back(fixedSignature(argument, "double"));
  }
  signatures.push_back(decimalSignature("double"));
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& config) {
        validateResultCenterType(resultType, DOUBLE());
        return makeByInputType<double>(step, argTypes, resultType, config);
      },
      {},
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerHistogramNumericAggregates(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerCanonical(
      prefix + "histogram_numeric", withCompanionFunctions, overwrite);
  registerLegacy(
      prefix + "histogram_numeric_legacy", withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
