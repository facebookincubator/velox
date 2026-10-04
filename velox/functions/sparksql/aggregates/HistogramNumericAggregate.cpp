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

#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/aggregates/HistogramNumericTypes.h"
#include "velox/functions/sparksql/aggregates/SparkNumericHistogram.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

enum class InputFamily {
  kByte,
  kShort,
  kInteger,
  kLong,
  kFloat,
  kDouble,
  kDecimal,
  kDate,
  kTimestamp,
  kTimestampNtz,
  kYearMonthInterval,
};

enum class OutputFamily {
  kByte,
  kShort,
  kInteger,
  kLong,
  kFloat,
  kDouble,
  kDecimal,
  kDate,
  kTimestamp,
  kTimestampNtz,
  kYearMonthInterval,
};

void validateNumBins(int32_t numBins) {
  VELOX_USER_CHECK_GE(
      numBins, 2, "histogram_numeric requires numBins at least 2");
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

template <
    typename TInput,
    typename TOutput,
    InputFamily kInput,
    OutputFamily kOutput>
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
      const TypePtr& resultType) {
    if constexpr (kInput == InputFamily::kDecimal) {
      for (const auto& type : argTypes) {
        if (type->isDecimal()) {
          setDecimalType(type);
          break;
        }
      }
    }
    if (resultType->isArray()) {
      const auto centerType = resultCenterType(resultType);
      if constexpr (kOutput == OutputFamily::kDecimal) {
        VELOX_USER_CHECK(
            centerType->isDecimal(),
            "histogram_numeric Decimal result must retain precision and scale");
        setDecimalType(centerType);
      }
    }
  }

  // Binds raw-input numBins even when the input is empty, masked, or all null.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    if (constantInputs.size() < 2) {
      return;
    }
    const auto numBins = readRequiredConstant(constantInputs, 1, "numBins");
    validateNumBins(numBins);
    if (rawNumBins_.has_value()) {
      VELOX_USER_CHECK_EQ(
          *rawNumBins_,
          numBins,
          "histogram_numeric numBins must be consistent");
    } else {
      rawNumBins_ = numBins;
    }
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
        exec::optional_arg_type<int32_t> numBins) {
      VELOX_USER_CHECK(
          numBins.has_value(),
          "histogram_numeric numBins argument must not be null");
      validateNumBins(numBins.value());
      function_->bindRawNumBins(numBins.value());
      histogram_.initialize(numBins.value());
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
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup && histogram_.numBins() == 0 &&
          !function_->rawNumBins_.has_value()) {
        return false;
      }
      if (histogram_.numBins() == 0) {
        VELOX_CHECK(
            function_->rawNumBins_.has_value(),
            "histogram_numeric numBins is unknown for an empty partial state");
        histogram_.initialize(*function_->rawNumBins_);
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
  void bindRawNumBins(int32_t numBins) {
    if (rawNumBins_.has_value()) {
      VELOX_USER_CHECK_EQ(
          *rawNumBins_,
          numBins,
          "histogram_numeric numBins must be consistent");
    } else {
      rawNumBins_ = numBins;
    }
  }

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
        kInput == InputFamily::kTimestamp ||
        kInput == InputFamily::kTimestampNtz) {
      return HistogramNumericTypes::timestampToDouble(value);
    } else if constexpr (kInput == InputFamily::kDecimal) {
      VELOX_CHECK(
          decimal_.has_value(),
          "histogram_numeric Decimal metadata is unavailable");
      return decimal_->decimalToDouble(static_cast<int128_t>(value));
    } else {
      return HistogramNumericTypes::numericToDouble(value);
    }
  }

  template <typename TRow>
  void writeCenter(TRow& row, double center) const {
    if constexpr (kOutput == OutputFamily::kByte) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToByte(center);
    } else if constexpr (kOutput == OutputFamily::kShort) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToShort(center);
    } else if constexpr (
        kOutput == OutputFamily::kInteger || kOutput == OutputFamily::kDate ||
        kOutput == OutputFamily::kYearMonthInterval) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToInt(center);
    } else if constexpr (kOutput == OutputFamily::kLong) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToLong(center);
    } else if constexpr (kOutput == OutputFamily::kFloat) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToFloat(center);
    } else if constexpr (kOutput == OutputFamily::kDouble) {
      row.template get_writer_at<0>() = center;
    } else if constexpr (
        kOutput == OutputFamily::kTimestamp ||
        kOutput == OutputFamily::kTimestampNtz) {
      row.template get_writer_at<0>() =
          HistogramNumericTypes::javaDoubleToTimestamp(center);
    } else if constexpr (kOutput == OutputFamily::kDecimal) {
      VELOX_CHECK(
          decimal_.has_value(),
          "histogram_numeric Decimal metadata is unavailable");
      const auto value = decimal_->tryReconstructNativeDecimal(center);
      if (!value.has_value()) {
        row.template set_null_at<0>();
        return;
      }
      if constexpr (std::is_same_v<exec::out_type<TOutput>, int64_t>) {
        row.template get_writer_at<0>() = static_cast<int64_t>(*value);
      } else {
        row.template get_writer_at<0>() = *value;
      }
    }
  }

  // Keeps only raw-input capacity; merged capacities remain per accumulator.
  std::optional<int32_t> rawNumBins_;
  std::optional<HistogramNumericDecimal> decimal_;
  TypePtr decimalType_;
};

template <
    typename TInput,
    typename TOutput,
    InputFamily kInput,
    OutputFamily kOutput>
std::unique_ptr<exec::Aggregate> makeAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const core::QueryConfig& config) {
  return std::make_unique<exec::SimpleAggregateAdapter<
      HistogramNumericAggregate<TInput, TOutput, kInput, kOutput>>>(
      step, argTypes, resultType, &config);
}

template <typename TOutput, OutputFamily kOutput>
std::unique_ptr<exec::Aggregate> makeByInputType(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const core::QueryConfig& config) {
  const auto inputType =
      !argTypes.empty() && !argTypes[0]->isVarbinary() ? argTypes[0] : DOUBLE();
  switch (inputType->kind()) {
    case TypeKind::TINYINT:
      return makeAggregate<int8_t, TOutput, InputFamily::kByte, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::SMALLINT:
      return makeAggregate<int16_t, TOutput, InputFamily::kShort, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::INTEGER:
      if (inputType->isDate()) {
        return makeAggregate<Date, TOutput, InputFamily::kDate, kOutput>(
            step, argTypes, resultType, config);
      }
      if (inputType->isIntervalYearMonth()) {
        return makeAggregate<
            IntervalYearMonth,
            TOutput,
            InputFamily::kYearMonthInterval,
            kOutput>(step, argTypes, resultType, config);
      }
      return makeAggregate<int32_t, TOutput, InputFamily::kInteger, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::BIGINT:
      if (inputType->isShortDecimal()) {
        return makeAggregate<
            ShortDecimal<P1, S1>,
            TOutput,
            InputFamily::kDecimal,
            kOutput>(step, argTypes, resultType, config);
      }
      return makeAggregate<int64_t, TOutput, InputFamily::kLong, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::HUGEINT:
      VELOX_CHECK(
          inputType->isLongDecimal(),
          "histogram_numeric supports HUGEINT only as Decimal");
      return makeAggregate<
          LongDecimal<P1, S1>,
          TOutput,
          InputFamily::kDecimal,
          kOutput>(step, argTypes, resultType, config);
    case TypeKind::REAL:
      return makeAggregate<float, TOutput, InputFamily::kFloat, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::DOUBLE:
      return makeAggregate<double, TOutput, InputFamily::kDouble, kOutput>(
          step, argTypes, resultType, config);
    case TypeKind::TIMESTAMP:
      if (inputType->equivalent(*TIMESTAMP_UTC())) {
        return makeAggregate<
            TimestampUtc,
            TOutput,
            InputFamily::kTimestampNtz,
            kOutput>(step, argTypes, resultType, config);
      }
      return makeAggregate<
          Timestamp,
          TOutput,
          InputFamily::kTimestamp,
          kOutput>(step, argTypes, resultType, config);
    default:
      VELOX_USER_FAIL(
          "Unsupported histogram_numeric input type: {}",
          inputType->toString());
  }
}

std::vector<exec::AggregateFunctionSignaturePtr> fixedSignature(
    std::string_view argument,
    std::string_view result) {
  return {exec::AggregateFunctionSignatureBuilder()
              .argumentType(std::string(argument))
              .constantArgumentType("integer")
              .intermediateType("varbinary")
              .returnType(fmt::format("array(row(x {},y double))", result))
              .build()};
}

std::vector<exec::AggregateFunctionSignaturePtr> decimalSignature(
    std::string_view result) {
  return {exec::AggregateFunctionSignatureBuilder()
              .integerVariable("p")
              .integerVariable("s")
              .argumentType("decimal(p,s)")
              .constantArgumentType("integer")
              .intermediateType("varbinary")
              .returnType(fmt::format("array(row(x {},y double))", result))
              .build()};
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
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return makeAggregate<
          int8_t,
          int8_t,
          InputFamily::kByte,
          OutputFamily::kByte>(step, argTypes, resultType, config);
    case TypeKind::SMALLINT:
      return makeAggregate<
          int16_t,
          int16_t,
          InputFamily::kShort,
          OutputFamily::kShort>(step, argTypes, resultType, config);
    case TypeKind::INTEGER:
      if (type->isDate()) {
        return makeAggregate<
            Date,
            Date,
            InputFamily::kDate,
            OutputFamily::kDate>(step, argTypes, resultType, config);
      }
      if (type->isIntervalYearMonth()) {
        return makeAggregate<
            IntervalYearMonth,
            IntervalYearMonth,
            InputFamily::kYearMonthInterval,
            OutputFamily::kYearMonthInterval>(
            step, argTypes, resultType, config);
      }
      return makeAggregate<
          int32_t,
          int32_t,
          InputFamily::kInteger,
          OutputFamily::kInteger>(step, argTypes, resultType, config);
    case TypeKind::BIGINT:
      if (type->isShortDecimal()) {
        return makeAggregate<
            ShortDecimal<P1, S1>,
            ShortDecimal<P1, S1>,
            InputFamily::kDecimal,
            OutputFamily::kDecimal>(step, argTypes, resultType, config);
      }
      return makeAggregate<
          int64_t,
          int64_t,
          InputFamily::kLong,
          OutputFamily::kLong>(step, argTypes, resultType, config);
    case TypeKind::HUGEINT:
      VELOX_CHECK(
          type->isLongDecimal(),
          "histogram_numeric supports HUGEINT only as Decimal");
      return makeAggregate<
          LongDecimal<P1, S1>,
          LongDecimal<P1, S1>,
          InputFamily::kDecimal,
          OutputFamily::kDecimal>(step, argTypes, resultType, config);
    case TypeKind::REAL:
      return makeAggregate<
          float,
          float,
          InputFamily::kFloat,
          OutputFamily::kFloat>(step, argTypes, resultType, config);
    case TypeKind::DOUBLE:
      return makeAggregate<
          double,
          double,
          InputFamily::kDouble,
          OutputFamily::kDouble>(step, argTypes, resultType, config);
    case TypeKind::TIMESTAMP:
      if (type->equivalent(*TIMESTAMP_UTC())) {
        return makeAggregate<
            TimestampUtc,
            TimestampUtc,
            InputFamily::kTimestampNtz,
            OutputFamily::kTimestampNtz>(step, argTypes, resultType, config);
      }
      return makeAggregate<
          Timestamp,
          Timestamp,
          InputFamily::kTimestamp,
          OutputFamily::kTimestamp>(step, argTypes, resultType, config);
    default:
      VELOX_USER_FAIL(
          "Unsupported histogram_numeric input type: {}", type->toString());
  }
}

std::vector<exec::AggregateFunctionSignaturePtr> canonicalSignatures() {
  std::vector<exec::AggregateFunctionSignaturePtr> signatures;
  for (const auto* type : {
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
       }) {
    auto signature = fixedSignature(type, type);
    signatures.push_back(std::move(signature.front()));
  }
  auto decimal = decimalSignature("decimal(p,s)");
  signatures.push_back(std::move(decimal.front()));
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
  for (const auto* argument : {
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
       }) {
    auto signature = fixedSignature(argument, "double");
    signatures.push_back(std::move(signature.front()));
  }
  auto decimal = decimalSignature("double");
  signatures.push_back(std::move(decimal.front()));
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& config) {
        validateResultCenterType(resultType, DOUBLE());
        return makeByInputType<double, OutputFamily::kDouble>(
            step, argTypes, resultType, config);
      },
      {},
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerHistogramNumericAggregates(
    const std::string& prefix,
    bool /*withCompanionFunctions*/,
    bool overwrite) {
  registerCanonical(
      prefix + "histogram_numeric",
      /*withCompanionFunctions=*/false,
      overwrite);
  registerLegacy(
      prefix + "histogram_numeric_legacy",
      /*withCompanionFunctions=*/false,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
