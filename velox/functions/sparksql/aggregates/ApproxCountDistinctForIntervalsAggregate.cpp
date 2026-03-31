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

#include "velox/functions/sparksql/aggregates/ApproxCountDistinctForIntervalsAggregate.h"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/functions/sparksql/XxHash64.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

constexpr uint64_t kXxHash64Seed = 42;

using HllAccumulator =
    common::hll::HllAccumulator<int64_t, false, HashStringAllocator>;

int8_t computeIndexBitLength(double relativeSD) {
  VELOX_USER_CHECK(
      std::isfinite(relativeSD) && relativeSD > 0.0,
      "relativeSD must be a positive finite value");
  const int32_t p = static_cast<int32_t>(
      std::ceil(2.0 * std::log(1.106 / relativeSD) / std::log(2.0)));
  VELOX_USER_CHECK_GE(
      p,
      4,
      "HLL++ requires at least 4 bits for addressing. Use a lower error, "
      "at most 39%.");
  VELOX_USER_CHECK_LE(
      p,
      16,
      "HLL++ requires at most 16 bits for addressing. Use a higher error.");
  return static_cast<int8_t>(p);
}

template <TypeKind kind>
double toDoubleDispatch(const exec::GenericView& value, const TypePtr& type) {
  if constexpr (
      kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
      kind == TypeKind::INTEGER || kind == TypeKind::BIGINT ||
      kind == TypeKind::REAL || kind == TypeKind::DOUBLE) {
    using T = typename TypeTraits<kind>::NativeType;
    auto converted =
        util::Converter<TypeKind::DOUBLE>::tryCast(value.template castTo<T>());
    VELOX_USER_CHECK(converted.hasValue(), "Failed to convert value to DOUBLE");
    return converted.value();
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
}

template <>
double toDoubleDispatch<TypeKind::TIMESTAMP>(
    const exec::GenericView& value,
    const TypePtr& /*type*/) {
  return static_cast<double>(value.castTo<Timestamp>().toMicros());
}

template <TypeKind kind>
uint64_t hashValueDispatch(
    const exec::GenericView& value,
    const TypePtr& type) {
  using T = typename TypeTraits<kind>::NativeType;
  if constexpr (kind == TypeKind::TINYINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        static_cast<int32_t>(value.template castTo<T>()), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::SMALLINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        static_cast<int32_t>(value.template castTo<T>()), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::INTEGER) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt32(
        value.template castTo<T>(), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::BIGINT) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashInt64(
        value.template castTo<T>(), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::REAL) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashFloat(
        value.template castTo<T>(), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::DOUBLE) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashDouble(
        value.template castTo<T>(), kXxHash64Seed);
  } else if constexpr (kind == TypeKind::TIMESTAMP) {
    return ::facebook::velox::functions::sparksql::XxHash64::hashTimestamp(
        value.template castTo<T>(), kXxHash64Seed);
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
}

template <>
uint64_t hashValueDispatch<TypeKind::HUGEINT>(
    const exec::GenericView& value,
    const TypePtr& /*type*/) {
  return ::facebook::velox::functions::sparksql::XxHash64::hashLongDecimal(
      value.castTo<int128_t>(), kXxHash64Seed);
}

class ApproxCountDistinctForIntervalsAggregate {
 public:
  using InputType = Row<Generic<T1>, Array<Generic<T2>>, double>;
  using IntermediateType = Row<Array<double>, Array<Varbinary>>;
  using OutputType = Array<int64_t>;

  static constexpr bool default_null_behavior_ = false;

  void initialize(
      core::AggregationNode::Step step,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& /*resultType*/) {
    if (exec::isRawInput(step)) {
      VELOX_CHECK_EQ(argTypes.size(), 3);
      inputType_ = argTypes[0];
      endpointsElementType_ = argTypes[1]->childAt(0);
    } else {
      VELOX_CHECK_EQ(argTypes.size(), 1);
    }
  }

  struct AccumulatorType {
    std::vector<HllAccumulator> hlls;
    ApproxCountDistinctForIntervalsAggregate* fn;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool is_aligned_ = true;

    AccumulatorType(
        HashStringAllocator* /*allocator*/,
        ApproxCountDistinctForIntervalsAggregate* fn)
        : fn(fn) {}

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Generic<T1>> data,
        exec::optional_arg_type<Array<Generic<T2>>> endpoints,
        exec::optional_arg_type<double> relativeSd) {
      fn->ensureRelativeSd(relativeSd);
      fn->ensureEndpoints(endpoints);

      if (!data.has_value()) {
        return false;
      }

      const double inputValue = fn->toDouble(data.value(), fn->inputType_);
      if (inputValue < fn->endpointsMin_ || inputValue > fn->endpointsMax_) {
        return false;
      }

      const auto intervalIndex = std::isnan(inputValue)
          ? (fn->intervalCount_ - 1)
          : fn->findIntervalIndex(inputValue);
      ensureSize(allocator, fn->intervalCount_, fn->indexBitLength_);
      const uint64_t hash = fn->hashValue(data.value(), fn->inputType_);
      hlls[intervalIndex].insertHash(hash);
      return true;
    }

    bool combine(
        HashStringAllocator* allocator,
        exec::optional_arg_type<IntermediateType> other) {
      if (!other.has_value()) {
        return false;
      }

      auto rowView = other.value();
      auto endpointsView = rowView.template at<0>();
      auto hllsView = rowView.template at<1>();
      if (!endpointsView.has_value() || !hllsView.has_value()) {
        return false;
      }

      fn->ensureEndpointsFromIntermediate(endpointsView.value());

      const auto& hllsArray = hllsView.value();
      VELOX_USER_CHECK_EQ(
          hllsArray.size(),
          fn->intervalCount_,
          "HLL array size {} does not match endpoints size {}",
          hllsArray.size(),
          fn->intervalCount_);

      for (const auto& entry : hllsArray) {
        if (entry.has_value()) {
          fn->maybeSetIndexBitLengthFromSerialized(entry.value());
          break;
        }
      }

      ensureSize(allocator, fn->intervalCount_, fn->indexBitLength_);
      for (size_t i = 0; i < hllsArray.size(); ++i) {
        const auto& entry = hllsArray[i];
        if (entry.has_value()) {
          hlls[i].mergeWith(entry.value(), allocator);
        }
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!fn->endpointsSet_ || fn->indexBitLength_ < 0) {
        return false;
      }

      fn->ensureEmptyHll();
      std::vector<std::string> serializedHlls;
      serializedHlls.reserve(fn->intervalCount_);
      for (int32_t interval = 0; interval < fn->intervalCount_; ++interval) {
        if (nonNullGroup && hlls.size() == fn->intervalCount_) {
          auto& hll = hlls[interval];
          const auto size = hll.serializedSize();
          std::string buffer(size, '\0');
          hll.serialize(buffer.data());
          serializedHlls.push_back(std::move(buffer));
        } else {
          serializedHlls.push_back(fn->emptyHll_);
        }
      }
      out.copy_from(std::make_tuple(fn->endpoints_, serializedHlls));
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!fn->endpointsSet_ || fn->indexBitLength_ < 0) {
        return false;
      }

      for (int32_t interval = 0; interval < fn->intervalCount_; ++interval) {
        int64_t count = 0;
        if (nonNullGroup && hlls.size() == fn->intervalCount_) {
          count = hlls[interval].cardinality();
        }
        if (fn->duplicateIntervals_[interval]) {
          count = 1;
        }
        out.add_item() = count;
      }
      return true;
    }

   private:
    void ensureSize(
        HashStringAllocator* allocator,
        int32_t targetSize,
        int8_t indexBitLength) {
      if (hlls.empty()) {
        hlls.reserve(targetSize);
        for (int32_t i = 0; i < targetSize; ++i) {
          if (indexBitLength >= 0) {
            hlls.emplace_back(indexBitLength, allocator);
          } else {
            hlls.emplace_back(allocator);
          }
        }
        return;
      }
      VELOX_USER_CHECK_EQ(hlls.size(), targetSize);
    }
  };

 private:
  void ensureRelativeSd(exec::optional_arg_type<double> relativeSd) {
    if (indexBitLength_ >= 0) {
      return;
    }
    VELOX_USER_CHECK(
        relativeSd.has_value(),
        "relativeSD must not be null for approx_count_distinct_for_intervals");
    indexBitLength_ = computeIndexBitLength(relativeSd.value());
    ensureEmptyHll();
  }

  void ensureEndpoints(exec::optional_arg_type<Array<Generic<T2>>> endpoints) {
    if (endpointsSet_) {
      return;
    }

    VELOX_CHECK_NOT_NULL(endpointsElementType_);
    VELOX_USER_CHECK(
        endpoints.has_value(),
        "Endpoints must not be null for approx_count_distinct_for_intervals");

    const auto& endpointsView = endpoints.value();
    VELOX_USER_CHECK_GE(
        endpointsView.size(),
        2,
        "approx_count_distinct_for_intervals requires at least 2 endpoints");

    std::vector<double> converted;
    converted.reserve(endpointsView.size());
    for (const auto& entry : endpointsView) {
      VELOX_USER_CHECK(
          entry.has_value(), "Endpoints must not contain null values");
      converted.push_back(toDouble(entry.value(), endpointsElementType_));
    }
    setEndpoints(converted);
  }

  void ensureEndpointsFromIntermediate(
      const exec::ArrayView<true, double>& endpointsView) {
    if (endpointsSet_) {
      return;
    }

    VELOX_USER_CHECK_GE(
        endpointsView.size(),
        2,
        "approx_count_distinct_for_intervals requires at least 2 endpoints");

    std::vector<double> converted;
    converted.reserve(endpointsView.size());
    for (const auto& entry : endpointsView) {
      VELOX_USER_CHECK(
          entry.has_value(), "Endpoints must not contain null values");
      converted.push_back(entry.value());
    }
    setEndpoints(converted);
  }

  void setEndpoints(const std::vector<double>& endpoints) {
    if (endpointsSet_) {
      VELOX_USER_CHECK_EQ(endpoints_.size(), endpoints.size());
      for (size_t i = 0; i < endpoints.size(); ++i) {
        VELOX_USER_CHECK_EQ(endpoints_[i], endpoints[i]);
      }
      return;
    }

    for (size_t i = 1; i < endpoints.size(); ++i) {
      VELOX_USER_CHECK_LE(
          endpoints[i - 1],
          endpoints[i],
          "Endpoints must be sorted in ascending order");
    }

    endpoints_ = endpoints;
    endpointsMin_ = endpoints_.front();
    endpointsMax_ = endpoints_.back();
    intervalCount_ = static_cast<int32_t>(endpoints_.size() - 1);
    duplicateIntervals_.resize(intervalCount_);
    for (int32_t i = 0; i < intervalCount_; ++i) {
      duplicateIntervals_[i] = (endpoints_[i] == endpoints_[i + 1]);
    }
    endpointsSet_ = true;
  }

  void ensureEmptyHll() {
    if (emptyHll_.empty() && indexBitLength_ >= 0) {
      emptyHll_ = common::hll::SparseHlls::serializeEmpty(indexBitLength_);
    }
  }

  void maybeSetIndexBitLengthFromSerialized(const StringView& serialized) {
    if (indexBitLength_ >= 0) {
      return;
    }

    const char* data = serialized.data();
    if (common::hll::SparseHlls::canDeserialize(data)) {
      indexBitLength_ =
          common::hll::SparseHlls::deserializeIndexBitLength(data);
    } else if (common::hll::DenseHlls::canDeserialize(data)) {
      indexBitLength_ = common::hll::DenseHlls::deserializeIndexBitLength(data);
    } else {
      VELOX_USER_FAIL("Unexpected type of HLL");
    }
    ensureEmptyHll();
  }

  static double toDouble(const exec::GenericView& value, const TypePtr& type) {
    const auto decimalToDouble = [](auto decimal, int32_t scale) {
      const auto scaleFactor = DecimalUtil::kPowersOfTen[scale];
      auto converted = util::Converter<TypeKind::DOUBLE>::tryCast(decimal);
      VELOX_USER_CHECK(
          converted.hasValue(), "Failed to convert decimal to DOUBLE");
      return converted.value() / scaleFactor;
    };

    if (type->isShortDecimal()) {
      return decimalToDouble(
          value.castTo<int64_t>(), type->asShortDecimal().scale());
    }
    if (type->isLongDecimal()) {
      return decimalToDouble(
          value.castTo<int128_t>(), type->asLongDecimal().scale());
    }

    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        toDoubleDispatch, type->kind(), value, type);
  }

  static uint64_t hashValue(
      const exec::GenericView& value,
      const TypePtr& type) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        hashValueDispatch, type->kind(), value, type);
  }

  int32_t findIntervalIndex(double value) const {
    auto it = std::lower_bound(endpoints_.begin(), endpoints_.end(), value);
    if (it != endpoints_.end() && *it == value) {
      auto index = static_cast<int32_t>(it - endpoints_.begin());
      while (index > 0 && endpoints_[index - 1] == value) {
        --index;
      }
      return index == 0 ? 0 : index - 1;
    }

    const auto insertionPoint = static_cast<int32_t>(it - endpoints_.begin());
    return insertionPoint == 0 ? 0 : insertionPoint - 1;
  }

  TypePtr inputType_;
  TypePtr endpointsElementType_;
  bool endpointsSet_{false};
  std::vector<double> endpoints_;
  std::vector<char> duplicateIntervals_;
  int32_t intervalCount_{0};
  double endpointsMin_{0};
  double endpointsMax_{0};
  int8_t indexBitLength_{-1};
  std::string emptyHll_;

  template <typename FUNC>
  friend class exec::SimpleAggregateAdapter;
};

class ApproxCountDistinctForIntervalsAggregateAdapter final
    : public exec::SimpleAggregateAdapter<
          ApproxCountDistinctForIntervalsAggregate> {
 public:
  using Base =
      exec::SimpleAggregateAdapter<ApproxCountDistinctForIntervalsAggregate>;

  ApproxCountDistinctForIntervalsAggregateAdapter(
      core::AggregationNode::Step step,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType)
      : Base(step, argTypes, resultType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    validateRawArguments(rows, args);
    Base::addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    validateRawArguments(rows, args);
    Base::addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 private:
  static void validateRawArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_USER_CHECK_EQ(
        args.size(),
        3,
        "approx_count_distinct_for_intervals requires relativeSD");

    const auto firstRow = rows.begin();

    DecodedVector decodedEndpoints(*args[1], rows);
    VELOX_USER_CHECK(
        decodedEndpoints.isConstantMapping(),
        "Endpoints must be constant for approx_count_distinct_for_intervals");
    VELOX_USER_CHECK(
        !decodedEndpoints.isNullAt(firstRow),
        "Endpoints must not be null for approx_count_distinct_for_intervals");

    DecodedVector decodedRelativeSd(*args[2], rows);
    VELOX_USER_CHECK(
        decodedRelativeSd.isConstantMapping(),
        "relativeSD must be constant for approx_count_distinct_for_intervals");
    VELOX_USER_CHECK(
        !decodedRelativeSd.isNullAt(firstRow),
        "relativeSD must not be null for approx_count_distinct_for_intervals");
  }
};

exec::AggregateRegistrationResult registerApproxCountDistinctForIntervals(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  const auto returnType = "array(bigint)";
  const auto intermediateType = "row(array(double), array(varbinary))";
  const std::vector<std::string> valueTypes = {
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real",
      "double",
      "date",
      "timestamp",
      "interval day to second",
      "interval year to month"};
  const std::vector<std::string> endpointTypes = valueTypes;

  auto addSignature = [&](exec::AggregateFunctionSignatureBuilder builder) {
    builder.constantArgumentType("double");
    signatures.push_back(builder.build());
  };

  for (const auto& valueType : valueTypes) {
    for (const auto& endpointType : endpointTypes) {
      addSignature(
          exec::AggregateFunctionSignatureBuilder()
              .returnType(returnType)
              .intermediateType(intermediateType)
              .argumentType(valueType)
              .constantArgumentType(fmt::format("array({})", endpointType)));
    }
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType(valueType)
            .constantArgumentType("array(DECIMAL(b_precision, b_scale))"));
  }

  for (const auto& endpointType : endpointTypes) {
    addSignature(
        exec::AggregateFunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .returnType(returnType)
            .intermediateType(intermediateType)
            .argumentType("DECIMAL(a_precision, a_scale)")
            .constantArgumentType(fmt::format("array({})", endpointType)));
  }

  addSignature(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .returnType(returnType)
          .intermediateType(intermediateType)
          .argumentType("DECIMAL(a_precision, a_scale)")
          .constantArgumentType("array(DECIMAL(b_precision, b_scale))"));

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK(
            argTypes.size() == 1 || argTypes.size() == 3,
            "{} takes 3 arguments",
            name);
        return std::make_unique<
            ApproxCountDistinctForIntervalsAggregateAdapter>(
            step, argTypes, resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerApproxCountDistinctForIntervalsAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerApproxCountDistinctForIntervals(
      prefix + "approx_count_distinct_for_intervals",
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
