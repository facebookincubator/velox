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
#include <limits>

#include <fmt/format.h>
#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/VectorReaders.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/functions/sparksql/Hash.h"
#include "velox/functions/sparksql/XxHash64.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Spark hashes values with xxhash64 seeded with 42 before feeding them to
// HLL++, see HyperLogLogPlusPlusHelper.
constexpr uint64_t kXxHash64Seed = 42;

using SparkXxHash64 = ::facebook::velox::functions::sparksql::XxHash64;

using HllAccumulator =
    common::hll::HllAccumulator<int64_t, false, HashStringAllocator>;

// Scales a decimal's unscaled value into a DOUBLE.
template <typename T>
double decimalToDouble(T unscaledValue, int32_t scale) {
  const auto scaleFactor = DecimalUtil::kPowersOfTen[scale];
  auto converted = util::Converter<TypeKind::DOUBLE>::tryCast(unscaledValue);
  VELOX_USER_CHECK(converted.hasValue(), "Failed to convert decimal to DOUBLE");
  return converted.value() / scaleFactor;
}

// Spark's DayTimeIntervalType holds microseconds while Velox's INTERVAL DAY TO
// SECOND holds milliseconds. Intervals that do not fit Spark's range cannot
// come from Spark, so they are rejected instead of silently overflowing.
int64_t toSparkIntervalMicros(int64_t millis) {
  int64_t micros;
  VELOX_USER_CHECK(
      !__builtin_mul_overflow(
          millis, Timestamp::kMicrosecondsInMillisecond, &micros),
      "Interval value of {} milliseconds is out of range for Spark's "
      "DayTimeIntervalType",
      millis);
  return micros;
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
  } else if constexpr (kind == TypeKind::TIMESTAMP) {
    return static_cast<double>(value.template castTo<Timestamp>().toMicros());
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
}

// Hashes a value the way Spark's HLL++ does, through the shared Spark xxhash64
// dispatch. Integer-like logical types (DATE, INTERVAL YEAR TO MONTH and short
// DECIMAL) hash their physical value, which matches Spark.
template <TypeKind kind>
uint64_t hashValueDispatch(
    const exec::GenericView& value,
    const TypePtr& type) {
  if constexpr (
      kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
      kind == TypeKind::INTEGER || kind == TypeKind::BIGINT ||
      kind == TypeKind::HUGEINT || kind == TypeKind::REAL ||
      kind == TypeKind::DOUBLE || kind == TypeKind::TIMESTAMP) {
    using T = typename TypeTraits<kind>::NativeType;
    return functions::sparksql::hashOne<SparkXxHash64>(
        value.template castTo<T>(), kXxHash64Seed);
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type for approx_count_distinct_for_intervals: {}",
        type->toString());
  }
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

  // Called once by the aggregation and window operators with the constant
  // arguments of the function call, before any input is processed.
  // Non-constant arguments are null.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    if (constantInputs.size() < 3) {
      return;
    }
    SelectivityVector rows(1);
    if (constantInputs[1] != nullptr) {
      checkEndpointsArgument(*constantInputs[1], rows);
    }
    if (constantInputs[2] != nullptr) {
      checkRelativeSdArgument(*constantInputs[2], rows);
    }
  }

  // Called by the adapter for every raw input batch. The first endpoints seen
  // initialize the intervals unless setConstantInputs() already did; every
  // selected row must then carry the same endpoints. A constant-encoded
  // argument is checked once, otherwise every selected row is checked, e.g.
  // when the constant was materialized by an exchange or a table scan.
  void checkEndpointsArgument(
      const BaseVector& endpointsVector,
      const SelectivityVector& rows) {
    VELOX_CHECK_NOT_NULL(endpointsElementType_);
    DecodedVector decodedEndpoints(endpointsVector, rows);
    if (decodedEndpoints.isConstantMapping()) {
      VELOX_USER_CHECK(
          !decodedEndpoints.isNullAt(rows.begin()),
          "Endpoints must not be null for approx_count_distinct_for_intervals");
    }

    const auto* arrayVector = decodedEndpoints.base()->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(arrayVector);
    DecodedVector decodedElements(*arrayVector->elements());
    exec::VectorReader<Generic<T2>> elementReader(&decodedElements);

    auto checkRow = [&](vector_size_t row) {
      VELOX_USER_CHECK(
          !decodedEndpoints.isNullAt(row),
          "Endpoints must not be null for approx_count_distinct_for_intervals");
      const auto arrayRow = decodedEndpoints.index(row);
      const auto offset = arrayVector->offsetAt(arrayRow);
      checkSetEndpoints(arrayVector->sizeAt(arrayRow), [&](vector_size_t i) {
        const auto elementRow = offset + i;
        VELOX_USER_CHECK(
            !decodedElements.isNullAt(elementRow),
            "Endpoints must not contain null values");
        return endpointToDouble(
            elementReader[elementRow], endpointsElementType_);
      });
    };

    if (decodedEndpoints.isConstantMapping()) {
      checkRow(rows.begin());
    } else {
      rows.applyToSelected(checkRow);
    }
  }

  // Same as checkEndpointsArgument() for the relativeSD argument.
  void checkRelativeSdArgument(
      const BaseVector& relativeSdVector,
      const SelectivityVector& rows) {
    DecodedVector decodedRelativeSd(relativeSdVector, rows);
    auto checkRow = [&](vector_size_t row) {
      VELOX_USER_CHECK(
          !decodedRelativeSd.isNullAt(row),
          "relativeSD must not be null for approx_count_distinct_for_intervals");
      checkSetRelativeSd(decodedRelativeSd.valueAt<double>(row));
    };

    if (decodedRelativeSd.isConstantMapping()) {
      checkRow(rows.begin());
    } else {
      rows.applyToSelected(checkRow);
    }
  }

  struct AccumulatorType {
    // Allocated through the HashStringAllocator so that the per-group HLL
    // array is accounted for by the query's memory pool.
    using HllVector = std::vector<HllAccumulator, StlAllocator<HllAccumulator>>;

    HllVector hlls;
    ApproxCountDistinctForIntervalsAggregate* fn;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool is_aligned_ = true;
    static constexpr bool use_external_memory_ = true;

    AccumulatorType(
        HashStringAllocator* allocator,
        ApproxCountDistinctForIntervalsAggregate* fn)
        : hlls{StlAllocator<HllAccumulator>(allocator)}, fn(fn) {}

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Generic<T1>> data,
        exec::optional_arg_type<Array<Generic<T2>>> /*endpoints*/,
        exec::optional_arg_type<double> /*relativeSd*/) {
      // The constant endpoints and relativeSD arguments are validated and
      // applied per batch by the adapter before any row is added.
      VELOX_CHECK(
          fn->endpointsSet_ && fn->indexBitLength_ >= 0,
          "approx_count_distinct_for_intervals received input rows before "
          "its constant arguments");

      if (!data.has_value()) {
        return false;
      }

      const double inputValue = fn->toDouble(data.value(), fn->inputType_);
      VELOX_USER_CHECK(
          !std::isnan(inputValue),
          "NaN input is rejected for approx_count_distinct_for_intervals");
      if (inputValue < fn->endpointsMin_ || inputValue > fn->endpointsMax_) {
        return false;
      }

      const auto intervalIndex = fn->findIntervalIndex(inputValue);
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
      VELOX_USER_CHECK(
          endpointsView.has_value() && hllsView.has_value(),
          "Malformed intermediate result for "
          "approx_count_distinct_for_intervals: endpoints and HLLs must not "
          "be null");

      // Every intermediate row carries the endpoints it was built with. Verify
      // they match, so that partial states of different calls are never merged.
      fn->checkSetEndpoints(endpointsView.value());

      const auto& hllsArray = hllsView.value();
      VELOX_USER_CHECK_EQ(
          hllsArray.size(),
          fn->intervalCount_,
          "HLL array size {} does not match the number of intervals {}",
          hllsArray.size(),
          fn->intervalCount_);

      for (const auto& entry : hllsArray) {
        VELOX_USER_CHECK(
            entry.has_value(),
            "Serialized HLL entries must not be null for "
            "approx_count_distinct_for_intervals");
        fn->checkIntermediateHll(entry.value());
      }

      ensureSize(allocator, fn->intervalCount_, fn->indexBitLength_);
      for (size_t i = 0; i < hllsArray.size(); ++i) {
        hlls[i].mergeWith(hllsArray[i].value(), allocator);
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!fn->endpointsSet_ || fn->indexBitLength_ < 0) {
        return false;
      }

      std::vector<std::string> serializedHlls;
      serializedHlls.reserve(fn->intervalCount_);
      for (int32_t interval = 0; interval < fn->intervalCount_; ++interval) {
        if (nonNullGroup &&
            hlls.size() == static_cast<size_t>(fn->intervalCount_)) {
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
        if (nonNullGroup &&
            hlls.size() == static_cast<size_t>(fn->intervalCount_)) {
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
    // Lazily creates one HLL per interval. Groups that never receive input
    // keep an empty vector.
    void ensureSize(
        HashStringAllocator* allocator,
        int32_t targetSize,
        int8_t indexBitLength) {
      if (!hlls.empty()) {
        VELOX_CHECK_EQ(hlls.size(), static_cast<size_t>(targetSize));
        return;
      }
      VELOX_CHECK_GE(indexBitLength, 0);
      hlls.reserve(targetSize);
      for (int32_t i = 0; i < targetSize; ++i) {
        hlls.emplace_back(indexBitLength, allocator);
      }
    }
  };

 private:
  // Initializes the HLL precision from relativeSD using the shared HLL
  // utilities, or verifies that 'relativeSd' matches the value already seen.
  void checkSetRelativeSd(double relativeSd) {
    if (!std::isnan(relativeSd_)) {
      VELOX_USER_CHECK_EQ(
          relativeSd,
          relativeSd_,
          "relativeSD must be constant for all input rows of "
          "approx_count_distinct_for_intervals");
      return;
    }

    common::hll::checkMaxStandardError(relativeSd);
    relativeSd_ = relativeSd;
    checkSetIndexBitLength(common::hll::toIndexBitLength(relativeSd));
  }

  // Sets the HLL precision, or verifies that 'indexBitLength' matches the
  // precision already in use.
  void checkSetIndexBitLength(int8_t indexBitLength) {
    if (indexBitLength_ < 0) {
      indexBitLength_ = indexBitLength;
      emptyHll_ = common::hll::SparseHlls::serializeEmpty(indexBitLength);
      return;
    }
    VELOX_USER_CHECK_EQ(
        static_cast<int32_t>(indexBitLength),
        static_cast<int32_t>(indexBitLength_),
        "Cannot merge HLLs with different number of buckets in "
        "approx_count_distinct_for_intervals");
  }

  // Validates a serialized HLL carried by an intermediate row, and verifies
  // that its precision matches the precision in use.
  void checkIntermediateHll(const StringView& serialized) {
    checkSetIndexBitLength(
        common::hll::checkSerializedHll(
            serialized.data(), static_cast<int32_t>(serialized.size())));
  }

  // Applies the endpoints carried by an intermediate row.
  void checkSetEndpoints(const exec::ArrayView<true, double>& endpointsView) {
    checkSetEndpoints(endpointsView.size(), [&](vector_size_t i) {
      const auto entry = endpointsView[i];
      VELOX_USER_CHECK(
          entry.has_value(), "Endpoints must not contain null values");
      return entry.value();
    });
  }

  // Initializes the intervals from the 'size' endpoints returned by
  // 'endpointAt', or verifies that they match the intervals already in use
  // without materializing them.
  template <typename TEndpointAt>
  void checkSetEndpoints(vector_size_t size, TEndpointAt endpointAt) {
    if (endpointsSet_) {
      VELOX_USER_CHECK_EQ(
          static_cast<size_t>(size),
          endpoints_.size(),
          "Endpoints must be constant for all input rows of "
          "approx_count_distinct_for_intervals");
      for (vector_size_t i = 0; i < size; ++i) {
        const double endpoint = endpointAt(i);
        VELOX_USER_CHECK_EQ(
            endpoint,
            endpoints_[i],
            "Endpoints must be constant for all input rows of "
            "approx_count_distinct_for_intervals");
      }
      return;
    }

    std::vector<double> endpoints;
    endpoints.reserve(size);
    for (vector_size_t i = 0; i < size; ++i) {
      endpoints.push_back(endpointAt(i));
    }
    setEndpoints(endpoints);
  }

  // Initializes the intervals from 'endpoints'.
  void setEndpoints(const std::vector<double>& endpoints) {
    VELOX_CHECK(!endpointsSet_);
    VELOX_USER_CHECK_GE(
        endpoints.size(),
        2,
        "approx_count_distinct_for_intervals requires at least 2 endpoints");
    for (const auto endpoint : endpoints) {
      VELOX_USER_CHECK(!std::isnan(endpoint), "Endpoints must not contain NaN");
    }
    for (size_t i = 1; i < endpoints.size(); ++i) {
      VELOX_USER_CHECK(
          !lessThan(endpoints[i], endpoints[i - 1]),
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

  static double toDouble(const exec::GenericView& value, const TypePtr& type) {
    if (type->isShortDecimal()) {
      return decimalToDouble(
          value.castTo<int64_t>(), type->asShortDecimal().scale());
    }
    if (type->isLongDecimal()) {
      return decimalToDouble(
          value.castTo<int128_t>(), type->asLongDecimal().scale());
    }
    if (type->isIntervalDayTime()) {
      return static_cast<double>(
          toSparkIntervalMicros(value.castTo<int64_t>()));
    }

    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        toDoubleDispatch, type->kind(), value, type);
  }

  // Spark converts endpoints with 'toString.toDouble', so a REAL endpoint takes
  // the value of its shortest decimal representation (0.1f becomes 0.1), not
  // the widened float (0.10000000149011612). Input values are widened, which
  // is what Spark does for them too.
  static double endpointToDouble(
      const exec::GenericView& value,
      const TypePtr& type) {
    if (type->kind() == TypeKind::REAL) {
      const auto floatValue = value.castTo<float>();
      if (!std::isfinite(floatValue)) {
        return static_cast<double>(floatValue);
      }
      return folly::to<double>(fmt::format("{}", floatValue));
    }
    return toDouble(value, type);
  }

  static uint64_t hashValue(
      const exec::GenericView& value,
      const TypePtr& type) {
    if (type->isIntervalDayTime()) {
      // Spark hashes DayTimeIntervalType's microseconds.
      return functions::sparksql::hashOne<SparkXxHash64>(
          toSparkIntervalMicros(value.castTo<int64_t>()), kXxHash64Seed);
    }
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        hashValueDispatch, type->kind(), value, type);
  }

  // Spark locates the interval with java.util.Arrays.binarySearch, which
  // orders -0.0 before 0.0. NaN never reaches here.
  static bool lessThan(double a, double b) {
    return a < b || (a == b && std::signbit(a) && !std::signbit(b));
  }

  // Mirrors Spark's findHllppIndex: values equal to an endpoint go to the
  // interval ending at the first endpoint with that value.
  int32_t findIntervalIndex(double value) const {
    auto it =
        std::lower_bound(endpoints_.begin(), endpoints_.end(), value, lessThan);
    if (it != endpoints_.end() && !lessThan(value, *it)) {
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
  // NaN until the constant relativeSD argument has been seen.
  double relativeSd_{std::numeric_limits<double>::quiet_NaN()};
  // -1 until the precision is known, either from relativeSD or from the
  // serialized HLLs of an intermediate row.
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
    checkConstantArguments(rows, args);
    Base::addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    checkConstantArguments(rows, args);
    Base::addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 private:
  // Verifies that the endpoints and relativeSD arguments of every selected
  // row match the values in use, initializing them from the first row when
  // the plan did not provide them as literals. Constant-encoded arguments are
  // checked once per batch.
  void checkConstantArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_USER_CHECK_EQ(
        args.size(),
        3,
        "approx_count_distinct_for_intervals requires relativeSD");

    if (!rows.hasSelections()) {
      return;
    }

    function().checkEndpointsArgument(*args[1], rows);
    function().checkRelativeSdArgument(*args[2], rows);
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
            "{} takes either 3 arguments (raw input) or 1 argument "
            "(intermediate input)",
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
