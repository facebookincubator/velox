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
#include <limits>
#include "velox/exec/AggregationHook.h"
#include "velox/functions/lib/CheckNestedNulls.h"
#include "velox/functions/lib/aggregates/Compare.h"
#include "velox/functions/lib/aggregates/SimpleNumericAggregate.h"
#include "velox/functions/lib/aggregates/SingleValueAccumulator.h"
#include "velox/type/FloatingPointUtil.h"

using namespace facebook::velox::aggregate;

namespace facebook::velox::functions::aggregate {

namespace {

template <typename T>
struct MinMaxTrait : public std::numeric_limits<T> {};

template <typename T>
class MinMaxAggregate : public SimpleNumericAggregate<T, T, T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit MinMaxAggregate(TypePtr resultType) : BaseAggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(T);
  }

  int32_t accumulatorAlignmentSize() const override {
    return 1;
  }

  bool supportsToIntermediate() const override {
    return true;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    this->singleInputAsIntermediate(rows, args, result);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<T>(
        groups, numGroups, result, [&](char* group) {
          return *BaseAggregate::Aggregate::template value<T>(group);
        });
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<T>(
        groups, numGroups, result, [&](char* group) {
          return *BaseAggregate::Aggregate::template value<T>(group);
        });
  }
};

/// Override 'accumulatorAlignmentSize' for UnscaledLongDecimal values as it
/// uses int128_t type. Some CPUs don't support misaligned access to int128_t
/// type.
template <>
inline int32_t MinMaxAggregate<int128_t>::accumulatorAlignmentSize() const {
  return static_cast<int32_t>(sizeof(int128_t));
}

// Truncate timestamps to milliseconds precision.
template <>
void MinMaxAggregate<Timestamp>::extractValues(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  BaseAggregate::template doExtractValues<Timestamp>(
      groups, numGroups, result, [&](char* group) {
        auto ts = *BaseAggregate::Aggregate::template value<Timestamp>(group);
        return Timestamp::fromMillis(ts.toMillis());
      });
}

template <typename T>
class MaxAggregate : public MinMaxAggregate<T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit MaxAggregate(TypePtr resultType) : MinMaxAggregate<T>(resultType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    // Re-enable pushdown for TIMESTAMP after
    // https://github.com/facebookincubator/velox/issues/6297 is fixed.
    if (args[0]->typeKind() == TypeKind::TIMESTAMP) {
      mayPushdown = false;
    }
    if (mayPushdown && args[0]->isLazy()) {
      BaseAggregate::template pushdown<MinMaxHook<T, false>>(
          groups, rows, args[0]);
      return;
    }
    BaseAggregate::template updateGroups<true, T>(
        groups, rows, args[0], updateGroup, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::updateOneGroup(
        group,
        rows,
        args[0],
        updateGroup,
        [](T& result, T value, int /* unused */) { result = value; },
        mayPushdown,
        kInitialValue_);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<T>(groups[i]) = kInitialValue_;
    }
  }

  static inline void updateGroup(T& result, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      if (util::floating_point::NaNAwareLessThan<T>{}(result, value)) {
        result = value;
      }
    } else {
      if (result < value) {
        result = value;
      }
    }
  }

 private:
  static const T kInitialValue_;
};

template <typename T>
const T MaxAggregate<T>::kInitialValue_ = MinMaxTrait<T>::lowest();

// Negative INF is the smallest value of floating point type.
template <>
const float MaxAggregate<float>::kInitialValue_ =
    -1 * MinMaxTrait<float>::infinity();

template <>
const double MaxAggregate<double>::kInitialValue_ =
    -1 * MinMaxTrait<double>::infinity();

template <typename T>
class MinAggregate : public MinMaxAggregate<T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit MinAggregate(TypePtr resultType) : MinMaxAggregate<T>(resultType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    // Re-enable pushdown for TIMESTAMP after
    // https://github.com/facebookincubator/velox/issues/6297 is fixed.
    if (args[0]->typeKind() == TypeKind::TIMESTAMP) {
      mayPushdown = false;
    }
    if (mayPushdown && args[0]->isLazy()) {
      BaseAggregate::template pushdown<MinMaxHook<T, true>>(
          groups, rows, args[0]);
      return;
    }
    BaseAggregate::template updateGroups<true, T>(
        groups, rows, args[0], updateGroup, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::updateOneGroup(
        group,
        rows,
        args[0],
        updateGroup,
        [](T& result, T value, int /* unused */) { result = value; },
        mayPushdown,
        kInitialValue_);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  static inline void updateGroup(T& result, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      if (util::floating_point::NaNAwareGreaterThan<T>{}(result, value)) {
        result = value;
      }
    } else {
      if (result > value) {
        result = value;
      }
    }
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<T>(groups[i]) = kInitialValue_;
    }
  }

 private:
  static const T kInitialValue_;
};

template <typename T>
const T MinAggregate<T>::kInitialValue_ = MinMaxTrait<T>::max();

// In velox, NaN is considered larger than infinity for floating point types.
template <>
const float MinAggregate<float>::kInitialValue_ =
    MinMaxTrait<float>::quiet_NaN();

template <>
const double MinAggregate<double>::kInitialValue_ =
    MinMaxTrait<double>::quiet_NaN();

class TimestampMicrosPrecisionMaxAggregate : public MaxAggregate<Timestamp> {
 public:
  explicit TimestampMicrosPrecisionMaxAggregate(TypePtr resultType)
      : MaxAggregate<Timestamp>(resultType) {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    using BaseAggregate =
        SimpleNumericAggregate<Timestamp, Timestamp, Timestamp>;
    BaseAggregate::template doExtractValues<Timestamp>(
        groups, numGroups, result, [&](char* group) {
          auto ts = *BaseAggregate::Aggregate::template value<Timestamp>(group);
          return Timestamp::fromMicros(ts.toMicros());
        });
  }
};

class TimestampMicrosPrecisionMinAggregate : public MinAggregate<Timestamp> {
 public:
  explicit TimestampMicrosPrecisionMinAggregate(TypePtr resultType)
      : MinAggregate<Timestamp>(resultType) {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    using BaseAggregate =
        SimpleNumericAggregate<Timestamp, Timestamp, Timestamp>;
    BaseAggregate::template doExtractValues<Timestamp>(
        groups, numGroups, result, [&](char* group) {
          auto ts = *BaseAggregate::Aggregate::template value<Timestamp>(group);
          return Timestamp::fromMicros(ts.toMicros());
        });
  }
};

class NonNumericMinMaxAggregateBase : public exec::Aggregate {
 public:
  explicit NonNumericMinMaxAggregateBase(
      const TypePtr& resultType,
      bool throwOnNestedNulls)
      : exec::Aggregate(resultType), throwOnNestedNulls_(throwOnNestedNulls) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SingleValueAccumulator);
  }

  bool supportsToIntermediate() const override {
    return true;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    const auto& input = args[0];

    if (throwOnNestedNulls_) {
      DecodedVector decoded(*input, rows, true);
      auto indices = decoded.indices();
      rows.applyToSelected([&](vector_size_t i) {
        velox::functions::checkNestedNulls(
            decoded, indices, i, throwOnNestedNulls_);
      });
    }

    if (rows.isAllSelected()) {
      result = input;
      return;
    }

    auto* pool = allocator_->pool();

    // Set result to NULL for rows that are masked out.
    BufferPtr nulls = allocateNulls(rows.size(), pool, bits::kNull);
    rows.clearNulls(nulls);

    BufferPtr indices = allocateIndices(rows.size(), pool);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    std::iota(rawIndices, rawIndices + rows.size(), 0);

    result = BaseVector::wrapInDictionary(nulls, indices, rows.size(), input);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    (*result)->resize(numGroups);

    uint64_t* rawNulls = nullptr;
    if ((*result)->mayHaveNulls()) {
      BufferPtr& nulls = (*result)->mutableNulls((*result)->size());
      rawNulls = nulls->asMutable<uint64_t>();
    }

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = value<SingleValueAccumulator>(group);
      if (!accumulator->hasValue()) {
        (*result)->setNull(i, true);
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }
        accumulator->read(*result, i);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    // partial and final aggregations are the same
    extractValues(groups, numGroups, result);
  }

 protected:
  template <typename TCompareTest, bool compareStopAtNull>
  void doUpdate(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TCompareTest compareTest) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    if (decoded.isConstantMapping() && decoded.isNullAt(0)) {
      // nothing to do; all values are nulls
      return;
    }

    rows.applyToSelected([&](vector_size_t i) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, i, throwOnNestedNulls_)) {
        return;
      }

      auto accumulator = value<SingleValueAccumulator>(groups[i]);
      if constexpr (compareStopAtNull) {
        if (!accumulator->hasValue() ||
            compareTest(compare(accumulator, decoded, i))) {
          accumulator->write(baseVector, indices[i], allocator_);
        }
      } else {
        if (!accumulator->hasValue() ||
            compareTest(compareWithNullAsValue(accumulator, decoded, i))) {
          accumulator->write(baseVector, indices[i], allocator_);
        }
      }
    });
  }

  template <typename TCompareTest, bool compareStopAtNull>
  void doUpdateSingleGroup(
      char* group,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      TCompareTest compareTest) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    if (decoded.isConstantMapping()) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, 0, throwOnNestedNulls_)) {
        return;
      }

      auto accumulator = value<SingleValueAccumulator>(group);
      if constexpr (compareStopAtNull) {
        if (!accumulator->hasValue() ||
            compareTest(compare(accumulator, decoded, 0))) {
          accumulator->write(baseVector, indices[0], allocator_);
        }
      } else {
        if (!accumulator->hasValue() ||
            compareTest(compareWithNullAsValue(accumulator, decoded, 0))) {
          accumulator->write(baseVector, indices[0], allocator_);
        }
      }
      return;
    }

    auto accumulator = value<SingleValueAccumulator>(group);
    rows.applyToSelected([&](vector_size_t i) {
      if (velox::functions::checkNestedNulls(
              decoded, indices, i, throwOnNestedNulls_)) {
        return;
      }

      if constexpr (compareStopAtNull) {
        if (!accumulator->hasValue() ||
            compareTest(compare(accumulator, decoded, i))) {
          accumulator->write(baseVector, indices[i], allocator_);
        }
      } else {
        if (!accumulator->hasValue() ||
            compareTest(compareWithNullAsValue(accumulator, decoded, i))) {
          accumulator->write(baseVector, indices[i], allocator_);
        }
      }
    });
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) SingleValueAccumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<SingleValueAccumulator>(group)->destroy(allocator_);
      }
    }
  }

 private:
  const bool throwOnNestedNulls_;
};

template <bool compareStopAtNull>
class NonNumericMaxAggregate : public NonNumericMinMaxAggregateBase {
 public:
  explicit NonNumericMaxAggregate(
      const TypePtr& resultType,
      bool throwOnNestedNulls)
      : NonNumericMinMaxAggregateBase(resultType, throwOnNestedNulls) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdate<std::function<bool(int32_t)>, compareStopAtNull>(
        groups, rows, args[0], [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdateSingleGroup<std::function<bool(int32_t)>, compareStopAtNull>(
        group, rows, args[0], [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }
};

template <bool compareStopAtNull>
class NonNumericMinAggregate : public NonNumericMinMaxAggregateBase {
 public:
  explicit NonNumericMinAggregate(
      const TypePtr& resultType,
      bool throwOnNestedNulls)
      : NonNumericMinMaxAggregateBase(resultType, throwOnNestedNulls) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdate<std::function<bool(int32_t)>, compareStopAtNull>(
        groups, rows, args[0], [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdateSingleGroup<std::function<bool(int32_t)>, compareStopAtNull>(
        group, rows, args[0], [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }
};

template <
    template <typename T>
    class TNumeric,
    template <bool compareStopAtNull>
    typename TNonNumeric,
    typename TTimestampMicros>
exec::AggregateFunctionFactory getMinMaxFunctionFactoryInternal(
    const std::string& name,
    bool nestedNullAllowed,
    bool mapTypeSupported,
    bool useMillisPrecision) {
  auto factory =
      [name, nestedNullAllowed, mapTypeSupported, useMillisPrecision](
          core::AggregationNode::Step step,
          std::vector<TypePtr> argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
      -> std::unique_ptr<exec::Aggregate> {
    const bool throwOnNestedNulls =
        !nestedNullAllowed && velox::exec::isRawInput(step);
    auto inputType = argTypes[0];
    switch (inputType->kind()) {
      case TypeKind::BOOLEAN:
        return std::make_unique<TNumeric<bool>>(resultType);
      case TypeKind::TINYINT:
        return std::make_unique<TNumeric<int8_t>>(resultType);
      case TypeKind::SMALLINT:
        return std::make_unique<TNumeric<int16_t>>(resultType);
      case TypeKind::INTEGER:
        return std::make_unique<TNumeric<int32_t>>(resultType);
      case TypeKind::BIGINT:
        return std::make_unique<TNumeric<int64_t>>(resultType);
      case TypeKind::REAL:
        return std::make_unique<TNumeric<float>>(resultType);
      case TypeKind::DOUBLE:
        return std::make_unique<TNumeric<double>>(resultType);
      case TypeKind::TIMESTAMP:
        if (useMillisPrecision) {
          return std::make_unique<TNumeric<Timestamp>>(resultType);
        } else {
          return std::make_unique<TTimestampMicros>(resultType);
        }
      case TypeKind::HUGEINT:
        return std::make_unique<TNumeric<int128_t>>(resultType);
      case TypeKind::VARBINARY:
        [[fallthrough]];
      case TypeKind::VARCHAR:
        return std::make_unique<TNonNumeric<false>>(inputType, false);
      case TypeKind::MAP:
        if (!mapTypeSupported) {
          VELOX_UNSUPPORTED(
              "Map type {} is not supported for aggregation {}",
              inputType->kindName(),
              name);
        }
        [[fallthrough]];
      case TypeKind::ARRAY:
        [[fallthrough]];
      case TypeKind::ROW:
        if (nestedNullAllowed) {
          return std::make_unique<TNonNumeric<false>>(
              inputType, throwOnNestedNulls);
        } else {
          return std::make_unique<TNonNumeric<true>>(
              inputType, throwOnNestedNulls);
        }
      case TypeKind::UNKNOWN:
        return std::make_unique<TNumeric<UnknownValue>>(resultType);
      default:
        VELOX_UNREACHABLE(
            "Unknown input type for {} aggregation {}",
            name,
            inputType->kindName());
    }
  };
  return factory;
}

} // namespace

exec::AggregateFunctionFactory getMinFunctionFactory(
    const std::string& name,
    bool nestedNullAllowed,
    bool mapTypeSupported,
    bool useMillisPrecision) {
  return getMinMaxFunctionFactoryInternal<
      MinAggregate,
      NonNumericMinAggregate,
      TimestampMicrosPrecisionMinAggregate>(
      name, nestedNullAllowed, mapTypeSupported, useMillisPrecision);
}

exec::AggregateFunctionFactory getMaxFunctionFactory(
    const std::string& name,
    bool nestedNullAllowed,
    bool mapTypeSupported,
    bool useMillisPrecision) {
  return getMinMaxFunctionFactoryInternal<
      MaxAggregate,
      NonNumericMaxAggregate,
      TimestampMicrosPrecisionMaxAggregate>(
      name, nestedNullAllowed, mapTypeSupported, useMillisPrecision);
}

} // namespace facebook::velox::functions::aggregate
