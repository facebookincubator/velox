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

#include <common/memory/HashStringAllocator.h>
#include <exec/SimpleAggregateAdapter.h>
#include <type/SimpleFunctionApi.h>
#include <type/StringView.h>
#include <type/Timestamp.h>
#include <type/Type.h>
#include <cstdint>
#include <memory>
#include <optional>
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/lib/aggregates/SingleValueAccumulator.h"
#include "velox/functions/lib/aggregates/ValueSet.h"

// This is used for ValueSet
using namespace facebook::velox::aggregate;

namespace facebook::velox::functions::aggregate {

template <typename T>
constexpr bool isNumeric() {
  return std::is_same_v<T, bool> || std::is_same_v<T, int8_t> ||
      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
      std::is_same_v<T, int64_t> || std::is_same_v<T, float> ||
      std::is_same_v<T, double> || std::is_same_v<T, Timestamp>;
}

template <typename T, typename = void>
struct AccumulatorInternalTypeTraits {};

template <typename T>
struct AccumulatorInternalTypeTraits<
    T,
    std::enable_if_t<isNumeric<T>(), void>> {
  using AccumulatorInternalType = T;
};

template <typename T>
struct AccumulatorInternalTypeTraits<
    T,
    std::enable_if_t<!isNumeric<T>(), void>> {
  using AccumulatorInternalType = SingleValueAccumulator;
};

template <
    typename V,
    typename C,
    bool isMaxFunc,
    template <bool B, typename C1, typename C2>
    class Comparator,
    bool throwOnNestedNulls>
class SimpleMinMaxByAggregate {
 public:
  using InputType = Row<Generic<T1>, Orderable<T2>>;
  using IntermediateType = Row<Generic<T1>, Orderable<T2>>;
  using OutputType = Generic<T1>;

  using ValueAccumulatorType =
      typename AccumulatorInternalTypeTraits<V>::AccumulatorType;
  using ComparisonAccumulatorType =
      typename AccumulatorInternalTypeTraits<C>::AccumulatorType;

  // Default null behavior is false because for both spark sql and presto sql,
  // max/min_by can still be done if value is null
  static constexpr bool default_null_behavior_ = false;

  struct AccumulatorType {
    std::optional<ValueAccumulatorType> currValue_;
    std::optional<ComparisonAccumulatorType> currComparison_;

    explicit AccumulatorType(HashStringAllocator* /*allocator*/) {}

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Generic<T1>> value,
        exec::optional_arg_type<Orderable<T2>> comparison) {
      // Input will be ignored if comparison is null
      if (!comparison.has_value()) {
        return false;
      }
      if (needUpdate(comparison.value())) {
        update(allocator, value, comparison);
      }

      return true;
    }

    bool writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      if (!currComparison_.has_value()) {
        return false;
      }

      if (!currValue_.has_value()) {
        out.set_null_at<0>();
      } else {
        auto& valueWriter = out.get_writer_at<0>();
        if (isNumeric<V>()) {
          valueWriter.castTo<V>() = currValue_;
        } else {
          currValue_->read(valueWriter.base(), valueWriter.offset());
        }
      }

      auto& comparisonWriter = out.get_writer_at<1>();

      if (isNumeric<C>()) {
        comparisonWriter.castTo<C>() = currComparison_;
      } else {
        currComparison_->read(
            comparisonWriter.base(), comparisonWriter.offset());
      }

      return true;
    }

    bool combine(
        HashStringAllocator* allocator,
        exec::optional_arg_type<IntermediateType> other) {
      if (!other.has_value()) {
        return false;
      }
      auto value = other->at<0>();
      auto comparison = other->at<1>();

      if (!comparison.has_value()) {
        return false;
      }

      if (needUpdate(comparison.value())) {
        update(allocator, value, comparison);
      }

      return true;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      if (!currValue_.has_value()) {
        return false;
      }

      if (isNumeric<V>()) {
        out.castTo<V>() = currValue_;
      } else {
        currValue_->read(out.base(), out.offset());
      }

      return true;
    }

   private:
    bool needUpdate(const exec::GenericView& newComparison) {
      if (!currComparison_.has_value()) {
        return true;
      }

      if (isNumeric<C>()) {
        return Comparator<isMaxFunc, C, ComparisonAccumulatorType>::compare(
            currComparison_.value(), newComparison.castTo<C>());
      }

      if constexpr (throwOnNestedNulls) {
        VELOX_USER_CHECK(
            !newComparison.decoded().base()->containsNullAt(
                newComparison.decodedIndex()),
            "{} comparison not supported for values that contain nulls",
            mapTypeKindToName(newComparison.kind()));
      }

      return Comparator<isMaxFunc, C, ComparisonAccumulatorType>::compare(
          currComparison_.value(),
          newComparison.decoded(),
          newComparison.decodedIndex());
    }

    void update(
        HashStringAllocator* allocator,
        const exec::optional_arg_type<Generic<T1>>& value,
        const exec::optional_arg_type<Orderable<T2>>& comparison) {
      if (isNumeric<C>()) {
        currComparison_.emplace(comparison->castTo<C>());
      } else {
        currComparison_->write(
            comparison->base(), comparison->decodedIndex(), allocator);
      }

      if (!value.has_value()) {
        currValue_.reset();
      } else {
        if (isNumeric<V>()) {
          currValue_.emplace(value->castTo<V>());
        } else {
          currValue_.write(value->base(), value->decodedIndex(), allocator);
        }
      }
    }
  };
};

template <
    template <bool C0, typename C1, typename C2>
    class Comparator,
    bool isMaxFunc,
    typename V,
    bool throwOnNestedNulls>
std::unique_ptr<exec::Aggregate> create(
    TypePtr resultType,
    TypePtr compareType,
    const std::string& errorMessage) {
  switch (compareType->kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              bool,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::TINYINT:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              int8_t,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              int16_t,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::INTEGER:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              int32_t,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::BIGINT:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              int64_t,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::REAL:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              float,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              double,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::VARBINARY:
      [[fallthrough]];
    case TypeKind::VARCHAR:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              StringView,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::TIMESTAMP:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              Timestamp,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::MAP:
      [[fallthrough]];
    case TypeKind::ROW:
      return std::make_unique<
          exec::SimpleAggregateAdapter<SimpleMinMaxByAggregate<
              V,
              ComplexType,
              isMaxFunc,
              Comparator,
              throwOnNestedNulls>>>(resultType);
    default:
      VELOX_FAIL("{}", errorMessage);
      return nullptr;
  }
}

template <
    template <bool C0, typename C1, typename C2>
    class Comparator,
    bool isMaxFunc,
    bool throwOnNestedNulls>
std::unique_ptr<exec::Aggregate> createAll(
    TypePtr resultType,
    TypePtr valueType,
    TypePtr compareType,
    const std::string& errorMessage) {
  switch (valueType->kind()) {
    case TypeKind::BOOLEAN:
      return create<Comparator, isMaxFunc, bool, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::TINYINT:
      return create<Comparator, isMaxFunc, int8_t, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::SMALLINT:
      return create<Comparator, isMaxFunc, int16_t, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::INTEGER:
      return create<Comparator, isMaxFunc, int32_t, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::BIGINT:
      return create<Comparator, isMaxFunc, int64_t, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::REAL:
      return create<Comparator, isMaxFunc, float, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::DOUBLE:
      return create<Comparator, isMaxFunc, double, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY:
      return create<Comparator, isMaxFunc, StringView, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::TIMESTAMP:
      return create<Comparator, isMaxFunc, Timestamp, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::MAP:
      [[fallthrough]];
    case TypeKind::ROW:
      return create<Comparator, isMaxFunc, ComplexType, throwOnNestedNulls>(
          resultType, compareType, errorMessage);
    default:
      VELOX_FAIL(errorMessage);
  }
}

} // namespace facebook::velox::functions::aggregate
