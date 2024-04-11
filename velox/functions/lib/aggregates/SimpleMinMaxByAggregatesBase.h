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
#include <optional>
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

struct ComplexStore {
  ValueSet valueSet_;
  HashStringAllocator::Position position_;

  explicit ComplexStore(HashStringAllocator* allocator)
      : valueSet_(allocator) {}

  void Store(const BaseVector& vector, vector_size_t index) {
    valueSet_.write(vector, index, position_);
  }

  void Store(const StringView& value) {
    valueSet_.write(value);
  }
};

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
  using AccumulatorInternalType = ComplexStore;
};

template <typename C, typename V, bool isMin>
class MinMaxByAggregate {
 public:
  using InputType = Row<Any, Orderable<T1>>;
  using IntermediateType = Row<Any, Orderable<T1>>;
  using OutputType = Array<Any>;

  using ValueAccumulatorType =
      typename AccumulatorInternalTypeTraits<V>::AccumulatorType;
  using ComparisonAccumulatorType =
      typename AccumulatorInternalTypeTraits<C>::AccumulatorType;

  struct AccumulatorType {
    std::optional<ValueAccumulatorType> value_;
    std::optional<ComparisonAccumulatorType> comparisons_;

    explicit AccumulatorType(HashStringAllocator* allocator);

    void addInput(
        HashStringAllocator* allocator,
        exec::arg_type<Any> value,
        exec::arg_type<Orderable<T1>> comparison) {}

   private:
    void storeValue(
        HashStringAllocator* allocator,
        const exec::arg_type<Any>& value) {
      if constexpr (isNumeric<V>()) {
        value_.emplace(value.castTo<V>());
        return;
      }

      if (!value_.has_value()) {
        value_.emplace(allocator);
      }

      // We can store StringView type more optimally compared to complex types
      if constexpr (std::is_same_v<V, StringView>) {
        value_.value().Store(value.castTo<StringView>());
        return;
      }

      value_.value().Store(*value.base(), value.decodedIndex());
    }
  };
};
} // namespace facebook::velox::functions::aggregate
