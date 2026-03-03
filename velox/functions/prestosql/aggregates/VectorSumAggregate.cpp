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
#include "velox/functions/prestosql/aggregates/VectorSumAggregate.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

template <typename T>
inline void addToSum(T& sum, T value) {
  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
    sum += value;
  } else {
    T checkedSum;
    auto overflow = __builtin_add_overflow(sum, value, &checkedSum);

    if (UNLIKELY(overflow)) {
      auto errorValue = (int128_t(sum) + int128_t(value));

      if (errorValue < 0) {
        VELOX_ARITHMETIC_ERROR(
            "Value {} is less than {}",
            errorValue,
            std::numeric_limits<T>::min());
      } else {
        VELOX_ARITHMETIC_ERROR(
            "Value {} exceeds {}", errorValue, std::numeric_limits<T>::max());
      }
    }
    sum = checkedSum;
  }
}

/// Simple aggregate implementation for vector_sum(array(T)) -> array(T).
/// This function takes arrays and sums corresponding elements at each position
/// across rows. The first array determines the expected size, and all
/// subsequent arrays must have the same size or an error is thrown.
/// Null elements are treated as 0.
template <typename T>
class SimpleVectorSumAggregate {
 public:
  using InputType = Row<Array<T>>;

  using IntermediateType = Array<T>;

  using OutputType = Array<T>;

  struct AccumulatorType {
    using SumsVector = std::vector<T, AlignedStlAllocator<T, 16>>;
    SumsVector sums_;
    bool initialized_{false};

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        SimpleVectorSumAggregate* /*fn*/)
        : sums_{AlignedStlAllocator<T, 16>(allocator)} {}

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Array<T>> arrayInput) {
      if (!initialized_) {
        sums_.resize(arrayInput.size());
        size_t idx = 0;
        for (const auto& element : arrayInput) {
          sums_[idx] = element.has_value() ? element.value() : T{0};
          ++idx;
        }
        initialized_ = true;
        return;
      }

      VELOX_USER_CHECK_EQ(
          arrayInput.size(),
          sums_.size(),
          "All arrays must have the same length. Expected {}, but got {}.",
          sums_.size(),
          arrayInput.size());

      size_t idx = 0;
      for (const auto& element : arrayInput) {
        if (element.has_value()) {
          auto value = element.value();
          if (value != T{0}) {
            addToSum(sums_[idx], value);
          }
        }
        ++idx;
      }
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Array<T>> other) {
      if (!initialized_) {
        sums_.resize(other.size());
        size_t idx = 0;
        for (const auto& element : other) {
          sums_[idx] = element.has_value() ? element.value() : T{0};
          ++idx;
        }
        initialized_ = true;
        return;
      }

      VELOX_USER_CHECK_EQ(
          other.size(),
          sums_.size(),
          "All arrays must have the same length. Expected {}, but got {}.",
          sums_.size(),
          other.size());

      size_t idx = 0;
      for (const auto& element : other) {
        if (element.has_value()) {
          auto value = element.value();
          if (value != T{0}) {
            addToSum(sums_[idx], value);
          }
        }
        ++idx;
      }
    }

    bool writeFinalResult(exec::out_type<Array<T>>& out) {
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }

    bool writeIntermediateResult(exec::out_type<Array<T>>& out) {
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }
  };
};

template <typename T>
std::unique_ptr<exec::Aggregate> createSimpleVectorSumAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  return std::make_unique<
      exec::SimpleAggregateAdapter<SimpleVectorSumAggregate<T>>>(
      step, argTypes, resultType);
}

} // namespace

void registerVectorSumAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  const std::vector<std::string> elementTypes = {
      "tinyint", "smallint", "integer", "bigint", "double", "real"};

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.reserve(elementTypes.size());
  for (const auto& elementType : elementTypes) {
    // array(T) -> array(T) signature.
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(fmt::format("array({})", elementType))
            .intermediateType(fmt::format("array({})", elementType))
            .argumentType(fmt::format("array({})", elementType))
            .build());
  }

  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1);
        VELOX_CHECK(argTypes[0]->isArray());

        auto& arrayType = argTypes[0]->asArray();
        auto elementTypeKind = arrayType.elementType()->kind();
        switch (elementTypeKind) {
          case TypeKind::TINYINT:
            return createSimpleVectorSumAggregate<int8_t>(
                step, argTypes, resultType);
          case TypeKind::SMALLINT:
            return createSimpleVectorSumAggregate<int16_t>(
                step, argTypes, resultType);
          case TypeKind::INTEGER:
            return createSimpleVectorSumAggregate<int32_t>(
                step, argTypes, resultType);
          case TypeKind::BIGINT:
            return createSimpleVectorSumAggregate<int64_t>(
                step, argTypes, resultType);
          case TypeKind::REAL:
            return createSimpleVectorSumAggregate<float>(
                step, argTypes, resultType);
          case TypeKind::DOUBLE:
            return createSimpleVectorSumAggregate<double>(
                step, argTypes, resultType);
          default:
            VELOX_UNREACHABLE(
                "Unexpected element type {}",
                TypeKindName::toName(elementTypeKind));
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
