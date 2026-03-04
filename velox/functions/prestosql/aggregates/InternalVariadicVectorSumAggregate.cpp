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
#include "velox/functions/prestosql/aggregates/InternalVariadicVectorSumAggregate.h"
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

/// Simple aggregate implementation for $internal$variadic_vector_sum.
/// Takes variadic scalar arguments (T, T, ..., T) and returns array(T).
/// Each argument position represents an element of a virtual array,
/// and sums are accumulated across rows.
/// Null values are treated as 0.
template <typename T>
class SimpleInternalVariadicVectorSumAggregate {
 public:
  using InputType = exec::AggregateInputType<Variadic<T>>;

  using IntermediateType = Array<T>;

  using OutputType = Array<T>;

  /// Disable default null behavior so that rows with null variadic elements
  /// are still processed (nulls are treated as 0).
  static constexpr bool default_null_behavior_ = false;

  struct AccumulatorType {
    using SumsVector = std::vector<T, AlignedStlAllocator<T, 16>>;
    SumsVector sums_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        SimpleInternalVariadicVectorSumAggregate* /*fn*/)
        : sums_{AlignedStlAllocator<T, 16>(allocator)} {}

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    /// Process variadic arguments, treating nulls as 0.
    /// Returns true to indicate the accumulator is non-null.
    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Variadic<T>> variadicArgs) {
      if (!variadicArgs.has_value()) {
        return !sums_.empty();
      }

      const auto& args = variadicArgs.value();
      const auto size = args.size();
      if (sums_.empty()) {
        sums_.resize(size, T{0});
        for (auto i = size; i-- > 0;) {
          if (args.at(i).has_value()) {
            sums_[i] = args.at(i).value();
          }
        }
        return !sums_.empty();
      }

      VELOX_USER_CHECK_EQ(
          size,
          sums_.size(),
          "All arrays must have the same length. Expected {}, but got {}.",
          sums_.size(),
          size);

      for (auto i = size; i-- > 0;) {
        if (args.at(i).has_value()) {
          auto value = args.at(i).value();
          if (value != T{0}) {
            addToSum(sums_[i], value);
          }
        }
      }
      return true;
    }

    /// Combine with intermediate result (array).
    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Array<T>> other) {
      if (!other.has_value()) {
        return !sums_.empty();
      }

      const auto& arr = other.value();
      const auto size = arr.size();
      if (sums_.empty()) {
        sums_.resize(size, T{0});
        for (auto i = size; i-- > 0;) {
          if (arr.at(i).has_value()) {
            sums_[i] = arr.at(i).value();
          }
        }
        return !sums_.empty();
      }

      VELOX_USER_CHECK_EQ(
          size,
          sums_.size(),
          "All arrays must have the same length. Expected {}, but got {}.",
          sums_.size(),
          size);

      for (auto i = size; i-- > 0;) {
        if (arr.at(i).has_value()) {
          auto value = arr.at(i).value();
          if (value != T{0}) {
            addToSum(sums_[i], value);
          }
        }
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<Array<T>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<Array<T>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }
  };
};

template <typename T>
std::unique_ptr<exec::Aggregate> createSimpleInternalVariadicVectorSumAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType) {
  return std::make_unique<exec::SimpleAggregateAdapter<
      SimpleInternalVariadicVectorSumAggregate<T>>>(step, argTypes, resultType);
}

} // namespace

void registerInternalVariadicVectorSumAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  const std::vector<std::string> elementTypes = {
      "tinyint", "smallint", "integer", "bigint", "double", "real"};

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.reserve(elementTypes.size());
  for (const auto& elementType : elementTypes) {
    // T, T... -> array(T) variadic signature.
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(fmt::format("array({})", elementType))
            .intermediateType(fmt::format("array({})", elementType))
            .argumentType(elementType)
            .variableArity(elementType)
            .build());
  }

  exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_GE(argTypes.size(), 1);

        // For merge companion functions, the input is already array(T),
        // so we need to extract the element type from the array.
        TypeKind elementTypeKind;
        if (argTypes[0]->kind() == TypeKind::ARRAY) {
          elementTypeKind = argTypes[0]->childAt(0)->kind();
        } else {
          elementTypeKind = argTypes[0]->kind();
        }

        switch (elementTypeKind) {
          case TypeKind::TINYINT:
            return createSimpleInternalVariadicVectorSumAggregate<int8_t>(
                step, argTypes, resultType);
          case TypeKind::SMALLINT:
            return createSimpleInternalVariadicVectorSumAggregate<int16_t>(
                step, argTypes, resultType);
          case TypeKind::INTEGER:
            return createSimpleInternalVariadicVectorSumAggregate<int32_t>(
                step, argTypes, resultType);
          case TypeKind::BIGINT:
            return createSimpleInternalVariadicVectorSumAggregate<int64_t>(
                step, argTypes, resultType);
          case TypeKind::REAL:
            return createSimpleInternalVariadicVectorSumAggregate<float>(
                step, argTypes, resultType);
          case TypeKind::DOUBLE:
            return createSimpleInternalVariadicVectorSumAggregate<double>(
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
