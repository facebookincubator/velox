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
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

/// Accumulator for variadic vector sum.
/// Stores element-wise sums where each argument position represents
/// an array element, summed across rows.
template <typename S>
struct VariadicAccumulator {
  using SumsVector = std::vector<S, AlignedStlAllocator<S, 16>>;
  SumsVector sums;
  bool initialized{false};

  explicit VariadicAccumulator(
      const TypePtr& /*type*/,
      HashStringAllocator* allocator)
      : sums{AlignedStlAllocator<S, 16>(allocator)} {}

  size_t size() const {
    return sums.size();
  }

  void combineValues(
      const ArrayVector* arrayVector,
      const VectorPtr& arrayElements,
      vector_size_t row) {
    auto elements = arrayElements->template as<SimpleVector<S>>();
    auto offset = arrayVector->offsetAt(row);
    auto arraySize = arrayVector->sizeAt(row);

    if (!initialized) {
      sums.resize(arraySize);
      for (auto i = 0; i < arraySize; ++i) {
        sums[i] = elements->valueAt(offset + i);
      }
      initialized = true;
      return;
    }

    VELOX_USER_CHECK_EQ(
        static_cast<size_t>(arraySize),
        sums.size(),
        "All arrays must have the same length. Expected {}, but got {}.",
        sums.size(),
        arraySize);

    for (auto i = 0; i < arraySize; ++i) {
      auto value = elements->valueAt(offset + i);
      if (value != S{0}) {
        addToSum(i, value);
      }
    }
  }

  vector_size_t extractValues(VectorPtr& arrayElements, vector_size_t offset) {
    auto elements = arrayElements->asFlatVector<S>();

    for (size_t i = 0; i < sums.size(); ++i) {
      elements->set(offset + static_cast<vector_size_t>(i), sums[i]);
    }

    return static_cast<vector_size_t>(sums.size());
  }

  void addToSum(vector_size_t i, S value) {
    if constexpr (std::is_same_v<S, double> || std::is_same_v<S, float>) {
      sums[i] += value;
    } else {
      S checkedSum;
      auto overflow = __builtin_add_overflow(sums[i], value, &checkedSum);

      if (UNLIKELY(overflow)) {
        auto errorValue = (int128_t(sums[i]) + int128_t(value));

        if (errorValue < 0) {
          VELOX_ARITHMETIC_ERROR(
              "Value {} is less than {}",
              errorValue,
              std::numeric_limits<S>::min());
        } else {
          VELOX_ARITHMETIC_ERROR(
              "Value {} exceeds {}", errorValue, std::numeric_limits<S>::max());
        }
      }
      sums[i] = checkedSum;
    }
  }
};

/// Vector API implementation for $internal$variadic_vector_sum(c1, c2, ..., cN)
/// where each ci is a scalar. Each argument position represents an element
/// of a virtual array, and sums are accumulated across rows.
/// This properly handles nulls in any argument position using DecodedVector.
template <typename S>
class InternalVariadicVectorSumAggregate : public exec::Aggregate {
 public:
  explicit InternalVariadicVectorSumAggregate(
      TypePtr resultType,
      size_t numArgs)
      : Aggregate(std::move(resultType)), numArgs_(numArgs) {}

  using AccumulatorType = VariadicAccumulator<S>;

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(AccumulatorType);
  }

  bool isFixedSize() const override {
    return false;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto arrayVector = (*result)->as<ArrayVector>();
    VELOX_CHECK(arrayVector);
    arrayVector->resize(numGroups);

    auto arrayElementsPtr = arrayVector->elements();

    auto numElements = countElements(groups, numGroups);
    arrayVector->elements()->as<FlatVector<S>>()->resize(numElements);

    auto rawNulls = arrayVector->mutableRawNulls();
    vector_size_t offset = 0;
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        bits::setNull(rawNulls, i, true);
        arrayVector->setOffsetAndSize(i, 0, 0);
      } else {
        clearNull(rawNulls, i);

        auto arraySize = value<AccumulatorType>(group)->extractValues(
            arrayElementsPtr, offset);
        arrayVector->setOffsetAndSize(i, offset, arraySize);
        offset += arraySize;
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArgs(args, rows);

    rows.applyToSelected([&](vector_size_t row) {
      auto* group = groups[row];
      clearNull(group);
      auto tracker = trackRowSize(group);
      auto* acc = value<AccumulatorType>(group);

      if (!acc->initialized) {
        acc->sums.resize(args.size(), S{0});
        for (size_t argIdx = 0; argIdx < args.size(); ++argIdx) {
          if (!decodedArgs_[argIdx].isNullAt(row)) {
            acc->sums[argIdx] = decodedArgs_[argIdx].template valueAt<S>(row);
          }
        }
        acc->initialized = true;
      } else {
        VELOX_USER_CHECK_EQ(
            acc->size(),
            args.size(),
            "All arrays must have the same length. Expected {}, but got {}.",
            acc->size(),
            args.size());

        for (size_t argIdx = 0; argIdx < args.size(); ++argIdx) {
          if (!decodedArgs_[argIdx].isNullAt(row)) {
            auto val = decodedArgs_[argIdx].template valueAt<S>(row);
            if (val != S{0}) {
              acc->addToSum(argIdx, val);
            }
          }
        }
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodeArgs(args, rows);

    auto* acc = value<AccumulatorType>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      clearNull(group);

      if (!acc->initialized) {
        acc->sums.resize(args.size(), S{0});
        for (size_t argIdx = 0; argIdx < args.size(); ++argIdx) {
          if (!decodedArgs_[argIdx].isNullAt(row)) {
            acc->sums[argIdx] = decodedArgs_[argIdx].template valueAt<S>(row);
          }
        }
        acc->initialized = true;
      } else {
        VELOX_USER_CHECK_EQ(
            acc->size(),
            args.size(),
            "All arrays must have the same length. Expected {}, but got {}.",
            acc->size(),
            args.size());

        for (size_t argIdx = 0; argIdx < args.size(); ++argIdx) {
          if (!decodedArgs_[argIdx].isNullAt(row)) {
            auto val = decodedArgs_[argIdx].template valueAt<S>(row);
            if (val != S{0}) {
              acc->addToSum(argIdx, val);
            }
          }
        }
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);
    auto arrayVector = decodedIntermediate_.base()->template as<ArrayVector>();
    auto arrayElements = arrayVector->elements();

    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedIntermediate_.isNullAt(row)) {
        auto* group = groups[row];
        clearNull(group);

        auto tracker = trackRowSize(group);
        auto* acc = value<AccumulatorType>(group);
        auto decodedRow = decodedIntermediate_.index(row);
        acc->combineValues(arrayVector, arrayElements, decodedRow);
      }
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedIntermediate_.decode(*args[0], rows);
    auto arrayVector = decodedIntermediate_.base()->template as<ArrayVector>();
    auto arrayElements = arrayVector->elements();

    auto* acc = value<AccumulatorType>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedIntermediate_.isNullAt(row)) {
        clearNull(group);
        auto decodedRow = decodedIntermediate_.index(row);
        acc->combineValues(arrayVector, arrayElements, decodedRow);
      }
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto index : indices) {
      new (groups[index] + offset_) AccumulatorType{resultType_, allocator_};
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<AccumulatorType>(groups);
  }

 private:
  void decodeArgs(
      const std::vector<VectorPtr>& args,
      const SelectivityVector& rows) {
    decodedArgs_.resize(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      decodedArgs_[i].decode(*args[i], rows);
    }
  }

  vector_size_t countElements(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      if (!isNull(groups[i])) {
        size += value<AccumulatorType>(groups[i])->size();
      }
    }
    return size;
  }

  size_t numArgs_;
  std::vector<DecodedVector> decodedArgs_;
  DecodedVector decodedIntermediate_;
};

template <typename T>
std::unique_ptr<exec::Aggregate> createInternalVariadicVectorSumAggregate(
    const TypePtr& resultType,
    size_t numArgs) {
  return std::make_unique<InternalVariadicVectorSumAggregate<T>>(
      resultType, numArgs);
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
      std::move(signatures),
      [](core::AggregationNode::Step /*step*/,
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
            return createInternalVariadicVectorSumAggregate<int8_t>(
                resultType, argTypes.size());
          case TypeKind::SMALLINT:
            return createInternalVariadicVectorSumAggregate<int16_t>(
                resultType, argTypes.size());
          case TypeKind::INTEGER:
            return createInternalVariadicVectorSumAggregate<int32_t>(
                resultType, argTypes.size());
          case TypeKind::BIGINT:
            return createInternalVariadicVectorSumAggregate<int64_t>(
                resultType, argTypes.size());
          case TypeKind::REAL:
            return createInternalVariadicVectorSumAggregate<float>(
                resultType, argTypes.size());
          case TypeKind::DOUBLE:
            return createInternalVariadicVectorSumAggregate<double>(
                resultType, argTypes.size());
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
