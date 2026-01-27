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
#include "velox/functions/prestosql/aggregates/ArrayUnionSumAggregate.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

template <typename S>
struct Accumulator {
  using SumsVector = std::vector<S, AlignedStlAllocator<S, 16>>;
  SumsVector sums;

  explicit Accumulator(const TypePtr& /*type*/, HashStringAllocator* allocator)
      : sums{AlignedStlAllocator<S, 16>(allocator)} {}

  size_t size() const {
    return sums.size();
  }

  void addValues(
      const ArrayVector* arrayVector,
      const VectorPtr& arrayElements,
      vector_size_t row) {
    auto elements = arrayElements->template as<SimpleVector<S>>();
    auto offset = arrayVector->offsetAt(row);
    auto arraySize = arrayVector->sizeAt(row);

    if (static_cast<size_t>(arraySize) > sums.size()) {
      sums.resize(arraySize, S{0});
    }

    for (auto i = 0; i < arraySize; ++i) {
      if (!elements->isNullAt(offset + i)) {
        auto value = elements->valueAt(offset + i);

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
                  "Value {} exceeds {}",
                  errorValue,
                  std::numeric_limits<S>::max());
            }
          }
          sums[i] = checkedSum;
        }
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
};

template <typename S>
class ArrayUnionSumAggregate : public exec::Aggregate {
 public:
  explicit ArrayUnionSumAggregate(TypePtr resultType)
      : Aggregate(std::move(resultType)) {}

  using AccumulatorType = Accumulator<S>;

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
    decodedArrays_.decode(*args[0], rows);
    auto arrayVector = decodedArrays_.base()->template as<ArrayVector>();
    auto arrayElements = arrayVector->elements();

    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedArrays_.isNullAt(row)) {
        auto* group = groups[row];
        clearNull(group);

        auto tracker = trackRowSize(group);
        auto groupAccumulator = value<AccumulatorType>(group);
        addArray(*groupAccumulator, arrayVector, arrayElements, row);
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedArrays_.decode(*args[0], rows);
    auto arrayVector = decodedArrays_.base()->template as<ArrayVector>();
    auto arrayElements = arrayVector->elements();

    auto groupAccumulator = value<AccumulatorType>(group);

    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedArrays_.isNullAt(row)) {
        clearNull(group);
        addArray(*groupAccumulator, arrayVector, arrayElements, row);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addRawInput(groups, rows, args, false);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    addSingleGroupRawInput(group, rows, args, false);
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
  void addArray(
      AccumulatorType& groupAccumulator,
      const ArrayVector* arrayVector,
      const VectorPtr& arrayElements,
      vector_size_t row) const {
    auto decodedRow = decodedArrays_.index(row);
    groupAccumulator.addValues(arrayVector, arrayElements, decodedRow);
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

  DecodedVector decodedArrays_;
};

std::unique_ptr<exec::Aggregate> createArrayUnionSumAggregate(
    TypeKind elementKind,
    const TypePtr& resultType) {
  switch (elementKind) {
    case TypeKind::TINYINT:
      return std::make_unique<ArrayUnionSumAggregate<int8_t>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<ArrayUnionSumAggregate<int16_t>>(resultType);
    case TypeKind::INTEGER:
      return std::make_unique<ArrayUnionSumAggregate<int32_t>>(resultType);
    case TypeKind::BIGINT:
      return std::make_unique<ArrayUnionSumAggregate<int64_t>>(resultType);
    case TypeKind::REAL:
      return std::make_unique<ArrayUnionSumAggregate<float>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<ArrayUnionSumAggregate<double>>(resultType);
    default:
      VELOX_UNREACHABLE(
          "Unexpected element type {}", TypeKindName::toName(elementKind));
  }
}

} // namespace

void registerArrayUnionSumAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  const std::vector<std::string> elementTypes = {
      "tinyint", "smallint", "integer", "bigint", "double", "real"};

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.reserve(elementTypes.size());
  for (const auto& elementType : elementTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(fmt::format("array({})", elementType))
            .intermediateType(fmt::format("array({})", elementType))
            .argumentType(fmt::format("array({})", elementType))
            .build());
  }

  auto name = prefix + kArrayUnionSum;
  exec::registerAggregateFunction(
      name,
      signatures,
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1);
        VELOX_CHECK(argTypes[0]->isArray());
        auto& arrayType = argTypes[0]->asArray();
        auto elementTypeKind = arrayType.elementType()->kind();

        return createArrayUnionSumAggregate(elementTypeKind, resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
