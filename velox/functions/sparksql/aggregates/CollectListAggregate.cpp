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

#include "velox/functions/sparksql/aggregates/CollectListAggregate.h"

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowFast.h"

using namespace facebook::velox::aggregate;
using namespace facebook::velox::exec;

namespace facebook::velox::functions::aggregate::sparksql {
namespace {
struct ArrayAccumulator {
  ValueList elements;
};

class CollectListAggregate : public exec::Aggregate {
 public:
  CollectListAggregate(TypePtr resultType, std::optional<TypePtr> elementType)
      : Aggregate(resultType), elementType_(elementType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(ArrayAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<ArrayVector>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);

    auto elements = vector->elements();
    elements->resize(countElements(groups, numGroups));

    uint64_t* rawNulls = getRawNulls(vector);
    vector_size_t offset = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto& values = value<ArrayAccumulator>(groups[i])->elements;
      auto arraySize = values.size();
      // If the group's accumulator is null, the corresponding result is an
      // empty array rather than null.
      clearNull(rawNulls, i);
      if (arraySize) {
        ValueListReader reader(values);
        for (auto index = 0; index < arraySize; ++index) {
          reader.next(*elements, offset + index);
        }
      }
      vector->setOffsetAndSize(i, offset, arraySize);
      offset += arraySize;
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto vector = BaseVector::create(
        ROW({"data", "type"}, {ARRAY(elementType_.value()), VARCHAR()}),
        numGroups,
        allocator_->pool());
    auto rowVector = vector->asChecked<RowVector>();
    auto arrayVector = rowVector->childAt(0)->asChecked<ArrayVector>();
    auto flatVector =
        rowVector->childAt(1)->asChecked<FlatVector<StringView>>();
    auto elements = arrayVector->elements();
    elements->resize(countElements(groups, numGroups));

    auto typeStr = folly::toJson(elementType_.value()->serialize());
    vector_size_t offset = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      flatVector->set(i, StringView(typeStr));

      // To align with Spark's intermediate data, if the group's accumulator is
      // null, the corresponding result is an empty array.
      auto& values = value<ArrayAccumulator>(groups[i])->elements;
      auto arraySize = values.size();
      arrayVector->setNull(i, false);
      if (arraySize) {
        ValueListReader reader(values);
        for (auto index = 0; index < arraySize; ++index) {
          reader.next(*elements, offset + index);
        }
      }
      arrayVector->setOffsetAndSize(i, offset, arraySize);
      offset += arraySize;
    }

    auto flatResult = (*result)->asUnchecked<FlatVector<StringView>>();
    flatResult->resize(numGroups);
    RowVectorPtr rowVectorPtr = std::dynamic_pointer_cast<RowVector>(vector);
    auto serializer = std::make_unique<row::UnsafeRowFast>(rowVectorPtr);
    size_t totalSize = 0;
    for (vector_size_t i = 0; i < numGroups; ++i) {
      int32_t rowSize = serializer->rowSize(i);
      totalSize += rowSize;
    }

    char* rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);
    // RawBuffer must be set to all zeros.
    std::memset(rawBuffer, 0, totalSize);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      auto size = serializer->serialize(i, rawBuffer);
      VELOX_DCHECK(!StringView::isInline(size));
      StringView serialized = StringView(rawBuffer, size);
      rawBuffer += size;
      flatResult->setNoCopy(i, serialized);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedElements_.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedElements_.isNullAt(row)) {
        return;
      }
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      value<ArrayAccumulator>(group)->elements.appendValue(
          decodedElements_, row, allocator_);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](auto row) {
      if (!decodedIntermediate_.isNullAt(row)) {
        auto group = groups[row];
        auto tracker = trackRowSize(group);
        auto serialized = decodedIntermediate_.valueAt<StringView>(row);
        if (!elementType_.has_value()) {
          VectorPtr strVector =
              row::UnsafeRowDeserializer::deserializeStructField(
                  {std::string_view(serialized.data())},
                  VARBINARY(),
                  kTypeIndex,
                  kFieldNum,
                  allocator_->pool());
          elementType_ = Type::create(folly::parseJson(strVector->toString(0)));
        }
        VectorPtr dataVector =
            row::UnsafeRowDeserializer::deserializeStructField(
                {std::string_view(serialized.data())},
                ARRAY(elementType_.value()),
                kDataIndex,
                kFieldNum,
                allocator_->pool());
        auto arrayVector = dataVector->as<ArrayVector>();
        value<ArrayAccumulator>(group)->elements.appendRange(
            arrayVector->elements(),
            arrayVector->offsetAt(0),
            arrayVector->sizeAt(0),
            allocator_);
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    auto& values = value<ArrayAccumulator>(group)->elements;

    decodedElements_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedElements_.isNullAt(row)) {
        return;
      }
      values.appendValue(decodedElements_, row, allocator_);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](auto row) {
      if (!decodedIntermediate_.isNullAt(row)) {
        auto serialized = decodedIntermediate_.valueAt<StringView>(row);
        if (!elementType_.has_value()) {
          VectorPtr strVector =
              row::UnsafeRowDeserializer::deserializeStructField(
                  {std::string_view(serialized.data())},
                  VARBINARY(),
                  kTypeIndex,
                  kFieldNum,
                  allocator_->pool());
          elementType_ = Type::create(folly::parseJson(strVector->toString(0)));
        }
        VectorPtr dataVector =
            row::UnsafeRowDeserializer::deserializeStructField(
                {std::string_view(serialized.data())},
                ARRAY(elementType_.value()),
                kDataIndex,
                kFieldNum,
                allocator_->pool());
        auto arrayVector = dataVector->as<ArrayVector>();
        value<ArrayAccumulator>(group)->elements.appendRange(
            arrayVector->elements(),
            arrayVector->offsetAt(0),
            arrayVector->sizeAt(0),
            allocator_);
      }
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      new (groups[index] + offset_) ArrayAccumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<ArrayAccumulator>(group)->elements.free(allocator_);
      }
    }
  }

 private:
  vector_size_t countElements(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      size += value<ArrayAccumulator>(groups[i])->elements.size();
    }
    return size;
  }

  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedElements_;
  DecodedVector decodedIntermediate_;
  std::optional<TypePtr> elementType_;
  static constexpr int32_t kDataIndex{0};
  static constexpr int32_t kTypeIndex{1};
  static constexpr int32_t kFieldNum{2};
};

AggregateRegistrationResult registerCollectList(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("varbinary")
          .argumentType("E")
          .build()};
  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 1, "{} takes at most one argument", name);
        if (step == core::AggregationNode::Step::kIntermediate) {
          return std::make_unique<CollectListAggregate>(
              resultType, std::nullopt);
        } else if (step == core::AggregationNode::Step::kFinal) {
          return std::make_unique<CollectListAggregate>(resultType, resultType);
        }
        return std::make_unique<CollectListAggregate>(resultType, argTypes[0]);
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace

void registerCollectListAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerCollectList(
      prefix + "collect_list", withCompanionFunctions, overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
