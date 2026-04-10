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
#include "velox/functions/lib/aggregates/SetBaseAggregate.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

// Spark collect_set aggregate with runtime ignoreNulls flag.
// The ignoreNulls_ flag is initialized via setConstantInputs() from the
// constant boolean argument provided at plan construction time.
template <
    typename T,
    typename AccumulatorType = velox::aggregate::prestosql::SetAccumulator<T>>
class SparkCollectSetAggregate
    : public SetAggAggregate<T, false, false, AccumulatorType> {
  using Base = SetAggAggregate<T, false, false, AccumulatorType>;
  using SBase = SetBaseAggregate<T, false, false, AccumulatorType>;

 public:
  explicit SparkCollectSetAggregate(const TypePtr& resultType)
      : Base(resultType) {}

  void setConstantInputs(
      const std::vector<VectorPtr>& constantInputs) override {
    // In the raw input step, constantInputs has 2 entries: [data, boolean].
    // In the intermediate/final step or companion functions, the boolean
    // constant may not be present — skip in those cases.
    if (constantInputs.size() >= 2 && constantInputs[1] != nullptr &&
        !constantInputs[1]->isNullAt(0)) {
      ignoreNulls_ = constantInputs[1]->as<ConstantVector<bool>>()->valueAt(0);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    SBase::decoded_.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t i) {
      auto* group = groups[i];
      SBase::clearNull(group);
      auto tracker = SBase::trackRowSize(group);
      if (ignoreNulls_) {
        SBase::value(group)->addNonNullValue(
            SBase::decoded_, i, SBase::allocator_);
      } else {
        SBase::value(group)->addValue(SBase::decoded_, i, SBase::allocator_);
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    SBase::decoded_.decode(*args[0], rows);
    SBase::clearNull(group);
    auto* accumulator = SBase::value(group);
    auto tracker = SBase::trackRowSize(group);
    rows.applyToSelected([&](vector_size_t i) {
      if (ignoreNulls_) {
        accumulator->addNonNullValue(SBase::decoded_, i, SBase::allocator_);
      } else {
        accumulator->addValue(SBase::decoded_, i, SBase::allocator_);
      }
    });
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    // When ignoreNulls is true, we must not wrap null input values into
    // [null] arrays, because the final/intermediate node uses the default
    // ignoreNulls_=false and would preserve those null elements. Instead,
    // output empty arrays for null elements so that addIntermediateResults
    // sees no null elements to preserve.
    if (!ignoreNulls_) {
      Base::toIntermediate(rows, args, result);
      return;
    }

    const auto& elements = args[0];
    const auto numRows = rows.size();

    auto* pool = SBase::allocator_->pool();
    BufferPtr nulls = allocateNulls(numRows, pool);
    memcpy(
        nulls->asMutable<uint64_t>(),
        rows.asRange().bits(),
        bits::nbytes(numRows));

    // For ignoreNulls=true: null input elements get size=0 (empty array),
    // non-null elements get size=1 as usual.
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();

    BufferPtr sizes = allocateSizes(numRows, pool);
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    vector_size_t offset = 0;
    rows.applyToSelected([&](vector_size_t i) {
      rawOffsets[i] = offset;
      if (elements->isNullAt(i)) {
        rawSizes[i] = 0;
      } else {
        rawSizes[i] = 1;
        offset++;
      }
    });

    // Build a compacted elements vector with only non-null values.
    auto nonNullCount = offset;
    auto compactedElements =
        BaseVector::create(elements->type(), nonNullCount, pool);
    vector_size_t idx = 0;
    rows.applyToSelected([&](vector_size_t i) {
      if (!elements->isNullAt(i)) {
        compactedElements->copy(elements.get(), idx++, i, 1);
      }
    });

    result = std::make_shared<ArrayVector>(
        pool,
        ARRAY(elements->type()),
        nulls,
        numRows,
        offsets,
        sizes,
        std::move(compactedElements));
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    SBase::decoded_.decode(*args[0], rows);
    auto baseArray = SBase::decoded_.base()->template as<ArrayVector>();
    SBase::decodedElements_.decode(*baseArray->elements());
    rows.applyToSelected([&](vector_size_t i) {
      if (SBase::decoded_.isNullAt(i)) {
        return;
      }
      auto* group = groups[i];
      SBase::clearNull(group);
      auto tracker = SBase::trackRowSize(group);
      auto decodedIndex = SBase::decoded_.index(i);
      // Intermediate results already have null filtering applied by the
      // partial step. Always preserve all elements (including nulls) here.
      SBase::value(group)->addValues(
          *baseArray, decodedIndex, SBase::decodedElements_, SBase::allocator_);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    SBase::decoded_.decode(*args[0], rows);
    auto baseArray = SBase::decoded_.base()->template as<ArrayVector>();
    SBase::decodedElements_.decode(*baseArray->elements());
    auto* accumulator = SBase::value(group);
    auto tracker = SBase::trackRowSize(group);
    rows.applyToSelected([&](vector_size_t i) {
      if (SBase::decoded_.isNullAt(i)) {
        return;
      }
      SBase::clearNull(group);
      auto decodedIndex = SBase::decoded_.index(i);
      // Intermediate results already have null filtering applied by the
      // partial step. Always preserve all elements (including nulls) here.
      accumulator->addValues(
          *baseArray, decodedIndex, SBase::decodedElements_, SBase::allocator_);
    });
  }

 private:
  // Default to true (Spark's default: IGNORE NULLS). Updated by
  // setConstantInputs() when a 2-arg signature provides explicit value.
  // Only used in addRawInput (partial/single step); intermediate/final
  // steps always preserve all elements from the partial output.
  bool ignoreNulls_{true};
};

std::unique_ptr<exec::Aggregate> createSetAgg(
    const TypeKind typeKind,
    const TypePtr& inputType,
    const TypePtr& resultType) {
  switch (typeKind) {
    case TypeKind::BOOLEAN:
      return std::make_unique<SparkCollectSetAggregate<bool>>(resultType);
    case TypeKind::TINYINT:
      return std::make_unique<SparkCollectSetAggregate<int8_t>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<SparkCollectSetAggregate<int16_t>>(resultType);
    case TypeKind::INTEGER:
      return std::make_unique<SparkCollectSetAggregate<int32_t>>(resultType);
    case TypeKind::BIGINT:
      return std::make_unique<SparkCollectSetAggregate<int64_t>>(resultType);
    case TypeKind::HUGEINT:
      VELOX_CHECK(
          inputType->isLongDecimal(),
          "Non-decimal use of HUGEINT is not supported");
      return std::make_unique<SparkCollectSetAggregate<int128_t>>(resultType);
    case TypeKind::REAL:
      return std::make_unique<SparkCollectSetAggregate<
          float,
          velox::aggregate::prestosql::FloatSetAccumulatorNaNUnaware<float>>>(
          resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<SparkCollectSetAggregate<
          double,
          velox::aggregate::prestosql::FloatSetAccumulatorNaNUnaware<double>>>(
          resultType);
    case TypeKind::TIMESTAMP:
      return std::make_unique<SparkCollectSetAggregate<Timestamp>>(resultType);
    case TypeKind::VARBINARY:
      [[fallthrough]];
    case TypeKind::VARCHAR:
      return std::make_unique<SparkCollectSetAggregate<StringView>>(resultType);
    case TypeKind::ARRAY:
      [[fallthrough]];
    case TypeKind::ROW:
      return std::make_unique<SparkCollectSetAggregate<ComplexType>>(
          resultType);
    case TypeKind::UNKNOWN:
      return std::make_unique<SparkCollectSetAggregate<UnknownValue>>(
          resultType);
    default:
      VELOX_UNSUPPORTED("Unsupported type {}", TypeKindName::toName(typeKind));
  }
}

} // namespace

void registerCollectSetAggAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures = {
      // collect_set(T) -> array(T): default ignoreNulls=true (Spark default).
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .intermediateType("array(T)")
          .argumentType("T")
          .build(),
      // collect_set(T, ignoreNulls) -> array(T): explicit ignoreNulls flag.
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .intermediateType("array(T)")
          .argumentType("T")
          .constantArgumentType("boolean")
          .build()};

  exec::registerAggregateFunction(
      prefix + "collect_set",
      std::move(signatures),
      [](core::AggregationNode::Step /*step*/,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        const TypePtr& inputType = argTypes[0];
        return createSetAgg(inputType->kind(), inputType, resultType);
      },
      {.ignoreDuplicates = true},
      withCompanionFunctions,
      overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
