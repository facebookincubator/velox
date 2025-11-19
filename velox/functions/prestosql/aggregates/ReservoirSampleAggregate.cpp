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

#include "velox/functions/prestosql/aggregates/ReservoirSampleAggregate.h"
#include "velox/functions/prestosql/aggregates/ReservoirSampleAccumulator.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

// Class definition
class ReservoirSampleAggregate : public exec::Aggregate {
 public:
  explicit ReservoirSampleAggregate(TypePtr resultType);

  int32_t accumulatorFixedWidthSize() const override;

  bool accumulatorUsesExternalMemory() const override;

  bool isFixedSize() const override;

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override;

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override;

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override;

  void destroyInternal(folly::Range<char**> groups) override;

 private:
  void shuffleVector(VectorPtr& vec);

  void initializeAccumulator(
      ReservoirSampleAccumulator* accumulator,
      const DecodedVector& initialSampleDecoded,
      const DecodedVector& initialProcessedCountDecoded,
      const DecodedVector& desiredSampleSizeDecoded,
      vector_size_t row,
      const TypePtr& elementType);

  void processInitialSample(
      ReservoirSampleAccumulator* accumulator,
      const DecodedVector& initialSampleDecoded,
      vector_size_t row,
      const TypePtr& elementType);

  void mergeIntermediateSamples(
      ReservoirSampleAccumulator* accumulator,
      const ArrayVector* sampleArrayVector,
      const ArrayVector* initialSampleArrayVector,
      const FlatVector<int64_t>* seenCountVector,
      vector_size_t decodedRow,
      const TypePtr& elementType);

  const TypePtr arrayType_;
  const TypePtr elementType_;
  const TypePtr extractValuesRowType_;
  const TypePtr extractAccumulatorRowType_;
};

// Constructor implementation
ReservoirSampleAggregate::ReservoirSampleAggregate(TypePtr resultType)
    : exec::Aggregate(resultType),
      arrayType_(resultType->childAt(1)),
      elementType_(arrayType_->asArray().elementType()),
      extractValuesRowType_(ROW({BIGINT(), arrayType_})),
      extractAccumulatorRowType_(ROW(
          {{"sample", ARRAY(elementType_)},
           {"initialSample", ARRAY(elementType_)},
           {"initialSeenCount", BIGINT()},
           {"seenCount", BIGINT()},
           {"maxSampleSize", INTEGER()}})) {}

int32_t ReservoirSampleAggregate::accumulatorFixedWidthSize() const {
  return sizeof(ReservoirSampleAccumulator);
}

bool ReservoirSampleAggregate::accumulatorUsesExternalMemory() const {
  return true;
}

bool ReservoirSampleAggregate::isFixedSize() const {
  return false;
}

void ReservoirSampleAggregate::extractValues(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  auto* pool = allocator_->pool();

  vector_size_t totalElements = 0;
  for (int32_t i = 0; i < numGroups; i++) {
    auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
    totalElements += accumulator->sampleCount();
  }

  auto processedCountVector = BaseVector::create(BIGINT(), numGroups, pool);
  auto* processedCounts =
      processedCountVector->asFlatVector<int64_t>()->mutableRawValues();

  auto elementVector = BaseVector::create(elementType_, totalElements, pool);
  auto offsets = allocateOffsets(numGroups, pool);
  auto sizes = allocateSizes(numGroups, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();

  vector_size_t offset = 0;
  for (int32_t i = 0; i < numGroups; i++) {
    auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
    auto sampleSize = accumulator->sampleCount();

    processedCounts[i] = accumulator->processedCount;
    rawOffsets[i] = offset;
    rawSizes[i] = sampleSize;

    if (sampleSize > 0) {
      elementVector->copy(accumulator->samples.get(), offset, 0, sampleSize);
      offset += sampleSize;
    }
  }

  auto sampleVector = std::make_shared<ArrayVector>(
      pool, arrayType_, nullptr, numGroups, offsets, sizes, elementVector);

  *result = std::make_shared<RowVector>(
      pool,
      extractValuesRowType_,
      nullptr,
      numGroups,
      std::vector<VectorPtr>{processedCountVector, sampleVector});
}

void ReservoirSampleAggregate::extractAccumulators(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  auto* pool = allocator_->pool();

  vector_size_t totalSampleElements = 0;
  vector_size_t totalInitialSampleElements = 0;

  for (int32_t i = 0; i < numGroups; i++) {
    auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
    totalSampleElements += accumulator->sampleCount();
    totalInitialSampleElements += accumulator->initialSampleCount();
  }

  auto elementsVector =
      BaseVector::create(elementType_, totalSampleElements, pool);
  auto offsets = allocateOffsets(numGroups, pool);
  auto sizes = allocateSizes(numGroups, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();

  auto initialSampleElementsVector =
      BaseVector::create(elementType_, totalInitialSampleElements, pool);
  auto initialSampleOffsets = allocateOffsets(numGroups, pool);
  auto initialSampleSizes = allocateSizes(numGroups, pool);
  auto* rawInitialSampleOffsets =
      initialSampleOffsets->asMutable<vector_size_t>();
  auto* rawInitialSampleSizes = initialSampleSizes->asMutable<vector_size_t>();

  auto initialSeenCountVector = BaseVector::create(BIGINT(), numGroups, pool);
  auto seenCountVector = BaseVector::create(BIGINT(), numGroups, pool);
  auto maxSampleSizeVector = BaseVector::create(INTEGER(), numGroups, pool);

  auto* initialSeenCounts =
      initialSeenCountVector->asFlatVector<int64_t>()->mutableRawValues();
  auto* seenCounts =
      seenCountVector->asFlatVector<int64_t>()->mutableRawValues();
  auto* maxSampleSizes =
      maxSampleSizeVector->asFlatVector<int32_t>()->mutableRawValues();

  vector_size_t offset = 0;
  vector_size_t initialSampleOffset = 0;

  for (int32_t i = 0; i < numGroups; i++) {
    auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
    auto sampleSize = accumulator->sampleCount();
    auto initialSampleSize = accumulator->initialSampleCount();

    rawOffsets[i] = offset;
    rawSizes[i] = sampleSize;
    if (sampleSize > 0) {
      elementsVector->copy(accumulator->samples.get(), offset, 0, sampleSize);
      offset += sampleSize;
    }

    rawInitialSampleOffsets[i] = initialSampleOffset;
    rawInitialSampleSizes[i] = initialSampleSize;
    if (initialSampleSize > 0) {
      initialSampleElementsVector->copy(
          accumulator->initialSamples.get(),
          initialSampleOffset,
          0,
          initialSampleSize);
      initialSampleOffset += initialSampleSize;
    }

    initialSeenCounts[i] = accumulator->initialSeenCount;
    seenCounts[i] = accumulator->processedCount;
    maxSampleSizes[i] = accumulator->maxSampleSize;
  }

  auto sampleVector = std::make_shared<ArrayVector>(
      pool,
      ARRAY(elementType_),
      nullptr,
      numGroups,
      offsets,
      sizes,
      elementsVector);

  auto initialSampleVector = std::make_shared<ArrayVector>(
      pool,
      ARRAY(elementType_),
      nullptr,
      numGroups,
      initialSampleOffsets,
      initialSampleSizes,
      initialSampleElementsVector);

  *result = std::make_shared<RowVector>(
      pool,
      extractAccumulatorRowType_,
      nullptr,
      numGroups,
      std::vector<VectorPtr>{
          sampleVector,
          initialSampleVector,
          initialSeenCountVector,
          seenCountVector,
          maxSampleSizeVector});
}

void ReservoirSampleAggregate::addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  VELOX_CHECK_EQ(args.size(), 4, "reservoir_sample requires 4 arguments");

  DecodedVector initialSampleDecoded(*args[0], rows);
  DecodedVector initialProcessedCountDecoded(*args[1], rows);
  DecodedVector valuesToSampleDecoded(*args[2], rows);
  DecodedVector desiredSampleSizeDecoded(*args[3], rows);

  auto elementType = args[2]->type();

  rows.applyToSelected([&](vector_size_t row_idx) {
    auto* accumulator = value<ReservoirSampleAccumulator>(groups[row_idx]);

    if (!accumulator->isInitialized()) {
      initializeAccumulator(
          accumulator,
          initialSampleDecoded,
          initialProcessedCountDecoded,
          desiredSampleSizeDecoded,
          row_idx,
          elementType);
    }

    accumulator->processedCount++;
    accumulator->addValueToReservoir(
        valuesToSampleDecoded, row_idx, elementType, allocator_->pool());
  });
}

void ReservoirSampleAggregate::addIntermediateResults(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  VELOX_CHECK_EQ(args.size(), 1, "intermediate results requires 1 argument");

  DecodedVector decodedIntermediate(*args[0], rows);
  auto* rowVector = decodedIntermediate.base()->as<RowVector>();
  auto* sampleArrayVector = rowVector->childAt(0)->as<ArrayVector>();
  auto* initialSampleArrayVector = rowVector->childAt(1)->as<ArrayVector>();
  auto* initialSeenCountVector = rowVector->childAt(2)->asFlatVector<int64_t>();
  auto* seenCountVector = rowVector->childAt(3)->asFlatVector<int64_t>();
  auto* maxSampleSizeVector = rowVector->childAt(4)->asFlatVector<int32_t>();

  auto elementType = sampleArrayVector->elements()->type();

  rows.applyToSelected([&](vector_size_t row) {
    if (decodedIntermediate.isNullAt(row)) {
      return;
    }

    auto* accumulator = value<ReservoirSampleAccumulator>(groups[row]);
    auto decodedRow = decodedIntermediate.index(row);

    int32_t maxSampleSize = maxSampleSizeVector->valueAt(decodedRow);
    if (maxSampleSize < 0) {
      return;
    }

    if (!accumulator->isInitialized()) {
      accumulator->maxSampleSize = maxSampleSize;
      accumulator->initialSeenCount =
          initialSeenCountVector->valueAt(decodedRow);
      accumulator->processedCount = seenCountVector->valueAt(decodedRow);
      auto initialSampleSize = initialSampleArrayVector->sizeAt(decodedRow);
      if (initialSampleSize > 0) {
        auto initialSampleOffset =
            initialSampleArrayVector->offsetAt(decodedRow);
        auto initialSampleElements = initialSampleArrayVector->elements();
        accumulator->initialSamples = BaseVector::create(
            elementType, initialSampleSize, allocator_->pool());
        accumulator->initialSamples->copy(
            initialSampleElements.get(),
            0,
            initialSampleOffset,
            initialSampleSize);
      }

      auto sampleSize = sampleArrayVector->sizeAt(decodedRow);
      if (sampleSize > 0) {
        auto sampleOffset = sampleArrayVector->offsetAt(decodedRow);
        auto sampleElements = sampleArrayVector->elements();

        accumulator->samples =
            BaseVector::create(elementType, sampleSize, allocator_->pool());
        accumulator->samples->copy(
            sampleElements.get(), 0, sampleOffset, sampleSize);
      }
      return;
    }

    mergeIntermediateSamples(
        accumulator,
        sampleArrayVector,
        initialSampleArrayVector,
        seenCountVector,
        decodedRow,
        elementType);
  });
}

void ReservoirSampleAggregate::addSingleGroupRawInput(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  VELOX_CHECK_EQ(args.size(), 4, "reservoir_sample requires 4 arguments");

  auto* accumulator = value<ReservoirSampleAccumulator>(group);

  DecodedVector initialSampleDecoded(*args[0], rows);
  DecodedVector initialProcessedCountDecoded(*args[1], rows);
  DecodedVector valuesToSampleDecoded(*args[2], rows);
  DecodedVector desiredSampleSizeDecoded(*args[3], rows);

  auto elementType = args[2]->type();

  if (!accumulator->isInitialized()) {
    vector_size_t firstRow = rows.begin();
    initializeAccumulator(
        accumulator,
        initialSampleDecoded,
        initialProcessedCountDecoded,
        desiredSampleSizeDecoded,
        firstRow,
        elementType);
  }

  rows.applyToSelected([&](vector_size_t row) {
    accumulator->processedCount++;
    accumulator->addValueToReservoir(
        valuesToSampleDecoded, row, elementType, allocator_->pool());
  });
}

void ReservoirSampleAggregate::addSingleGroupIntermediateResults(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  VELOX_CHECK_EQ(args.size(), 1, "intermediate results requires 1 argument");

  auto* accumulator = value<ReservoirSampleAccumulator>(group);
  DecodedVector decodedIntermediate(*args[0], rows);
  auto* rowVector = decodedIntermediate.base()->as<RowVector>();
  auto* sampleArrayVector = rowVector->childAt(0)->as<ArrayVector>();
  auto* initialSampleArrayVector = rowVector->childAt(1)->as<ArrayVector>();
  auto* initialSeenCountVector = rowVector->childAt(2)->asFlatVector<int64_t>();
  auto* seenCountVector = rowVector->childAt(3)->asFlatVector<int64_t>();
  auto* maxSampleSizeVector = rowVector->childAt(4)->asFlatVector<int32_t>();

  auto elementType = sampleArrayVector->elements()->type();

  rows.applyToSelected([&](vector_size_t row) {
    if (decodedIntermediate.isNullAt(row)) {
      return;
    }

    auto decodedRow = decodedIntermediate.index(row);
    int32_t maxSampleSize = maxSampleSizeVector->valueAt(decodedRow);

    if (maxSampleSize < 0) {
      return;
    }

    if (!accumulator->isInitialized()) {
      accumulator->maxSampleSize = maxSampleSize;
      accumulator->initialSeenCount =
          initialSeenCountVector->valueAt(decodedRow);
      accumulator->processedCount = seenCountVector->valueAt(decodedRow);
      auto initialSampleSize = initialSampleArrayVector->sizeAt(decodedRow);
      if (initialSampleSize > 0) {
        auto initialSampleOffset =
            initialSampleArrayVector->offsetAt(decodedRow);
        auto initialSampleElements = initialSampleArrayVector->elements();
        accumulator->initialSamples = BaseVector::create(
            elementType, initialSampleSize, allocator_->pool());
        accumulator->initialSamples->copy(
            initialSampleElements.get(),
            0,
            initialSampleOffset,
            initialSampleSize);
      }

      auto sampleSize = sampleArrayVector->sizeAt(decodedRow);
      if (sampleSize > 0) {
        auto sampleOffset = sampleArrayVector->offsetAt(decodedRow);
        auto sampleElements = sampleArrayVector->elements();
        accumulator->samples =
            BaseVector::create(elementType, sampleSize, allocator_->pool());
        accumulator->samples->copy(
            sampleElements.get(), 0, sampleOffset, sampleSize);
      }
      return;
    }

    mergeIntermediateSamples(
        accumulator,
        sampleArrayVector,
        initialSampleArrayVector,
        seenCountVector,
        decodedRow,
        elementType);
  });
}

void ReservoirSampleAggregate::initializeNewGroupsInternal(
    char** groups,
    folly::Range<const vector_size_t*> indices) {
  for (auto index : indices) {
    new (groups[index] + offset_) ReservoirSampleAccumulator();
  }
}

void ReservoirSampleAggregate::destroyInternal(folly::Range<char**> groups) {
  for (auto group : groups) {
    if (isInitialized(group)) {
      auto* accumulator = value<ReservoirSampleAccumulator>(group);
      std::destroy_at(accumulator);
    }
  }
}

void ReservoirSampleAggregate::shuffleVector(VectorPtr& vec) {
  if (!vec || vec->size() == 0) {
    return;
  }
  auto size = vec->size();
  auto tempVec = BaseVector::create(vec->type(), size, allocator_->pool());
  std::vector<vector_size_t> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  folly::ThreadLocalPRNG rng;
  std::shuffle(indices.begin(), indices.end(), rng);
  for (vector_size_t i = 0; i < size; i++) {
    tempVec->copy(vec.get(), i, indices[i], 1);
  }
  vec = tempVec;
}

void ReservoirSampleAggregate::initializeAccumulator(
    ReservoirSampleAccumulator* accumulator,
    const DecodedVector& initialSampleDecoded,
    const DecodedVector& initialProcessedCountDecoded,
    const DecodedVector& desiredSampleSizeDecoded,
    vector_size_t row,
    const TypePtr& elementType) {
  int32_t sampleSize = desiredSampleSizeDecoded.valueAt<int32_t>(row);
  VELOX_CHECK_GT(sampleSize, 0, "sample size must be positive");

  accumulator->maxSampleSize = sampleSize;
  accumulator->initialSeenCount =
      initialProcessedCountDecoded.valueAt<int64_t>(row);
  accumulator->processedCount = accumulator->initialSeenCount;

  if (!initialSampleDecoded.isNullAt(row)) {
    processInitialSample(accumulator, initialSampleDecoded, row, elementType);
  }
}

void ReservoirSampleAggregate::processInitialSample(
    ReservoirSampleAccumulator* accumulator,
    const DecodedVector& initialSampleDecoded,
    vector_size_t row,
    const TypePtr& elementType) {
  auto* arrayVector = initialSampleDecoded.base()->as<ArrayVector>();
  auto arrayIndex = initialSampleDecoded.index(row);
  auto arraySize = arrayVector->sizeAt(arrayIndex);
  auto arrayOffset = arrayVector->offsetAt(arrayIndex);

  if (arraySize > 0) {
    VELOX_CHECK(
        accumulator->initialSeenCount >= arraySize,
        "initialProcessedCount must be greater than or equal to the number of positions in the initial sample");
    auto elementVector = arrayVector->elements();
    accumulator->initialSamples =
        BaseVector::create(elementType, arraySize, allocator_->pool());
    accumulator->initialSamples->copy(
        elementVector.get(), 0, arrayOffset, arraySize);
    accumulator->samples =
        BaseVector::create(elementType, arraySize, allocator_->pool());
    accumulator->samples->copy(elementVector.get(), 0, arrayOffset, arraySize);
  }
}

void ReservoirSampleAggregate::mergeIntermediateSamples(
    ReservoirSampleAccumulator* accumulator,
    const ArrayVector* sampleArrayVector,
    const ArrayVector* initialSampleArrayVector,
    const FlatVector<int64_t>* seenCountVector,
    vector_size_t decodedRow,
    const TypePtr& elementType) {
  int64_t otherProcessedCount = seenCountVector->valueAt(decodedRow);
  auto otherSampleSize = sampleArrayVector->sizeAt(decodedRow);
  auto otherSampleOffset = sampleArrayVector->offsetAt(decodedRow);

  if (!accumulator->initialSamples) {
    auto initialSampleSize = initialSampleArrayVector->sizeAt(decodedRow);
    if (initialSampleSize > 0) {
      auto initialSampleOffset = initialSampleArrayVector->offsetAt(decodedRow);
      auto initialSampleElements = initialSampleArrayVector->elements();

      accumulator->initialSamples = BaseVector::create(
          elementType, initialSampleSize, allocator_->pool());
      accumulator->initialSamples->copy(
          initialSampleElements.get(),
          0,
          initialSampleOffset,
          initialSampleSize);
    }
  }
  if (otherSampleSize == 0) {
    return;
  }

  auto sampleElements = sampleArrayVector->elements();
  DecodedVector decodedSampleElements(*sampleElements);

  if (otherProcessedCount < accumulator->maxSampleSize) {
    for (vector_size_t i = 0; i < otherSampleSize; i++) {
      accumulator->processedCount++;
      auto elementIndex = otherSampleOffset + i;
      accumulator->addValueToReservoir(
          decodedSampleElements, elementIndex, elementType, allocator_->pool());
    }
    return;
  }

  if (accumulator->processedCount < accumulator->maxSampleSize) {
    auto thisSampleSize = accumulator->sampleCount();
    if (thisSampleSize > 0) {
      auto tempSamples =
          BaseVector::create(elementType, thisSampleSize, allocator_->pool());
      tempSamples->resize(thisSampleSize);
      tempSamples->copy(accumulator->samples.get(), 0, 0, thisSampleSize);
      accumulator->samples =
          BaseVector::create(elementType, otherSampleSize, allocator_->pool());
      accumulator->samples->copy(
          sampleElements.get(), 0, otherSampleOffset, otherSampleSize);
      accumulator->processedCount = otherProcessedCount;

      DecodedVector decodedTempSamples(*tempSamples);
      for (vector_size_t i = 0; i < thisSampleSize; i++) {
        accumulator->processedCount++;
        accumulator->addValueToReservoir(
            decodedTempSamples, i, elementType, allocator_->pool());
      }
    } else {
      accumulator->samples =
          BaseVector::create(elementType, otherSampleSize, allocator_->pool());
      accumulator->samples->copy(
          sampleElements.get(), 0, otherSampleOffset, otherSampleSize);
      accumulator->processedCount = otherProcessedCount;
    }
    return;
  }

  shuffleVector(accumulator->samples);

  auto otherSamplesCopy =
      BaseVector::create(elementType, otherSampleSize, allocator_->pool());
  otherSamplesCopy->copy(
      sampleElements.get(), 0, otherSampleOffset, otherSampleSize);

  shuffleVector(otherSamplesCopy);

  int64_t thisProcessedCount = accumulator->processedCount;
  int64_t totalProcessedCount = thisProcessedCount + otherProcessedCount;
  int otherNextIndex = 0;
  folly::ThreadLocalPRNG rng;
  for (vector_size_t i = 0; i < accumulator->maxSampleSize; i++) {
    std::uniform_int_distribution<int64_t> dist(0, totalProcessedCount - 1);
    int64_t randomValue = dist(rng);
    if (randomValue >= thisProcessedCount) {
      accumulator->samples->copy(otherSamplesCopy.get(), i, otherNextIndex, 1);
      otherNextIndex++;
    }
  }

  accumulator->processedCount = totalProcessedCount;
}

} // namespace

void registerReservoirSampleAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  for (const auto& inputType : {
           "tinyint",
           "smallint",
           "integer",
           "bigint",
           "real",
           "double",
           "varchar",
           "boolean",
           "timestamp",
       }) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(
                fmt::format(
                    "row(processed_count bigint, sample array({}))", inputType))
            .intermediateType(
                fmt::format(
                    "row(sample array({}), initialSample array({}), initialSeenCount bigint, "
                    "seenCount bigint, maxSampleSize integer)",
                    inputType,
                    inputType))
            .argumentType(fmt::format("array({})", inputType))
            .argumentType("bigint")
            .argumentType(inputType)
            .argumentType("integer")
            .build());
  }

  auto name = prefix + "reservoir_sample";
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 4, "reservoir_sample requires 4 arguments");

        return std::make_unique<ReservoirSampleAggregate>(resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
