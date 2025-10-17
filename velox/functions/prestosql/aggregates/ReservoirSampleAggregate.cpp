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

#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/functions/prestosql/aggregates/ReservoirSampleAccumulator.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

using facebook::velox::aggregate::ValueList;
using facebook::velox::aggregate::ValueListReader;

namespace facebook::velox::aggregate::prestosql {

namespace {

class ReservoirSampleGenericAggregate : public exec::Aggregate {
 public:
  explicit ReservoirSampleGenericAggregate(TypePtr resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(ReservoirSampleAccumulator);
  }

  bool accumulatorUsesExternalMemory() const override {
    return true;
  }

  bool isFixedSize() const override {
    return false;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* pool = allocator_->pool();
    auto arrayType = this->resultType()->childAt(1);
    auto rowType = ROW({BIGINT(), arrayType});

    auto processedCountVector = BaseVector::create(BIGINT(), numGroups, pool);
    auto totalSampleElements = calculateTotalSampleElements(groups, numGroups);

    // Create ArrayVector for samples
    auto sampleVector = createSampleArrayVector(
        groups, numGroups, arrayType, totalSampleElements, pool);

    // Fill processed counts
    fillProcessedCounts(groups, numGroups, processedCountVector);

    *result = std::make_shared<RowVector>(
        pool,
        rowType,
        nullptr,
        numGroups,
        std::vector<VectorPtr>{processedCountVector, sampleVector});
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* pool = allocator_->pool();
    auto elementType = getElementTypeFromIntermediate();

    auto rowType = ROW(
        {{"initialSample", ARRAY(elementType)},
         {"initialSeenCount", BIGINT()},
         {"seenCount", BIGINT()},
         {"maxSampleSize", INTEGER()},
         {"sample", ARRAY(elementType)}});

    // Calculate total elements for both arrays
    auto [totalInitialElements, totalSampleElements] =
        calculateTotalElements(groups, numGroups);

    // Create vectors
    auto initialSampleVector = createArrayVectorFromValueLists(
        groups, numGroups, elementType, totalInitialElements, pool, true);
    auto sampleVector = createArrayVectorFromValueLists(
        groups, numGroups, elementType, totalSampleElements, pool, false);

    auto initialSeenCountVector = BaseVector::create(BIGINT(), numGroups, pool);
    auto seenCountVector = BaseVector::create(BIGINT(), numGroups, pool);
    auto maxSampleSizeVector = BaseVector::create(INTEGER(), numGroups, pool);

    // Fill scalar values
    fillIntermediateScalarData(
        groups,
        numGroups,
        initialSeenCountVector,
        seenCountVector,
        maxSampleSizeVector);

    *result = std::make_shared<RowVector>(
        pool,
        rowType,
        nullptr,
        numGroups,
        std::vector<VectorPtr>{
            initialSampleVector,
            initialSeenCountVector,
            seenCountVector,
            maxSampleSizeVector,
            sampleVector});
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK_EQ(args.size(), 4, "reservoir_sample requires 4 arguments");

    DecodedVector initialSampleDecoded(*args[0], rows);
    DecodedVector initialProcessedCountDecoded(*args[1], rows);
    DecodedVector valuesToSampleDecoded(*args[2], rows);
    DecodedVector desiredSampleSizeDecoded(*args[3], rows);

    rows.applyToSelected([&](vector_size_t row_idx) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[row_idx]);

      if (!accumulator->isInitialized()) {
        LOG(INFO) << "[addRawInput] Initializing accumulator for row "
                  << row_idx;
        initializeAccumulator(
            accumulator,
            initialSampleDecoded,
            initialProcessedCountDecoded,
            desiredSampleSizeDecoded,
            row_idx);
      }

      // Add new value using reservoir sampling
      // Note: We process null values too, matching Java's behavior
      accumulator->processedCount++;
      LOG(INFO) << "[addRawInput] Adding value for row " << row_idx
                << ", processedCount now: " << accumulator->processedCount;
      accumulator->addValueToReservoir(
          valuesToSampleDecoded, row_idx, allocator_);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK_EQ(args.size(), 1, "intermediate results requires 1 argument");

    DecodedVector decodedIntermediate(*args[0], rows);
    auto* rowVector = decodedIntermediate.base()->as<RowVector>();

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<ReservoirSampleAccumulator>(groups[row]);
      auto decodedRow = decodedIntermediate.index(row);

      mergeIntermediateResult(accumulator, rowVector, decodedRow);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK_EQ(args.size(), 4, "reservoir_sample requires 4 arguments");

    auto* accumulator = value<ReservoirSampleAccumulator>(group);

    DecodedVector initialSampleDecoded(*args[0], rows);
    DecodedVector initialProcessedCountDecoded(*args[1], rows);
    DecodedVector valuesToSampleDecoded(*args[2], rows);
    DecodedVector desiredSampleSizeDecoded(*args[3], rows);

    if (!accumulator->isInitialized()) {
      vector_size_t firstRow = rows.begin();
      initializeAccumulator(
          accumulator,
          initialSampleDecoded,
          initialProcessedCountDecoded,
          desiredSampleSizeDecoded,
          firstRow);
    }

    // Process all input values
    // Note: We process null values too, matching Java's behavior
    rows.applyToSelected([&](vector_size_t row) {
      accumulator->processedCount++;
      accumulator->addValueToReservoir(valuesToSampleDecoded, row, allocator_);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK_EQ(args.size(), 1, "intermediate results requires 1 argument");

    auto* accumulator = value<ReservoirSampleAccumulator>(group);
    DecodedVector decodedIntermediate(*args[0], rows);
    auto* rowVector = decodedIntermediate.base()->as<RowVector>();

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate.isNullAt(row)) {
        return;
      }

      auto decodedRow = decodedIntermediate.index(row);
      mergeIntermediateResult(accumulator, rowVector, decodedRow);
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      new (groups[index] + offset_) ReservoirSampleAccumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        auto* accumulator = value<ReservoirSampleAccumulator>(group);
        accumulator->samples.free(allocator_);
        accumulator->initialSamples.free(allocator_);
        std::destroy_at(accumulator);
      }
    }
  }

 private:
  TypePtr getElementTypeFromIntermediate() const {
    auto child = this->resultType()->childAt(0);
    VELOX_CHECK(
        child->isArray(),
        "Expected first child of intermediate type to be array, got: {}",
        child->toString());
    return child->asArray().elementType();
  }

  vector_size_t calculateTotalSampleElements(char** groups, int32_t numGroups)
      const {
    vector_size_t total = 0;
    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
      total += accumulator->samples.size();
    }
    return total;
  }

  std::pair<vector_size_t, vector_size_t> calculateTotalElements(
      char** groups,
      int32_t numGroups) const {
    vector_size_t totalInitial = 0;
    vector_size_t totalSample = 0;

    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
      totalInitial += accumulator->initialSamples.size();
      totalSample += accumulator->samples.size();
    }
    return {totalInitial, totalSample};
  }

  ArrayVectorPtr createSampleArrayVector(
      char** groups,
      int32_t numGroups,
      const TypePtr& arrayType,
      vector_size_t totalElements,
      memory::MemoryPool* pool) const {
    auto elementType = arrayType->asArray().elementType();
    auto sampleElementsVector =
        BaseVector::create(elementType, totalElements, pool);

    BufferPtr offsets =
        AlignedBuffer::allocate<vector_size_t>(numGroups + 1, pool);
    BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(numGroups, pool);

    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    vector_size_t elementIndex = 0;
    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);

      rawOffsets[i] = elementIndex;
      rawSizes[i] = accumulator->samples.size();

      if (accumulator->samples.size() > 0) {
        ValueListReader reader(const_cast<ValueList&>(accumulator->samples));
        for (vector_size_t j = 0; j < accumulator->samples.size(); j++) {
          reader.next(*sampleElementsVector, elementIndex + j);
        }
      }
      elementIndex += accumulator->samples.size();
    }
    rawOffsets[numGroups] = totalElements;

    return std::make_shared<ArrayVector>(
        pool,
        arrayType,
        nullptr,
        numGroups,
        offsets,
        sizes,
        sampleElementsVector);
  }

  ArrayVectorPtr createArrayVectorFromValueLists(
      char** groups,
      int32_t numGroups,
      const TypePtr& elementType,
      vector_size_t totalElements,
      memory::MemoryPool* pool,
      bool useInitialSamples) const {
    auto elementsVector = BaseVector::create(elementType, totalElements, pool);

    BufferPtr offsets =
        AlignedBuffer::allocate<vector_size_t>(numGroups + 1, pool);
    BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(numGroups, pool);

    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    vector_size_t elementIndex = 0;
    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);

      const ValueList& valueList = useInitialSamples
          ? accumulator->initialSamples
          : accumulator->samples;

      rawOffsets[i] = elementIndex;
      rawSizes[i] = valueList.size();

      if (valueList.size() > 0) {
        ValueListReader reader(const_cast<ValueList&>(valueList));
        for (vector_size_t j = 0; j < valueList.size(); j++) {
          reader.next(*elementsVector, elementIndex + j);
        }
      }
      elementIndex += valueList.size();
    }
    rawOffsets[numGroups] = totalElements;

    return std::make_shared<ArrayVector>(
        pool,
        ARRAY(elementType),
        nullptr,
        numGroups,
        offsets,
        sizes,
        elementsVector);
  }

  void fillProcessedCounts(
      char** groups,
      int32_t numGroups,
      const VectorPtr& processedCountVector) const {
    auto* flatVector = processedCountVector->as<FlatVector<int64_t>>();
    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
      flatVector->set(i, accumulator->processedCount);
    }
  }

  void fillIntermediateScalarData(
      char** groups,
      int32_t numGroups,
      const VectorPtr& initialSeenCountVector,
      const VectorPtr& seenCountVector,
      const VectorPtr& maxSampleSizeVector) const {
    auto* initialSeenCountFlat =
        initialSeenCountVector->as<FlatVector<int64_t>>();
    auto* seenCountFlat = seenCountVector->as<FlatVector<int64_t>>();
    auto* maxSampleSizeFlat = maxSampleSizeVector->as<FlatVector<int32_t>>();

    for (int32_t i = 0; i < numGroups; i++) {
      auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
      initialSeenCountFlat->set(i, accumulator->initialSeenCount);
      seenCountFlat->set(i, accumulator->processedCount);
      maxSampleSizeFlat->set(
          i, static_cast<int32_t>(accumulator->maxSampleSize));
    }
  }

  void initializeAccumulator(
      ReservoirSampleAccumulator* accumulator,
      const DecodedVector& initialSampleDecoded,
      const DecodedVector& initialProcessedCountDecoded,
      const DecodedVector& desiredSampleSizeDecoded,
      vector_size_t row) {
    new (accumulator) ReservoirSampleAccumulator();

    int32_t sampleSize = desiredSampleSizeDecoded.valueAt<int32_t>(row);
    VELOX_CHECK_GT(sampleSize, 0, "sample size must be positive");

    accumulator->maxSampleSize = sampleSize;
    accumulator->initialSeenCount =
        initialProcessedCountDecoded.valueAt<int64_t>(row);
    accumulator->processedCount = accumulator->initialSeenCount;

    // Process initial sample if present
    if (!initialSampleDecoded.isNullAt(row)) {
      processInitialSample(accumulator, initialSampleDecoded, row);
    }
  }

  void processInitialSample(
      ReservoirSampleAccumulator* accumulator,
      const DecodedVector& initialSampleDecoded,
      vector_size_t row) {
    auto* arrayVector = initialSampleDecoded.base()->as<ArrayVector>();
    auto arrayIndex = initialSampleDecoded.index(row);
    auto arraySize = arrayVector->sizeAt(arrayIndex);
    auto arrayOffset = arrayVector->offsetAt(arrayIndex);

    LOG(INFO) << "[processInitialSample] Processing initial sample. "
              << "arraySize: " << arraySize
              << ", initialSeenCount: " << accumulator->initialSeenCount
              << ", maxSampleSize: " << accumulator->maxSampleSize;

    // When initial_processed_count > 0, the initial sample size must equal
    // min(maxSampleSize, initialSeenCount)
    if (accumulator->initialSeenCount > 0) {
      auto expectedInitialSize = std::min<int64_t>(
          accumulator->maxSampleSize, accumulator->initialSeenCount);
      VELOX_CHECK_EQ(
          arraySize,
          expectedInitialSize,
          "when a positive initial_processed_count is provided the size of "
          "the initial sample must be equal to desired_sample_size parameter");
    }

    auto elementsVector = arrayVector->elements();
    DecodedVector decodedElements(*elementsVector);

    for (vector_size_t i = 0; i < arraySize; i++) {
      auto elementIndex = arrayOffset + i;
      LOG(INFO) << "[processInitialSample] Adding initial element " << i
                << " to initialSamples";
      // Add to initial samples (including nulls, matching Java behavior)
      accumulator->initialSamples.appendValue(
          decodedElements, elementIndex, allocator_);

      if (accumulator->samples.size() < accumulator->maxSampleSize) {
        LOG(INFO) << "[processInitialSample] Adding initial element " << i
                  << " to samples. Current samples.size: "
                  << accumulator->samples.size();
        accumulator->samples.appendValue(
            decodedElements, elementIndex, allocator_);
      }
    }

    LOG(INFO) << "[processInitialSample] Done. "
              << "initialSamples.size: " << accumulator->initialSamples.size()
              << ", samples.size: " << accumulator->samples.size();
  }

  void mergeIntermediateResult(
      ReservoirSampleAccumulator* accumulator,
      const RowVector* rowVector,
      vector_size_t decodedRow) {
    auto* initialSampleVector = rowVector->childAt(0)->as<ArrayVector>();
    auto* initialSeenCountVector =
        rowVector->childAt(1)->asFlatVector<int64_t>();
    auto* seenCountVector = rowVector->childAt(2)->asFlatVector<int64_t>();
    auto* maxSampleSizeVector = rowVector->childAt(3)->asFlatVector<int32_t>();
    auto* sampleArrayVector = rowVector->childAt(4)->as<ArrayVector>();

    int32_t maxSampleSize = maxSampleSizeVector->valueAt(decodedRow);

    // Skip uninitialized accumulators
    if (maxSampleSize < 0) {
      return;
    }

    // Initialize accumulator if needed
    if (!accumulator->isInitialized()) {
      initializeFromIntermediate(
          accumulator,
          initialSampleVector,
          initialSeenCountVector,
          seenCountVector,
          maxSampleSizeVector,
          sampleArrayVector,
          decodedRow);
    }

    // Merge samples
    mergeIntermediateSamples(
        accumulator, sampleArrayVector, seenCountVector, decodedRow);
  }

  void initializeFromIntermediate(
      ReservoirSampleAccumulator* accumulator,
      const ArrayVector* initialSampleVector,
      const FlatVector<int64_t>* initialSeenCountVector,
      const FlatVector<int64_t>* seenCountVector,
      const FlatVector<int32_t>* maxSampleSizeVector,
      const ArrayVector* sampleArrayVector,
      vector_size_t decodedRow) {
    new (accumulator) ReservoirSampleAccumulator();

    accumulator->maxSampleSize = maxSampleSizeVector->valueAt(decodedRow);
    accumulator->initialSeenCount = initialSeenCountVector->valueAt(decodedRow);
    accumulator->processedCount = accumulator->initialSeenCount;

    // Copy initial samples
    if (!initialSampleVector->isNullAt(decodedRow)) {
      copyInitialSamplesFromIntermediate(
          accumulator, initialSampleVector, decodedRow);
    }
  }

  void copyInitialSamplesFromIntermediate(
      ReservoirSampleAccumulator* accumulator,
      const ArrayVector* initialSampleVector,
      vector_size_t decodedRow) {
    auto initialSampleOffset = initialSampleVector->offsetAt(decodedRow);
    auto initialSampleSize = initialSampleVector->sizeAt(decodedRow);
    auto initialElements = initialSampleVector->elements();

    DecodedVector decodedInitialElements(*initialElements);

    for (vector_size_t i = 0; i < initialSampleSize; i++) {
      auto elementIndex = initialSampleOffset + i;
      accumulator->initialSamples.appendValue(
          decodedInitialElements, elementIndex, allocator_);

      if (accumulator->samples.size() < accumulator->maxSampleSize) {
        accumulator->samples.appendValue(
            decodedInitialElements, elementIndex, allocator_);
      }
    }
  }

  void mergeIntermediateSamples(
      ReservoirSampleAccumulator* accumulator,
      const ArrayVector* sampleArrayVector,
      const FlatVector<int64_t>* seenCountVector,
      vector_size_t decodedRow) {
    int64_t otherSeenCount = seenCountVector->valueAt(decodedRow);
    auto sampleOffset = sampleArrayVector->offsetAt(decodedRow);
    auto sampleSize = sampleArrayVector->sizeAt(decodedRow);

    LOG(INFO) << "[mergeIntermediateSamples] Called. "
              << "otherSeenCount: " << otherSeenCount
              << ", sampleSize: " << sampleSize
              << ", accumulator->processedCount: "
              << accumulator->processedCount
              << ", accumulator->initialSeenCount: "
              << accumulator->initialSeenCount
              << ", accumulator->initialSamples.size: "
              << accumulator->initialSamples.size();

    // Calculate how many additional items were processed in the other partition
    int64_t additionalCount = otherSeenCount - accumulator->initialSeenCount;

    // Skip if the other partition didn't process any new items
    if (additionalCount <= 0 || sampleSize == 0) {
      LOG(INFO) << "[mergeIntermediateSamples] Skipping. additionalCount: "
                << additionalCount;
      return;
    }

    // Update total processed count
    accumulator->processedCount += additionalCount;

    LOG(INFO) << "[mergeIntermediateSamples] After update, processedCount: "
              << accumulator->processedCount;

    // Both accumulators were initialized with the same initial samples.
    // The incoming sample array contains both initial samples and potentially
    // new samples. We should skip the initial samples since they're already
    // in our accumulator, and only merge the non-initial samples.
    auto initialSampleCount = accumulator->initialSamples.size();
    auto nonInitialStartIndex = std::min<vector_size_t>(initialSampleCount, sampleSize);

    auto sampleElements = sampleArrayVector->elements();
    DecodedVector decodedSampleElements(*sampleElements);

    LOG(INFO) << "[mergeIntermediateSamples] Merging " << sampleSize
              << " samples, skipping first " << nonInitialStartIndex
              << " initial samples";

    // Only process samples beyond the initial samples
    for (vector_size_t i = nonInitialStartIndex; i < sampleSize; i++) {
      auto elementIndex = sampleOffset + i;
      LOG(INFO) << "[mergeIntermediateSamples] Processing incoming sample " << i
                << " at elementIndex " << elementIndex;
      accumulator->addValueToReservoir(
          decodedSampleElements, elementIndex, allocator_);
    }

    LOG(INFO) << "[mergeIntermediateSamples] Done. Final samples.size: "
              << accumulator->samples.size();
  }
};

} // namespace

void registerReservoirSampleAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  // Add support for primitive types
  for (const auto& inputType :
       {"tinyint",
        "smallint",
        "integer",
        "bigint",
        "real",
        "double",
        "varchar"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(fmt::format(
                "row(processed_count bigint, sample array({}))", inputType))
            .intermediateType(fmt::format(
                "row(initialSample array({}), initialSeenCount bigint, "
                "seenCount bigint, maxSampleSize integer, sample array({}))",
                inputType,
                inputType))
            .argumentType(fmt::format("array({})", inputType))
            .argumentType("bigint")
            .argumentType(inputType)
            .argumentType("integer")
            .build());
  }

  // Add support for generic types using type variables - this will handle ROW
  // and other complex types
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("row(processed_count bigint, sample array(T))")
          .intermediateType(
              "row(initialSample array(T), initialSeenCount bigint, "
              "seenCount bigint, maxSampleSize integer, sample array(T))")
          .argumentType("array(T)")
          .argumentType("bigint")
          .argumentType("T")
          .argumentType("integer")
          .build());

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

        // Always use the generic implementation that works with any type
        return std::make_unique<ReservoirSampleGenericAggregate>(resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
