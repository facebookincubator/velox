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

#include <random>
#include "velox/common/base/BitUtil.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

using facebook::velox::aggregate::ValueList;
using facebook::velox::aggregate::ValueListReader;

namespace facebook::velox::aggregate::prestosql {

namespace {

constexpr int32_t kUninitializedMarker = -1;

struct ReservoirSampleAccumulator {
  ValueList samples;
  ValueList initialSamples;
  int64_t initialSeenCount = 0;
  int64_t processedCount = 0;
  int32_t maxSampleSize = 0;
  bool initialized = false;

  explicit ReservoirSampleAccumulator() {}

  void addValueToReservoir(
      const DecodedVector& decodedVector,
      vector_size_t row,
      HashStringAllocator* allocator) {
    if (samples.size() < static_cast<size_t>(maxSampleSize)) {
      samples.appendValue(decodedVector, row, allocator);
    } else {
      uint64_t valueHash = computeValueHash(decodedVector, row);
      uint64_t seed =
          bits::hashMix(static_cast<uint64_t>(maxSampleSize), valueHash);

      std::mt19937_64 generator(seed);
      std::uniform_int_distribution<int64_t> distribution(1, processedCount);
      int64_t randomPos = distribution(generator);

      if (randomPos <= maxSampleSize) {
        vector_size_t replaceIndex = randomPos - 1;

        ValueList newSamples;
        ValueListReader reader(const_cast<ValueList&>(samples));

        for (vector_size_t i = 0; i < samples.size(); i++) {
          if (i == replaceIndex) {
            newSamples.appendValue(decodedVector, row, allocator);
          } else {
            auto tempVector = BaseVector::create(
                decodedVector.base()->type(), 1, allocator->pool());
            reader.next(*tempVector, 0);
            DecodedVector tempDecoded(*tempVector);
            newSamples.appendValue(tempDecoded, 0, allocator);
          }
        }

        samples.free(allocator);
        samples = std::move(newSamples);
      }
    }
  }

 private:
  uint64_t computeValueHash(
      const DecodedVector& decodedVector,
      vector_size_t row) const {
    // Compute hash based on the value to ensure deterministic sampling
    if (decodedVector.isNullAt(row)) {
      return 0; // Consistent hash for null values
    }

    auto baseVector = decodedVector.base();
    auto index = decodedVector.index(row);

    // Simple hash based on vector type and value
    switch (baseVector->typeKind()) {
      case TypeKind::BIGINT:
        return static_cast<uint64_t>(
            baseVector->as<FlatVector<int64_t>>()->valueAt(index));
      case TypeKind::INTEGER:
        return static_cast<uint64_t>(
            baseVector->as<FlatVector<int32_t>>()->valueAt(index));
      case TypeKind::SMALLINT:
        return static_cast<uint64_t>(
            baseVector->as<FlatVector<int16_t>>()->valueAt(index));
      case TypeKind::TINYINT:
        return static_cast<uint64_t>(
            baseVector->as<FlatVector<int8_t>>()->valueAt(index));
      case TypeKind::REAL: {
        float value = baseVector->as<FlatVector<float>>()->valueAt(index);
        return bits::hashBytes(
            0, reinterpret_cast<const char*>(&value), sizeof(float));
      }
      case TypeKind::DOUBLE: {
        double value = baseVector->as<FlatVector<double>>()->valueAt(index);
        return bits::hashBytes(
            0, reinterpret_cast<const char*>(&value), sizeof(double));
      }
      case TypeKind::VARCHAR: {
        auto stringView =
            baseVector->as<FlatVector<StringView>>()->valueAt(index);
        return folly::Hash{}(stringView);
      }
      default:
        // For complex types (ROW, ARRAY, etc.), use the value's hash if
        // available Otherwise, use a combination of type and position
        return bits::hashMix(
            static_cast<uint64_t>(baseVector->typeKind()),
            static_cast<uint64_t>(index));
    }
  }
};

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

      if (!accumulator->initialized) {
        initializeAccumulator(
            accumulator,
            initialSampleDecoded,
            initialProcessedCountDecoded,
            desiredSampleSizeDecoded,
            row_idx);
      }

      // Add new value using reservoir sampling
      if (!valuesToSampleDecoded.isNullAt(row_idx)) {
        accumulator->processedCount++;
        accumulator->addValueToReservoir(
            valuesToSampleDecoded, row_idx, allocator_);
      }
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

    if (!accumulator->initialized) {
      vector_size_t firstRow = rows.begin();
      initializeAccumulator(
          accumulator,
          initialSampleDecoded,
          initialProcessedCountDecoded,
          desiredSampleSizeDecoded,
          firstRow);
    }

    // Process all input values
    rows.applyToSelected([&](vector_size_t row) {
      if (!valuesToSampleDecoded.isNullAt(row)) {
        accumulator->processedCount++;
        accumulator->addValueToReservoir(
            valuesToSampleDecoded, row, allocator_);
      }
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
      if (!accumulator->initialized) {
        const_cast<ReservoirSampleAccumulator*>(accumulator)->initialized =
            true;
      }
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
      if (!accumulator->initialized) {
        // Mark uninitialized accumulators with special marker
        const_cast<ReservoirSampleAccumulator*>(accumulator)->initialized =
            true;
        const_cast<ReservoirSampleAccumulator*>(accumulator)->maxSampleSize =
            kUninitializedMarker;
      }
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

      // Copy samples from ValueList to output vector
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

      // Copy values from ValueList to output vector
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
      maxSampleSizeFlat->set(i, accumulator->maxSampleSize);
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

    accumulator->initialized = true;
  }

  void processInitialSample(
      ReservoirSampleAccumulator* accumulator,
      const DecodedVector& initialSampleDecoded,
      vector_size_t row) {
    auto* arrayVector = initialSampleDecoded.base()->as<ArrayVector>();
    auto arrayIndex = initialSampleDecoded.index(row);
    auto arraySize = arrayVector->sizeAt(arrayIndex);
    auto arrayOffset = arrayVector->offsetAt(arrayIndex);

    auto elementsVector = arrayVector->elements();
    DecodedVector decodedElements(*elementsVector);

    for (vector_size_t i = 0; i < arraySize; i++) {
      auto elementIndex = arrayOffset + i;
      if (!decodedElements.isNullAt(elementIndex)) {
        // Add to initial samples
        accumulator->initialSamples.appendValue(
            decodedElements, elementIndex, allocator_);

        // Add to current samples if there's space
        if (accumulator->samples.size() <
            static_cast<size_t>(accumulator->maxSampleSize)) {
          accumulator->samples.appendValue(
              decodedElements, elementIndex, allocator_);
        }
      }
    }
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
    if (maxSampleSize == kUninitializedMarker) {
      return;
    }

    // Initialize accumulator if needed
    if (!accumulator->initialized) {
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

    accumulator->initialized = true;
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
      if (!decodedInitialElements.isNullAt(elementIndex)) {
        // Add to initial samples
        accumulator->initialSamples.appendValue(
            decodedInitialElements, elementIndex, allocator_);

        // Add to current samples if there's space
        if (accumulator->samples.size() <
            static_cast<size_t>(accumulator->maxSampleSize)) {
          accumulator->samples.appendValue(
              decodedInitialElements, elementIndex, allocator_);
        }
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

    // Update processed count (avoid double counting)
    int64_t additionalCount = otherSeenCount - accumulator->initialSeenCount;
    if (additionalCount > 0) {
      accumulator->processedCount += additionalCount;
    }

    // Merge samples using reservoir sampling
    if (otherSeenCount > accumulator->initialSeenCount && sampleSize > 0) {
      auto sampleElements = sampleArrayVector->elements();
      DecodedVector decodedSampleElements(*sampleElements);

      for (vector_size_t i = 0; i < sampleSize; i++) {
        auto elementIndex = sampleOffset + i;
        if (!decodedSampleElements.isNullAt(elementIndex)) {
          accumulator->addValueToReservoir(
              decodedSampleElements, elementIndex, allocator_);
        }
      }
    }
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
