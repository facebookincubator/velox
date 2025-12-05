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

#include <folly/Random.h>

#include "velox/exec/Aggregate.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

struct ReservoirSampleAccumulator {
  VectorPtr samples = nullptr;
  std::optional<int32_t> maxSampleSize = std::nullopt;
  std::optional<int64_t> processedCount = std::nullopt;
  VectorPtr initialSamples = nullptr;
  // -1 indicates no initialSamples. Use -1 for simplicity for writing and
  // reading intermediate results. Allowed values of initialSeenCount are -1, 0,
  // and positive integers.
  int64_t initialSeenCount = -1;

  void initialize(
      int32_t desiredSampleSize,
      const TypePtr& elementType,
      const HashStringAllocator* allocator);

  bool isInitialized() const {
    if (samples == nullptr) {
      return false;
    }
    VELOX_DCHECK(maxSampleSize.has_value());
    VELOX_DCHECK(processedCount.has_value());
    return true;
  }

  vector_size_t initialSampleCount() const {
    return initialSamples ? initialSamples->size() : 0;
  }

  vector_size_t sampleCount() const {
    return processedCount.value_or(0) <= maxSampleSize.value_or(0)
        ? processedCount.value_or(0)
        : samples->size();
  }

  void addValueToReservoir(
      const BaseVector* vector,
      vector_size_t row,
      bool isNull = false);

  void mergeSamples(
      const ArrayVector& sampleArrayVector,
      vector_size_t row,
      int64_t otherProcessedCount,
      int32_t otherMaxSampleSize,
      const HashStringAllocator* allocator);

  void mergeSamples(
      const VectorPtr& sampleVector,
      vector_size_t offset,
      vector_size_t size,
      int64_t otherProcessedCount,
      int32_t otherMaxSampleSize,
      const HashStringAllocator* allocator);

  void setSamples(
      const VectorPtr& sampleVector,
      vector_size_t offset,
      vector_size_t size,
      int64_t seenCount,
      int32_t desiredSampleSize,
      const HashStringAllocator* allocator);

  void setInitialSamples(
      const VectorPtr initialSampleVector,
      vector_size_t offset,
      vector_size_t size,
      int64_t initialProcessedCount,
      const HashStringAllocator* allocator);

 private:
  static VectorPtr shuffleByDictionary(
      VectorPtr vector,
      vector_size_t offset,
      vector_size_t size,
      folly::ThreadLocalPRNG& randGen,
      const HashStringAllocator* allocator);
};

void ReservoirSampleAccumulator::initialize(
    int32_t desiredSampleSize,
    const TypePtr& elementType,
    const HashStringAllocator* allocator) {
  VELOX_USER_CHECK_GT(desiredSampleSize, 0, "sample size must be positive");
  maxSampleSize = desiredSampleSize;
  samples =
      BaseVector::create(elementType, desiredSampleSize, allocator->pool());
  VELOX_DCHECK(samples->isFlatEncoding());
  processedCount = 0;

  initialSamples = nullptr;
  initialSeenCount = -1;
}

void ReservoirSampleAccumulator::setSamples(
    const VectorPtr& sampleVector,
    vector_size_t offset,
    vector_size_t size,
    int64_t seenCount,
    int32_t desiredSampleSize,
    const HashStringAllocator* allocator) {
  VELOX_CHECK_NOT_NULL(sampleVector);
  VELOX_CHECK_GE(seenCount, 0);
  VELOX_DCHECK(size == seenCount || size == desiredSampleSize);

  // const VectorPtr& sampleElements = sampleArrayVector.elements();
  samples = BaseVector::create(
      sampleVector->type(), desiredSampleSize, allocator->pool());
  samples->copy(sampleVector.get(), 0, offset, size);
  processedCount = seenCount;
  maxSampleSize = desiredSampleSize;
}

void ReservoirSampleAccumulator::setInitialSamples(
    const VectorPtr initialSampleVector,
    vector_size_t offset,
    vector_size_t size,
    int64_t initialProcessedCount,
    const HashStringAllocator* allocator) {
  if (initialProcessedCount <= 0) {
    VELOX_USER_CHECK(
        initialSampleVector == nullptr || size == 0,
        "Initial state array must be null or empty "
        "when initial processed count is <= 0.");
  }

  if (initialSeenCount >= 0) {
    return;
  }

  if (initialSampleVector && size > 0) {
    VELOX_USER_CHECK_GE(
        initialProcessedCount,
        size,
        "InitialProcessedCount must be greater than or equal to the number "
        "of positions in the initial sample.");
  }

  // When initialSamples == nullptr, initialSeenCount can be any of
  // -1, 0, or positive.
  initialSeenCount = initialProcessedCount;

  if (initialProcessedCount <= 0 || initialSampleVector == nullptr) {
    initialSamples = nullptr;
    return;
  }
  initialSamples =
      BaseVector::create(initialSampleVector->type(), size, allocator->pool());
  initialSamples->copy(initialSampleVector.get(), 0, offset, size);
}

void ReservoirSampleAccumulator::addValueToReservoir(
    const BaseVector* vector,
    vector_size_t row,
    bool isNull) {
  VELOX_CHECK(isInitialized());

  processedCount = processedCount.value_or(0) + 1;

  vector_size_t targetIndex = -1;
  if (processedCount.value() <= maxSampleSize) {
    targetIndex = processedCount.value() - 1;
  } else {
    const uint64_t random = folly::Random::rand64(processedCount.value());
    if (random < samples->size()) {
      targetIndex = random;
    }
  }

  if (targetIndex >= 0) {
    if (isNull) {
      samples->setNull(targetIndex, true);
    } else {
      samples->copy(vector, targetIndex, row, 1);
    }
  }
}

void ReservoirSampleAccumulator::mergeSamples(
    const ArrayVector& sampleArrayVector,
    vector_size_t row,
    int64_t otherProcessedCount,
    int32_t otherMaxSampleSize,
    const HashStringAllocator* allocator) {
  if (sampleArrayVector.isNullAt(row)) {
    return;
  }
  mergeSamples(
      sampleArrayVector.elements(),
      sampleArrayVector.offsetAt(row),
      sampleArrayVector.sizeAt(row),
      otherProcessedCount,
      otherMaxSampleSize,
      allocator);
}

void ReservoirSampleAccumulator::mergeSamples(
    const VectorPtr& sampleVector,
    vector_size_t offset,
    vector_size_t size,
    int64_t otherProcessedCount,
    int32_t otherMaxSampleSize,
    const HashStringAllocator* allocator) {
  VELOX_DCHECK_GE(otherProcessedCount, 0);
  VELOX_DCHECK_GT(otherMaxSampleSize, 0);
  VELOX_CHECK_NOT_NULL(sampleVector);

  const TypePtr& elementType = sampleVector->type();

  if (!isInitialized()) {
    initialize(otherMaxSampleSize, elementType, allocator);
  }

  VELOX_DCHECK(
      samples->type()->equivalent(*elementType),
      "Samples to be merged are of different types.");
  VELOX_CHECK_EQ(
      maxSampleSize.value(),
      otherMaxSampleSize,
      "maximum number of samples {} must be equal to that of other {}",
      maxSampleSize.value(),
      otherMaxSampleSize);

  if (otherProcessedCount < maxSampleSize.value()) {
    for (vector_size_t i = 0; i < size; ++i) {
      addValueToReservoir(sampleVector.get(), offset + i);
    }
    return;
  }

  if (processedCount.value() < maxSampleSize.value()) {
    const VectorPtr tempSamples = std::move(samples);
    const int64_t tempProcessedCount = processedCount.value();

    setSamples(
        sampleVector,
        offset,
        size,
        otherProcessedCount,
        otherMaxSampleSize,
        allocator);

    for (vector_size_t i = 0; i < tempProcessedCount; ++i) {
      addValueToReservoir(tempSamples.get(), i);
    }
    return;
  }

  folly::ThreadLocalPRNG rng;
  const VectorPtr oneShuffledVector =
      shuffleByDictionary(samples, 0, sampleCount(), rng, allocator);
  const VectorPtr otherShuffledVector =
      shuffleByDictionary(sampleVector, offset, size, rng, allocator);
  const int64_t oneProcessedCount = processedCount.value();

  samples =
      BaseVector::create(elementType, maxSampleSize.value(), allocator->pool());
  processedCount = 0;
  vector_size_t oneNext = 0;
  vector_size_t otherNext = 0;
  for (vector_size_t i = 0; i < maxSampleSize.value(); ++i) {
    if (folly::Random::rand64(oneProcessedCount + otherProcessedCount) <
        oneProcessedCount) {
      addValueToReservoir(oneShuffledVector.get(), oneNext++);
    } else {
      addValueToReservoir(otherShuffledVector.get(), otherNext++);
    }
  }
  processedCount = oneProcessedCount + otherProcessedCount;
}

// static
VectorPtr ReservoirSampleAccumulator::shuffleByDictionary(
    VectorPtr vector,
    vector_size_t offset,
    vector_size_t size,
    folly::ThreadLocalPRNG& randGen,
    const HashStringAllocator* allocator) {
  BufferPtr shuffledIndices = allocateIndices(size, allocator->pool());
  vector_size_t* rawIndices = shuffledIndices->asMutable<vector_size_t>();

  std::iota(rawIndices, rawIndices + size, offset);
  std::shuffle(rawIndices, rawIndices + size, randGen);

  return BaseVector::wrapInDictionary(
      nullptr, shuffledIndices, size, std::move(vector));
}

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
  template <bool kSingleGroup>
  void addRaw(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args);

  template <bool kSingleGroup>
  void addIntermediate(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args);

  static void validateArguments(const std::vector<VectorPtr>& args);

  std::tuple<vector_size_t, vector_size_t, vector_size_t> computeTotalSamples(
      char** groups,
      int32_t numGroups) const;
};

ReservoirSampleAggregate::ReservoirSampleAggregate(TypePtr resultType)
    : exec::Aggregate(std::move(resultType)) {}

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
  auto* rowVector = (*result)->asChecked<RowVector>();
  rowVector->resetNulls();
  rowVector->resize(numGroups);

  auto* processedCountVector =
      rowVector->childAt(0)->asChecked<FlatVector<int64_t>>();
  processedCountVector->resetNulls();

  auto* sampleArrayVector = rowVector->childAt(1)->asChecked<ArrayVector>();
  sampleArrayVector->resetNulls();
  const auto [_, __, totalMergedSamples] =
      computeTotalSamples(groups, numGroups);
  const VectorPtr& sampleElementsVector = sampleArrayVector->elements();
  sampleElementsVector->resize(totalMergedSamples);

  vector_size_t offset = 0;
  for (int32_t i = 0; i < numGroups; i++) {
    const auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);

    if (!accumulator->isInitialized()) {
      sampleArrayVector->setNull(i, true);
      processedCountVector->setNull(i, true);
      continue;
    }

    // Merge the initial sample at last in the final aggregation before
    // extracting output.
    VELOX_USER_CHECK(
        accumulator->initialSeenCount <= 0 ||
            (accumulator->initialSeenCount ==
                 accumulator->initialSampleCount() &&
             accumulator->initialSampleCount() <=
                 accumulator->maxSampleSize.value()) ||
            accumulator->maxSampleSize.value() ==
                accumulator->initialSampleCount(),
        "When a positive initial_processed_count is provided, "
        "the size of the initial sample must be equal to desired_sample_size parameter.");

    ReservoirSampleAccumulator finalAccumulator;
    const VectorPtr nonNullInitialSample = BaseVector::getOrCreateEmpty(
        accumulator->initialSamples,
        accumulator->samples->type(),
        allocator_->pool());
    finalAccumulator.setSamples(
        nonNullInitialSample,
        0,
        accumulator->initialSampleCount(),
        std::max<int64_t>(accumulator->initialSeenCount, 0),
        accumulator->maxSampleSize.value(),
        allocator_);
    finalAccumulator.mergeSamples(
        accumulator->samples,
        0,
        accumulator->sampleCount(),
        accumulator->processedCount.value(),
        accumulator->maxSampleSize.value(),
        allocator_);

    sampleArrayVector->setNull(i, false);
    const vector_size_t sampleSize = finalAccumulator.sampleCount();
    sampleElementsVector->copy(
        finalAccumulator.samples.get(), offset, 0, sampleSize);
    sampleArrayVector->setOffsetAndSize(i, offset, sampleSize);
    offset += sampleSize;

    processedCountVector->set(i, finalAccumulator.processedCount.value());
  }
}

void ReservoirSampleAggregate::extractAccumulators(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  // Order of RowVector children is determined by protocol.
  auto* rowVector = (*result)->asChecked<RowVector>();
  rowVector->resetNulls();
  rowVector->resize(numGroups);

  const auto [totalSamples, totalInitialSamples, _] =
      computeTotalSamples(groups, numGroups);

  auto* initialSampleVector = rowVector->childAt(0)->asChecked<ArrayVector>();
  initialSampleVector->resetNulls();
  const VectorPtr& initialSampleElementsVector =
      initialSampleVector->elements();
  initialSampleElementsVector->resize(totalInitialSamples);

  auto* initialSeenCountVector =
      rowVector->childAt(1)->asChecked<FlatVector<int64_t>>();
  initialSeenCountVector->resetNulls();

  auto* seenCountVector =
      rowVector->childAt(2)->asChecked<FlatVector<int64_t>>();
  seenCountVector->resetNulls();

  auto* maxSampleSizeVector =
      rowVector->childAt(3)->asChecked<FlatVector<int32_t>>();
  maxSampleSizeVector->resetNulls();

  auto* sampleVector = rowVector->childAt(4)->asChecked<ArrayVector>();
  sampleVector->resetNulls();
  const VectorPtr& sampleElementsVector = sampleVector->elements();
  sampleElementsVector->resize(totalSamples);

  vector_size_t offset = 0;
  vector_size_t initialSampleOffset = 0;

  for (int32_t i = 0; i < numGroups; i++) {
    const auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);

    if (!accumulator->isInitialized()) {
      rowVector->setNull(i, true);
      sampleVector->setNull(i, true);
      maxSampleSizeVector->setNull(i, true);
      seenCountVector->setNull(i, true);
      initialSampleVector->setNull(i, true);
      initialSeenCountVector->setNull(i, true);
      continue;
    }

    rowVector->setNull(i, false);
    sampleVector->setNull(i, false);
    const vector_size_t sampleSize = accumulator->sampleCount();
    sampleElementsVector->copy(
        accumulator->samples.get(), offset, 0, sampleSize);
    sampleVector->setOffsetAndSize(i, offset, sampleSize);
    offset += sampleSize;

    maxSampleSizeVector->set(i, accumulator->maxSampleSize.value());
    seenCountVector->set(i, accumulator->processedCount.value());

    if (accumulator->initialSamples == nullptr) {
      initialSampleVector->setNull(i, true);
    } else {
      initialSampleVector->setNull(i, false);
      const vector_size_t initialSampleSize = accumulator->initialSampleCount();
      initialSampleElementsVector->copy(
          accumulator->initialSamples.get(),
          initialSampleOffset,
          0,
          initialSampleSize);
      initialSampleVector->setOffsetAndSize(
          i, initialSampleOffset, initialSampleSize);
      initialSampleOffset += initialSampleSize;
    }

    initialSeenCountVector->set(i, accumulator->initialSeenCount);
  }
}

template <bool kSingleGroup>
void ReservoirSampleAggregate::addRaw(
    std::conditional_t<kSingleGroup, char*, char**> group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  validateArguments(args);

  const DecodedVector initialSampleDecoded(*args[0], rows);
  const DecodedVector initialProcessedCountDecoded(*args[1], rows);
  const DecodedVector valuesToSampleDecoded(*args[2], rows);
  const DecodedVector desiredSampleSizeDecoded(*args[3], rows);

  const auto& elementType = args[2]->type();

  rows.applyToSelected([&](vector_size_t row) {
    if (desiredSampleSizeDecoded.isNullAt(row) ||
        initialProcessedCountDecoded.isNullAt(row)) {
      return;
    }

    ReservoirSampleAccumulator* accumulator;
    if constexpr (kSingleGroup) {
      accumulator = value<ReservoirSampleAccumulator>(group);
    } else {
      accumulator = value<ReservoirSampleAccumulator>(group[row]);
    }

    if (!accumulator->isInitialized()) {
      accumulator->initialize(
          desiredSampleSizeDecoded.valueAt<int32_t>(row),
          elementType,
          allocator_);
    }

    const auto* initialSampleBase =
        initialSampleDecoded.base()->asChecked<ArrayVector>();
    accumulator->setInitialSamples(
        initialSampleDecoded.isNullAt(row) ? nullptr
                                           : initialSampleBase->elements(),
        initialSampleBase->offsetAt(initialSampleDecoded.index(row)),
        initialSampleBase->sizeAt(initialSampleDecoded.index(row)),
        initialProcessedCountDecoded.valueAt<int64_t>(row),
        allocator_);

    accumulator->addValueToReservoir(
        valuesToSampleDecoded.base(),
        valuesToSampleDecoded.index(row),
        valuesToSampleDecoded.isNullAt(row));
  });
}

template <bool kSingleGroup>
void ReservoirSampleAggregate::addIntermediate(
    std::conditional_t<kSingleGroup, char*, char**> group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  VELOX_CHECK_EQ(args.size(), 1, "Intermediate results requires 1 argument");

  const DecodedVector decodedIntermediate(*args[0], rows);
  const auto* rowVector = decodedIntermediate.base()->asChecked<RowVector>();
  VELOX_CHECK_EQ(rowVector->childrenSize(), 5);
  const auto* initialSampleArrayVector =
      rowVector->childAt(0)->asChecked<ArrayVector>();
  const auto* initialSeenCountVector =
      rowVector->childAt(1)->asChecked<FlatVector<int64_t>>();
  const auto* seenCountVector =
      rowVector->childAt(2)->asChecked<FlatVector<int64_t>>();
  const auto* maxSampleSizeVector =
      rowVector->childAt(3)->asChecked<FlatVector<int32_t>>();
  const auto* sampleArrayVector =
      rowVector->childAt(4)->asChecked<ArrayVector>();

  const auto& elementType = sampleArrayVector->elements()->type();

  rows.applyToSelected([&](vector_size_t row) {
    if (decodedIntermediate.isNullAt(row)) {
      return;
    }

    const auto decodedRow = decodedIntermediate.index(row);

    if (sampleArrayVector->isNullAt(decodedRow)) {
      return;
    } /// order with accumulator->initialize
    VELOX_DCHECK(!maxSampleSizeVector->isNullAt(decodedRow));
    VELOX_DCHECK(!seenCountVector->isNullAt(decodedRow));
    VELOX_DCHECK(!initialSeenCountVector->isNullAt(decodedRow));

    ReservoirSampleAccumulator* accumulator;
    if constexpr (kSingleGroup) {
      accumulator = value<ReservoirSampleAccumulator>(group);
    } else {
      accumulator = value<ReservoirSampleAccumulator>(group[row]);
    }

    if (!accumulator->isInitialized()) {
      accumulator->initialize(
          maxSampleSizeVector->valueAt(decodedRow), elementType, allocator_);
    }

    accumulator->mergeSamples(
        *sampleArrayVector,
        decodedRow,
        seenCountVector->valueAt(decodedRow),
        maxSampleSizeVector->valueAt(decodedRow),
        allocator_);
    accumulator->setInitialSamples(
        initialSampleArrayVector->isNullAt(decodedRow)
            ? nullptr
            : initialSampleArrayVector->elements(),
        initialSampleArrayVector->offsetAt(decodedRow),
        initialSampleArrayVector->sizeAt(decodedRow),
        initialSeenCountVector->valueAt(decodedRow),
        allocator_);
  });
}

void ReservoirSampleAggregate::addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  addRaw<false>(groups, rows, args);
}

void ReservoirSampleAggregate::addIntermediateResults(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  addIntermediate<false>(groups, rows, args);
}

void ReservoirSampleAggregate::addSingleGroupRawInput(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  addRaw<true>(group, rows, args);
}

void ReservoirSampleAggregate::addSingleGroupIntermediateResults(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /* mayPushdown */) {
  addIntermediate<true>(group, rows, args);
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

// static
void ReservoirSampleAggregate::validateArguments(
    const std::vector<VectorPtr>& args) {
  static constexpr size_t kInitialSampleIndex = 0;
  static constexpr size_t kInitialProcessedCountIndex = 1;
  static constexpr size_t kSampleIndex = 2;
  static constexpr size_t kMaxSampleSizeIndex = 3;

  VELOX_USER_CHECK_EQ(args.size(), 4, "reservoir_sample requires 4 arguments");
  VELOX_USER_CHECK(args[kInitialProcessedCountIndex]->type()->isBigint());
  VELOX_USER_CHECK(args[kMaxSampleSizeIndex]->type()->isInteger());

  auto elementType = args[kSampleIndex]->type();
  // TODO: extend to ComplexType
  VELOX_USER_CHECK(elementType->isPrimitiveType());
  auto initialSampleType = args[kInitialSampleIndex]->type();
  VELOX_USER_CHECK(initialSampleType->isArray());
  VELOX_USER_CHECK(
      initialSampleType->asArray().elementType()->equivalent(*elementType),
      "Initial samples and values to sample must have the same type.");
}

std::tuple<vector_size_t, vector_size_t, vector_size_t>
ReservoirSampleAggregate::computeTotalSamples(char** groups, int32_t numGroups)
    const {
  vector_size_t totalSamples = 0;
  vector_size_t totalInitialSamples = 0;
  vector_size_t totalMergedSamples = 0;

  for (int32_t i = 0; i < numGroups; i++) {
    const auto* accumulator = value<ReservoirSampleAccumulator>(groups[i]);
    totalSamples += accumulator->sampleCount();
    totalInitialSamples += accumulator->initialSampleCount();
    totalMergedSamples += std::min<vector_size_t>(
        accumulator->sampleCount() + accumulator->initialSampleCount(),
        accumulator->maxSampleSize.value_or(0));
  }

  return std::make_tuple(totalSamples, totalInitialSamples, totalMergedSamples);
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
    // argumentType, intermediateType and returnType and their field order need
    // to be consistent with those defined in protocol.
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(
                fmt::format(
                    "row(processed_count bigint, sample array({}))", inputType))
            .intermediateType(
                fmt::format(
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

  auto name = prefix + kReservoirSample;
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
