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

#pragma once

#include <folly/Random.h>
#include <memory>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::dwio::common {

using namespace facebook::velox::common;

using SubfieldFilters = std::unordered_map<Subfield, std::unique_ptr<Filter>>;

struct FilterSpec {
  FilterSpec() {}

  explicit FilterSpec(
      std::string field,
      float startPct = 50,
      float selectPct = 20,
      FilterKind filterKind = FilterKind::kBigintRange,
      bool isForRowGroupSkip = true,
      bool allowNulls = true)
      : field(field),
        startPct(startPct),
        selectPct(selectPct),
        filterKind(filterKind),
        isForRowGroupSkip(isForRowGroupSkip),
        allowNulls_(allowNulls) {}

  std::string field;
  float startPct = 50;
  float selectPct = 20;
  FilterKind filterKind = FilterKind::kBigintRange;
  // If true, makes a filter that matches max value in the column so as to skip
  // row groups on min/max.
  bool isForRowGroupSkip{false};
  bool allowNulls_{true};
};

// Encodes a batch number and an index into the batch into an int32_t
uint64_t batchPosition(uint32_t batchNumber, vector_size_t batchRow);
uint32_t batchNumber(uint64_t position);
vector_size_t batchRow(uint64_t position);
VectorPtr getChildBySubfield(
    const RowVector* rowVector,
    const Subfield& subfield,
    const RowTypePtr& rowType = nullptr);

class AbstractColumnStats {
 public:
  // ASCII string greater than test data values. Used for row group skipping
  // tests.
  static constexpr const char* kMaxString = "~~~~~";
  AbstractColumnStats(TypePtr type, RowTypePtr rootType)
      : type_(type), rootType_(rootType) {}

  virtual ~AbstractColumnStats() = default;

  virtual void sample(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& rows) = 0;

  virtual std::unique_ptr<Filter> filter(
      const std::vector<RowVectorPtr>& batches,
      const FilterSpec& filterSpec,
      std::vector<uint64_t>& hits) = 0;

  virtual std::unique_ptr<Filter> rowGroupSkipFilter(
      const std::vector<RowVectorPtr>& /*batches*/,
      const Subfield& /*subfield*/,
      std::vector<uint64_t>& /*hits*/) {
    VELOX_NYI();
  }

 protected:
  const TypePtr type_;
  const RowTypePtr rootType_;
  int32_t numDistinct_ = 0;
  int32_t numNulls_ = 0;
  int32_t numSamples_ = 0;
  std::unordered_map<size_t, int> uniques_;
  static uint32_t counter_;
};

template <typename T>
class ColumnStats : public AbstractColumnStats {
 public:
  explicit ColumnStats(TypePtr type, RowTypePtr rootTypePtr)
      : AbstractColumnStats(type, rootTypePtr) {}

  void sample(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& rows) override {
    int32_t previousBatch = -1;
    SimpleVector<T>* values = nullptr;
    for (auto row : rows) {
      auto batch = batchNumber(row);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];

        values = getChildBySubfield(vector.get(), subfield, rootType_)
                     ->template asUnchecked<SimpleVector<T>>();
      }

      addSample(values, batchRow(row));
    }
    if constexpr (!std::is_same_v<T, ComplexType>) {
      std::sort(values_.begin(), values_.end());
    }
  }

  std::unique_ptr<Filter> filter(
      const std::vector<RowVectorPtr>& batches,
      const FilterSpec& filterSpec,
      std::vector<uint64_t>& hits) override {
    Subfield subfield(filterSpec.field);
    std::unique_ptr<Filter> filter;
    switch (filterSpec.filterKind) {
      case FilterKind::kIsNull:
        filter = std::make_unique<velox::common::IsNull>();
        break;
      case FilterKind::kIsNotNull:
        filter = std::make_unique<velox::common::IsNotNull>();
        break;
      default:
        filter = makeRangeFilter(filterSpec);
        break;
    }

    size_t numHits = 0;
    SimpleVector<T>* values = nullptr;
    int32_t previousBatch = -1;
    for (auto hit : hits) {
      auto batch = batchNumber(hit);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];
        values = getChildBySubfield(batches[batch].get(), subfield, rootType_)
                     ->template as<SimpleVector<T>>();
      }
      auto row = batchRow(hit);
      if (values->isNullAt(row)) {
        if (filter->testNull()) {
          hits[numHits++] = hit;
        }
        continue;
      }
      if (velox::common::applyFilter(*filter, values->valueAt(row))) {
        hits[numHits++] = hit;
      }
    }
    hits.resize(numHits);
    return filter;
  }

  std::unique_ptr<Filter> rowGroupSkipFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& hits) override {
    std::unique_ptr<Filter> filter;
    filter = makeRowGroupSkipRangeFilter(batches, subfield);
    size_t numHits = 0;
    SimpleVector<T>* values = nullptr;
    int32_t previousBatch = -1;
    for (auto hit : hits) {
      auto batch = batchNumber(hit);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];
        values = getChildBySubfield(batches[batch].get(), subfield, rootType_)
                     ->template as<SimpleVector<T>>();
      }
      auto row = batchRow(hit);
      if (values->isNullAt(row)) {
        if (filter->testNull()) {
          hits[numHits++] = hit;
        }
        continue;
      }
      if (velox::common::applyFilter(*filter, values->valueAt(row))) {
        hits[numHits++] = hit;
      }
    }
    hits.resize(numHits);
    return filter;
  }

 private:
  void addSample(SimpleVector<T>* vector, vector_size_t index) {
    ++numSamples_;
    if (vector->isNullAt(index)) {
      ++numNulls_;
      return;
    }
    T value = vector->valueAt(index);
    size_t hash = folly::hasher<T>()(value) & kUniquesMask;
    if (uniques_.find(hash) != uniques_.end()) {
      uniques_[hash]++;
      return;
    }
    uniques_[hash]++;
    ++numDistinct_;
    values_.push_back(value);
  }

  int32_t findIndex(float pct) {
    int32_t index = 0;
    int32_t sampleCount = 0;

    for (; index < values_.size(); index++) {
      auto value = values_[index];
      size_t hash = folly::hasher<T>()(value) & kUniquesMask;
      sampleCount += uniques_[hash];
      if (sampleCount >= (pct / 100) * (numSamples_ - numNulls_)) {
        break;
      }
    }
    return index;
  }

  T valueAtPct(float pct, int32_t* indexOut = nullptr) {
    int32_t index = findIndex(pct);
    int32_t boundedIndex =
        std::min<int32_t>(values_.size() - 1, std::max<int32_t>(0, index));
    if (indexOut) {
      *indexOut = boundedIndex;
    }
    return values_[boundedIndex];
  }

  int64_t getIntegerValue(const T& value) {
    return value;
  }

  std::unique_ptr<Filter> makeRangeFilter(const FilterSpec& filterSpec) {
    if (values_.empty()) {
      if (filterSpec.allowNulls_) {
        return std::make_unique<velox::common::IsNull>();
      } else {
        return std::make_unique<velox::common::BigintRange>(0, 0, false);
      }
    }
    int32_t lowerIndex;
    int32_t upperIndex;
    T lower = valueAtPct(filterSpec.startPct, &lowerIndex);
    T upper =
        valueAtPct(filterSpec.startPct + filterSpec.selectPct, &upperIndex);
    if (!filterSpec.allowNulls_) {
      return std::make_unique<velox::common::BigintRange>(
          getIntegerValue(lower), getIntegerValue(upper), false);
    }
    if (upperIndex - lowerIndex < 1000 && ++counter_ % 10 <= 3) {
      std::vector<int64_t> in;
      for (auto i = lowerIndex; i <= upperIndex; ++i) {
        in.push_back(getIntegerValue(values_[i]));
      }
      // make sure we don't accidentally generate an AlwaysFalse filter
      if (counter_ % 2 == 1 && filterSpec.selectPct < 100.0) {
        return velox::common::createNegatedBigintValues(in, true);
      }
      return velox::common::createBigintValues(in, true);
    }
    // sometimes make a negated filter instead (1/4 chance)
    if (counter_ % 4 == 1 && filterSpec.selectPct < 100.0) {
      return std::make_unique<velox::common::NegatedBigintRange>(
          getIntegerValue(lower),
          getIntegerValue(upper),
          filterSpec.selectPct < 75);
    }
    return std::make_unique<velox::common::BigintRange>(
        getIntegerValue(lower),
        getIntegerValue(upper),
        filterSpec.selectPct > 25);
  }

  std::unique_ptr<Filter> makeRowGroupSkipRangeFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield) {
    T max;
    bool hasMax = false;
    for (auto batch : batches) {
      auto values = getChildBySubfield(batch.get(), subfield, rootType_)
                        ->template as<SimpleVector<T>>();
      DWIO_ENSURE_NOT_NULL(
          values,
          "Failed to convert to SimpleVector<",
          typeid(T).name(),
          "> for batch of kind ",
          batch->type()->kindName());
      for (auto i = 0; i < values->size(); ++i) {
        if (values->isNullAt(i)) {
          continue;
        }
        if (hasMax && max < values->valueAt(i)) {
          max = values->valueAt(i);
        } else if (!hasMax) {
          max = values->valueAt(i);
          hasMax = true;
        }
      }
    }
    return std::make_unique<velox::common::BigintRange>(
        getIntegerValue(max), getIntegerValue(max), false);
  }

  // The sample size is 65536.
  static constexpr size_t kUniquesMask = 0xffff;
  std::vector<T> values_;
};

class ComplexColumnStats : public AbstractColumnStats {
 public:
  explicit ComplexColumnStats(TypePtr type, RowTypePtr rootTypePtr)
      : AbstractColumnStats(type, rootTypePtr) {}

  void sample(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& rows) override {
    int32_t previousBatch = -1;
    VectorPtr values = nullptr;
    for (auto row : rows) {
      auto batch = batchNumber(row);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];

        values = getChildBySubfield(vector.get(), subfield, rootType_);
      }
      ++numSamples_;
      if (values->isNullAt(batchRow(row))) {
        ++numNulls_;
      }
    }
  }

  std::unique_ptr<Filter> filter(
      const std::vector<RowVectorPtr>& batches,
      const FilterSpec& filterSpec,
      std::vector<uint64_t>& hits) override {
    Subfield subfield(filterSpec.field);
    std::unique_ptr<Filter> filter;
    // A complex type can only have is null and is not null filters. make an is
    // null if selective.
    if (filterSpec.selectPct < 20) {
      filter = std::make_unique<velox::common::IsNull>();
    } else {
      filter = std::make_unique<velox::common::IsNotNull>();
    }
    size_t numHits = 0;
    BaseVector* values = nullptr;
    bool isNull = filter->kind() == velox::common::FilterKind::kIsNull;
    int32_t previousBatch = -1;
    for (auto hit : hits) {
      auto batch = batchNumber(hit);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];
        values =
            getChildBySubfield(batches[batch].get(), subfield, rootType_).get();
      }
      auto row = batchRow(hit);
      if (values->isNullAt(row) == isNull) {
        hits[numHits++] = hit;
      }
    }
    if (!numHits) {
      // Do not make a filter that selects nothing.
      return nullptr;
    }
    hits.resize(numHits);
    return filter;
  }

  std::unique_ptr<Filter> rowGroupSkipFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& hits) override {
    VELOX_FAIL("N/A in ComplexType");
  }

 private:
  std::unique_ptr<Filter> makeRangeFilter(const FilterSpec&) {
    VELOX_FAIL("N/A in ComplexType");
  }

  std::unique_ptr<Filter> makeRowGroupSkipRangeFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield) {
    VELOX_FAIL("N/A in ComplexType");
  }
};

template <>
std::unique_ptr<Filter> ColumnStats<bool>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<float>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<double>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeRowGroupSkipRangeFilter(
    const std::vector<RowVectorPtr>& /*batches*/,
    const Subfield& /*subfield*/);

template <TypeKind Kind>
std::unique_ptr<AbstractColumnStats> makeStats(
    TypePtr type,
    RowTypePtr rootType) {
  using T = typename TypeTraits<Kind>::NativeType;
  return std::make_unique<ColumnStats<T>>(type, rootType);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::ROW>(
    TypePtr type,
    RowTypePtr rootType) {
  return std::make_unique<ComplexColumnStats>(type, rootType);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::ARRAY>(
    TypePtr type,
    RowTypePtr rootType) {
  return std::make_unique<ComplexColumnStats>(type, rootType);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::MAP>(
    TypePtr type,
    RowTypePtr rootType) {
  return std::make_unique<ComplexColumnStats>(type, rootType);
}

class FilterGenerator {
 public:
  static std::string specsToString(const std::vector<FilterSpec>& specs);
  static SubfieldFilters cloneSubfieldFilters(const SubfieldFilters& src);

  explicit FilterGenerator(
      std::shared_ptr<const RowType>& rowType,
      folly::Random::DefaultGenerator::result_type seed =
          folly::Random::DefaultGenerator::default_seed)
      : rowType_(rowType), seed_(seed), rng_(seed) {}

  SubfieldFilters makeSubfieldFilters(
      const std::vector<FilterSpec>& filterSpecs,
      const std::vector<RowVectorPtr>& batches,
      std::vector<uint64_t>& hitRows);
  std::vector<std::string> makeFilterables(uint32_t count, float pct);
  std::vector<FilterSpec> makeRandomSpecs(
      const std::vector<std::string>& filterable,
      int32_t countX100);

  // Make a ScanSpec with random prunings on columns included in 'prunable'.
  // Only complex typed columns are prunable.
  std::shared_ptr<ScanSpec> makeScanSpec(
      const std::vector<std::string>& prunable,
      std::vector<RowVectorPtr>& batches,
      memory::MemoryPool* pool);

  // Make a ScanSpec with the filters specified.
  std::shared_ptr<ScanSpec> makeScanSpec(const SubfieldFilters& filters);

  // Add the filter to an existing ScanSpec.
  static void addToScanSpec(const SubfieldFilters& filters, ScanSpec&);

  inline folly::Random::DefaultGenerator& rng() {
    return rng_;
  }

  inline void reseedRng() {
    rng_.seed(seed_);
  }

  inline const std::unordered_map<std::string, std::array<int32_t, 2>>&
  filterCoverage() {
    return filterCoverage_;
  }

 private:
  static void collectFilterableSubFields(
      const RowType* rowType,
      std::vector<std::string>& subFields);

  std::shared_ptr<const RowType> rowType_;
  folly::Random::DefaultGenerator::result_type seed_;
  folly::Random::DefaultGenerator rng_;
  std::unordered_map<std::string, std::array<int32_t, 2>> filterCoverage_;
};

} // namespace facebook::velox::dwio::common
