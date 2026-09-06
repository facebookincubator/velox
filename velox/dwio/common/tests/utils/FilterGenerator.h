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

#include <fmt/format.h>
#include <folly/Random.h>
#include <memory>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::dwio::common {

using namespace facebook::velox::common;

struct FilterSpec {
  FilterSpec() {}

  explicit FilterSpec(
      std::string field,
      float startPct = 50,
      float selectPct = 20,
      FilterKind filterKind = FilterKind::kBigintRange,
      bool isForRowGroupSkip = false,
      bool allowNulls = true)
      : field(field),
        startPct(startPct),
        selectPct(selectPct),
        filterKind(filterKind),
        isForRowGroupSkip(isForRowGroupSkip),
        allowNulls_(allowNulls) {}

  std::string toString() const {
    return fmt::format(
        "FilterSpec(field={}, startPct={}, selectPct={}, filterKind={}, isForRowGroupSkip={}, isForEmptyResult={}, allowNulls={})",
        field,
        startPct,
        selectPct,
        filterKind,
        isForRowGroupSkip,
        isForEmptyResult,
        allowNulls_);
  }

  std::string field;
  float startPct = 50;
  float selectPct = 20;
  FilterKind filterKind = FilterKind::kBigintRange;
  // If true, makes a filter that matches max value in the column so as to skip
  // row groups on min/max.
  bool isForRowGroupSkip{false};
  // If true, makes a filter positioned just past the column maximum, so that no
  // row survives it. Opt in with FilterGenerator::setEmptyResultProbability.
  bool isForEmptyResult{false};
  bool allowNulls_{true};
};

struct MutationSpec {
  std::vector<int64_t> deletedRows;
};

// Encodes a batch number and an index into the batch into an int32_t
uint64_t batchPosition(uint32_t batchNumber, vector_size_t batchRow);
uint32_t batchNumber(uint64_t position);
vector_size_t batchRow(uint64_t position);
VectorPtr getChildBySubfield(
    const RowVector* rowVector,
    const Subfield& subfield,
    const RowTypePtr& rowType = nullptr);

// As above, and additionally marks in 'ancestorNulls' the rows where a struct
// enclosing the subfield is null. Needed because a null struct leaves its
// children's values undefined instead of marking them null, so the leaf's own
// null flags do not say what a reader will produce for the subfield. Left
// empty when the path crosses an ARRAY or a MAP, where positions stop being
// row indices and nothing collected can be lined up with the caller's rows.
VectorPtr getChildBySubfieldWithAncestorNulls(
    const RowVector* rowVector,
    const Subfield& subfield,
    const RowTypePtr& rootType,
    std::vector<uint8_t>* ancestorNulls);

class AbstractColumnStats {
 public:
  // ASCII string greater than test data values. Used for row group skipping
  // tests.
  static constexpr const char* kMaxString = "~~~~~";
  AbstractColumnStats(
      TypePtr type,
      RowTypePtr rootType,
      folly::Random::DefaultGenerator& rng)
      : type_(type), rootType_(rootType), rng_(rng) {}

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

  // Returns a filter that no row passes, or nullptr when the column admits no
  // such filter (every value null, or the maximum is already the type maximum).
  virtual std::unique_ptr<Filter> emptyResultFilter(
      const std::vector<RowVectorPtr>& /*batches*/,
      const Subfield& /*subfield*/,
      std::vector<uint64_t>& /*hits*/) {
    VELOX_NYI();
  }

 protected:
  // Empty whenever the subfield has no struct above it, or the path crossed an
  // ARRAY or a MAP and the positions stopped meaning rows.
  static bool isAncestorNull(
      const std::vector<uint8_t>& ancestorNulls,
      vector_size_t row) {
    return !ancestorNulls.empty() && ancestorNulls[row] != 0;
  }

  // Whether the generated filter should pass nulls. `allowNulls_` on the spec
  // is a hard constraint from the caller; when it permits nulls the answer is
  // drawn independently rather than derived from selectPct, which used to tie
  // the two together and make "selective filter that also passes nulls"
  // unreachable.
  bool drawNullAllowed(const FilterSpec& filterSpec) {
    return filterSpec.allowNulls_ && folly::Random::oneIn(2, rng_);
  }

  // Carves the selected percentile band into two or three sub-bands for an
  // OR of disjoint ranges. Each sub-band covers only the first half of its
  // slice; the second half is the gap that makes the result an OR rather than
  // one wide range. Returned as percentiles so both the integer and the string
  // side can resolve them against their own sorted samples.
  std::vector<std::pair<float, float>> multiRangeBands(
      const FilterSpec& filterSpec) {
    const int32_t count =
        2 + static_cast<int32_t>(folly::Random::rand32(2, rng_));
    const float width = filterSpec.selectPct / static_cast<float>(count);
    std::vector<std::pair<float, float>> bands;
    bands.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
      const float from = filterSpec.startPct + static_cast<float>(i) * width;
      bands.emplace_back(from, from + width / 2);
    }
    return bands;
  }

  const TypePtr type_;
  const RowTypePtr rootType_;
  int32_t numDistinct_ = 0;
  int32_t numNulls_ = 0;
  int32_t numSamples_ = 0;
  std::unordered_map<size_t, int> uniques_;
  // Borrowed from the owning FilterGenerator so that filter kind selection is
  // driven by the run's seed. This used to be a process-global counter, which
  // made the choice depend on how many columns happened to be processed
  // earlier and left it outside the seed's control entirely.
  folly::Random::DefaultGenerator& rng_;
};

template <typename T>
class ColumnStats : public AbstractColumnStats {
 public:
  ColumnStats(
      TypePtr type,
      RowTypePtr rootTypePtr,
      folly::Random::DefaultGenerator& rng)
      : AbstractColumnStats(type, rootTypePtr, rng) {}

  void sample(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& rows) override {
    int32_t previousBatch = -1;
    SimpleVector<T>* values = nullptr;
    std::vector<uint8_t> ancestorNulls;
    for (auto row : rows) {
      auto batch = batchNumber(row);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];

        ancestorNulls.clear();
        values = getChildBySubfieldWithAncestorNulls(
                     vector.get(), subfield, rootType_, &ancestorNulls)
                     ->template asUnchecked<SimpleVector<T>>();
      }

      // A row under a null struct contributes nothing to the value
      // distribution: the leaf holds a value there, but no reader will ever
      // return it.
      if (isAncestorNull(ancestorNulls, batchRow(row))) {
        ++numSamples_;
        ++numNulls_;
        continue;
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
      case FilterKind::kBytesRange:
        filter = makeRangeFilter(filterSpec);
        break;
      default:
        if (type_->kind() == TypeKind::VARCHAR) {
          filter = makeRandomFilter(filterSpec);
        } else {
          filter = makeRangeFilter(filterSpec);
        }
        break;
    }

    narrowHits(batches, subfield, *filter, hits);
    return filter;
  }

  std::unique_ptr<Filter> rowGroupSkipFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& hits) override {
    std::unique_ptr<Filter> filter;
    filter = makeRowGroupSkipRangeFilter(batches, subfield);
    narrowHits(batches, subfield, *filter, hits);
    return filter;
  }

  std::unique_ptr<Filter> emptyResultFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& hits) override {
    auto filter = makeEmptyResultRangeFilter(batches, subfield);
    if (filter == nullptr) {
      return nullptr;
    }
    narrowHits(batches, subfield, *filter, hits);
    return filter;
  }

 private:
  // Narrows 'hits' to the rows the filter passes. Every filter flavor shares
  // this: the reference row set has to be derived from the filter that is
  // actually handed to the reader, not from the spec that produced it.
  void narrowHits(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      const Filter& filter,
      std::vector<uint64_t>& hits) {
    size_t numHits = 0;
    SimpleVector<T>* values = nullptr;
    int32_t previousBatch = -1;
    std::vector<uint8_t> ancestorNulls;
    for (auto hit : hits) {
      auto batch = batchNumber(hit);
      if (batch != previousBatch) {
        previousBatch = batch;
        ancestorNulls.clear();
        values = getChildBySubfieldWithAncestorNulls(
                     batches[batch].get(), subfield, rootType_, &ancestorNulls)
                     ->template as<SimpleVector<T>>();
      }
      auto row = batchRow(hit);
      if (values->isNullAt(row) || isAncestorNull(ancestorNulls, row)) {
        if (filter.testNull()) {
          hits[numHits++] = hit;
        }
        continue;
      }
      if (velox::common::applyFilter(filter, values->valueAt(row))) {
        hits[numHits++] = hit;
      }
    }
    hits.resize(numHits);
  }

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
    const bool nullAllowed = drawNullAllowed(filterSpec);
    if (upperIndex - lowerIndex < 1000 &&
        folly::Random::rand32(10, rng_) <= 3) {
      std::vector<int64_t> in;
      for (auto i = lowerIndex; i <= upperIndex; ++i) {
        in.push_back(getIntegerValue(values_[i]));
      }
      // make sure we don't accidentally generate an AlwaysFalse filter
      if (folly::Random::oneIn(2, rng_) && filterSpec.selectPct < 100.0) {
        return velox::common::createNegatedBigintValues(in, nullAllowed);
      }
      return velox::common::createBigintValues(in, nullAllowed);
    }
    // sometimes make a negated filter instead (1/4 chance)
    if (folly::Random::oneIn(4, rng_) && filterSpec.selectPct < 100.0) {
      return std::make_unique<velox::common::NegatedBigintRange>(
          getIntegerValue(lower), getIntegerValue(upper), nullAllowed);
    }
    // An OR of disjoint ranges. Its own testInt64 and testInt64Range mean a
    // reader takes a different path than for one contiguous range, and a
    // min/max check can no longer be a single comparison against the bounds.
    if (folly::Random::oneIn(4, rng_)) {
      auto multiRange = makeMultiRangeFilter(filterSpec, nullAllowed);
      if (multiRange != nullptr) {
        return multiRange;
      }
    }
    return std::make_unique<velox::common::BigintRange>(
        getIntegerValue(lower), getIntegerValue(upper), nullAllowed);
  }

  // Returns nullptr when the sample is too coarse for the sub-ranges to come
  // out ascending and disjoint, which BigintMultiRange requires and a column
  // with few distinct values cannot always satisfy -- valueAtPct then hands
  // back the same value for neighbouring percentiles.
  std::unique_ptr<Filter> makeMultiRangeFilter(
      const FilterSpec& filterSpec,
      bool nullAllowed) {
    std::vector<std::unique_ptr<velox::common::BigintRange>> ranges;
    int64_t previousUpper = 0;
    for (const auto& [from, to] : multiRangeBands(filterSpec)) {
      const int64_t lower = getIntegerValue(valueAtPct(from));
      const int64_t upper = getIntegerValue(valueAtPct(to));
      if (lower > upper || (!ranges.empty() && lower <= previousUpper)) {
        return nullptr;
      }
      previousUpper = upper;
      ranges.push_back(
          std::make_unique<velox::common::BigintRange>(lower, upper, false));
    }
    return std::make_unique<velox::common::BigintMultiRange>(
        std::move(ranges), nullAllowed);
  }

  // Scans every row rather than the sample in 'values_': a filter that has to
  // sit at or past the column maximum cannot be built from a sample. Returns
  // false, leaving 'max' untouched, if every value is null.
  bool columnMax(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      T& max) {
    bool hasMax = false;
    std::vector<uint8_t> ancestorNulls;
    for (auto batch : batches) {
      ancestorNulls.clear();
      auto values = getChildBySubfieldWithAncestorNulls(
                        batch.get(), subfield, rootType_, &ancestorNulls)
                        ->template as<SimpleVector<T>>();
      DWIO_ENSURE_NOT_NULL(
          values,
          "Failed to convert to SimpleVector<",
          typeid(T).name(),
          "> for batch of kind ",
          batch->type()->kindName());
      for (auto i = 0; i < values->size(); ++i) {
        if (values->isNullAt(i) || isAncestorNull(ancestorNulls, i)) {
          continue;
        }
        if (!hasMax || max < values->valueAt(i)) {
          max = values->valueAt(i);
          hasMax = true;
        }
      }
    }
    return hasMax;
  }

  std::unique_ptr<Filter> makeRowGroupSkipRangeFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield) {
    T max{};
    columnMax(batches, subfield, max);
    return std::make_unique<velox::common::BigintRange>(
        getIntegerValue(max), getIntegerValue(max), false);
  }

  // One past the column maximum. Only reached for the integral kinds, where
  // getIntegerValue is exact -- see supportsEmptyResult in the .cpp -- so the
  // resulting range cannot accidentally overlap a value.
  std::unique_ptr<Filter> makeEmptyResultRangeFilter(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield) {
    T max{};
    if (!columnMax(batches, subfield, max)) {
      return nullptr;
    }
    const int64_t bound = getIntegerValue(max);
    if (bound == std::numeric_limits<int64_t>::max()) {
      return nullptr;
    }
    return std::make_unique<velox::common::BigintRange>(
        bound + 1, bound + 1, false);
  }

  std::unique_ptr<Filter> makeRandomFilter(const FilterSpec& filterSpec) {
    VELOX_FAIL("This method is only used in specific types.");
  }

  // The sample size is 65536.
  static constexpr size_t kUniquesMask = 0xffff;
  std::vector<T> values_;
};

class ComplexColumnStats : public AbstractColumnStats {
 public:
  ComplexColumnStats(
      TypePtr type,
      RowTypePtr rootTypePtr,
      folly::Random::DefaultGenerator& rng)
      : AbstractColumnStats(type, rootTypePtr, rng) {}

  void sample(
      const std::vector<RowVectorPtr>& batches,
      const Subfield& subfield,
      std::vector<uint64_t>& rows) override {
    int32_t previousBatch = -1;
    VectorPtr values = nullptr;
    std::vector<uint8_t> ancestorNulls;
    for (auto row : rows) {
      auto batch = batchNumber(row);
      if (batch != previousBatch) {
        previousBatch = batch;
        auto vector = batches[batch];

        ancestorNulls.clear();
        values = getChildBySubfieldWithAncestorNulls(
            vector.get(), subfield, rootType_, &ancestorNulls);
      }
      ++numSamples_;
      if (values->isNullAt(batchRow(row)) ||
          isAncestorNull(ancestorNulls, batchRow(row))) {
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
    std::vector<uint8_t> ancestorNulls;
    VectorPtr held;
    for (auto hit : hits) {
      auto batch = batchNumber(hit);
      if (batch != previousBatch) {
        previousBatch = batch;
        ancestorNulls.clear();
        held = getChildBySubfieldWithAncestorNulls(
            batches[batch].get(), subfield, rootType_, &ancestorNulls);
        values = held.get();
      }
      auto row = batchRow(hit);
      const bool rowIsNull =
          values->isNullAt(row) || isAncestorNull(ancestorNulls, row);
      if (rowIsNull == isNull) {
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

  // A complex type only admits is null / is not null, so whether either matches
  // nothing depends on the data rather than on how the filter is built.
  std::unique_ptr<Filter> emptyResultFilter(
      const std::vector<RowVectorPtr>& /*batches*/,
      const Subfield& /*subfield*/,
      std::vector<uint64_t>& /*hits*/) override {
    VELOX_FAIL("N/A in ComplexType");
  }

 private:
  std::unique_ptr<Filter> makeRangeFilter(const FilterSpec&) {
    VELOX_FAIL("N/A in ComplexType");
  }

  std::unique_ptr<Filter> makeRandomFilter(const FilterSpec&) {
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
std::unique_ptr<Filter> ColumnStats<int128_t>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeRandomFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<Timestamp>::makeRangeFilter(
    const FilterSpec& filterSpec);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeRowGroupSkipRangeFilter(
    const std::vector<RowVectorPtr>& /*batches*/,
    const Subfield& /*subfield*/);

template <>
std::unique_ptr<Filter> ColumnStats<Timestamp>::makeRowGroupSkipRangeFilter(
    const std::vector<RowVectorPtr>& /*batches*/,
    const Subfield& /*subfield*/);

template <>
std::unique_ptr<Filter> ColumnStats<StringView>::makeEmptyResultRangeFilter(
    const std::vector<RowVectorPtr>& /*batches*/,
    const Subfield& /*subfield*/);

template <>
std::unique_ptr<Filter> ColumnStats<Timestamp>::makeEmptyResultRangeFilter(
    const std::vector<RowVectorPtr>& /*batches*/,
    const Subfield& /*subfield*/);

template <TypeKind Kind>
std::unique_ptr<AbstractColumnStats> makeStats(
    TypePtr type,
    RowTypePtr rootType,
    folly::Random::DefaultGenerator& rng) {
  using T = typename TypeTraits<Kind>::NativeType;
  return std::make_unique<ColumnStats<T>>(type, rootType, rng);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::ROW>(
    TypePtr type,
    RowTypePtr rootType,
    folly::Random::DefaultGenerator& rng) {
  return std::make_unique<ComplexColumnStats>(type, rootType, rng);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::ARRAY>(
    TypePtr type,
    RowTypePtr rootType,
    folly::Random::DefaultGenerator& rng) {
  return std::make_unique<ComplexColumnStats>(type, rootType, rng);
}

template <>
inline std::unique_ptr<AbstractColumnStats> makeStats<TypeKind::MAP>(
    TypePtr type,
    RowTypePtr rootType,
    folly::Random::DefaultGenerator& rng) {
  return std::make_unique<ComplexColumnStats>(type, rootType, rng);
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
      MutationSpec*,
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

  // Probability that a generated spec set asks for a filter no row passes.
  // Zero by default, and at zero not a single random number is drawn for it, so
  // callers that do not opt in keep their exact filter sequence.
  //
  // Ordinary selective filters already intersect to nothing now and then, but
  // only by accident and only from inside the value range, so the reader still
  // descends into every row group and rejects row by row. The filter this asks
  // for sits past the column maximum instead, which is what makes min/max
  // pruning skip every row group. Opt-in because it is a trade: the columns
  // filtered alongside it contribute nothing, having no row left to accept or
  // reject.
  void setEmptyResultProbability(double probability) {
    VELOX_CHECK_GE(probability, 0.0);
    VELOX_CHECK_LE(probability, 1.0);
    emptyResultProbability_ = probability;
  }

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
  double emptyResultProbability_{0.0};
  std::unordered_map<std::string, std::array<int32_t, 2>> filterCoverage_;
};

} // namespace facebook::velox::dwio::common
