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

#include <optional>
#include <vector>

#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/KllSketchAccumulator.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate {

enum ApproxPercentileIntermediateTypeChildIndex {
  kPercentiles = 0,
  kPercentilesIsArray = 1,
  kAccuracy = 2,
  kK = 3,
  kN = 4,
  kMinValue = 5,
  kMaxValue = 6,
  kItems = 7,
  kLevels = 8,
};

template <typename T, bool kHasWeight, bool kAccuracyIsErrorBound>
class ApproxPercentileAggregateBase : public exec::Aggregate {
 public:
  ApproxPercentileAggregateBase(
      bool hasWeight,
      bool hasAccuracy,
      const TypePtr& resultType,
      std::optional<uint32_t> fixedRandomSeed)
      : exec::Aggregate(resultType),
        hasWeight_(hasWeight),
        hasAccuracy_(hasAccuracy),
        fixedRandomSeed_(fixedRandomSeed) {}

  struct Percentiles {
    std::vector<double> values;
    bool isArray;
  };

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(KllSketchAccumulator<T>);
  }

  bool isFixedSize() const override {
    return false;
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);

    if (kHasWeight && hasWeight_) {
      rows.applyToSelected([&](auto row) {
        if (decodedValue_.isNullAt(row) || decodedWeight_.isNullAt(row)) {
          return;
        }

        auto tracker = trackRowSize(groups[row]);
        auto accumulator = initRawAccumulator(groups[row]);
        auto value = decodedValue_.valueAt<T>(row);
        auto weight = decodedWeight_.valueAt<int64_t>(row);
        checkWeight(weight);
        accumulator->append(value, weight, allocator_, fixedRandomSeed_);
      });
    } else {
      if (decodedValue_.mayHaveNulls()) {
        rows.applyToSelected([&](auto row) {
          if (decodedValue_.isNullAt(row)) {
            return;
          }

          auto accumulator = initRawAccumulator(groups[row]);
          accumulator->append(decodedValue_.valueAt<T>(row));
        });
      } else {
        rows.applyToSelected([&](auto row) {
          auto accumulator = initRawAccumulator(groups[row]);
          accumulator->append(decodedValue_.valueAt<T>(row));
        });
      }
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediate<false>(groups, rows, args);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);

    auto tracker = trackRowSize(group);
    auto accumulator = initRawAccumulator(group);

    if (kHasWeight && hasWeight_) {
      rows.applyToSelected([&](auto row) {
        if (decodedValue_.isNullAt(row) || decodedWeight_.isNullAt(row)) {
          return;
        }

        auto value = decodedValue_.valueAt<T>(row);
        auto weight = decodedWeight_.valueAt<int64_t>(row);
        checkWeight(weight);
        accumulator->append(value, weight, allocator_, fixedRandomSeed_);
      });
    } else {
      if (decodedValue_.mayHaveNulls()) {
        rows.applyToSelected([&](auto row) {
          if (decodedValue_.isNullAt(row)) {
            return;
          }

          accumulator->append(decodedValue_.valueAt<T>(row));
        });
      } else {
        rows.applyToSelected([&](auto row) {
          accumulator->append(decodedValue_.valueAt<T>(row));
        });
      }
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediate<true>(group, rows, args);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    for (auto i = 0; i < numGroups; ++i) {
      value<KllSketchAccumulator<T>>(groups[i])->flush(
          allocator_, fixedRandomSeed_);
    }

    VELOX_CHECK(result);
    // When all inputs are nulls or masked out, percentiles_ can be
    // uninitialized. The result should be nulls in this case.
    if (!percentiles_.has_value()) {
      *result = BaseVector::createNullConstant(
          (*result)->type(), numGroups, (*result)->pool());
      return;
    }

    if (percentiles_ && percentiles_->isArray) {
      folly::Range percentiles(
          percentiles_->values.begin(), percentiles_->values.end());
      auto arrayResult = (*result)->asUnchecked<ArrayVector>();
      vector_size_t elementsCount = 0;
      for (auto i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        auto accumulator = value<KllSketchAccumulator<T>>(group);
        if (accumulator->getSketch().totalCount() > 0) {
          elementsCount += percentiles.size();
        }
      }
      arrayResult->elements()->resize(elementsCount);
      elementsCount = 0;
      auto rawValues =
          arrayResult->elements()->asFlatVector<T>()->mutableRawValues();
      extractHelper(
          groups,
          numGroups,
          arrayResult,
          [&](const KllSketch<T>& digest,
              ArrayVector* result,
              vector_size_t index) {
            digest.estimateQuantiles(percentiles, rawValues + elementsCount);
            result->setOffsetAndSize(index, elementsCount, percentiles.size());
            result->setNull(index, false);
            elementsCount += percentiles.size();
          });
    } else {
      extractHelper(
          groups,
          numGroups,
          (*result)->asFlatVector<T>(),
          [&](const KllSketch<T>& digest,
              FlatVector<T>* result,
              vector_size_t index) {
            VELOX_DCHECK_EQ(percentiles_->values.size(), 1);
            result->set(
                index, digest.estimateQuantile(percentiles_->values.back()));
          });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    std::vector<KllSketch<T, std::allocator<T>>> sketches;
    sketches.reserve(numGroups);
    for (auto i = 0; i < numGroups; ++i) {
      sketches.push_back(
          value<KllSketchAccumulator<T>>(groups[i])->compact(fixedRandomSeed_));
    }

    VELOX_CHECK(result);
    auto rowResult = (*result)->as<RowVector>();
    VELOX_CHECK(rowResult);
    auto pool = rowResult->pool();

    // percentiles_ can be uninitialized during an intermediate aggregation step
    // when all input intermediate states are nulls. Result should be nulls in
    // this case.
    if (!percentiles_) {
      rowResult->ensureWritable(SelectivityVector{numGroups});
      // rowResult->childAt(i) for i = kPercentiles, kPercentilesIsArray, and
      // kAccuracy are expected to be constant in addIntermediateResults.
      rowResult->childAt(kPercentiles) =
          BaseVector::createNullConstant(ARRAY(DOUBLE()), numGroups, pool);
      rowResult->childAt(kPercentilesIsArray) =
          BaseVector::createNullConstant(BOOLEAN(), numGroups, pool);
      rowResult->childAt(kAccuracy) =
          BaseVector::createNullConstant(DOUBLE(), numGroups, pool);

      // Set nulls for all rows.
      auto rawNulls = rowResult->mutableRawNulls();
      bits::fillBits(rawNulls, 0, rowResult->size(), bits::kNull);
      return;
    }
    auto& values = percentiles_->values;
    auto size = values.size();
    auto elements =
        BaseVector::create<FlatVector<double>>(DOUBLE(), size, pool);
    std::copy(values.begin(), values.end(), elements->mutableRawValues());
    auto array = std::make_shared<ArrayVector>(
        pool,
        ARRAY(DOUBLE()),
        nullptr,
        1,
        AlignedBuffer::allocate<vector_size_t>(1, pool, 0),
        AlignedBuffer::allocate<vector_size_t>(1, pool, size),
        std::move(elements));
    rowResult->childAt(kPercentiles) =
        BaseVector::wrapInConstant(numGroups, 0, std::move(array));
    rowResult->childAt(kPercentilesIsArray) =
        std::make_shared<ConstantVector<bool>>(
            pool,
            numGroups,
            false,
            BOOLEAN(),
            static_cast<bool&&>(percentiles_->isArray));

    bool isDefaultAccuracy = false;
    if constexpr (kAccuracyIsErrorBound) {
      isDefaultAccuracy = (accuracy_ == kMissingNormalizedValue);
    }
    rowResult->childAt(kAccuracy) = std::make_shared<ConstantVector<double>>(
        pool,
        numGroups,
        isDefaultAccuracy,
        DOUBLE(),
        static_cast<double&&>(accuracy_));
    auto k = rowResult->childAt(kK)->asFlatVector<int32_t>();
    auto n = rowResult->childAt(kN)->asFlatVector<int64_t>();
    auto minValue = rowResult->childAt(kMinValue)->asFlatVector<T>();
    auto maxValue = rowResult->childAt(kMaxValue)->asFlatVector<T>();
    auto items = rowResult->childAt(kItems)->as<ArrayVector>();
    auto levels = rowResult->childAt(kLevels)->as<ArrayVector>();

    rowResult->resize(numGroups);
    k->resize(numGroups);
    n->resize(numGroups);
    minValue->resize(numGroups);
    maxValue->resize(numGroups);
    items->resize(numGroups);
    levels->resize(numGroups);

    auto itemsElements = items->elements()->asFlatVector<T>();
    auto levelsElements = levels->elements()->asFlatVector<int32_t>();
    size_t itemsCount = 0;
    vector_size_t levelsCount = 0;
    for (auto& sketch : sketches) {
      auto v = sketch.toView();
      itemsCount += v.items.size();
      levelsCount += v.levels.size();
    }
    VELOX_CHECK_LE(itemsCount, std::numeric_limits<vector_size_t>::max());
    itemsElements->resetNulls();
    itemsElements->resize(itemsCount);
    levelsElements->resetNulls();
    levelsElements->resize(levelsCount);

    auto rawItems = itemsElements->mutableRawValues();
    auto rawLevels = levelsElements->mutableRawValues();
    itemsCount = 0;
    levelsCount = 0;
    for (int i = 0; i < sketches.size(); ++i) {
      auto v = sketches[i].toView();
      if (v.n == 0) {
        rowResult->setNull(i, true);
      } else {
        rowResult->setNull(i, false);
        k->set(i, v.k);
        n->set(i, v.n);
        minValue->set(i, v.minValue);
        maxValue->set(i, v.maxValue);
        std::copy(v.items.begin(), v.items.end(), rawItems + itemsCount);
        items->setOffsetAndSize(i, itemsCount, v.items.size());
        itemsCount += v.items.size();
        std::copy(v.levels.begin(), v.levels.end(), rawLevels + levelsCount);
        levels->setOffsetAndSize(i, levelsCount, v.levels.size());
        levelsCount += v.levels.size();
      }
    }
  }

 protected:
  const bool hasWeight_;
  const bool hasAccuracy_;
  const std::optional<uint32_t> fixedRandomSeed_;

  static constexpr double kMissingNormalizedValue = -1;
  static constexpr int32_t kSparkDefaultAccuracy = 10000;
  double accuracy_{
      kAccuracyIsErrorBound ? kMissingNormalizedValue
                            : static_cast<double>(kSparkDefaultAccuracy)};

  std::optional<Percentiles> percentiles_;
  DecodedVector decodedValue_;
  DecodedVector decodedWeight_;
  DecodedVector decodedAccuracy_;

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_)
          KllSketchAccumulator<T>(allocator_, fixedRandomSeed_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<KllSketchAccumulator<T>>(group)->~KllSketchAccumulator<T>();
      }
    }
  }

  KllSketchAccumulator<T>* initRawAccumulator(char* group) {
    auto accumulator = value<KllSketchAccumulator<T>>(group);
    if constexpr (kAccuracyIsErrorBound) {
      if (accuracy_ != kMissingNormalizedValue) {
        accumulator->setPrestoAccuracy(accuracy_);
      }
    } else {
      accumulator->setSparkAccuracy(static_cast<int32_t>(accuracy_));
    }
    return accumulator;
  }

  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    size_t argIndex = 0;
    decodedValue_.decode(*args[argIndex++], rows, true);
    if (kHasWeight && hasWeight_) {
      decodedWeight_.decode(*args[argIndex++], rows, true);
    }
    checkSetPercentile(rows, *args[argIndex++]);
    if (hasAccuracy_) {
      decodedAccuracy_.decode(*args[argIndex++], rows, true);
      checkSetAccuracy(rows);
    }
    VELOX_CHECK_EQ(argIndex, args.size());
  }

  void checkSetPercentile(
      const SelectivityVector& rows,
      const BaseVector& vec) {
    DecodedVector decoded(vec, rows);

    auto* base = decoded.base();
    auto baseFirstRow = decoded.index(rows.begin());
    if (!decoded.isConstantMapping()) {
      rows.applyToSelected([&](vector_size_t row) {
        VELOX_USER_CHECK(!decoded.isNullAt(row), "Percentile cannot be null");
        auto baseRow = decoded.index(row);
        VELOX_USER_CHECK(
            base->equalValueAt(base, baseRow, baseFirstRow),
            "Percentile argument must be constant for all input rows: {} vs. {}",
            base->toString(baseRow),
            base->toString(baseFirstRow));
      });
    }

    bool isArray;
    vector_size_t offset;
    vector_size_t len;
    if (base->typeKind() == TypeKind::DOUBLE) {
      isArray = false;
      if constexpr (kHasWeight) {
        offset = rows.begin();
      } else {
        offset = baseFirstRow;
      }
      len = 1;
    } else if (base->typeKind() == TypeKind::ARRAY) {
      isArray = true;
      auto arrays = base->asUnchecked<ArrayVector>();
      if constexpr (!kHasWeight) {
        auto actualNullCount = arrays->nulls()
            ? BaseVector::countNulls(arrays->nulls(), arrays->size())
            : 0;
        VELOX_USER_CHECK_EQ(
            actualNullCount, 0, "Percentile array cannot contain nulls");
      }
      decoded.decode(*arrays->elements());
      offset = arrays->offsetAt(baseFirstRow);
      len = arrays->sizeAt(baseFirstRow);
    } else {
      VELOX_USER_FAIL(
          "Incorrect type for percentile: {}", base->type()->toString());
    }
    checkSetPercentileValues(isArray, decoded, offset, len);
  }

  void checkSetPercentileValues(
      bool isArray,
      const DecodedVector& percentiles,
      vector_size_t offset,
      vector_size_t len) {
    if (!percentiles_) {
      VELOX_USER_CHECK_GT(len, 0, "Percentile cannot be empty");
      percentiles_ = {
          .values = std::vector<double>(len),
          .isArray = isArray,
      };
      for (vector_size_t i = 0; i < len; ++i) {
        VELOX_USER_CHECK(!percentiles.isNullAt(i), "Percentile cannot be null");
        auto value = percentiles.valueAt<double>(offset + i);
        VELOX_USER_CHECK_GE(value, 0, "Percentile must be between 0 and 1");
        VELOX_USER_CHECK_LE(value, 1, "Percentile must be between 0 and 1");
        percentiles_->values[i] = value;
      }
    } else {
      VELOX_USER_CHECK_EQ(
          isArray,
          percentiles_->isArray,
          "Percentile argument must be constant for all input rows");
      VELOX_USER_CHECK_EQ(
          len,
          percentiles_->values.size(),
          "Percentile argument must be constant for all input rows");
      for (vector_size_t i = 0; i < len; ++i) {
        VELOX_USER_CHECK_EQ(
            percentiles.valueAt<double>(offset + i),
            percentiles_->values[i],
            "Percentile argument must be constant for all input rows");
      }
    }
  }

  void checkSetPercentileValuesFromVector(
      bool isArray,
      const std::vector<double>& percentiles) {
    vector_size_t len = static_cast<vector_size_t>(percentiles.size());
    if (!percentiles_) {
      VELOX_USER_CHECK_GT(len, 0, "Percentile cannot be empty");
      percentiles_ = {.values = percentiles, .isArray = isArray};
      for (const auto& val : percentiles) {
        VELOX_USER_CHECK_GE(val, 0.0, "Percentile must be between 0 and 1");
        VELOX_USER_CHECK_LE(val, 1.0, "Percentile must be between 0 and 1");
      }
    } else {
      VELOX_USER_CHECK_EQ(
          isArray, percentiles_->isArray, "Percentile type must be constant");
      VELOX_USER_CHECK_EQ(
          len,
          static_cast<vector_size_t>(percentiles_->values.size()),
          "Percentile size must be constant");
      for (vector_size_t i = 0; i < len; ++i) {
        VELOX_USER_CHECK_EQ(
            percentiles[i],
            percentiles_->values[i],
            "Percentile values must be constant");
      }
    }
  }

  void checkSetAccuracy(const SelectivityVector& rows) {
    if (!hasAccuracy_) {
      return;
    }

    if constexpr (kAccuracyIsErrorBound) {
      if (decodedAccuracy_.isConstantMapping()) {
        VELOX_USER_CHECK(
            !decodedAccuracy_.isNullAt(0), "Accuracy cannot be null");
        checkSetPrestoAccuracy(decodedAccuracy_.valueAt<double>(0));
      } else {
        rows.applyToSelected([&](auto row) {
          VELOX_USER_CHECK(
              !decodedAccuracy_.isNullAt(row), "Accuracy cannot be null");
          const auto accuracy = decodedAccuracy_.valueAt<double>(row);
          if (accuracy_ == kMissingNormalizedValue) {
            checkSetPrestoAccuracy(accuracy);
          }
          VELOX_USER_CHECK_EQ(
              accuracy,
              accuracy_,
              "Accuracy argument must be constant for all input rows");
        });
      }
    } else {
      if (decodedAccuracy_.isConstantMapping()) {
        VELOX_USER_CHECK(
            !decodedAccuracy_.isNullAt(0), "Accuracy cannot be null");
        setValidSparkAccuracy(decodedAccuracy_.valueAt<int32_t>(0));
      } else {
        rows.applyToSelected([&](auto row) {
          VELOX_USER_CHECK(
              !decodedAccuracy_.isNullAt(row), "Accuracy cannot be null");
          const auto currentAccuracy = decodedAccuracy_.valueAt<int32_t>(row);
          if (accuracy_ == static_cast<double>(kSparkDefaultAccuracy)) {
            setValidSparkAccuracy(currentAccuracy);
          }
          VELOX_USER_CHECK_EQ(
              currentAccuracy,
              static_cast<int32_t>(accuracy_),
              "Accuracy argument must be constant");
        });
      }
    }
  }

  static void checkWeight(int64_t weight) {
    constexpr int64_t kMaxWeight = (1ll << 60) - 1;
    VELOX_USER_CHECK(
        1 <= weight && weight <= kMaxWeight,
        "approx_percentile: weight must be in range [1, {}], got {}",
        kMaxWeight,
        weight);
  }

  template <typename VectorType, typename ExtractFunc>
  void extractHelper(
      char** groups,
      int32_t numGroups,
      VectorType* result,
      ExtractFunc extractFunction) {
    VELOX_CHECK(result);
    result->resize(numGroups);

    uint64_t* rawNulls = nullptr;
    if (result->mayHaveNulls()) {
      BufferPtr& nulls = result->mutableNulls(result->size());
      rawNulls = nulls->asMutable<uint64_t>();
    }

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = value<KllSketchAccumulator<T>>(group);
      if (accumulator->getSketch().totalCount() == 0) {
        result->setNull(i, true);
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }
        extractFunction(accumulator->getSketch(), result, i);
      }
    }
  }

 private:
  void checkSetPrestoAccuracy(double accuracy) {
    VELOX_USER_CHECK(
        0 < accuracy && accuracy <= 1, "Accuracy must be between 0 and 1");
    if (accuracy_ == kMissingNormalizedValue) {
      accuracy_ = accuracy;
    } else {
      VELOX_USER_CHECK_EQ(
          accuracy,
          accuracy_,
          "Accuracy argument must be constant for all input rows");
    }
  }

  void setValidSparkAccuracy(int32_t inputAccuracy) {
    VELOX_USER_CHECK(
        inputAccuracy > 0 &&
            inputAccuracy <= std::numeric_limits<int32_t>::max(),
        "Accuracy must be greater than 0 and less than or equal to "
        "Int32.MaxValue, got {}",
        inputAccuracy);
    accuracy_ = static_cast<double>(inputAccuracy);
  }

  template <bool kSingleGroup>
  void addIntermediate(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    if (validateIntermediateInputs_) {
      addIntermediateImpl<kSingleGroup, true>(group, rows, args);
    } else {
      addIntermediateImpl<kSingleGroup, false>(group, rows, args);
    }
  }

  template <bool kSingleGroup, bool checkIntermediateInputs>
  void addIntermediateImpl(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_CHECK_EQ(args.size(), 1);
    DecodedVector decoded(*args[0], rows);
    auto rowVec = decoded.base()->as<RowVector>();
    if constexpr (checkIntermediateInputs) {
      VELOX_USER_CHECK(rowVec);
      if constexpr (kHasWeight) {
        for (int i = kPercentiles; i <= kAccuracy; ++i) {
          VELOX_USER_CHECK(rowVec->childAt(i)->isConstantEncoding());
        }
      }
      for (int i = kK; i <= kMaxValue; ++i) {
        VELOX_USER_CHECK(rowVec->childAt(i)->isFlatEncoding());
      }
      for (int i = kItems; i <= kLevels; ++i) {
        VELOX_USER_CHECK(
            rowVec->childAt(i)->encoding() == VectorEncoding::Simple::ARRAY);
      }
    } else {
      VELOX_CHECK(rowVec);
    }

    const SelectivityVector* baseRows = &rows;
    SelectivityVector innerRows{rowVec->size(), false};
    if (!decoded.isIdentityMapping()) {
      if (decoded.isConstantMapping()) {
        innerRows.setValid(decoded.index(0), true);
        innerRows.updateBounds();
      } else {
        velox::translateToInnerRows(
            rows, decoded.indices(), decoded.nulls(&rows), innerRows);
      }
      baseRows = &innerRows;
    }

    DecodedVector percentiles(*rowVec->childAt(kPercentiles), *baseRows);
    auto percentileIsArray =
        rowVec->childAt(kPercentilesIsArray)->asUnchecked<SimpleVector<bool>>();
    auto accuracy =
        rowVec->childAt(kAccuracy)->asUnchecked<SimpleVector<double>>();
    auto k = rowVec->childAt(kK)->asUnchecked<SimpleVector<int32_t>>();
    auto n = rowVec->childAt(kN)->asUnchecked<SimpleVector<int64_t>>();
    auto minValue = rowVec->childAt(kMinValue)->asUnchecked<SimpleVector<T>>();
    auto maxValue = rowVec->childAt(kMaxValue)->asUnchecked<SimpleVector<T>>();
    auto items = rowVec->childAt(kItems)->asUnchecked<ArrayVector>();
    auto levels = rowVec->childAt(kLevels)->asUnchecked<ArrayVector>();

    auto itemsElements = items->elements()->asFlatVector<T>();
    auto levelElements = levels->elements()->asFlatVector<int32_t>();
    if constexpr (checkIntermediateInputs) {
      VELOX_USER_CHECK(itemsElements);
      VELOX_USER_CHECK(levelElements);
    } else {
      VELOX_CHECK(itemsElements);
      VELOX_CHECK(levelElements);
    }
    auto rawItems = itemsElements->rawValues();
    auto rawLevels = levelElements->rawValues<uint32_t>();

    KllSketchAccumulator<T>* accumulator = nullptr;
    std::vector<KllView<T>> views;
    if constexpr (kSingleGroup) {
      views.reserve(rows.end());
    }
    rows.applyToSelected([&](auto row) {
      if (decoded.isNullAt(row)) {
        return;
      }
      int i = decoded.index(row);
      if (percentileIsArray->isNullAt(i)) {
        return;
      }
      if (!accumulator) {
        if constexpr (kHasWeight) {
          int indexInBaseVector = percentiles.index(i);
          auto percentilesBase = percentiles.base()->asUnchecked<ArrayVector>();
          auto percentileBaseElements =
              percentilesBase->elements()->asFlatVector<double>();
          if constexpr (checkIntermediateInputs) {
            VELOX_USER_CHECK(percentileBaseElements);
            VELOX_USER_CHECK(!percentilesBase->isNullAt(indexInBaseVector));
          }

          bool isArray = percentileIsArray->valueAt(i);
          DecodedVector decodedElements(*percentilesBase->elements());
          checkSetPercentileValues(
              isArray,
              decodedElements,
              percentilesBase->offsetAt(indexInBaseVector),
              percentilesBase->sizeAt(indexInBaseVector));
        } else {
          auto pctArrayVec = percentiles.base()->asUnchecked<ArrayVector>();
          auto pctElementsVec =
              pctArrayVec->elements()->asUnchecked<FlatVector<double>>();
          int32_t pctOffset = pctArrayVec->offsetAt(percentiles.index(i));
          int32_t pctSize = pctArrayVec->sizeAt(percentiles.index(i));
          std::vector<double> pctVals(pctSize);
          for (int32_t j = 0; j < pctSize; ++j) {
            pctVals[j] = pctElementsVec->valueAt(pctOffset + j);
          }
          checkSetPercentileValuesFromVector(
              percentileIsArray->valueAt(i), pctVals);
        }

        if (!accuracy->isNullAt(i)) {
          if constexpr (kAccuracyIsErrorBound) {
            checkSetPrestoAccuracy(accuracy->valueAt(i));
          } else {
            setValidSparkAccuracy(static_cast<int32_t>(accuracy->valueAt(i)));
          }
        }
      }
      if constexpr (kSingleGroup) {
        if (!accumulator) {
          accumulator = initRawAccumulator(group);
        }
      } else {
        accumulator = initRawAccumulator(group[row]);
      }

      if constexpr (checkIntermediateInputs) {
        VELOX_USER_CHECK(
            !(k->isNullAt(i) || n->isNullAt(i) || minValue->isNullAt(i) ||
              maxValue->isNullAt(i) || items->isNullAt(i) ||
              levels->isNullAt(i)));
      }
      KllView<T> v{
          .k = static_cast<uint32_t>(k->valueAt(i)),
          .n = static_cast<size_t>(n->valueAt(i)),
          .minValue = minValue->valueAt(i),
          .maxValue = maxValue->valueAt(i),
          .items =
              {rawItems + items->offsetAt(i),
               static_cast<size_t>(items->sizeAt(i))},
          .levels =
              {rawLevels + levels->offsetAt(i),
               static_cast<size_t>(levels->sizeAt(i))},
      };
      if constexpr (kSingleGroup) {
        views.push_back(v);
      } else {
        auto tracker = trackRowSize(group[row]);
        accumulator->append(v);
      }
    });
    if constexpr (kSingleGroup) {
      if (!views.empty()) {
        auto tracker = trackRowSize(group);
        accumulator->append(views);
      }
    }
  }
};

} // namespace facebook::velox::aggregate
