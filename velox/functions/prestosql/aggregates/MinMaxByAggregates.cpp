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

#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/SingleValueAccumulator.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate {

namespace {

// NumericMinMaxByAggregate is the base class for min_by and max_by functions
// with numeric value and comparison types. These functions return the value of
// X associated with the minimum/maximum value of Y over all input values.
// Partial aggregation produces a pair of X and min/max Y. Final aggregation
// takes a pair of X and min/max Y and returns X. T is the type of X and U is
// the type of Y.
template <typename T, typename U>
class NumericMinMaxByAggregate : public exec::Aggregate {
 public:
  NumericMinMaxByAggregate(TypePtr resultType, U initialValue)
      : exec::Aggregate(resultType), initialValue_(initialValue) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(T) + sizeof(U) + sizeof(bool);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      comparisonValue(group) = initialValue_;
      valueIsNull(group) = true;
    }
  }

  void finalize(char** /* unused */, int32_t /* unused */) override {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<T>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    T* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group) || valueIsNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        rawValues[i] = this->value(group);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto valueVector = rowVector->childAt(0)->asFlatVector<T>();
    auto comparisonVector = rowVector->childAt(1)->asFlatVector<U>();

    rowVector->resize(numGroups);
    valueVector->resize(numGroups);
    comparisonVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    T* rawValues = valueVector->mutableRawValues();
    U* rawComparisonValues = comparisonVector->mutableRawValues();
    BufferPtr nulls = valueVector->mutableNulls(rowVector->size());
    uint64_t* nullValues = nulls->asMutable<uint64_t>();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        bits::setNull(nullValues, i, valueIsNull(group));
        if (!valueIsNull(group)) {
          rawValues[i] = value(group);
        }
        rawComparisonValues[i] = comparisonValue(group);
      }
    }
  }

 protected:
  template <typename MayUpdate>
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      auto value = decodedValue.valueAt<T>(0);
      auto comparisonValue = decodedComparisonValue.valueAt<U>(0);
      auto nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(groups[i], value, comparisonValue, nullValue, mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            groups[i],
            decodedValue.valueAt<T>(i),
            decodedComparisonValue.valueAt<U>(i),
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            decodedValue.valueAt<T>(i),
            decodedComparisonValue.valueAt<U>(i),
            false,
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPairs.base());
    auto baseValueVector = baseRowVector->childAt(0)->as<FlatVector<T>>();
    auto baseComparisonVector = baseRowVector->childAt(1)->as<FlatVector<U>>();

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      auto decodedIndex = decodedPairs.index(0);
      auto value = baseValueVector->valueAt(decodedIndex);
      auto comparisonValue = baseComparisonVector->valueAt(decodedIndex);
      auto nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(groups[i], value, comparisonValue, nullValue, mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            baseValueVector->valueAt(decodedIndex),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            baseValueVector->valueAt(decodedIndex),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      auto value = decodedValue.valueAt<T>(0);
      auto comparisonValue = decodedComparisonValue.valueAt<U>(0);
      auto nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(group, value, comparisonValue, nullValue, mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            group,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue.valueAt<U>(i),
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            group,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue.valueAt<U>(i),
            false,
            mayUpdate);
      });
    }
  }

  // Final aggregation will take (Value, comparisonValue) structs as inputs. It
  // will produce the Value associated with the maximum/minimum of
  // comparisonValue over all structs.
  template <typename MayUpdate>
  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // Decode struct(Value, ComparisonValue) as individual vectors.
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPairs.base());
    auto baseValueVector = baseRowVector->childAt(0)->as<FlatVector<T>>();
    auto baseComparisonVector = baseRowVector->childAt(1)->as<FlatVector<U>>();

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      auto decodedIndex = decodedPairs.index(0);
      auto value = baseValueVector->valueAt(decodedIndex);
      auto comparisonValue = baseComparisonVector->valueAt(decodedIndex);
      auto nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(group, value, comparisonValue, nullValue, mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            baseValueVector->valueAt(decodedIndex),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            baseValueVector->valueAt(decodedIndex),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

 private:
  template <typename MayUpdate>
  inline void updateValues(
      char* group,
      T newValue,
      U newComparisonValue,
      bool isValueNull,
      MayUpdate mayUpdate) {
    clearNull(group);
    if (mayUpdate(comparisonValue(group), newComparisonValue)) {
      valueIsNull(group) = isValueNull;
      if (LIKELY(!isValueNull)) {
        value(group) = newValue;
      }
      comparisonValue(group) = newComparisonValue;
    }
  }

  inline T& value(char* group) {
    return *reinterpret_cast<T*>(group + Aggregate::offset_);
  }

  inline U& comparisonValue(char* group) {
    return *reinterpret_cast<U*>(group + Aggregate::offset_ + sizeof(T));
  }

  inline bool& valueIsNull(char* group) {
    return *reinterpret_cast<bool*>(
        group + Aggregate::offset_ + sizeof(T) + sizeof(U));
  }

  // Initial value will take the minimum and maximum values of the numerical
  // limits.
  const U initialValue_;
};

template <typename T, typename U>
class NumericMaxByAggregate : public NumericMinMaxByAggregate<T, U> {
 public:
  explicit NumericMaxByAggregate(TypePtr resultType)
      : NumericMinMaxByAggregate<T, U>(
            resultType,
            std::numeric_limits<U>::min()) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NumericMinMaxByAggregate<T, U>::addRawInput(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NumericMinMaxByAggregate<T, U>::addIntermediateResults(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NumericMinMaxByAggregate<T, U>::addSingleGroupRawInput(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NumericMinMaxByAggregate<T, U>::addSingleGroupIntermediateResults(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }
};

template <typename T, typename U>
class NumericMinByAggregate : public NumericMinMaxByAggregate<T, U> {
 public:
  explicit NumericMinByAggregate(TypePtr resultType)
      : NumericMinMaxByAggregate<T, U>(
            resultType,
            std::numeric_limits<U>::max()) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NumericMinMaxByAggregate<T, U>::addRawInput(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NumericMinMaxByAggregate<T, U>::addIntermediateResults(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NumericMinMaxByAggregate<T, U>::addSingleGroupRawInput(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NumericMinMaxByAggregate<T, U>::addSingleGroupIntermediateResults(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }
};

// Similar to NumericMinMaxByAggregate but with a non-numeric value type.
template <typename U>
class MinMaxByAggregateWithNonNumericValue : public exec::Aggregate {
 public:
  MinMaxByAggregateWithNonNumericValue(TypePtr resultType, U initialValue)
      : exec::Aggregate(resultType), initialValue_(initialValue) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SingleValueAccumulator) + sizeof(U) + sizeof(bool);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      comparisonValue(group) = initialValue_;
      new (groups[i] + valueAccumulatorOffset()) SingleValueAccumulator();
      valueIsNull(group) = true;
    }
  }

  void finalize(char** /* unused */, int32_t /* unused */) override {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    (*result)->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(result->get());

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group) || valueIsNull(group)) {
        (*result)->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* accumulator = valueAccumulator(group);
        VELOX_DCHECK(accumulator->hasValue());
        accumulator->read(*result, i);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto valueVector = rowVector->childAt(0);
    auto comparisonVector = rowVector->childAt(1)->asFlatVector<U>();

    rowVector->resize(numGroups);
    valueVector->resize(numGroups);
    comparisonVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    U* rawComparisonValues = comparisonVector->mutableRawValues();
    BufferPtr nulls = valueVector->mutableNulls(rowVector->size());
    uint64_t* nullValues = nulls->asMutable<uint64_t>();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        bits::setNull(nullValues, i, valueIsNull(group));
        if (!valueIsNull(group)) {
          auto* accumulator = valueAccumulator(group);
          VELOX_DCHECK(accumulator->hasValue());
          accumulator->read(valueVector, i);
        }
        rawComparisonValues[i] = comparisonValue(group);
      }
    }
  }

  void destroy(folly::Range<char**> groups) override {
    for (auto group : groups) {
      valueAccumulator(group)->destroy(allocator_);
    }
  }

 protected:
  template <typename MayUpdate>
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const auto comparisonValue = decodedComparisonValue.valueAt<U>(0);
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            i,
            decodedValue.base(),
            comparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            groups[i],
            i,
            decodedValue.base(),
            decodedComparisonValue.valueAt<U>(i),
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            i,
            decodedValue.base(),
            decodedComparisonValue.valueAt<U>(i),
            false,
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->as<RowVector>();
    auto baseValueVector = baseRowVector->childAt(0);
    auto baseComparisonVector = baseRowVector->childAt(1)->as<FlatVector<U>>();

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      auto decodedIndex = decodedPairs.index(0);
      const auto comparisonValue = baseComparisonVector->valueAt(decodedIndex);
      const bool nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            decodedIndex,
            baseValueVector.get(),
            comparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            baseValueVector.get(),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            baseValueVector.get(),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const auto comparisonValue = decodedComparisonValue.valueAt<U>(0);
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group,
            0,
            decodedValue.base(),
            comparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            group,
            i,
            decodedValue.base(),
            decodedComparisonValue.valueAt<U>(i),
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            group,
            i,
            decodedValue.base(),
            decodedComparisonValue.valueAt<U>(i),
            false,
            mayUpdate);
      });
    }
  }

  // Final aggregation will take (Value, comparisonValue) structs as inputs. It
  // will produce the Value associated with the maximum/minimum of
  // comparisonValue over all structs.
  template <typename MayUpdate>
  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // Decode struct(Value, ComparisonValue) as individual vectors.
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->as<RowVector>();
    auto baseValueVector = baseRowVector->childAt(0);
    auto baseComparisonVector = baseRowVector->childAt(1)->as<FlatVector<U>>();

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      const auto decodedIndex = decodedPairs.index(0);
      const auto comparisonValue = baseComparisonVector->valueAt(decodedIndex);
      const bool nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group,
            decodedIndex,
            baseValueVector.get(),
            comparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            baseValueVector.get(),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            baseValueVector.get(),
            baseComparisonVector->valueAt(decodedIndex),
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

 private:
  template <typename MayUpdate>
  inline void updateValues(
      char* group,
      vector_size_t index,
      const BaseVector* valueVector,
      U newComparisonValue,
      bool isValueNull,
      MayUpdate mayUpdate) {
    clearNull(group);
    if (mayUpdate(comparisonValue(group), newComparisonValue)) {
      valueIsNull(group) = isValueNull;
      if (LIKELY(!isValueNull)) {
        auto* accumulator = valueAccumulator(group);
        accumulator->write(valueVector, index, allocator_);
      }
      comparisonValue(group) = newComparisonValue;
    }
  }

  inline int32_t valueAccumulatorOffset() const {
    return Aggregate::offset_;
  }

  inline SingleValueAccumulator* valueAccumulator(char* group) {
    return reinterpret_cast<SingleValueAccumulator*>(
        group + Aggregate::offset_);
  }

  inline U& comparisonValue(char* group) {
    return *reinterpret_cast<U*>(
        group + Aggregate::offset_ + sizeof(SingleValueAccumulator));
  }

  inline bool& valueIsNull(char* group) {
    return *reinterpret_cast<bool*>(
        group + Aggregate::offset_ + sizeof(SingleValueAccumulator) +
        sizeof(U));
  }

  // Initial value will take the minimum and maximum values of the numerical
  // limits.
  const U initialValue_;
};

template <typename U>
class MaxByAggregateWithNonNumericValue
    : public MinMaxByAggregateWithNonNumericValue<U> {
 public:
  explicit MaxByAggregateWithNonNumericValue(TypePtr resultType)
      : MinMaxByAggregateWithNonNumericValue<U>(
            resultType,
            std::numeric_limits<U>::min()) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addRawInput(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addIntermediateResults(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addSingleGroupRawInput(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addSingleGroupIntermediateResults(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue > currentValue;
        });
  }
};

template <typename U>
class MinByAggregateWithNonNumericValue
    : public MinMaxByAggregateWithNonNumericValue<U> {
 public:
  explicit MinByAggregateWithNonNumericValue(TypePtr resultType)
      : MinMaxByAggregateWithNonNumericValue<U>(
            resultType,
            std::numeric_limits<U>::max()) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addRawInput(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addIntermediateResults(
        groups, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addSingleGroupRawInput(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericValue<U>::addSingleGroupIntermediateResults(
        group, rows, args, [](U& currentValue, U newValue) {
          return newValue < currentValue;
        });
  }
};

// Similar to NumericMinMaxByAggregate but with a non-numeric comparison type.
template <typename T>
class MinMaxByAggregateWithNonNumericComparison : public exec::Aggregate {
 public:
  MinMaxByAggregateWithNonNumericComparison(
      TypePtr resultType,
      TypePtr comparisonType)
      : exec::Aggregate(resultType), comparisonType_(comparisonType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(T) + sizeof(SingleValueAccumulator) + sizeof(bool);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (groups[i] + comparisonAccumulatorOffset()) SingleValueAccumulator();
      valueIsNull(group) = true;
    }
  }

  void finalize(char** /* unused */, int32_t /* unused */) override {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<T>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    T* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group) || valueIsNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        rawValues[i] = this->value(group);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto valueVector = rowVector->childAt(0)->asFlatVector<T>();
    auto comparisonVector = rowVector->childAt(1);

    rowVector->resize(numGroups);
    valueVector->resize(numGroups);
    comparisonVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    T* rawValues = valueVector->mutableRawValues();
    BufferPtr nulls = valueVector->mutableNulls(rowVector->size());
    uint64_t* nullValues = nulls->asMutable<uint64_t>();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        bits::setNull(nullValues, i, valueIsNull(group));
        if (!valueIsNull(group)) {
          rawValues[i] = value(group);
        }
        auto* accumulator = comparisonAccumulator(group);
        VELOX_DCHECK(accumulator->hasValue());
        accumulator->read(comparisonVector, i);
      }
    }
  }

  void destroy(folly::Range<char**> groups) override {
    for (auto group : groups) {
      comparisonAccumulator(group)->destroy(allocator_);
    }
  }

 protected:
  template <typename MayUpdate>
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const auto value = decodedValue.valueAt<T>(0);
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i], i, value, decodedComparisonValue, nullValue, mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            groups[i],
            i,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue,
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            i,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue,
            false,
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->template as<RowVector>();
    auto baseValueVector = baseRowVector->childAt(0)->as<FlatVector<T>>();
    DecodedVector decodedComparisonVector(*baseRowVector->childAt(1), rows);

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      const auto decodedIndex = decodedPairs.index(0);
      const auto value = baseValueVector->valueAt(decodedIndex);
      const bool nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            decodedIndex,
            value,
            decodedComparisonVector,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            baseValueVector->valueAt(decodedIndex),
            decodedComparisonVector,
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            baseValueVector->valueAt(decodedIndex),
            decodedComparisonVector,
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const auto value = decodedValue.valueAt<T>(0);
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group, 0, value, decodedComparisonValue, nullValue, mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            group,
            i,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue,
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            group,
            i,
            decodedValue.valueAt<T>(i),
            decodedComparisonValue,
            false,
            mayUpdate);
      });
    }
  }

  // Final aggregation will take (Value, comparisonValue) structs as inputs. It
  // will produce the Value associated with the maximum/minimum of
  // comparisonValue over all structs.
  template <typename MayUpdate>
  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // Decode struct(Value, ComparisonValue) as individual vectors.
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->as<RowVector>();
    auto baseValueVector = baseRowVector->childAt(0)->as<FlatVector<T>>();
    DecodedVector decodedComparisonVector(*baseRowVector->childAt(1), rows);

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      const auto decodedIndex = decodedPairs.index(0);
      const auto value = baseValueVector->valueAt(decodedIndex);
      const bool nullValue = baseValueVector->isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group,
            decodedIndex,
            value,
            decodedComparisonVector,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            baseValueVector->valueAt(decodedIndex),
            decodedComparisonVector,
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            baseValueVector->valueAt(decodedIndex),
            decodedComparisonVector,
            baseValueVector->isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

 private:
  template <typename MayUpdate>
  inline void updateValues(
      char* group,
      vector_size_t index,
      T newValue,
      const DecodedVector& decodedComparisonVector,
      bool isValueNull,
      MayUpdate mayUpdate) {
    clearNull(group);
    auto* accumulator = comparisonAccumulator(group);
    if (accumulator->hasValue() &&
        !mayUpdate(accumulator->compare(decodedComparisonVector, index))) {
      return;
    }
    valueIsNull(group) = isValueNull;
    if (LIKELY((!isValueNull))) {
      value(group) = newValue;
    }
    accumulator->write(decodedComparisonVector.base(), index, allocator_);
  }

  inline T& value(char* group) {
    return *reinterpret_cast<T*>(group + Aggregate::offset_);
  }

  inline int32_t comparisonAccumulatorOffset() const {
    return Aggregate::offset_ + sizeof(T);
  }

  inline SingleValueAccumulator* comparisonAccumulator(char* group) {
    return reinterpret_cast<SingleValueAccumulator*>(
        group + comparisonAccumulatorOffset());
  }

  inline bool& valueIsNull(char* group) {
    return *reinterpret_cast<bool*>(
        group + Aggregate::offset_ + sizeof(T) +
        sizeof(SingleValueAccumulator));
  }

  const TypePtr comparisonType_;
};

template <typename T>
class MaxByAggregateWithNonNumericComparison
    : public MinMaxByAggregateWithNonNumericComparison<T> {
 public:
  explicit MaxByAggregateWithNonNumericComparison(
      TypePtr resultType,
      TypePtr comparisonType)
      : MinMaxByAggregateWithNonNumericComparison<T>(
            resultType,
            comparisonType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addRawInput(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addIntermediateResults(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addSingleGroupRawInput(
        group, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::
        addSingleGroupIntermediateResults(
            group, rows, args, [](int32_t compareResult) {
              return compareResult < 0;
            });
  }
};

template <typename T>
class MinByAggregateWithNonNumericComparison
    : public MinMaxByAggregateWithNonNumericComparison<T> {
 public:
  explicit MinByAggregateWithNonNumericComparison(
      TypePtr resultType,
      TypePtr comparisonType)
      : MinMaxByAggregateWithNonNumericComparison<T>(
            resultType,
            comparisonType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addRawInput(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addIntermediateResults(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::addSingleGroupRawInput(
        group, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    MinMaxByAggregateWithNonNumericComparison<T>::
        addSingleGroupIntermediateResults(
            group, rows, args, [](int32_t compareResult) {
              return compareResult > 0;
            });
  }
};

// Similar to NumericMinMaxByAggregate but with bot non-numeric value and
// comparison types.
class NonNumericMinMaxByAggregate : public exec::Aggregate {
 public:
  NonNumericMinMaxByAggregate(TypePtr resultType, TypePtr comparisonType)
      : exec::Aggregate(resultType), comparisonType_(comparisonType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SingleValueAccumulator) + sizeof(SingleValueAccumulator) +
        sizeof(bool);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (groups[i] + valueAccumulatorOffset()) SingleValueAccumulator();
      new (groups[i] + comparisonAccumulatorOffset()) SingleValueAccumulator();
      valueIsNull(group) = true;
    }
  }

  void finalize(char** /* unused */, int32_t /* unused */) override {}

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    (*result)->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(result->get());

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group) || valueIsNull(group)) {
        (*result)->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* accumulator = valueAccumulator(group);
        VELOX_DCHECK(accumulator->hasValue());
        accumulator->read(*result, i);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto valueVector = rowVector->childAt(0);
    auto comparisonVector = rowVector->childAt(1);

    rowVector->resize(numGroups);
    valueVector->resize(numGroups);
    comparisonVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    BufferPtr nulls = valueVector->mutableNulls(rowVector->size());
    uint64_t* nullValues = nulls->asMutable<uint64_t>();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        bits::setNull(nullValues, i, valueIsNull(group));
        if (!valueIsNull(group)) {
          auto* valueAcc = valueAccumulator(group);
          VELOX_DCHECK(valueAcc->hasValue());
          valueAcc->read(valueVector, i);
        }
        auto* comparisonAcc = comparisonAccumulator(group);
        VELOX_DCHECK(comparisonAcc->hasValue());
        comparisonAcc->read(comparisonVector, i);
      }
    }
  }

  void destroy(folly::Range<char**> groups) override {
    for (auto group : groups) {
      valueAccumulator(group)->destroy(allocator_);
      comparisonAccumulator(group)->destroy(allocator_);
    }
  }

 protected:
  template <typename MayUpdate>
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            i,
            decodedValue,
            decodedComparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            groups[i],
            i,
            decodedValue,
            decodedComparisonValue,
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            i,
            decodedValue,
            decodedComparisonValue,
            false,
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->template as<RowVector>();
    DecodedVector decodedValueVector(*baseRowVector->childAt(0), rows);
    DecodedVector decodedComparisonVector(*baseRowVector->childAt(1), rows);

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      const auto decodedIndex = decodedPairs.index(0);
      const bool nullValue = decodedValueVector.isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            groups[i],
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            decodedValueVector.isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPairs.index(i);
        updateValues(
            groups[i],
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            decodedValueVector.isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

  template <typename MayUpdate>
  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // decodedValue will contain the values of column X.
    // decodedComparisonValue will contain the values of column Y which will be
    // used to select the minimum or the maximum.
    DecodedVector decodedValue(*args[0], rows);
    DecodedVector decodedComparisonValue(*args[1], rows);

    if (decodedValue.isConstantMapping() &&
        decodedComparisonValue.isConstantMapping()) {
      if (decodedComparisonValue.isNullAt(0)) {
        return;
      }
      const bool nullValue = decodedValue.isNullAt(0);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group,
            0,
            decodedValue,
            decodedComparisonValue,
            nullValue,
            mayUpdate);
      });
    } else if (
        decodedValue.mayHaveNulls() || decodedComparisonValue.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedComparisonValue.isNullAt(i)) {
          return;
        }
        updateValues(
            group,
            i,
            decodedValue,
            decodedComparisonValue,
            decodedValue.isNullAt(i),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateValues(
            group, i, decodedValue, decodedComparisonValue, false, mayUpdate);
      });
    }
  }

  // Final aggregation will take (Value, comparisonValue) structs as inputs. It
  // will produce the Value associated with the maximum/minimum of
  // comparisonValue over all structs.
  template <typename MayUpdate>
  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      MayUpdate mayUpdate) {
    // Decode struct(Value, ComparisonValue) as individual vectors.
    DecodedVector decodedPairs(*args[0], rows);
    auto baseRowVector = decodedPairs.base()->as<RowVector>();
    DecodedVector decodedValueVector(*baseRowVector->childAt(0), rows);
    DecodedVector decodedComparisonVector(*baseRowVector->childAt(1), rows);

    if (decodedPairs.isConstantMapping()) {
      if (decodedPairs.isNullAt(0)) {
        return;
      }
      const auto decodedIndex = decodedPairs.index(0);
      const bool nullValue = decodedValueVector.isNullAt(decodedIndex);
      rows.applyToSelected([&](vector_size_t /*i*/) {
        updateValues(
            group,
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            nullValue,
            mayUpdate);
      });
    } else if (decodedPairs.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPairs.isNullAt(i)) {
          return;
        }
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            decodedValueVector.isNullAt(decodedIndex),
            mayUpdate);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        const auto decodedIndex = decodedPairs.index(i);
        updateValues(
            group,
            decodedIndex,
            decodedValueVector,
            decodedComparisonVector,
            decodedValueVector.isNullAt(decodedIndex),
            mayUpdate);
      });
    }
  }

 private:
  template <typename MayUpdate>
  inline void updateValues(
      char* group,
      vector_size_t index,
      const DecodedVector& decodedValueVector,
      const DecodedVector& decodedComparisonVector,
      bool isValueNull,
      MayUpdate mayUpdate) {
    clearNull(group);
    auto* comparisonAcc = comparisonAccumulator(group);
    if (comparisonAcc->hasValue() &&
        !mayUpdate(comparisonAcc->compare(decodedComparisonVector, index))) {
      return;
    }
    valueIsNull(group) = isValueNull;
    if (LIKELY((!isValueNull))) {
      auto* valueAcc = valueAccumulator(group);
      valueAcc->write(decodedValueVector.base(), index, allocator_);
    }
    comparisonAcc->write(decodedComparisonVector.base(), index, allocator_);
  }

  inline int32_t valueAccumulatorOffset() const {
    return Aggregate::offset_;
  }
  inline SingleValueAccumulator* valueAccumulator(char* group) {
    auto* object = reinterpret_cast<SingleValueAccumulator*>(
        group + valueAccumulatorOffset());
    SingleValueAccumulator& objectRef = *object;
    return reinterpret_cast<SingleValueAccumulator*>(
        group + valueAccumulatorOffset());
  }
  inline int32_t comparisonAccumulatorOffset() const {
    return Aggregate::offset_ + sizeof(SingleValueAccumulator);
  }
  inline SingleValueAccumulator* comparisonAccumulator(char* group) {
    return reinterpret_cast<SingleValueAccumulator*>(
        group + comparisonAccumulatorOffset());
  }
  inline bool& valueIsNull(char* group) {
    return *reinterpret_cast<bool*>(
        group + Aggregate::offset_ + sizeof(SingleValueAccumulator) +
        sizeof(SingleValueAccumulator));
  }

  const TypePtr comparisonType_;
};

class NonNumericMaxByAggregate : public NonNumericMinMaxByAggregate {
 public:
  explicit NonNumericMaxByAggregate(TypePtr resultType, TypePtr comparisonType)
      : NonNumericMinMaxByAggregate(resultType, comparisonType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NonNumericMinMaxByAggregate::addRawInput(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NonNumericMinMaxByAggregate::addIntermediateResults(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NonNumericMinMaxByAggregate::addSingleGroupRawInput(
        group, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NonNumericMinMaxByAggregate::addSingleGroupIntermediateResults(
        group, rows, args, [](int32_t compareResult) {
          return compareResult < 0;
        });
  }
};

class NonNumericMinByAggregate : public NonNumericMinMaxByAggregate {
 public:
  explicit NonNumericMinByAggregate(TypePtr resultType, TypePtr comparisonType)
      : NonNumericMinMaxByAggregate(resultType, comparisonType) {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NonNumericMinMaxByAggregate::addRawInput(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NonNumericMinMaxByAggregate::addIntermediateResults(
        groups, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    NonNumericMinMaxByAggregate::addSingleGroupRawInput(
        group, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    NonNumericMinMaxByAggregate::addSingleGroupIntermediateResults(
        group, rows, args, [](int32_t compareResult) {
          return compareResult > 0;
        });
  }
};

template <
    template <typename U, typename V>
    class NumericAggregate,
    template <typename V>
    class AggregateWithNonNumericComparison,
    typename W>
std::unique_ptr<exec::Aggregate> createWithNumericValue(
    TypePtr resultType,
    TypePtr compareType,
    const std::string& errorMessage) {
  switch (compareType->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<NumericAggregate<W, int8_t>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<NumericAggregate<W, int16_t>>(resultType);
    case TypeKind::INTEGER:
      return std::make_unique<NumericAggregate<W, int32_t>>(resultType);
    case TypeKind::BIGINT:
      return std::make_unique<NumericAggregate<W, int64_t>>(resultType);
    case TypeKind::REAL:
      return std::make_unique<NumericAggregate<W, float>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<NumericAggregate<W, double>>(resultType);
    case TypeKind::VARCHAR:
      return std::make_unique<AggregateWithNonNumericComparison<W>>(
          resultType, compareType);
    default:
      VELOX_FAIL("{}", errorMessage);
      return nullptr;
  }
}

template <
    template <typename U>
    class AggregateWithNonNumericValue,
    class NonNumericAggregate>
std::unique_ptr<exec::Aggregate> createWithNonNumericValue(
    TypePtr resultType,
    TypePtr compareType,
    const std::string& errorMessage) {
  switch (compareType->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<AggregateWithNonNumericValue<int8_t>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<AggregateWithNonNumericValue<int16_t>>(
          resultType);
    case TypeKind::INTEGER:
      return std::make_unique<AggregateWithNonNumericValue<int32_t>>(
          resultType);
    case TypeKind::BIGINT:
      return std::make_unique<AggregateWithNonNumericValue<int64_t>>(
          resultType);
    case TypeKind::REAL:
      return std::make_unique<AggregateWithNonNumericValue<float>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<AggregateWithNonNumericValue<double>>(resultType);
    case TypeKind::VARCHAR:
      return std::make_unique<NonNumericAggregate>(resultType, compareType);
    default:
      VELOX_FAIL("{}", errorMessage);
      return nullptr;
  }
}

template <
    template <typename U, typename V>
    class NumericAggregate,
    template <typename U>
    class AggregateWithNonNumericValue,
    template <typename V>
    class AggregateWithNonNumericComparison,
    class NonNumericAggregate>
bool registerMinMaxByAggregate(const std::string& name) {
  // TODO(spershin): Need to add support for varchar and other types of
  // variable length. For both arguments. See MinMaxAggregates for
  // reference.
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  for (const auto& valueType :
       {"tinyint",
        "smallint",
        "integer",
        "bigint",
        "real",
        "double",
        "varchar"}) {
    for (const auto& compareType :
         {"tinyint",
          "smallint",
          "integer",
          "bigint",
          "real",
          "double",
          "varchar"}) {
      signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                               .returnType(valueType)
                               .intermediateType(fmt::format(
                                   "row({},{})", valueType, compareType))
                               .argumentType(valueType)
                               .argumentType(compareType)
                               .build());
    }
  }

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        auto isRawInput = exec::isRawInput(step);
        if (isRawInput) {
          VELOX_CHECK_EQ(
              argTypes.size(),
              2,
              "{} partial aggregation takes 2 arguments",
              name);
        } else {
          VELOX_CHECK_EQ(
              argTypes.size(),
              1,
              "{} final aggregation takes one argument",
              name);
          VELOX_CHECK_EQ(
              argTypes[0]->kind(),
              TypeKind::ROW,
              "{} final aggregation takes ROW({NUMERIC,NUMERIC}) structs as input",
              name);
        }

        auto valueType = isRawInput ? argTypes[0] : argTypes[0]->childAt(0);
        auto compareType = isRawInput ? argTypes[1] : argTypes[0]->childAt(1);
        std::string errorMessage = fmt::format(
            "Unknown input types for {} ({}) aggregation: {}, {}",
            name,
            mapAggregationStepToName(step),
            valueType->kindName(),
            compareType->kindName());

        switch (valueType->kind()) {
          case TypeKind::TINYINT:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                int8_t>(resultType, compareType, errorMessage);
          case TypeKind::SMALLINT:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                int16_t>(resultType, compareType, errorMessage);
          case TypeKind::INTEGER:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                int32_t>(resultType, compareType, errorMessage);
          case TypeKind::BIGINT:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                int64_t>(resultType, compareType, errorMessage);
          case TypeKind::REAL:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                float>(resultType, compareType, errorMessage);
          case TypeKind::DOUBLE:
            return createWithNumericValue<
                NumericAggregate,
                AggregateWithNonNumericComparison,
                double>(resultType, compareType, errorMessage);
          case TypeKind::VARCHAR:
            return createWithNonNumericValue<
                AggregateWithNonNumericValue,
                NonNumericAggregate>(resultType, compareType, errorMessage);
          default:
            VELOX_FAIL(errorMessage);
        }
      });

  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_AggregateFunction) =
    registerMinMaxByAggregate<
        NumericMaxByAggregate,
        MaxByAggregateWithNonNumericValue,
        MaxByAggregateWithNonNumericComparison,
        NonNumericMaxByAggregate>(kMaxBy);
static bool FB_ANONYMOUS_VARIABLE(g_AggregateFunction) =
    registerMinMaxByAggregate<
        NumericMinByAggregate,
        MinByAggregateWithNonNumericValue,
        MinByAggregateWithNonNumericComparison,
        NonNumericMinByAggregate>(kMinBy);

} // namespace
} // namespace facebook::velox::aggregate
