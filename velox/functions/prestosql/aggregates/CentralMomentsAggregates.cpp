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
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {
// Indices into RowType representing intermediate results of skewness and
// kurtosis. Columns appear in alphabetical order.
struct CentralMomentsIndices {
  int32_t count;
  int32_t m1;
  int32_t m2;
  int32_t m3;
  int32_t m4;
};
constexpr CentralMomentsIndices kCentralMomentsIndices{0, 1, 2, 3, 4};

struct CentralMomentsAccumulator {
  double count() const {
    return count_;
  }

  double m1() const {
    return m1_;
  }

  double m2() const {
    return m2_;
  }

  double m3() const {
    return m3_;
  }

  double m4() const {
    return m4_;
  }

  void update(double value) {
    double oldCount = count();
    count_ += 1;
    double oldM1 = m1();
    double oldM2 = m2();
    double oldM3 = m3();
    double delta = value - oldM1;
    double deltaN = delta / count();
    double deltaN2 = deltaN * deltaN;
    double dm2 = delta * deltaN * oldCount;

    m1_ += deltaN;
    m2_ += dm2;
    m3_ += dm2 * deltaN * (count() - 2) - 3 * deltaN * oldM2;
    m4_ += dm2 * deltaN2 * (count() * (double)count() - 3 * count() + 3) +
        6 * deltaN2 * oldM2 - 4 * deltaN * oldM3;
  }

  inline void merge(const CentralMomentsAccumulator& other) {
    merge(other.count(), other.m1(), other.m2(), other.m3(), other.m4());
  }

  void merge(
      double otherCount,
      double otherM1,
      double otherM2,
      double otherM3,
      double otherM4) {
    if (otherCount == 0) {
      return;
    }

    double oldCount = count();
    count_ += otherCount;

    double oldM1 = m1();
    double oldM2 = m2();
    double oldM3 = m3();
    double delta = otherM1 - oldM1;
    double delta2 = delta * delta;
    double delta3 = delta * delta2;
    double delta4 = delta2 * delta2;

    m1_ = (oldCount * oldM1 + otherCount * otherM1) / count();
    m2_ += otherM2 + delta2 * oldCount * otherCount / count();
    m3_ += otherM3 +
        delta3 * oldCount * otherCount * (oldCount - otherCount) /
            (count() * count()) +
        3 * delta * (oldCount * otherM2 - otherCount * oldM2) / count();
    m4_ += otherM4 +
        delta4 * oldCount * otherCount *
            (oldCount * oldCount - oldCount * otherCount +
             otherCount * otherCount) /
            (count() * count() * count()) +
        6 * delta2 *
            (oldCount * oldCount * otherM2 + otherCount * otherCount * oldM2) /
            (count() * count()) +
        4 * delta * (oldCount * otherM3 - otherCount * oldM3) / count();
  }

 private:
  int64_t count_{0};
  double m1_{0};
  double m2_{0};
  double m3_{0};
  double m4_{0};
};

struct SkewnessResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    return accumulator.count() >= 3;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    return std::sqrt(accumulator.count()) * accumulator.m3() /
        std::pow(accumulator.m2(), 1.5);
  }
};

struct KurtosisResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    return accumulator.count() >= 4;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    double count = accumulator.count();
    double m2 = accumulator.m2();
    double m4 = accumulator.m4();
    return ((count - 1) * count * (count + 1)) / ((count - 2) * (count - 3)) *
        m4 / (m2 * m2) -
        3 * ((count - 1) * (count - 1)) / ((count - 2) * (count - 3));
  }
};

template <typename T>
SimpleVector<T>* asSimpleVector(
    const RowVector* rowVector,
    int32_t childIndex) {
  auto result = rowVector->childAt(childIndex)->as<SimpleVector<T>>();
  VELOX_CHECK_NOT_NULL(result);
  return result;
}

class CentralMomentsIntermediateInput {
 public:
  explicit CentralMomentsIntermediateInput(
      const RowVector* rowVector,
      const CentralMomentsIndices& indices = kCentralMomentsIndices)
      : count_{asSimpleVector<int64_t>(rowVector, indices.count)},
        m1_{asSimpleVector<double>(rowVector, indices.m1)},
        m2_{asSimpleVector<double>(rowVector, indices.m2)},
        m3_{asSimpleVector<double>(rowVector, indices.m3)},
        m4_{asSimpleVector<double>(rowVector, indices.m4)} {}

  void mergeInto(CentralMomentsAccumulator& accumulator, vector_size_t row) {
    accumulator.merge(
        count_->valueAt(row),
        m1_->valueAt(row),
        m2_->valueAt(row),
        m3_->valueAt(row),
        m4_->valueAt(row));
  }

 protected:
  SimpleVector<int64_t>* count_;
  SimpleVector<double>* m1_;
  SimpleVector<double>* m2_;
  SimpleVector<double>* m3_;
  SimpleVector<double>* m4_;
};

template <typename T>
T* mutableRawValues(const RowVector* rowVector, int32_t childIndex) {
  return rowVector->childAt(childIndex)
      ->as<FlatVector<T>>()
      ->mutableRawValues();
}

class CentralMomentsIntermediateResult {
 public:
  explicit CentralMomentsIntermediateResult(
      const RowVector* rowVector,
      const CentralMomentsIndices& indices = kCentralMomentsIndices)
      : count_{mutableRawValues<int64_t>(rowVector, indices.count)},
        m1_{mutableRawValues<double>(rowVector, indices.m1)},
        m2_{mutableRawValues<double>(rowVector, indices.m2)},
        m3_{mutableRawValues<double>(rowVector, indices.m3)},
        m4_{mutableRawValues<double>(rowVector, indices.m4)} {}

  static std::string type() {
    return "row(bigint,double,double,double,double)";
  }

  void set(vector_size_t row, const CentralMomentsAccumulator& accumulator) {
    count_[row] = accumulator.count();
    m1_[row] = accumulator.m1();
    m2_[row] = accumulator.m2();
    m3_[row] = accumulator.m3();
    m4_[row] = accumulator.m4();
  }

 private:
  int64_t* count_;
  double* m1_;
  double* m2_;
  double* m3_;
  double* m4_;
};

// T is the input type for partial aggregation, it can be integer, double or
// float. Not used for final aggregation. TResultAccessor is the type of the
// static struct that will access the result in a certain way from the
// CentralMoments Accumulator.
template <typename T, typename TResultAccessor>
class CentralMomentsAggregate : public exec::Aggregate {
 public:
  explicit CentralMomentsAggregate(TypePtr resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorAlignmentSize() const override {
    return alignof(CentralMomentsAccumulator);
  }

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(CentralMomentsAccumulator);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) CentralMomentsAccumulator();
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<T>(0);
        rows.applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(groups[i], value); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedRaw_.isNullAt(i)) {
          return;
        }
        updateNonNullValue(groups[i], decodedRaw_.valueAt<T>(i));
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<T>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], data[i]);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(groups[i], decodedRaw_.valueAt<T>(i));
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);

    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<T>(0);
        rows.applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(group, value); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(group, decodedRaw_.valueAt<T>(i));
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<T>();
      CentralMomentsAccumulator accData;
      rows.applyToSelected([&](vector_size_t i) { accData.update(data[i]); });
      updateNonNullValue<false>(group, accData);
    } else {
      CentralMomentsAccumulator accData;
      rows.applyToSelected(
          [&](vector_size_t i) { accData.update(decodedRaw_.valueAt<T>(i)); });
      updateNonNullValue(group, accData);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);

    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    CentralMomentsIntermediateInput input{baseRowVector};

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        rows.applyToSelected([&](vector_size_t i) {
          exec::Aggregate::clearNull(groups[i]);
          input.mergeInto(*accumulator(groups[i]), decodedIndex);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        exec::Aggregate::clearNull(groups[i]);
        input.mergeInto(*accumulator(groups[i]), decodedIndex);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        exec::Aggregate::clearNull(groups[i]);
        input.mergeInto(*accumulator(groups[i]), decodedIndex);
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    CentralMomentsIntermediateInput input{baseRowVector};

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        CentralMomentsAccumulator accData;
        rows.applyToSelected(
            [&](vector_size_t i) { input.mergeInto(accData, decodedIndex); });
        updateNonNullValue(group, accData);
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        exec::Aggregate::clearNull(group);
        input.mergeInto(*accumulator(group), decodedIndex);
      });
    } else {
      CentralMomentsAccumulator accData;
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        input.mergeInto(accData, decodedIndex);
      });
      updateNonNullValue(group, accData);
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<double>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    double* rawValues = vector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        auto* accData = accumulator(group);
        if (TResultAccessor::hasResult(*accData)) {
          clearNull(rawNulls, i);
          rawValues[i] = TResultAccessor::result(*accData);
        } else {
          vector->setNull(i, true);
        }
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    rowVector->resize(numGroups);
    for (auto& child : rowVector->children()) {
      child->resize(numGroups);
    }

    uint64_t* rawNulls = getRawNulls(rowVector);

    CentralMomentsIntermediateResult centralMomentsResult{rowVector};

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        centralMomentsResult.set(i, *accumulator(group));
      }
    }
  }

 private:
  template <bool tableHasNulls = true>
  inline void updateNonNullValue(char* group, T value) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    CentralMomentsAccumulator* accData = accumulator(group);
    accData->update((double)value);
  }

  template <bool tableHasNulls = true>
  inline void updateNonNullValue(
      char* group,
      const CentralMomentsAccumulator& accData) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    CentralMomentsAccumulator* thisAccData = accumulator(group);
    thisAccData->merge(accData);
  }

  inline CentralMomentsAccumulator* accumulator(char* group) {
    return exec::Aggregate::value<CentralMomentsAccumulator>(group);
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
};

void checkAccumulatorRowType(
    const TypePtr& type,
    const std::string& errorMessage) {
  VELOX_CHECK_EQ(type->kind(), TypeKind::ROW, "{}", errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCentralMomentsIndices.count)->kind(),
      TypeKind::BIGINT,
      "{}",
      errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCentralMomentsIndices.m1)->kind(),
      TypeKind::DOUBLE,
      "{}",
      errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCentralMomentsIndices.m2)->kind(),
      TypeKind::DOUBLE,
      "{}",
      errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCentralMomentsIndices.m3)->kind(),
      TypeKind::DOUBLE,
      "{}",
      errorMessage);
  VELOX_CHECK_EQ(
      type->childAt(kCentralMomentsIndices.m4)->kind(),
      TypeKind::DOUBLE,
      "{}",
      errorMessage);
}

template <typename TResultAccessor>
exec::AggregateRegistrationResult registerCentralMoments(
    const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  std::vector<std::string> inputTypes = {
      "smallint", "integer", "bigint", "real", "double"};
  for (const auto& inputType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType("double")
            .intermediateType(CentralMomentsIntermediateResult::type())
            .argumentType(inputType)
            .build());
  }

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (exec::isRawInput(step)) {
          switch (inputType->kind()) {
            case TypeKind::SMALLINT:
              return std::make_unique<
                  CentralMomentsAggregate<int16_t, TResultAccessor>>(
                  resultType);
            case TypeKind::INTEGER:
              return std::make_unique<
                  CentralMomentsAggregate<int32_t, TResultAccessor>>(
                  resultType);
            case TypeKind::BIGINT:
              return std::make_unique<
                  CentralMomentsAggregate<int64_t, TResultAccessor>>(
                  resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<
                  CentralMomentsAggregate<double, TResultAccessor>>(resultType);
            case TypeKind::REAL:
              return std::make_unique<
                  CentralMomentsAggregate<float, TResultAccessor>>(resultType);
            default:
              VELOX_UNSUPPORTED(
                  "Unsupported input type: {}. "
                  "Expected SMALLINT, INTEGER, BIGINT, DOUBLE or REAL.",
                  inputType->toString())
          }
        } else {
          checkAccumulatorRowType(
              inputType,
              "Input type for final aggregation must be "
              "(count:bigint, m1:double, m2:double, m3:double, m4:double) struct");
          // final agg not use template T, int64_t here has no effect.
          return std::make_unique<
              CentralMomentsAggregate<int64_t, TResultAccessor>>(resultType);
        }
      });
}

} // namespace

void registerCentralMomentsAggregates(const std::string& prefix) {
  registerCentralMoments<KurtosisResultAccessor>(prefix + kKurtosis);
  registerCentralMoments<SkewnessResultAccessor>(prefix + kSkewness);
}

} // namespace facebook::velox::aggregate::prestosql
