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
#include "velox/functions/prestosql/aggregates/PrestoHasher.h"
#include "velox/functions/prestosql/aggregates/SimpleNumericAggregate.h"

namespace facebook::velox::aggregate {

namespace {

/// ChecksumAggregate is the aggregate function to compute checksum aggregate on
/// a vector.
/// Checksum will return an order-insensitive checksum of the input vector.
///
/// checksum(T)-> varbinary
class ChecksumAggregate : public exec::Aggregate {
 public:
  static const long kPrime64 = 0x9E3779B185EBCA87L;

  explicit ChecksumAggregate(const TypePtr& resultType)
      : Aggregate(resultType) {}

  void finalize(char** /* groups */, int32_t /* numGroups */) override {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(int64_t);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto i : indices) {
      *value<int64_t>(groups[i]) = 0;
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* vector = (*result)->as<FlatVector<StringView>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);

    auto* rawValues = vector->mutableRawValues();
    for (vector_size_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      clearNull(group);
      rawValues[i] = StringView(value<char>(group), sizeof(int64_t));
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK_EQ((*result)->encoding(), VectorEncoding::Simple::FLAT);
    auto vector = (*result)->as<FlatVector<int64_t>>();
    VELOX_CHECK(
        vector,
        "Unexpected type of the result vector: {}",
        (*result)->type()->toString());
    vector->resize(numGroups);

    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        vector->set(i, *value<int64_t>(group));
      }
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /**/) override {
    const auto& arg = args[0];
    auto hasher = getPrestoHasher(arg->type());
    BufferPtr hashes =
        AlignedBuffer::allocate<int64_t>(rows.end(), arg->pool());
    hasher->hash(arg, rows, hashes);
    auto rawHashes = hashes->as<int64_t>();

    rows.template applyToSelected([&](vector_size_t row) {
      if (arg->isNullAt(row)) {
        computeHashForNull(groups[row]);
      } else {
        computeHash(groups[row], rawHashes, row);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /**/) override {
    VELOX_CHECK_EQ(args[0]->encoding(), VectorEncoding::Simple::FLAT);

    auto vector = args[0]->asUnchecked<FlatVector<int64_t>>();
    auto rawValues = vector->rawValues();

    rows.applyToSelected(
        [&](vector_size_t i) { *value<int64_t>(groups[i]) += rawValues[i]; });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /**/) override {
    const auto& arg = args[0];
    auto hasher = getPrestoHasher(arg->type());
    BufferPtr hashes =
        AlignedBuffer::allocate<int64_t>(rows.end(), arg->pool());
    hasher->hash(arg, rows, hashes);
    auto rawHashes = hashes->as<int64_t>();

    rows.template applyToSelected([&](vector_size_t row) {
      if (arg->isNullAt(row)) {
        computeHashForNull(group);
      } else {
        computeHash(group, rawHashes, row);
      }
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /**/) override {
    VELOX_CHECK_EQ(args[0]->encoding(), VectorEncoding::Simple::FLAT);
    clearNull(group);

    auto vector = args[0]->asUnchecked<FlatVector<int64_t>>();
    auto rawValues = vector->rawValues();
    int64_t result = 0;
    if (vector->mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!vector->isNullAt(i)) {
          result += rawValues[i];
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) { result += rawValues[i]; });
    }

    *value<int64_t>(group) += result;
  }

 private:
  FOLLY_ALWAYS_INLINE void
  computeHash(char* group, const int64_t* hashes, const vector_size_t& row) {
    clearNull(group);

    auto hashVal = hashes[row];
    auto result = *value<int64_t>(group);
    result += hashVal * kPrime64;
    *value<int64_t>(group) = result;
  }

  FOLLY_ALWAYS_INLINE void computeHashForNull(char* group) {
    clearNull(group);
    *value<int64_t>(group) += kPrime64;
  }

  FOLLY_ALWAYS_INLINE PrestoHasher* getPrestoHasher(TypePtr typePtr) {
    if (prestoHasher_ == nullptr) {
      prestoHasher_ = std::make_unique<PrestoHasher>(typePtr);
    }
    return prestoHasher_.get();
  }

  FOLLY_ALWAYS_INLINE const int64_t* getHashValue(
      const StringView& stringView) {
    return reinterpret_cast<const int64_t*>(stringView.data());
  }

  std::unique_ptr<PrestoHasher> prestoHasher_;
};

bool registerChecksumAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("varbinary")
          .intermediateType("bigint")
          .argumentType("T")
          .build(),
  };

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [&name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& /*resultType*/) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes one argument", name);

        if (step == core::AggregationNode::Step::kFinal or
            step == core::AggregationNode::Step::kSingle) {
          return std::make_unique<ChecksumAggregate>(VARBINARY());
        }

        return std::make_unique<ChecksumAggregate>(BIGINT());
      });

  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_checksumAggregateFunction) =
    registerChecksumAggregate(kChecksum);
} // namespace
} // namespace facebook::velox::aggregate