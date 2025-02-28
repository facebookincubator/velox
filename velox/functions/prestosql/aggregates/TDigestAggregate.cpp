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
// #include <functional>
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/TDigest.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate::prestosql {

namespace {

struct TDigestAccumulator {
  explicit TDigestAccumulator(HashStringAllocator* allocator)
      : digest_(StlAllocator<double>(allocator)) {}
  double compression_ = 0.0;
  facebook::velox::functions::TDigest<StlAllocator<double>> digest_;

  template <typename Func, typename... Args>
  void executeWithCatch(Func&& func, Args&&... args) {
    try {
      std::invoke(
          std::forward<Func>(func), digest_, std::forward<Args>(args)...);
    } catch (const std::exception& e) {
      VELOX_USER_FAIL("TDigest operation failed: {}", e.what());
    }
  }

  void mergeDeserialized(std::vector<int16_t>& positions, const char* input) {
    executeWithCatch(
        &facebook::velox::functions::TDigest<
            StlAllocator<double>>::mergeDeserialized,
        positions,
        input);
  }

  void add(std::vector<int16_t>& positions, double value, int64_t weight = 1) {
    executeWithCatch(
        &facebook::velox::functions::TDigest<StlAllocator<double>>::add,
        positions,
        value,
        weight);
  }

  void compress(std::vector<int16_t>& positions) {
    executeWithCatch(
        &facebook::velox::functions::TDigest<StlAllocator<double>>::compress,
        positions);
  }

  void serialize(char* out) const {
    try {
      digest_.serialize(out);
    } catch (const std::exception& e) {
      VELOX_USER_FAIL("TDigest serialization failed: {}", e.what());
    }
  }

  int64_t serializedByteSize() const {
    try {
      return digest_.serializedByteSize();
    } catch (const std::exception& e) {
      VELOX_USER_FAIL("TDigest serializedByteSize failed: {}", e.what());
    }
  }

  void setCompression(double compression) {
    try {
      digest_.setCompression(compression);
    } catch (const std::exception& e) {
      VELOX_USER_FAIL("TDigest compression failed: {}", e.what());
    }
  }
};

template <typename T>
class TDigestAggregate : public exec::Aggregate {
 private:
  bool hasWeight_;
  bool hasCompression_;
  double compression_ = 0;
  DecodedVector decodedValue_;
  DecodedVector decodedWeight_;
  DecodedVector decodedCompression_;

 public:
  TDigestAggregate(
      bool hasWeight,
      bool hasCompression,
      const TypePtr& resultType)
      : exec::Aggregate(resultType),
        hasWeight_{hasWeight},
        hasCompression_{hasCompression} {}

  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    size_t argIndex = 0;
    decodedValue_.decode(*args[argIndex++], rows, true);
    if (hasWeight_) {
      decodedWeight_.decode(*args[argIndex++], rows, true);
    }
    if (hasCompression_) {
      decodedCompression_.decode(*args[argIndex++], rows, true);
    }
    VELOX_CHECK_EQ(argIndex, args.size());
  }

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(TDigestAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(TDigestAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractCommon(groups, numGroups, result);
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractCommon(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      auto accumulator = value<TDigestAccumulator>(groups[row]);
      if (hasCompression_) {
        checkAndSetCompression(accumulator, row);
      }
      double value = decodedValue_.valueAt<double>(row);
      if (hasWeight_) {
        int64_t weight = decodedWeight_.valueAt<int64_t>(row);
        VELOX_USER_CHECK_GT(weight, 0, "Weight must be > 0.");
        accumulator->add(positions, value, weight);
      } else {
        accumulator->add(positions, value);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    DecodedVector decodedSerializedDigests;
    decodedSerializedDigests.decode(*args[0], rows);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      // Skip null serialized digests
      if (decodedSerializedDigests.isNullAt(row)) {
        return;
      }
      auto group = groups[row];
      auto accumulator = value<TDigestAccumulator>(group);
      auto serialized =
          decodedSerializedDigests.valueAt<facebook::velox::StringView>(row);
      accumulator->mergeDeserialized(positions, serialized.data());
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    auto accumulator = value<TDigestAccumulator>(group);
    std::vector<int16_t> positions;
    rows.applyToSelected([&](vector_size_t row) {
      if (hasCompression_) {
        checkAndSetCompression(accumulator, row);
      }
      double value = decodedValue_.valueAt<double>(row);
      if (hasWeight_) {
        int64_t weight = decodedWeight_.valueAt<int64_t>(row);
        VELOX_USER_CHECK_GT(
            weight, 0, "Weight value must be greater than zero.");
        accumulator->add(positions, value, weight);
      } else {
        accumulator->add(positions, value);
      }
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    DecodedVector decodedSerializedDigests;
    decodedSerializedDigests.decode(*args[0], rows);
    std::vector<int16_t> positions;
    auto accumulator = value<TDigestAccumulator>(group);
    accumulator->compress(positions);
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedSerializedDigests.isNullAt(row)) {
        return;
      }
      auto serialized =
          decodedSerializedDigests.valueAt<facebook::velox::StringView>(row);
      accumulator->mergeDeserialized(positions, serialized.data());
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_) TDigestAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<TDigestAccumulator>(group)->~TDigestAccumulator();
      }
    }
  }

  void checkAndSetCompression(
      TDigestAccumulator* accumulator,
      vector_size_t row) {
    double compression = decodedCompression_.valueAt<double>(row);
    VELOX_USER_CHECK_GT(compression, 0, "Compression factor must be positive.");
    VELOX_USER_CHECK_LE(compression, 1000, "Compression must be at most 1000");
    // Ensure compression is at least 10.
    compression = std::max(compression, 10.0);
    // Set compression if not set
    if (compression_ == 0) {
      compression_ = compression;
    } else if (compression_ != compression) {
      VELOX_USER_FAIL("Compression factor must be same for all rows");
    }
    if (accumulator->compression_ == 0) {
      accumulator->compression_ = compression;
      // Set compression at most once.
      if (accumulator->digest_.compression() != compression) {
        accumulator->setCompression(compression);
      }
    } else if (accumulator->compression_ != compression) {
      VELOX_USER_FAIL("Compression factor must be same for all rows");
    }
  }

  void extractCommon(char** groups, int32_t numGroups, VectorPtr* result) {
    // Check for valid input
    if (!groups || !result) {
      VELOX_USER_FAIL(
          "Null groups or result vector passed to extractValues() or extractAccumulators()");
      return;
    }
    // If there are no groups, ensure the result vector is empty or
    // appropriately initialized
    if (numGroups == 0) {
      (*result)->resize(0);
      return;
    }
    auto flatResult = (*result)->asFlatVector<facebook::velox::StringView>();
    flatResult->resize(numGroups);
    std::vector<int16_t> positions;
    size_t totalSerializedSize = 0;
    std::vector<size_t> serializedSizes(numGroups);
    for (int32_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      if (!group) {
        VELOX_USER_FAIL("Null group encountered in extractCommon.");
        continue;
      }
      auto accumulator = value<TDigestAccumulator>(group);
      accumulator->compress(positions);
      serializedSizes[i] = accumulator->serializedByteSize();
      totalSerializedSize += serializedSizes[i];
    }
    BufferPtr buffer = flatResult->getBufferWithSpace(totalSerializedSize);
    char* currentPtr = buffer->asMutable<char>();
    for (int32_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      if (!group) {
        continue;
      }
      auto accumulator = value<TDigestAccumulator>(group);
      accumulator->serialize(currentPtr);
      flatResult->set(
          i, facebook::velox::StringView(currentPtr, serializedSizes[i]));
      currentPtr += serializedSizes[i];
    }
  }
};
} // namespace

void registerTDigestAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<AggregateFunctionSignature>> signatures;
  for (const auto& signature :
       {AggregateFunctionSignatureBuilder()
            .returnType("tdigest(double)")
            .intermediateType("varbinary")
            .argumentType("double")
            .build(),
        AggregateFunctionSignatureBuilder()
            .returnType("tdigest(double)")
            .intermediateType("varbinary")
            .argumentType("double")
            .argumentType("bigint")
            .build(),
        AggregateFunctionSignatureBuilder()
            .returnType("tdigest(double)")
            .intermediateType("varbinary")
            .argumentType("double")
            .argumentType("bigint")
            .argumentType("double")
            .build()}) {
    signatures.push_back(signature);
  }
  auto name = prefix + kTDigestAgg;
  exec::registerAggregateFunction(
      name,
      signatures,
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultTypes,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        if (argTypes.empty() || argTypes[0]->kind() != TypeKind::DOUBLE) {
          VELOX_USER_FAIL(
              "The first argument of {} must be of type DOUBLE", name);
        }
        bool hasWeight =
            argTypes.size() >= 2 && argTypes[1]->kind() == TypeKind::BIGINT;
        bool hasCompression =
            argTypes.size() >= 3 && argTypes[2]->kind() == TypeKind::DOUBLE;
        VELOX_USER_CHECK_EQ(
            argTypes.size(),
            1 + hasWeight + hasCompression,
            "Wrong number of arguments passed to {}",
            name);
        if (hasWeight) {
          VELOX_USER_CHECK_EQ(
              argTypes[1]->kind(),
              TypeKind::BIGINT,
              "The type of the weight argument of {} must be BIGINT",
              name);
        }
        if (hasCompression) {
          VELOX_USER_CHECK_EQ(
              argTypes[2]->kind(),
              TypeKind::DOUBLE,
              "The type of the compression argument of {} must be DOUBLE",
              name);
        }
        return std::make_unique<TDigestAggregate<double>>(
            hasWeight, hasCompression, resultTypes);
      },
      {false /*orderSensitive*/, false /*companionFunction*/},
      false /*companionFunction*/,
      overwrite);
}
} // namespace facebook::velox::aggregate::prestosql
