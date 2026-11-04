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

#include "velox/functions/sparksql/aggregates/BloomFilterAggAggregate.h"

#include <cstring>
#include <optional>

#include <folly/lang/Bits.h>

#include "velox/common/base/BloomFilter.h"
#include "velox/common/base/SplitBlockBloomFilter.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate::sparksql {

using functions::sparksql::SparkQueryConfig;

BloomFilterAccumulator::BloomFilterAccumulator(HashStringAllocator* allocator)
    : blocks_{BlockAllocator(allocator)} {}

int32_t BloomFilterAccumulator::serializedSize() const {
  return static_cast<int32_t>(blocks_.size() * sizeof(Block));
}

void BloomFilterAccumulator::serialize(char* output) const {
  std::memcpy(output, blocks_.data(), blocks_.size() * sizeof(Block));
}

void BloomFilterAccumulator::merge(const StringView& serialized) {
  VELOX_USER_CHECK_GT(
      serialized.size(), 0, "Serialized split-block Bloom filter is empty");
  VELOX_USER_CHECK_EQ(
      serialized.size() % sizeof(Block),
      0,
      "Invalid serialized split-block Bloom filter size: {}",
      serialized.size());
  const auto numBlocks =
      static_cast<int32_t>(serialized.size() / sizeof(Block));
  if (blocks_.empty()) {
    init(numBlocks);
  } else {
    VELOX_USER_CHECK_EQ(
        blocks_.size(),
        numBlocks,
        "Cannot merge split-block Bloom filters of different sizes");
  }

  for (int64_t i = 0; i < numBlocks; ++i) {
    for (int32_t j = 0; j < xsimd::batch<uint32_t>::size; ++j) {
      const auto offset = i * sizeof(Block) + j * sizeof(uint32_t);
      blocks_[i].data[j] |=
          folly::loadUnaligned<uint32_t>(serialized.data() + offset);
    }
  }
}

bool BloomFilterAccumulator::initialized() const {
  return !blocks_.empty();
}

void BloomFilterAccumulator::init(int32_t numBlocks) {
  if (blocks_.empty()) {
    VELOX_CHECK_GT(numBlocks, 0);
    blocks_.resize(numBlocks);
    bloomFilter_.emplace(blocks_);
  }
}

void BloomFilterAccumulator::insert(int64_t value) {
  VELOX_DCHECK(bloomFilter_.has_value());
  bloomFilter_->insert(folly::hasher<int64_t>()(value));
}

const SplitBlockBloomFilter* BloomFilterAccumulator::bloomFilter() const {
  VELOX_CHECK(bloomFilter_.has_value());
  return &bloomFilter_.value();
}

namespace {

class BloomFilterAggAggregate : public exec::Aggregate {
 public:
  explicit BloomFilterAggAggregate(
      const TypePtr& resultType,
      const core::QueryConfig& config)
      : Aggregate(resultType),
        defaultExpectedNumItems_(
            SparkQueryConfig{config}.bloomFilterExpectedNumItems()),
        defaultNumBits_(SparkQueryConfig{config}.bloomFilterNumBits()),
        maxNumBits_(SparkQueryConfig{config}.bloomFilterMaxNumBits()),
        maxNumItems_(SparkQueryConfig{config}.bloomFilterMaxNumItems()) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(BloomFilterAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  static FOLLY_ALWAYS_INLINE void checkBloomFilterNotNull(
      DecodedVector& decoded,
      vector_size_t idx) {
    VELOX_USER_CHECK(
        !decoded.isNullAt(idx),
        "First argument of bloom_filter_agg cannot be null");
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    const auto numBlocks = this->numBlocks();
    auto mayHaveNulls = decodedRaw_.mayHaveNulls();
    rows.applyToSelected([&](vector_size_t row) {
      if (mayHaveNulls) {
        checkBloomFilterNotNull(decodedRaw_, row);
      }
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      auto accumulator = value<BloomFilterAccumulator>(group);
      accumulator->init(numBlocks);
      accumulator->insert(decodedRaw_.valueAt<int64_t>(row));
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_EQ(args.size(), 1);
    decodedIntermediate_.decode(*args[0], rows);
    rows.applyToSelected([&](auto row) {
      if (UNLIKELY(decodedIntermediate_.isNullAt(row))) {
        return;
      }
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      auto accumulator = value<BloomFilterAccumulator>(group);
      accumulator->merge(serialized);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    const auto numBlocks = this->numBlocks();
    auto tracker = trackRowSize(group);
    auto accumulator = value<BloomFilterAccumulator>(group);
    accumulator->init(numBlocks);
    if (decodedRaw_.isConstantMapping()) {
      // All values are same, just do for the first.
      checkBloomFilterNotNull(decodedRaw_, 0);
      accumulator->insert(decodedRaw_.valueAt<int64_t>(0));
      return;
    }
    auto mayHaveNulls = decodedRaw_.mayHaveNulls();
    rows.applyToSelected([&](vector_size_t row) {
      if (mayHaveNulls) {
        checkBloomFilterNotNull(decodedRaw_, row);
      }
      accumulator->insert(decodedRaw_.valueAt<int64_t>(row));
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_EQ(args.size(), 1);
    decodedIntermediate_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    auto accumulator = value<BloomFilterAccumulator>(group);
    rows.applyToSelected([&](auto row) {
      if (UNLIKELY(decodedIntermediate_.isNullAt(row))) {
        return;
      }
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      accumulator->merge(serialized);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto flatResult = (*result)->asUnchecked<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    int32_t totalSize = getTotalSize(groups, numGroups);
    char* rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      auto accumulator = value<BloomFilterAccumulator>(group);
      if (UNLIKELY(!accumulator->initialized())) {
        flatResult->setNull(i, true);
        continue;
      }

      auto size = accumulator->serializedSize();
      VELOX_DCHECK(!StringView::isInline(size));
      accumulator->serialize(rawBuffer);
      StringView serialized = StringView(rawBuffer, size);
      rawBuffer += size;
      flatResult->setNoCopy(i, serialized);
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) BloomFilterAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto* group : groups) {
      if (isInitialized(group)) {
        value<BloomFilterAccumulator>(group)->~BloomFilterAccumulator();
      }
    }
  }

 private:
  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_USER_CHECK(args.size() > 0);
    decodedRaw_.decode(*args[0], rows);
    if (args.size() > 1) {
      DecodedVector decodedEstimatedNumItems(*args[1], rows);
      setConstantArgument(
          "estimatedNumItems", estimatedNumItems_, decodedEstimatedNumItems);
      if (args.size() > 2) {
        VELOX_CHECK_EQ(args.size(), 3);
        DecodedVector decodedNumBits(*args[2], rows);
        setConstantArgument("numBits", numBits_, decodedNumBits);
      } else {
        numBits_ =
            BloomFilter<>::optimalNumOfBits(estimatedNumItems_, maxNumItems_);
      }
    } else {
      estimatedNumItems_ = defaultExpectedNumItems_;
      numBits_ = defaultNumBits_;
    }
  }

  int32_t numBlocks() const {
    const int64_t numBits = std::min(numBits_, maxNumBits_);
    return static_cast<int32_t>(std::max<int64_t>(
        1, numBits / (8 * sizeof(SplitBlockBloomFilter::Block))));
  }

  int32_t getTotalSize(char** groups, int32_t numGroups) const {
    int32_t totalSize = 0;
    for (vector_size_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      auto accumulator = value<BloomFilterAccumulator>(group);
      if (UNLIKELY(!accumulator->initialized())) {
        continue;
      }

      auto size = accumulator->serializedSize();
      VELOX_DCHECK(!StringView::isInline(size));
      totalSize += size;
    }
    return totalSize;
  }

  static void setConstantArgument(
      const char* name,
      int64_t& currentValue,
      const DecodedVector& vector) {
    VELOX_CHECK(
        vector.isConstantMapping(),
        "{} argument must be constant for all input rows",
        name);
    int64_t newValue = vector.valueAt<int64_t>(0);
    VELOX_USER_CHECK_GT(newValue, 0, "{} must be positive", name);
    if (currentValue == kMissingArgument) {
      currentValue = newValue;
    } else {
      VELOX_USER_CHECK_EQ(
          newValue,
          currentValue,
          "{} argument must be constant for all input rows",
          name);
    }
  }

  static constexpr int64_t kMissingArgument = -1;
  const int64_t defaultExpectedNumItems_;
  const int64_t defaultNumBits_;
  const int64_t maxNumBits_;
  const int64_t maxNumItems_;

  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedRaw_;
  DecodedVector decodedIntermediate_;
  int64_t estimatedNumItems_ = kMissingArgument;
  int64_t numBits_ = kMissingArgument;
};

} // namespace

exec::AggregateRegistrationResult registerBloomFilterAggAggregate(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .argumentType("bigint")
          .constantArgumentType("bigint")
          .constantArgumentType("bigint")
          .intermediateType("varbinary")
          .returnType("varbinary")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .argumentType("bigint")
          .constantArgumentType("bigint")
          .intermediateType("varbinary")
          .returnType("varbinary")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .argumentType("bigint")
          .intermediateType("varbinary")
          .returnType("varbinary")
          .build()};

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step /* step */,
          const std::vector<TypePtr>& /* argTypes */,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<BloomFilterAggAggregate>(resultType, config);
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
