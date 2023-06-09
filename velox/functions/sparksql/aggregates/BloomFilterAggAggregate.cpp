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

#include "velox/common/base/BloomFilter.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql::aggregates {

namespace {

struct BloomFilterAccumulator {
  explicit BloomFilterAccumulator(HashStringAllocator* allocator)
      : bloomFilter_{StlAllocator<uint64_t>(allocator)} {}

  int32_t serializedSize() {
    return bloomFilter_.serializedSize();
  }

  void serialize(StringView& output) {
    return bloomFilter_.serialize(const_cast<char*>(output.data()));
  }

  void mergeWith(StringView& serialized) {
    bloomFilter_.merge(serialized.data());
  }

  void init(int32_t capacity) {
    if (!bloomFilter_.isSet()) {
      bloomFilter_.reset(capacity);
    }
  }

  void insert(int64_t value) {
    bloomFilter_.insert(folly::hasher<int64_t>()(value));
  }

  BloomFilter<StlAllocator<uint64_t>> bloomFilter_;
}; // namespace

template <typename T>
class BloomFilterAggAggregate : public exec::Aggregate {
 public:
  explicit BloomFilterAggAggregate(const TypePtr& resultType)
      : Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(BloomFilterAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  /// Initialize each group.
  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) BloomFilterAccumulator(allocator_);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    // ignore the estimatedNumItems, this config is not used in
    // velox bloom filter implementation
    decodeArguments(rows, args);
    VELOX_CHECK(!decodedRaw_.mayHaveNulls());
    rows.applyToSelected([&](vector_size_t row) {
      auto accumulator = value<BloomFilterAccumulator>(groups[row]);
      accumulator->init(capacity_);
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
      accumulator->mergeWith(serialized);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    auto accumulator = value<BloomFilterAccumulator>(group);
    // VELOX_CHECK(!decodedRaw_.mayHaveNulls());
    if (decodedRaw_.isConstantMapping()) {
      // all values are same, just do for the first
      accumulator->init(capacity_);
      accumulator->insert(decodedRaw_.valueAt<int64_t>(0));
      return;
    }
    rows.applyToSelected([&](vector_size_t row) {
      accumulator->init(capacity_);
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
      accumulator->mergeWith(serialized);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto flatResult = (*result)->asUnchecked<FlatVector<StringView>>();
    flatResult->resize(numGroups);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      auto group = groups[i];
      auto accumulator = value<BloomFilterAccumulator>(group);
      auto size = accumulator->serializedSize();
      if (UNLIKELY(!accumulator->bloomFilter_.isSet())) {
        flatResult->setNull(i, true);
        continue;
      }
      if (StringView::isInline(size)) {
        char buffer[StringView::kInlineSize];
        StringView serialized = StringView(buffer, size);
        accumulator->serialize(serialized);
        flatResult->setNoCopy(i, serialized);
      } else {
        Buffer* buffer = flatResult->getBufferWithSpace(size);
        StringView serialized(buffer->as<char>() + buffer->size(), size);
        accumulator->serialize(serialized);
        buffer->setSize(buffer->size() + size);
        flatResult->setNoCopy(i, serialized);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 private:
  const int64_t DEFAULT_ESPECTED_NUM_ITEMS = 1000000;
  const int64_t MAX_NUM_ITEMS = 4000000;
  const int64_t MAX_NUM_BITS = 67108864;

  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    if (args.size() > 0) {
      decodedRaw_.decode(*args[0], rows);
      if (args.size() > 1) {
        DecodedVector decodedEstimatedNumItems(*args[1], rows);
        setConstantArgument(
            "originalEstimatedNumItems",
            originalEstimatedNumItems_,
            decodedEstimatedNumItems);
        if (args.size() > 2) {
          DecodedVector decodedNumBits(*args[2], rows);
          setConstantArgument(
              "originalNumBits", originalNumBits_, decodedNumBits);
        } else {
          VELOX_CHECK_EQ(args.size(), 3);
          originalNumBits_ = originalEstimatedNumItems_ * 8;
        }
      } else {
        originalEstimatedNumItems_ = DEFAULT_ESPECTED_NUM_ITEMS;
        originalNumBits_ = originalEstimatedNumItems_ * 8;
      }
    } else {
      VELOX_USER_FAIL("Function args size must be more than 0")
    }
    estimatedNumItems_ = std::min(originalEstimatedNumItems_, MAX_NUM_ITEMS);
    numBits_ = std::min(originalNumBits_, MAX_NUM_BITS);
    capacity_ = numBits_ / 16;
  }

  static void
  setConstantArgument(const char* name, int64_t& val, int64_t newVal) {
    VELOX_USER_CHECK_GT(newVal, 0, "{} must be positive", name);
    if (val == kMissingArgument) {
      val = newVal;
    } else {
      VELOX_USER_CHECK_EQ(
          newVal, val, "{} argument must be constant for all input rows", name);
    }
  }

  static void setConstantArgument(
      const char* name,
      int64_t& val,
      const DecodedVector& vec) {
    VELOX_CHECK(
        vec.isConstantMapping(),
        "{} argument must be constant for all input rows",
        name);
    setConstantArgument(name, val, vec.valueAt<int64_t>(0));
  }

  static constexpr int64_t kMissingArgument = -1;
  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedRaw_;
  DecodedVector decodedIntermediate_;
  int64_t originalEstimatedNumItems_ = kMissingArgument;
  int64_t originalNumBits_ = kMissingArgument;
  int64_t estimatedNumItems_ = kMissingArgument;
  int64_t numBits_ = kMissingArgument;
  int32_t capacity_ = kMissingArgument;
};

} // namespace

bool registerBloomFilterAggAggregate(const std::string& name) {
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
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<BloomFilterAggAggregate<uint64_t>>(resultType);
      });
}
} // namespace facebook::velox::functions::sparksql::aggregates
