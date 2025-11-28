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

#include "velox/common/hyperloglog/KHyperLogLog.h"
#include "velox/exec/Aggregate.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

struct MergeKHllAccumulator {
  explicit MergeKHllAccumulator(HashStringAllocator* allocator)
      : khll_{allocator} {}

  void mergeWith(StringView serialized, HashStringAllocator* allocator) {
    auto other = common::hll::KHyperLogLog<HashStringAllocator>::deserialize(
        serialized.data(), serialized.size(), allocator);
    khll_.mergeWith(*other);
  }

  size_t serializedSize() {
    return khll_.estimatedSerializedSize();
  }

  void serialize(char* output) {
    std::string serialized = khll_.serialize();
    std::memcpy(output, serialized.data(), serialized.size());
  }

  common::hll::KHyperLogLog<HashStringAllocator> khll_;
};

class MergeKHyperLogLogAggregate : public exec::Aggregate {
 public:
  explicit MergeKHyperLogLogAggregate(const TypePtr& resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(MergeKHllAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(MergeKHllAccumulator);
  }

  bool accumulatorUsesExternalMemory() const override {
    return true;
  }

  bool isFixedSize() const override {
    return false;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const final {
    singleInputAsIntermediate(rows, args, result);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractAccumulators(groups, numGroups, result);
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto* flatResult = (*result)->asFlatVector<StringView>();
    flatResult->resize(numGroups);

    uint64_t* rawNulls = nullptr;
    if (flatResult->mayHaveNulls()) {
      BufferPtr& nulls = flatResult->mutableNulls(flatResult->size());
      rawNulls = nulls->asMutable<uint64_t>();
    }

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }

        auto* accumulator = value<MergeKHllAccumulator>(group);
        auto size = accumulator->serializedSize();

        StringView serialized;
        if (StringView::isInline(size)) {
          std::string buffer(size, '\0');
          accumulator->serialize(buffer.data());
          serialized = StringView::makeInline(buffer);
        } else {
          char* rawBuffer = flatResult->getRawStringBufferWithSpace(size);
          accumulator->serialize(rawBuffer);
          serialized = StringView(rawBuffer, size);
        }
        flatResult->setNoCopy(i, serialized);
      }
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediateResults(groups, rows, args, false);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedKHll_.decode(*args[0], rows, true);

    rows.applyToSelected([&](auto row) {
      if (decodedKHll_.isNullAt(row)) {
        return;
      }

      auto group = groups[row];
      auto tracker = trackRowSize(group);
      clearNull(group);

      auto serialized = decodedKHll_.valueAt<StringView>(row);
      auto* accumulator = value<MergeKHllAccumulator>(group);
      accumulator->mergeWith(serialized, allocator_);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto tracker = trackRowSize(group);
    addSingleGroupIntermediateResults(group, rows, args, false);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedKHll_.decode(*args[0], rows, true);

    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](auto row) {
      if (decodedKHll_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedKHll_.valueAt<StringView>(row);
      auto* accumulator = value<MergeKHllAccumulator>(group);
      accumulator->mergeWith(serialized, allocator_);
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_) MergeKHllAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<MergeKHllAccumulator>(groups);
  }

 private:
  DecodedVector decodedKHll_;
};

inline std::unique_ptr<exec::Aggregate> createMergeKHyperLogLogAggregate(
    const TypePtr& resultType) {
  return std::make_unique<MergeKHyperLogLogAggregate>(resultType);
}

} // namespace facebook::velox::aggregate::prestosql
