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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

template <typename TValue, typename TUii>
class KHyperLogLogAggregate : public exec::Aggregate {
  using KHllAccumulator = common::hll::KHyperLogLog<TUii, HashStringAllocator>;

 public:
  explicit KHyperLogLogAggregate(const TypePtr& resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(KHllAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(KHllAccumulator);
  }

  bool isFixedSize() const override {
    return false;
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

    BufferPtr& nulls = flatResult->mutableNulls(flatResult->size());
    uint64_t* rawNulls = nulls->asMutable<uint64_t>();

    // Calculate total size needed for non-inline strings.
    int64_t totalSize = 0;
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (!isNull(group)) {
        auto* accumulator = value<KHllAccumulator>(group);
        auto size = accumulator->estimatedSerializedSize();
        if (!StringView::isInline(size)) {
          totalSize += size;
        }
      }
    }

    // Allocate all required space at once.
    char* rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);

    // Serialize accumulators.
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        bits::clearBit(rawNulls, i);

        auto* accumulator = value<KHllAccumulator>(group);
        size_t size = accumulator->estimatedSerializedSize();

        StringView serialized;
        if (StringView::isInline(size)) {
          std::string buffer(size, '\0');
          accumulator->serialize(buffer.data());
          serialized = StringView::makeInline(buffer);
        } else {
          accumulator->serialize(rawBuffer);
          serialized = StringView(rawBuffer, size);
          rawBuffer += size;
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
    decodedValue_.decode(*args[0], rows, true);
    decodedUii_.decode(*args[1], rows, true);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row) || decodedUii_.isNullAt(row)) {
        return;
      }

      auto group = groups[row];
      auto tracker = trackRowSize(group);
      auto* accumulator = value<KHllAccumulator>(group);
      clearNull(group);

      auto val = decodedValue_.valueAt<TValue>(row);
      auto uii = decodedUii_.valueAt<TUii>(row);
      accumulator->add(val, uii);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows, true);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      auto group = groups[row];
      auto tracker = trackRowSize(group);
      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      auto* accumulator = value<KHllAccumulator>(group);
      accumulator->mergeWith(serialized, allocator_);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto tracker = trackRowSize(group);

    decodedValue_.decode(*args[0], rows, true);
    decodedUii_.decode(*args[1], rows, true);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row) || decodedUii_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<KHllAccumulator>(group);
      clearNull(group);

      auto val = decodedValue_.valueAt<TValue>(row);
      auto uii = decodedUii_.valueAt<TUii>(row);
      accumulator->add(val, uii);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows, true);

    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      auto* accumulator = value<KHllAccumulator>(group);
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
      new (group + offset_) KHllAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<KHllAccumulator>(groups);
  }

 private:
  DecodedVector decodedValue_;
  DecodedVector decodedUii_;
  DecodedVector decodedIntermediate_;
};

} // namespace facebook::velox::aggregate::prestosql
