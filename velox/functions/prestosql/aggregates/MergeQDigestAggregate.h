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

#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/QuantileDigest.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

template <typename T>
struct QDigestAccumulator {
  explicit QDigestAccumulator(HashStringAllocator* allocator)
      : digest_(
            StlAllocator<T>(allocator),
            facebook::velox::functions::qdigest::kUninitializedMaxError) {}

  void mergeWith(
      StringView serialized,
      HashStringAllocator* /*allocator*/,
      std::vector<int16_t>& /*positions*/) {
    if (serialized.empty()) {
      return;
    }
    digest_.mergeSerialized(serialized.data());
  }

  int64_t serializedSize() {
    return digest_.serializedByteSize();
  }

  void serialize(char* outputBuffer) {
    // If maxError is still uninitialized, set it to the default value of 0.01
    if (digest_.getMaxError() ==
        facebook::velox::functions::qdigest::kUninitializedMaxError) {
      digest_.setMaxError(0.01);
    }
    digest_.serialize(outputBuffer);
  }

 private:
  facebook::velox::functions::qdigest::QuantileDigest<T, StlAllocator<T>>
      digest_;
};

template <typename T>
class MergeQDigestAggregate : public exec::Aggregate {
 public:
  explicit MergeQDigestAggregate(const TypePtr& resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(QDigestAccumulator<T>);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(QDigestAccumulator<T>);
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
    extract(
        groups,
        numGroups,
        flatResult,
        [&](QDigestAccumulator<T>* accumulator,
            FlatVector<StringView>* result,
            vector_size_t index) {
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
          result->setNoCopy(index, serialized);
        });
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addIntermediateResults(groups, rows, args, false /*unused*/);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedQDigest_.decode(*args[0], rows, true);

    rows.applyToSelected([&](auto row) {
      if (decodedQDigest_.isNullAt(row)) {
        return;
      }

      auto group = groups[row];
      auto tracker = trackRowSize(group);
      clearNull(group);

      mergeToAccumulator(group, row);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto tracker = trackRowSize(group);
    addSingleGroupIntermediateResults(group, rows, args, false /*unused*/);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedQDigest_.decode(*args[0], rows, true);

    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](auto row) {
      if (decodedQDigest_.isNullAt(row)) {
        return;
      }

      clearNull(group);
      mergeToAccumulator(group, row);
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_) QDigestAccumulator<T>(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<QDigestAccumulator<T>>(groups);
  }

 private:
  void mergeToAccumulator(char* group, const vector_size_t row) {
    auto serialized = decodedQDigest_.valueAt<StringView>(row);
    QDigestAccumulator<T>* accumulator = value<QDigestAccumulator<T>>(group);
    accumulator->mergeWith(serialized, allocator_, positions_);
  }

  template <typename ExtractResult, typename ExtractFunc>
  void extract(
      char** groups,
      int32_t numGroups,
      FlatVector<ExtractResult>* result,
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
      if (isNull(group)) {
        // Set null for null input
        result->setNull(i, true);
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }
        auto accumulator = value<QDigestAccumulator<T>>(group);
        extractFunction(accumulator, result, i);
      }
    }
  }
  DecodedVector decodedQDigest_;
  std::vector<int16_t> positions_;
};

inline std::unique_ptr<exec::Aggregate> createMergeQDigestAggregate(
    const TypePtr& resultType,
    const TypePtr& argType = nullptr) {
  if (argType != nullptr) {
    if (*argType == *QDIGEST(BIGINT())) {
      return std::make_unique<MergeQDigestAggregate<int64_t>>(resultType);
    } else if (*argType == *QDIGEST(REAL())) {
      return std::make_unique<MergeQDigestAggregate<float>>(resultType);
    } else if (*argType == *QDIGEST(DOUBLE())) {
      return std::make_unique<MergeQDigestAggregate<double>>(resultType);
    }
    VELOX_UNSUPPORTED("QDigest {} type is not supported.", argType->toString());
  }
  VELOX_UNREACHABLE("Arg type is null");
}
} // namespace facebook::velox::aggregate::prestosql
