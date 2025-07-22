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
#include "velox/functions/prestosql/aggregates/sfm/SfmSketch.h"
#include "velox/functions/prestosql/aggregates/sfm/SfmSketchAccumulator.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

template <bool sketchAsFinalResult>
class SfmSketchAggregate : public exec::Aggregate {
  using SfmSketchAccumulator =
      facebook::velox::functions::aggregate::SfmSketchAccumulator;

 public:
  explicit SfmSketchAggregate(TypePtr resultType)
      : exec::Aggregate(std::move(resultType)) {}

  int32_t accumulatorFixedWidthSize() const override {
    return static_cast<int32_t>(sizeof(SfmSketchAccumulator));
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(SfmSketchAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      [[maybe_unused]] bool mayPushdown) override {
    decodeInput(rows, args);
    rows.applyToSelected([&](vector_size_t i) {
      auto group = groups[i];
      updateFromInput(group, i);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      [[maybe_unused]] bool mayPushdown) override {
    decodeInput(rows, args);
    rows.applyToSelected([&](vector_size_t i) { updateFromInput(group, i); });
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    serializeToStringVector(
        groups,
        numGroups,
        result,
        [](const SfmSketchAccumulator& accumulator) -> size_t {
          return accumulator.serializedSize();
        },
        [](const SfmSketchAccumulator& accumulator, char* buffer) -> void {
          accumulator.serialize(buffer);
        });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    if constexpr (sketchAsFinalResult) {
      // For final sketch result, serialize only the SfmSketch, not the
      // SfmSketchAccumulator.
      serializeToStringVector(
          groups,
          numGroups,
          result,
          [](SfmSketchAccumulator& accumulator) -> size_t {
            return accumulator.sketch().serializedSize();
          },
          [](SfmSketchAccumulator& accumulator, char* buffer) -> void {
            accumulator.sketch().serialize(buffer);
          });
      return;
    }

    auto flatResult = (*result)->asFlatVector<int64_t>();
    flatResult->resize(numGroups);

    for (auto i = 0; i < numGroups; i++) {
      auto group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto accumulator = value<SfmSketchAccumulator>(group);
        if (!accumulator->isInitialized()) {
          flatResult->setNull(i, true);
          continue;
        }
        flatResult->set(i, accumulator->cardinality());
      }
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      [[maybe_unused]] bool mayPushdown) override {
    DecodedVector decodedVector(*args[0], rows);

    rows.applyToSelected([&](vector_size_t i) {
      auto group = groups[i];
      updateFromIntermediate(decodedVector, group, i);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      [[maybe_unused]] bool mayPushdown) override {
    DecodedVector decodedVector(*args[0], rows);

    rows.applyToSelected([&](vector_size_t i) {
      updateFromIntermediate(decodedVector, group, i);
    });
  }

 protected:
  DecodedVector decodedValue_;
  DecodedVector decodedEpsilon_;
  DecodedVector decodedBuckets_;
  DecodedVector decodedPrecision_;
  size_t numArgs_ = 0;

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      auto* group = groups[i];
      new (value<SfmSketchAccumulator>(group)) SfmSketchAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<SfmSketchAccumulator>(groups);
  }

 private:
  // Helper function to serialize data to StringView result.
  template <typename SizeFunc, typename SerializeFunc>
  void serializeToStringVector(
      char** groups,
      int32_t numGroups,
      VectorPtr* result,
      SizeFunc sizeFunc,
      SerializeFunc serializeFunc) {
    auto flatResult = (*result)->asFlatVector<StringView>();
    flatResult->resize(numGroups);

    // Calculate total size needed for all valid groups.
    size_t totalSize = 0;
    std::vector<size_t> groupSizes(numGroups, 0);

    for (auto i = 0; i < numGroups; i++) {
      auto group = groups[i];
      if (!isNull(group)) {
        auto* accumulator = value<SfmSketchAccumulator>(group);
        if (accumulator->isInitialized()) {
          groupSizes[i] = sizeFunc(*accumulator);
          totalSize += groupSizes[i];
        }
      }
    }

    // Allocate buffer for all serializations.
    auto rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);
    size_t offset = 0;

    for (auto i = 0; i < numGroups; i++) {
      auto group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto* accumulator = value<SfmSketchAccumulator>(group);
        if (!accumulator->isInitialized()) {
          flatResult->setNull(i, true);
        } else {
          // Serialize the data.
          serializeFunc(*accumulator, rawBuffer + offset);
          flatResult->setNoCopy(
              i,
              StringView(
                  rawBuffer + offset, static_cast<int32_t>(groupSizes[i])));
          offset += groupSizes[i];
        }
      }
    }
  }

  void decodeInput(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    numArgs_ = args.size();
    VELOX_CHECK_GE(numArgs_, 2);
    decodedValue_.decode(*args[0], rows);
    decodedEpsilon_.decode(*args[1], rows);
    if (args.size() > 2) {
      decodedBuckets_.decode(*args[2], rows);
    }
    if (args.size() > 3) {
      decodedPrecision_.decode(*args[3], rows);
    }
  }

  void updateFromInput(char* group, vector_size_t i) {
    if (!decodedValue_.isNullAt(i)) {
      auto tracker = trackRowSize(group);
      auto* accumulator = value<SfmSketchAccumulator>(group);
      clearNull(group);
      std::optional<int32_t> buckets =
          (numArgs_ > 2 && !decodedBuckets_.isNullAt(i))
          ? std::optional<int32_t>(decodedBuckets_.valueAt<int64_t>(i))
          : std::nullopt;
      std::optional<int32_t> precision =
          (numArgs_ > 3 && !decodedPrecision_.isNullAt(i))
          ? std::optional<int32_t>(decodedPrecision_.valueAt<int64_t>(i))
          : std::nullopt;

      if (!accumulator->isInitialized()) {
        accumulator->initialize(buckets, precision);
      }

      accumulator->setEpsilon(decodedEpsilon_.valueAt<double>(i));

      // Handle different input types.
      auto inputType = decodedValue_.base()->type();
      switch (inputType->kind()) {
        case TypeKind::BIGINT:
          accumulator->add(decodedValue_.valueAt<int64_t>(i));
          break;
        case TypeKind::DOUBLE:
          accumulator->add(decodedValue_.valueAt<double>(i));
          break;
        case TypeKind::VARCHAR:
        case TypeKind::VARBINARY: {
          auto stringValue = decodedValue_.valueAt<StringView>(i);
          accumulator->add(stringValue);
          break;
        }
        default:
          VELOX_UNSUPPORTED(
              "Unsupported input type for SfmSketch: {}",
              inputType->toString());
      }
    }
  }

  void updateFromIntermediate(
      DecodedVector& decodedVector,
      char* group,
      vector_size_t i) {
    if (decodedVector.isNullAt(i)) {
      return;
    }
    auto tracker = trackRowSize(group);
    auto* accumulator = value<SfmSketchAccumulator>(group);
    clearNull(group);
    auto serialized = decodedVector.valueAt<StringView>(i);
    auto otherAccumulator =
        SfmSketchAccumulator::deserialize(serialized.data(), allocator_);
    accumulator->mergeWith(otherAccumulator);
  }
};

} // namespace facebook::velox::aggregate::prestosql
