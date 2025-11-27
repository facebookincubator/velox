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
#include "velox/common/hyperloglog/Murmur3Hash128.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

/// Accumulator for khyperloglog_agg(x, y).
/// Creates a KHyperLogLog instance that tracks the relationship between
/// x values (hashed as keys into MinHash) and y values (tracked in HyperLogLog
/// sketches).
struct KHllAccumulator {
  explicit KHllAccumulator(HashStringAllocator* allocator) : khll_{allocator} {}

  template <typename TJoinKey>
  static int64_t convertKeys(TJoinKey joinKey) {
    if constexpr (std::is_same_v<TJoinKey, int64_t>) {
      return joinKey;
    } else if constexpr (std::is_integral_v<TJoinKey>) {
      return static_cast<int64_t>(joinKey);
    } else if constexpr (std::is_same_v<TJoinKey, float>) {
      // Cast to double first, then extract bits, based on implicit coercion
      double dbl = static_cast<double>(joinKey);
      return *reinterpret_cast<int64_t*>(&dbl);
    } else if constexpr (std::is_same_v<TJoinKey, double>) {
      // Extract bits
      return *reinterpret_cast<int64_t*>(&joinKey);
    } else if constexpr (std::is_same_v<TJoinKey, StringView>) {
      return common::hll::Murmur3Hash128::hash64(
          joinKey.data(), joinKey.size(), 0);
    } else if constexpr (std::is_same_v<TJoinKey, Timestamp>) {
      return joinKey.toMillis();
    } else {
      VELOX_UNREACHABLE("Unsupported input type: {}", typeid(TJoinKey).name());
    }
  }

  template <typename TUii>
  static int64_t convertUii(TUii uii) {
    if constexpr (std::is_same_v<TUii, int64_t>) {
      return uii;
    } else if constexpr (
        std::is_integral_v<TUii> || std::is_same_v<TUii, double> ||
        std::is_same_v<TUii, float>) {
      return static_cast<int64_t>(uii);
    } else if constexpr (std::is_same_v<TUii, StringView>) {
      return common::hll::Murmur3Hash128::hash64(uii.data(), uii.size(), 0);
    } else if constexpr (std::is_same_v<TUii, Timestamp>) {
      return uii.toMillis();
    } else {
      VELOX_UNREACHABLE("ahh");
    }
  }

  /// Adds a (joinKey, uii) pair to the KHyperLogLog.
  /// The 2 input parameters are treated differently for certain input
  /// parameter types because DOUBLE is only explicitly accepted for the first
  /// parameter joinKey in the Java implementation of the function. So different
  /// conversion functions are used to convert the inputs to int64_t.
  template <typename TJoinKey, typename TUii>
  void add(TJoinKey joinKey, TUii uii) {
    const int64_t convertedJoinKey = convertKeys(joinKey);
    const int64_t convertedUii = convertUii(uii);
    khll_.add(convertedJoinKey, convertedUii);
  }

  /// Merges a serialized KHyperLogLog into this accumulator.
  void mergeWith(StringView serialized, HashStringAllocator* allocator) {
    auto other = common::hll::KHyperLogLog<HashStringAllocator>::deserialize(
        serialized.data(), serialized.size(), allocator);
    khll_.mergeWith(*other);
  }

  /// Returns serialized size of the KHyperLogLog.
  size_t serializedSize() {
    return khll_.estimatedSerializedSize();
  }

  /// Serializes the KHyperLogLog to a buffer.
  void serialize(char* output) {
    khll_.serialize(output);
  }

  common::hll::KHyperLogLog<HashStringAllocator> khll_;
};

template <typename TValue, typename TUii>
class KHyperLogLogAggregate : public exec::Aggregate {
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

        auto* accumulator = value<KHllAccumulator>(group);
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
    decodedValue_.decode(*args[0], rows, true);
    decodedUii_.decode(*args[1], rows, true);

    rows.applyToSelected([&](auto row) {
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

    rows.applyToSelected([&](auto row) {
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

    rows.applyToSelected([&](auto row) {
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
    rows.applyToSelected([&](auto row) {
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

/// Factory function for creating KHyperLogLogAggregate instances
/// based on input types. Both types are provided as template parameters.
template <TypeKind TValueKind, TypeKind TUiiKind>
std::unique_ptr<exec::Aggregate> createKHyperLogLogAggregate(
    const TypePtr& resultType) {
  using TValue = typename TypeTraits<TValueKind>::NativeType;
  using TUii = typename TypeTraits<TUiiKind>::NativeType;
  return std::make_unique<KHyperLogLogAggregate<TValue, TUii>>(resultType);
}

/// Intermediate dispatch function for the second type parameter.
/// This is necessary because VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH requires
/// a template function to receive the first dispatched type.
template <TypeKind TValueKind>
std::unique_ptr<exec::Aggregate> dispatchOnUiiType(
    TypeKind uiiKind,
    const TypePtr& resultType) {
  return VELOX_DYNAMIC_SCALAR_TEMPLATE_TYPE_DISPATCH(
      createKHyperLogLogAggregate, TValueKind, uiiKind, resultType);
}

} // namespace facebook::velox::aggregate::prestosql
