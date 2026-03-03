/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/lib/SetDigest.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

// SetDigest only supports int64_t and StringView as element types.
// Template specializations ensure compile-time errors for unsupported types.
template <typename T>
struct SetDigestElementType;

template <>
struct SetDigestElementType<bool> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<int8_t> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<int16_t> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<int32_t> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<int64_t> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<float> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<double> {
  using type = int64_t;
};

template <>
struct SetDigestElementType<StringView> {
  using type = StringView;
};

template <typename T>
class SetDigestAggregate : public exec::Aggregate {
 private:
  using SetDigestAccumulator =
      functions::SetDigest<typename SetDigestElementType<T>::type>;

 public:
  explicit SetDigestAggregate(const TypePtr& resultType)
      : Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SetDigestAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) SetDigestAccumulator(allocator_);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      addValueToAccumulator(accumulator, row);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);

    auto* accumulator = value<SetDigestAccumulator>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      clearNull(group);
      addValueToAccumulator(accumulator, row);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    auto* accumulator = value<SetDigestAccumulator>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* flatResult = (*result)->as<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    // Calculate total buffer size needed for all non-inline
    // strings.
    int64_t totalBufferSize = 0;
    std::vector<int32_t> sizes(numGroups);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (!isNull(group)) {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = accumulator->estimatedSerializedSize();
        sizes[i] = size;
        if (!StringView::isInline(size)) {
          totalBufferSize += size;
        }
      }
    }

    // Allocate buffer once for all non-inline strings.
    char* rawBuffer = nullptr;
    if (totalBufferSize > 0) {
      rawBuffer = flatResult->getRawStringBufferWithSpace(totalBufferSize);
    }

    // Serialize into pre-allocated buffer.
    for (vector_size_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = sizes[i];

        StringView serialized;
        if (StringView::isInline(size)) {
          // For small serialized data (<= 12 bytes), use inline StringView
          // storage to avoid heap allocation overhead.
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

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<SetDigestAccumulator>(groups);
  }

 private:
  void addValueToAccumulator(
      SetDigestAccumulator* accumulator,
      vector_size_t row) {
    if constexpr (std::is_same_v<T, StringView>) {
      accumulator->add(decodedValue_.valueAt<StringView>(row));
    } else if constexpr (std::is_same_v<T, float>) {
      float val = decodedValue_.valueAt<float>(row);
      int32_t bits;
      std::memcpy(&bits, &val, sizeof(float));
      accumulator->add(static_cast<int64_t>(bits));
    } else if constexpr (std::is_same_v<T, double>) {
      double val = decodedValue_.valueAt<double>(row);
      int64_t bits;
      std::memcpy(&bits, &val, sizeof(double));
      accumulator->add(bits);
    } else {
      accumulator->add(static_cast<int64_t>(decodedValue_.valueAt<T>(row)));
    }
  }

  DecodedVector decodedValue_;
  DecodedVector decodedIntermediate_;
};

template <typename T>
std::unique_ptr<exec::Aggregate> createSetDigestAggregate(
    const TypePtr& resultType) {
  return std::make_unique<SetDigestAggregate<T>>(resultType);
}

std::vector<exec::AggregateRegistrationResult> registerMakeSetDigest(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  // Support all common input types
  std::vector<std::string> inputTypes = {
      "boolean",
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real", // float (32-bit)
      "double", // double (64-bit)
      "date", // int32 days since epoch (handled by INTEGER case)
      "varchar",
      "varbinary",
  };

  signatures.reserve(inputTypes.size());
  for (const auto& inputType : inputTypes) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType("varbinary")
            .intermediateType("varbinary")
            .argumentType(inputType)
            .build());
  }

  return exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step /*step*/,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        const auto& inputType = argTypes[0];

        switch (inputType->kind()) {
          case TypeKind::BOOLEAN:
            return createSetDigestAggregate<bool>(resultType);
          case TypeKind::TINYINT:
            return createSetDigestAggregate<int8_t>(resultType);
          case TypeKind::SMALLINT:
            return createSetDigestAggregate<int16_t>(resultType);
          case TypeKind::INTEGER:
            return createSetDigestAggregate<int32_t>(resultType);
          case TypeKind::BIGINT:
            return createSetDigestAggregate<int64_t>(resultType);
          case TypeKind::REAL:
            return createSetDigestAggregate<float>(resultType);
          case TypeKind::DOUBLE:
            return createSetDigestAggregate<double>(resultType);
          case TypeKind::VARCHAR:
          case TypeKind::VARBINARY:
            return createSetDigestAggregate<StringView>(resultType);
          default:
            VELOX_UNREACHABLE(
                "Unsupported input type for make_set_digest: {}",
                inputType->toString());
        }
      },
      withCompanionFunctions,
      overwrite);
}

void registerMakeSetDigestAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMakeSetDigest(names, withCompanionFunctions, overwrite);
}

class MergeSetDigestAggregate : public exec::Aggregate {
 private:
  using SetDigestAccumulator = functions::SetDigest<int64_t>;

 public:
  explicit MergeSetDigestAggregate(const TypePtr& resultType)
      : Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SetDigestAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) SetDigestAccumulator(allocator_);
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedValue_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);

    auto* accumulator = value<SetDigestAccumulator>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedValue_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    auto* accumulator = value<SetDigestAccumulator>(group);
    auto tracker = trackRowSize(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      auto status = other.deserialize(serialized.data(), serialized.size());
      VELOX_CHECK(
          status.ok(), "Failed to deserialize SetDigest: {}", status.message());
      accumulator->mergeWith(other);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* flatResult = (*result)->as<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    // Calculate total buffer size needed for all non-inline
    // strings.
    int64_t totalBufferSize = 0;
    std::vector<int32_t> sizes(numGroups);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (!isNull(group)) {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = accumulator->estimatedSerializedSize();
        sizes[i] = size;
        if (!StringView::isInline(size)) {
          totalBufferSize += size;
        }
      }
    }

    // Allocate buffer once for all non-inline strings.
    char* rawBuffer = nullptr;
    if (totalBufferSize > 0) {
      rawBuffer = flatResult->getRawStringBufferWithSpace(totalBufferSize);
    }

    // Serialize into pre-allocated buffer.
    for (vector_size_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = sizes[i];

        StringView serialized;
        if (StringView::isInline(size)) {
          // For small serialized data (<= 12 bytes), use inline StringView
          // storage to avoid heap allocation overhead.
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

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<SetDigestAccumulator>(groups);
  }

 private:
  DecodedVector decodedValue_;
  DecodedVector decodedIntermediate_;
};

std::vector<exec::AggregateRegistrationResult> registerMergeSetDigest(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("setdigest")
          .intermediateType("varbinary")
          .argumentType("setdigest")
          .build());

  return exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step /*step*/,
         const std::vector<TypePtr>& /*argTypes*/,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<MergeSetDigestAggregate>(resultType);
      },
      withCompanionFunctions,
      overwrite);
}

void registerMergeSetDigestAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMergeSetDigest(names, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
