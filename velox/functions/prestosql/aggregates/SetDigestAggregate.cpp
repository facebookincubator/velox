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
#include "velox/functions/lib/SetDigest.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

using SetDigestAccumulator = functions::SetDigest;

template <typename T>
class SetDigestAggregate : public exec::Aggregate {
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

    rows.applyToSelected([&](auto row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      if constexpr (std::is_same_v<T, StringView>) {
        accumulator->add(decodedValue_.valueAt<StringView>(row));
      } else {
        accumulator->add(static_cast<int64_t>(decodedValue_.valueAt<T>(row)));
      }
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

    rows.applyToSelected([&](auto row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      if constexpr (std::is_same_v<T, StringView>) {
        accumulator->add(decodedValue_.valueAt<StringView>(row));
      } else {
        accumulator->add(static_cast<int64_t>(decodedValue_.valueAt<T>(row)));
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](auto row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
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

    rows.applyToSelected([&](auto row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
      accumulator->mergeWith(other);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* flatResult = (*result)->as<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = accumulator->estimatedSerializedSize();

        auto* rawBuffer = flatResult->getRawStringBufferWithSpace(size);
        accumulator->serialize(rawBuffer);
        flatResult->setNoCopy(i, StringView(rawBuffer, size));
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  void destroyInternal(folly::Range<char**> groups) override {
    for (auto* group : groups) {
      if (!isNull(group)) {
        value<SetDigestAccumulator>(group)->~SetDigestAccumulator();
      }
    }
  }

 private:
  DecodedVector decodedValue_;
  DecodedVector decodedIntermediate_;
};

template <typename T>
std::unique_ptr<exec::Aggregate> createSetDigestAggregate(
    const TypePtr& resultType) {
  return std::make_unique<SetDigestAggregate<T>>(resultType);
}

exec::AggregateRegistrationResult registerMakeSetDigest(
    const std::string& name,
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
      name,
      std::move(signatures),
      [](core::AggregationNode::Step /*step*/,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        auto inputType = argTypes[0];

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
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMakeSetDigest(
      prefix + kMakeSetDigest, withCompanionFunctions, overwrite);
}

class MergeSetDigestAggregate : public exec::Aggregate {
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

    rows.applyToSelected([&](auto row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedValue_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
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

    rows.applyToSelected([&](auto row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedValue_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
      accumulator->mergeWith(other);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    rows.applyToSelected([&](auto row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      auto* accumulator = value<SetDigestAccumulator>(groups[row]);
      clearNull(groups[row]);

      auto tracker = trackRowSize(groups[row]);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
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

    rows.applyToSelected([&](auto row) {
      if (decodedIntermediate_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedIntermediate_.valueAt<StringView>(row);

      SetDigestAccumulator other(allocator_);
      other.deserialize(serialized.data(), serialized.size());
      accumulator->mergeWith(other);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* flatResult = (*result)->as<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        flatResult->setNull(i, true);
      } else {
        auto* accumulator = value<SetDigestAccumulator>(group);
        auto size = accumulator->estimatedSerializedSize();

        auto* rawBuffer = flatResult->getRawStringBufferWithSpace(size);
        accumulator->serialize(rawBuffer);
        flatResult->setNoCopy(i, StringView(rawBuffer, size));
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  void destroyInternal(folly::Range<char**> groups) override {
    for (auto* group : groups) {
      if (!isNull(group)) {
        value<SetDigestAccumulator>(group)->~SetDigestAccumulator();
      }
    }
  }

 private:
  DecodedVector decodedValue_;
  DecodedVector decodedIntermediate_;
};

exec::AggregateRegistrationResult registerMergeSetDigest(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .returnType("varbinary")
          .intermediateType("varbinary")
          .argumentType("varbinary")
          .build());

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
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
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMergeSetDigest(
      prefix + kMergeSetDigest, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
