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

#include "velox/functions/prestosql/aggregates/ApproxMostFrequentAggregate.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/exec/Strings.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/ApproxMostFrequentStreamSummary.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

template <typename T, typename A = AlignedStlAllocator<T, 16>>
using Summary = functions::ApproxMostFrequentStreamSummary<T, A>;

template <typename T>
struct Accumulator {
  Summary<T> summary;

  explicit Accumulator(HashStringAllocator* allocator)
      : summary(AlignedStlAllocator<T, 16>(allocator)) {}

  void insert(T value, int64_t count) {
    summary.insert(value, count);
  }
};

template <>
struct Accumulator<StringView> {
  Summary<StringView> summary;
  HashStringAllocator* allocator;
  Strings strings;
  size_t activeBytes_{0};
  size_t evictedBytes_{0};

  explicit Accumulator(HashStringAllocator* _allocator)
      : summary(AlignedStlAllocator<StringView, 16>(_allocator)),
        allocator(_allocator) {}

  ~Accumulator() {
    strings.free(*allocator);
  }

  void insert(
      StringView value,
      int64_t count,
      uint64_t compactionBytesThreshold = 0,
      double compactionUnusedMemoryRatio = 0) {
    if (!value.isInline() && !summary.contains(value)) {
      activeBytes_ += value.size();
      value = strings.append(value, *allocator);
    }

    auto evicted = summary.insert(value, count);
    if (evicted.has_value() && !evicted->isInline()) {
      const auto evictedSize = evicted->size();
      evictedBytes_ += evictedSize;
      VELOX_CHECK_GE(activeBytes_, evictedSize);
      activeBytes_ -= evictedSize;

      // Trigger compaction to reclaim memory from evicted strings.
      // Compact only when:
      // 1. Total string storage exceeds compactionBytesThreshold, AND
      // 2. Evicted bytes exceed compactionUnusedMemoryRatio of the threshold.
      if (compactionBytesThreshold > 0) {
        if (FOLLY_UNLIKELY(
                (evictedBytes_ + activeBytes_ > compactionBytesThreshold) &&
                (evictedBytes_ >
                 compactionBytesThreshold * compactionUnusedMemoryRatio))) {
          compact();
        }
      }
    }
  }

  uint64_t compact() {
    if (summary.size() == 0 || evictedBytes_ == 0) {
      return 0;
    }

    const auto bytesFreed = evictedBytes_;
    const auto capacity = summary.capacity();
    const auto currentSize = summary.size();

    Strings compactedStrings;
    Summary<StringView> compactedSummary{
        AlignedStlAllocator<StringView, 16>{allocator}};
    compactedSummary.setCapacity(capacity);

    uint64_t compactedActiveBytes = 0;
    for (auto i = 0; i < currentSize; ++i) {
      StringView v = summary.values()[i];
      int64_t cnt = summary.counts()[i];
      if (!v.isInline()) {
        compactedActiveBytes += v.size();
        v = compactedStrings.append(v, *allocator);
      }
      compactedSummary.insert(v, cnt);
    }

    VELOX_CHECK_EQ(compactedActiveBytes, activeBytes_);

    // Free old storage.
    strings.free(*allocator);

    strings = std::move(compactedStrings);
    summary = std::move(compactedSummary);
    evictedBytes_ = 0;
    return bytesFreed;
  }
};

template <typename T>
struct ApproxMostFrequentAggregate : exec::Aggregate {
  explicit ApproxMostFrequentAggregate(
      const TypePtr& resultType,
      uint64_t compactionBytesThreshold,
      double compactionUnusedMemoryRatio)
      : Aggregate(resultType),
        compactionBytesThreshold_(compactionBytesThreshold),
        compactionUnusedMemoryRatio_(compactionUnusedMemoryRatio) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(Accumulator<T>);
  }

  bool isFixedSize() const override {
    return false;
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool) override {
    decodeArguments(rows, args);
    rows.applyToSelected([&](auto row) {
      if (!decodedValues_.isNullAt(row)) {
        auto* accumulator = initAccumulator(groups[row]);
        auto tracker = trackRowSize(groups[row]);
        accumulator->insert(decodedValues_.valueAt<T>(row), 1);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool) override {
    addIntermediate<false>(groups, rows, args);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool) override {
    decodeArguments(rows, args);
    auto* accumulator = initAccumulator(group);
    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](auto row) {
      if (!decodedValues_.isNullAt(row)) {
        if constexpr (std::is_same_v<T, StringView>) {
          accumulator->insert(
              decodedValues_.valueAt<T>(row),
              1,
              compactionBytesThreshold_,
              compactionUnusedMemoryRatio_);
        } else {
          accumulator->insert(decodedValues_.valueAt<T>(row), 1);
        }
      }
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool) override {
    addIntermediate<true>(group, rows, args);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    (*result)->resize(numGroups);
    if (buckets_ == kMissingArgument) {
      // No data has been added.
      for (int i = 0; i < numGroups; ++i) {
        VELOX_DCHECK_EQ(value<Accumulator<T>>(groups[i])->summary.size(), 0);
        (*result)->setNull(i, true);
      }
      return;
    }
    VELOX_USER_CHECK_LE(buckets_, std::numeric_limits<int>::max());
    auto mapVector = (*result)->as<MapVector>();
    auto [keys, values] = prepareFinalResult(groups, numGroups, mapVector);
    vector_size_t entryCount = 0;
    for (int i = 0; i < numGroups; ++i) {
      auto* summary = &value<Accumulator<T>>(groups[i])->summary;
      const int size = std::min<int>(buckets_, summary->size());
      if (size == 0) {
        mapVector->setNull(i, true);
      } else {
        summary->topK(
            buckets_,
            keys->mutableRawValues() + entryCount,
            values->mutableRawValues() + entryCount);
        if constexpr (std::is_same_v<T, StringView>) {
          // Populate the string buffers.
          for (int j = 0; j < size; ++j) {
            keys->set(entryCount + j, keys->valueAtFast(entryCount + j));
          }
        }
        entryCount += size;
      }
      mapVector->setOffsetAndSize(i, entryCount - size, size);
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* rowVec = (*result)->as<RowVector>();
    VELOX_CHECK_NOT_NULL(rowVec);
    rowVec->childAt(0) = std::make_shared<ConstantVector<int64_t>>(
        rowVec->pool(),
        numGroups,
        false,
        BIGINT(),
        static_cast<int64_t&&>(buckets_));
    rowVec->childAt(1) = std::make_shared<ConstantVector<int64_t>>(
        rowVec->pool(),
        numGroups,
        false,
        BIGINT(),
        static_cast<int64_t&&>(capacity_));
    auto* values = rowVec->childAt(2)->as<ArrayVector>();
    auto* counts = rowVec->childAt(3)->as<ArrayVector>();
    rowVec->resize(numGroups);
    values->resize(numGroups);
    counts->resize(numGroups);

    auto* v = values->elements()->template asFlatVector<T>();
    auto* c = counts->elements()->template asFlatVector<int64_t>();
    vector_size_t entryCount = 0;
    for (int i = 0; i < numGroups; ++i) {
      auto* accumulator = value<const Accumulator<T>>(groups[i]);
      entryCount += accumulator->summary.size();
    }
    v->resize(entryCount);
    c->resize(entryCount);
    v->resetNulls();
    c->resetNulls();

    entryCount = 0;
    for (int i = 0; i < numGroups; ++i) {
      auto* summary = &value<const Accumulator<T>>(groups[i])->summary;
      if (summary->size() == 0) {
        rowVec->setNull(i, true);
      } else {
        if constexpr (std::is_same_v<T, StringView>) {
          for (int j = 0; j < summary->size(); ++j) {
            v->set(entryCount + j, summary->values()[j]);
          }
        } else {
          memcpy(
              v->mutableRawValues() + entryCount,
              summary->values(),
              sizeof(T) * summary->size());
        }
        memcpy(
            c->mutableRawValues() + entryCount,
            summary->counts(),
            sizeof(int64_t) * summary->size());
        values->setOffsetAndSize(i, entryCount, summary->size());
        counts->setOffsetAndSize(i, entryCount, summary->size());
        entryCount += summary->size();
      }
    }
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      new (groups[index] + offset_) Accumulator<T>(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<Accumulator<T>>(groups);
  }

  bool supportsCompact() const override {
    return std::is_same_v<T, StringView>;
  }

  uint64_t compact(folly::Range<char**> groups) override {
    if constexpr (!std::is_same_v<T, StringView>) {
      return 0;
    } else {
      uint64_t freedBytes = 0;
      for (auto* group : groups) {
        if (isInitialized(group)) {
          freedBytes += value<Accumulator<T>>(group)->compact();
        }
      }
      return freedBytes;
    }
  }

 private:
  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_CHECK_EQ(args.size(), 3);
    DecodedVector decodedBuckets(*args[0], rows);
    decodedValues_.decode(*args[1], rows);
    DecodedVector decodedCapacity(*args[2], rows);
    setConstantArgument("Buckets", buckets_, decodedBuckets);
    setConstantArgument("Capacity", capacity_, decodedCapacity);
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
    VELOX_USER_CHECK(
        vec.isConstantMapping(),
        "{} argument must be constant for all input rows",
        name);
    setConstantArgument(name, val, vec.valueAt<int64_t>(0));
  }

  Accumulator<T>* initAccumulator(char* group) {
    auto accumulator = value<Accumulator<T>>(group);
    VELOX_USER_CHECK_LE(capacity_, std::numeric_limits<int>::max());
    accumulator->summary.setCapacity(capacity_);
    return accumulator;
  }

  template <bool kSingleGroup>
  void addIntermediate(
      std::conditional_t<kSingleGroup, char*, char**> group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_CHECK_EQ(args.size(), 1);
    DecodedVector decoded(*args[0], rows);
    auto rowVec = static_cast<const RowVector*>(decoded.base());
    auto* buckets = rowVec->childAt(0)->as<SimpleVector<int64_t>>();
    VELOX_CHECK_NOT_NULL(buckets);
    auto* capacity = rowVec->childAt(1)->as<SimpleVector<int64_t>>();
    VELOX_CHECK_NOT_NULL(capacity);
    auto* values = rowVec->childAt(2)->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(values);
    auto* counts = rowVec->childAt(3)->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(counts);

    auto* v = values->elements()->template asFlatVector<T>();
    VELOX_CHECK_NOT_NULL(v);
    auto* c = counts->elements()->template asFlatVector<int64_t>();
    VELOX_CHECK_NOT_NULL(c);

    Accumulator<T>* accumulator{nullptr};
    rows.applyToSelected([&](auto row) {
      if (decoded.isNullAt(row)) {
        return;
      }
      const int i = decoded.index(row);
      setConstantArgument("Buckets", buckets_, buckets->valueAt(i));
      setConstantArgument("Capacity", capacity_, capacity->valueAt(i));
      if constexpr (kSingleGroup) {
        if (accumulator == nullptr) {
          accumulator = initAccumulator(group);
        }
        auto tracker = trackRowSize(group);
        const auto size = values->sizeAt(i);
        VELOX_DCHECK_EQ(counts->sizeAt(i), size);
        const auto vo = values->offsetAt(i);
        const auto co = counts->offsetAt(i);
        if constexpr (std::is_same_v<T, StringView>) {
          for (int j = 0; j < size; ++j) {
            accumulator->insert(
                v->valueAt(vo + j),
                c->valueAt(co + j),
                compactionBytesThreshold_,
                compactionUnusedMemoryRatio_);
          }
        } else {
          for (int j = 0; j < size; ++j) {
            accumulator->insert(v->valueAt(vo + j), c->valueAt(co + j));
          }
        }
      } else {
        accumulator = initAccumulator(group[row]);
        auto tracker = trackRowSize(group[row]);
        const auto size = values->sizeAt(i);
        VELOX_DCHECK_EQ(counts->sizeAt(i), size);
        const auto vo = values->offsetAt(i);
        const auto co = counts->offsetAt(i);
        for (int j = 0; j < size; ++j) {
          accumulator->insert(v->valueAt(vo + j), c->valueAt(co + j));
        }
      }
    });
  }

  std::pair<FlatVector<T>*, FlatVector<int64_t>*>
  prepareFinalResult(char** groups, int32_t numGroups, MapVector* result) {
    VELOX_CHECK_NOT_NULL(result);
    auto* keys = result->mapKeys()->asUnchecked<FlatVector<T>>();
    VELOX_CHECK_NOT_NULL(keys);
    auto* values = result->mapValues()->asUnchecked<FlatVector<int64_t>>();
    VELOX_CHECK_NOT_NULL(values);
    vector_size_t entryCount = 0;
    for (int i = 0; i < numGroups; ++i) {
      auto* summary = &value<const Accumulator<T>>(groups[i])->summary;
      entryCount += std::min<int>(buckets_, summary->size());
    }
    keys->resize(entryCount);
    values->resize(entryCount);
    return std::make_pair(keys, values);
  }

  static constexpr int64_t kMissingArgument{-1};

  // NOTE: compaction is currently only applied for global aggregation
  // addSingleGroupRawInput with StringView type
  const uint64_t compactionBytesThreshold_;
  const double compactionUnusedMemoryRatio_;

  DecodedVector decodedValues_;
  int64_t buckets_{kMissingArgument};
  int64_t capacity_{kMissingArgument};
};

class ApproxMostFrequentBooleanAggregate {
 public:
  using InputType =
      Row</*buckets*/ int64_t, /*value*/ bool, /*capacity*/ int64_t>;

  using IntermediateType = Row</*buckets*/ int64_t,
                               /*capacity*/ int64_t,
                               /*values*/ Array<bool>,
                               /*counts*/ Array<int64_t>>;

  using OutputType = Map<bool, int64_t>;

  static bool toIntermediate(
      exec::out_type<IntermediateType>& out,
      int64_t buckets,
      bool value,
      int64_t capacity) {
    out.get_writer_at<0>() = buckets;
    out.get_writer_at<1>() = capacity;

    auto& valuesWriter = out.get_writer_at<2>();
    valuesWriter.add_item() = true;
    valuesWriter.add_item() = false;

    auto& countsWriter = out.get_writer_at<3>();
    countsWriter.add_item() = value ? 1 : 0;
    countsWriter.add_item() = value ? 0 : 1;

    return true;
  }

  struct AccumulatorType {
    int64_t numTrue{0};
    int64_t numFalse{0};

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        ApproxMostFrequentBooleanAggregate* /*fn*/) {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        int64_t /*buckets*/,
        bool value,
        int64_t /*capacity*/) {
      if (value) {
        ++numTrue;
      } else {
        ++numFalse;
      }
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<IntermediateType> other) {
      VELOX_CHECK(other.at<2>().has_value());
      VELOX_CHECK(other.at<3>().has_value());

      const auto& values = *other.at<2>();
      VELOX_CHECK_EQ(2, values.size());

      VELOX_CHECK_EQ(values[0].value(), true);
      VELOX_CHECK_EQ(values[1].value(), false);

      const auto& counts = *other.at<3>();
      VELOX_CHECK_EQ(2, counts.size());

      numTrue += counts[0].value();
      numFalse += counts[1].value();
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      if (numTrue > 0) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter = true;
        valueWriter = numTrue;
      }

      if (numFalse > 0) {
        auto [keyWriter, valueWriter] = out.add_item();
        keyWriter = false;
        valueWriter = numFalse;
      }

      return true;
    }

    bool writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      // Write some hard-coded values for 'buckets' and 'capacity'. These are
      // not used.
      out.get_writer_at<0>() = 2;
      out.get_writer_at<1>() = 2;

      auto& valuesWriter = out.get_writer_at<2>();
      valuesWriter.add_item() = true;
      valuesWriter.add_item() = false;

      auto& countsWriter = out.get_writer_at<3>();
      countsWriter.add_item() = numTrue;
      countsWriter.add_item() = numFalse;

      return true;
    }
  };
};

template <TypeKind kKind>
std::unique_ptr<exec::Aggregate> makeApproxMostFrequentAggregate(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    const TypePtr& valueType,
    uint64_t compactionBytesThreshold,
    double compactionUnusedMemoryRatio) {
  if constexpr (
      kKind == TypeKind::TINYINT || kKind == TypeKind::SMALLINT ||
      kKind == TypeKind::INTEGER || kKind == TypeKind::BIGINT ||
      kKind == TypeKind::VARCHAR) {
    return std::make_unique<
        ApproxMostFrequentAggregate<typename TypeTraits<kKind>::NativeType>>(
        resultType, compactionBytesThreshold, compactionUnusedMemoryRatio);
  }

  if (kKind == TypeKind::BOOLEAN) {
    return std::make_unique<
        exec::SimpleAggregateAdapter<ApproxMostFrequentBooleanAggregate>>(
        step, argTypes, resultType);
  }

  VELOX_USER_FAIL(
      "Unsupported value type for {} aggregation {}",
      name,
      valueType->toString());
}

} // namespace

void registerApproxMostFrequentAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  for (const auto& valueType :
       {"boolean",
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "varchar",
        "json"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(fmt::format("map({},bigint)", valueType))
            .intermediateType(
                fmt::format(
                    "row(bigint, bigint, array({}), array(bigint))", valueType))
            .argumentType("bigint")
            .argumentType(valueType)
            .argumentType("bigint")
            .build());
  }
  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [names](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        auto& valueType = exec::isPartialOutput(step)
            ? resultType->childAt(2)->childAt(0)
            : resultType->childAt(0);
        return VELOX_DYNAMIC_TYPE_DISPATCH(
            makeApproxMostFrequentAggregate,
            valueType->kind(),
            names.front(),
            step,
            argTypes,
            resultType,
            valueType,
            config.aggregationCompactionBytesThreshold(),
            config.aggregationCompactionUnusedMemoryRatio());
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
