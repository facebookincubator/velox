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
#include "velox/dwio/nimble/writer/FieldWriter.h"

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include <folly/system/HardwareConcurrency.h>
#include "velox/common/base/CompareFlags.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/writer/DeduplicationUtils.h"
#include "velox/dwio/nimble/writer/RawSizeUtils.h"
#include "velox/dwio/nimble/writer/VectorAdapters.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::nimble {
namespace {

template <velox::TypeKind KIND>
struct NimbleTypeTraits {};

template <>
struct NimbleTypeTraits<velox::TypeKind::BOOLEAN> {
  static constexpr ScalarKind scalarKind = ScalarKind::Bool;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::TINYINT> {
  static constexpr ScalarKind scalarKind = ScalarKind::Int8;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::SMALLINT> {
  static constexpr ScalarKind scalarKind = ScalarKind::Int16;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::INTEGER> {
  static constexpr ScalarKind scalarKind = ScalarKind::Int32;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::BIGINT> {
  static constexpr ScalarKind scalarKind = ScalarKind::Int64;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::REAL> {
  static constexpr ScalarKind scalarKind = ScalarKind::Float;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::DOUBLE> {
  static constexpr ScalarKind scalarKind = ScalarKind::Double;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::VARCHAR> {
  static constexpr ScalarKind scalarKind = ScalarKind::String;
};

template <>
struct NimbleTypeTraits<velox::TypeKind::VARBINARY> {
  static constexpr ScalarKind scalarKind = ScalarKind::Binary;
};

constexpr uint64_t kTimestampLogicalSize = 12;

template <bool addNulls, typename Vector, typename Consumer, typename IndexOp>
uint64_t iterateNonNulls(
    const OrderedRanges& ranges,
    nimble::Vector<bool>& nonNulls,
    const Vector& vector,
    const Consumer& consumer,
    const IndexOp& indexOp) {
  uint64_t nonNullCount{0};
  if (vector.hasNulls()) {
    ranges.applyEach([&](auto offset) {
      const auto notNull = !vector.isNullAt(offset);
      if constexpr (addNulls) {
        nonNulls.push_back(notNull);
      }
      if (notNull) {
        ++nonNullCount;
        consumer(indexOp(offset));
      }
    });
  } else {
    ranges.applyEach([&](auto offset) { consumer(indexOp(offset)); });
    nonNullCount = ranges.size();
  }
  return nonNullCount;
}

template <bool addNulls, typename Vector, typename Op>
uint64_t iterateNonNullIndices(
    const OrderedRanges& ranges,
    nimble::Vector<bool>& nonNulls,
    const Vector& vector,
    const Op& op) {
  return iterateNonNulls<addNulls>(
      ranges, nonNulls, vector, op, [&](auto offset) {
        return vector.index(offset);
      });
}

template <typename Vector, typename Op>
uint64_t iterateNonNullValues(
    const OrderedRanges& ranges,
    nimble::Vector<bool>& nonNulls,
    const Vector& vector,
    const Op& op) {
  return iterateNonNulls<true>(ranges, nonNulls, vector, op, [&](auto offset) {
    return vector.valueAt(offset);
  });
}

template <typename T>
bool equalDecodedVectorIndices(
    const velox::DecodedVector& vec,
    velox::vector_size_t index,
    velox::vector_size_t otherIndex) {
  bool otherNull = vec.isNullAt(otherIndex);
  bool thisNull = vec.isNullAt(index);

  if (thisNull && otherNull) {
    return true;
  }

  if (thisNull || otherNull) {
    return false;
  }

  return vec.valueAt<T>(index) == vec.valueAt<T>(otherIndex);
}

template <typename T>
bool compareDecodedVectorToCache(
    const velox::DecodedVector& thisVec,
    velox::vector_size_t index,
    velox::FlatVector<T>* cachedFlatVec,
    velox::vector_size_t cacheIndex,
    velox::CompareFlags flags) {
  bool thisNull = thisVec.isNullAt(index);
  bool otherNull = cachedFlatVec->isNullAt(cacheIndex);

  if (thisNull && otherNull) {
    return true;
  }

  if (thisNull || otherNull) {
    return false;
  }

  return thisVec.valueAt<T>(index) == cachedFlatVec->valueAt(cacheIndex);
}

template <typename T>
std::string_view convert(const Vector<T>& input) {
  return {
      reinterpret_cast<const char*>(input.data()), input.size() * sizeof(T)};
}

template <typename T, typename = std::enable_if_t<std::is_trivial_v<T>>>
struct IdentityConverter {
  static T convert(T t, Buffer&, uint64_t&) {
    return t;
  }
};

struct StringConverter {
  static std::string_view
  convert(velox::StringView sv, Buffer& buffer, uint64_t& memoryUsed) {
    memoryUsed += sv.size();
    return buffer.writeString({sv.data(), sv.size()});
  }
};

namespace {
template <velox::TypeKind K>
void collectScalarTypeStats(
    StatisticsCollector* statsBuilder,
    std::span<typename velox::TypeTraits<K>::NativeType> values) {
  auto intStatsBuilder = statsBuilder->as<IntegralStatisticsCollector>();
  NIMBLE_CHECK_NOT_NULL(intStatsBuilder);
  intStatsBuilder->addValues(values);
}

template <>
void collectScalarTypeStats<velox::TypeKind::BOOLEAN>(
    StatisticsCollector* statsBuilder,
    std::span<bool> values) {
  statsBuilder->addValues(values);
}

template <>
void collectScalarTypeStats<velox::TypeKind::REAL>(
    StatisticsCollector* statsBuilder,
    std::span<float> values) {
  auto floatStatsBuilder = statsBuilder->as<FloatingPointStatisticsCollector>();
  NIMBLE_CHECK_NOT_NULL(floatStatsBuilder);
  floatStatsBuilder->addValues(values);
}

template <>
void collectScalarTypeStats<velox::TypeKind::DOUBLE>(
    StatisticsCollector* statsBuilder,
    std::span<double> values) {
  auto floatStatsBuilder = statsBuilder->as<FloatingPointStatisticsCollector>();
  NIMBLE_CHECK_NOT_NULL(floatStatsBuilder);
  floatStatsBuilder->addValues(values);
}

template <>
void collectScalarTypeStats<velox::TypeKind::VARCHAR>(
    StatisticsCollector* statsBuilder,
    std::span<velox::StringView> values) {
  NIMBLE_UNREACHABLE(
      "Wrong call site for string stats collection. Use collectStringTypeStats instead");
}

template <>
void collectScalarTypeStats<velox::TypeKind::VARBINARY>(
    StatisticsCollector* statsBuilder,
    std::span<velox::StringView> values) {
  NIMBLE_UNREACHABLE(
      "Wrong call site for string stats collection. Use collectStringTypeStats instead");
}

void collectStringTypeStats(
    StatisticsCollector* statsBuilder,
    std::span<std::string_view> values) {
  auto stringStatsBuilder = statsBuilder->as<StringStatisticsCollector>();
  NIMBLE_CHECK_NOT_NULL(stringStatsBuilder);
  stringStatsBuilder->addValues(values);
}
} // namespace

template <
    velox::TypeKind K,
    typename C = IdentityConverter<typename velox::TypeTraits<K>::NativeType>>
class SimpleFieldWriter : public FieldWriter {
  using SourceType = typename velox::TypeTraits<K>::NativeType;
  using TargetType = decltype(C::convert(
      SourceType(),
      std::declval<Buffer&>(),
      std::declval<uint64_t&>()));

 public:
  explicit SimpleFieldWriter(FieldWriterContext& context, uint32_t nodeId)
      : FieldWriter(
            context,
            context.schemaBuilder().createScalarTypeBuilder(
                NimbleTypeTraits<K>::scalarKind)),
        valuesStream_{context.createNullableContentStreamData<TargetType>(
            typeBuilder_->asScalar().scalarDescriptor(),
            nodeId)},
        statisticsCollector_{context.getStatsCollector(nodeId)} {}

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    auto size = ranges.size();
    auto& buffer = context_.stringBuffer();
    auto& data = valuesStream_.mutableData();
    uint64_t nullCount = 0;

    if (auto flat = vector->asFlatVector<SourceType>()) {
      valuesStream_.ensureAdditionalNullsCapacity(flat->mayHaveNulls(), size);
      bool rangeCopied = false;
      if (!flat->mayHaveNulls()) {
        if constexpr (
            std::is_same_v<C, IdentityConverter<SourceType, void>> &&
            K != velox::TypeKind::BOOLEAN) {
          uint64_t newSize = data.size() + size;
          valuesStream_.ensureMutableDataCapacity(newSize);
          ranges.apply([&](auto offset, auto count) {
            data.insert(
                data.end(),
                flat->rawValues() + offset,
                flat->rawValues() + offset + count);
          });
          rangeCopied = true;
        }
      }

      if (!rangeCopied) {
        auto nonNullCount = iterateNonNullValues(
            ranges,
            valuesStream_.mutableNonNulls(),
            FlatAdapter<SourceType>{vector},
            [&](SourceType value) {
              data.push_back(
                  C::convert(value, buffer, valuesStream_.extraMemory()));
            });
        nullCount = size - nonNullCount;
      }
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      valuesStream_.ensureAdditionalNullsCapacity(decoded.mayHaveNulls(), size);
      auto nonNullCount = iterateNonNullValues(
          ranges,
          valuesStream_.mutableNonNulls(),
          DecodedAdapter<SourceType>{decoded},
          [&](SourceType value) {
            data.push_back(
                C::convert(value, buffer, valuesStream_.extraMemory()));
          });
      nullCount = size - nonNullCount;
    }

    // NOTE: Since the stream chunker logic can compact the buffered value
    // streams and change its content we need to collect stats immediately after
    // writes.
    collectStatistics(nullCount, size);
  }

  void reset() override {
    valuesStream_.reset();
  }

 private:
  void collectStatistics(uint64_t nullCount, uint64_t valueCount) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(nullCount);

    const auto batchNonNullValueCount = valueCount - nullCount;
    if (batchNonNullValueCount == 0) {
      return;
    }

    auto totalNonNullCount = valuesStream_.mutableData().size();
    auto rangeStart = totalNonNullCount - batchNonNullValueCount;
    if (statisticsCollector_->shared()) {
      // TODO(T253295607): consider using converter for stats collection.
      auto sharedBuilder =
          statisticsCollector_->as<SharedStatisticsCollector>();
      sharedBuilder->updateBaseCollector([&](StatisticsCollector* builder) {
        if constexpr (
            K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
          collectStringTypeStats(
              builder,
              std::span(
                  valuesStream_.mutableData().data() + rangeStart,
                  batchNonNullValueCount));
        } else {
          collectScalarTypeStats<K>(
              builder,
              std::span<typename velox::TypeTraits<K>::NativeType>(
                  valuesStream_.mutableData().data() + rangeStart,
                  batchNonNullValueCount));
        }
      });
    } else {
      if constexpr (
          K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
        collectStringTypeStats(
            statisticsCollector_,
            std::span(
                valuesStream_.mutableData().data() + rangeStart,
                batchNonNullValueCount));
      } else {
        collectScalarTypeStats<K>(
            statisticsCollector_,
            std::span<typename velox::TypeTraits<K>::NativeType>(
                valuesStream_.mutableData().data() + rangeStart,
                batchNonNullValueCount));
      }
    }
  }

  NullableContentStreamData<TargetType>& valuesStream_;
  StatisticsCollector* statisticsCollector_;
};

template <velox::TypeKind K>
class StringFieldWriter : public FieldWriter {
 public:
  explicit StringFieldWriter(FieldWriterContext& context, uint32_t nodeId)
      : FieldWriter(
            context,
            context.schemaBuilder().createScalarTypeBuilder(
                NimbleTypeTraits<K>::scalarKind)),
        valuesStream_{context.createNullableContentStringStreamData(
            typeBuilder_->asScalar().scalarDescriptor(),
            nodeId)},
        statisticsCollector_{context.getStatsCollector(nodeId)} {
    static_assert(
        K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY,
        "StringFieldWriter only supports VARCHAR and VARBINARY types");
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    // Ensure string buffer capacity.
    auto size = ranges.size();
    const uint64_t totalBytes = getRawSizeFromVector(vector, ranges);
    valuesStream_.ensureStringBufferCapacity(size, totalBytes);

    auto stringBuffer = valuesStream_.mutableData();
    // Track the starting position in the buffer and lengths array before this
    // batch. We'll use these to build string_views after all strings are
    // copied.
    const size_t lengthsStartIdx = stringBuffer.lengths.size();
    const size_t bufferStartOffset = stringBuffer.buffer.size();

    auto appendToStringBuffer = [&](velox::StringView sv) {
      auto& buffer = stringBuffer.buffer;
      buffer.insert(buffer.end(), sv.begin(), sv.end());
      auto& mutableLengths = stringBuffer.lengths;
      mutableLengths.push_back(sv.size());
    };

    uint64_t nonNullCount = 0;
    if (auto flat = vector->asFlatVector<velox::StringView>()) {
      valuesStream_.ensureAdditionalNullsCapacity(flat->mayHaveNulls(), size);
      nonNullCount = iterateNonNullValues(
          ranges,
          valuesStream_.mutableNonNulls(),
          FlatAdapter<velox::StringView>{vector},
          appendToStringBuffer);
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      valuesStream_.ensureAdditionalNullsCapacity(decoded.mayHaveNulls(), size);
      nonNullCount = iterateNonNullValues(
          ranges,
          valuesStream_.mutableNonNulls(),
          DecodedAdapter<velox::StringView>{decoded},
          appendToStringBuffer);
    }

    // Build string_views from the copied data in the buffer.
    // This avoids holding pointers to velox::StringView inline storage which
    // may be destroyed when the lambda returns (stack-use-after-return).
    const size_t batchNonNullCount = nonNullCount;
    Vector<std::string_view> tempStringViews{context_.bufferMemoryPool().get()};
    tempStringViews.reserve(batchNonNullCount);
    size_t runningOffset = bufferStartOffset;
    for (size_t i = lengthsStartIdx; i < lengthsStartIdx + batchNonNullCount;
         ++i) {
      const size_t len = stringBuffer.lengths[i];
      tempStringViews.push_back(
          std::string_view(stringBuffer.buffer.data() + runningOffset, len));
      runningOffset += len;
    }

    uint64_t nullCount = size - nonNullCount;
    collectStatistics(nullCount, size, tempStringViews);
  }

  void reset() override {
    valuesStream_.reset();
  }

 private:
  void collectStatistics(
      uint64_t nullCount,
      uint64_t valueCount,
      std::span<std::string_view> values) {
    if (!statisticsCollector_) {
      return;
    }

    // bump the logical size by null count.
    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(nullCount);

    const auto batchNonNullValueCount = valueCount - nullCount;
    if (batchNonNullValueCount == 0) {
      return;
    }

    if (statisticsCollector_->shared()) {
      // TODO: looks like we can just use the same converter pattern.
      auto sharedBuilder =
          statisticsCollector_->as<SharedStatisticsCollector>();
      sharedBuilder->updateBaseCollector([&](StatisticsCollector* builder) {
        collectStringTypeStats(builder, values);
      });
    } else {
      collectStringTypeStats(statisticsCollector_, values);
    }
  }

  NullableContentStringStreamData& valuesStream_;
  StatisticsCollector* statisticsCollector_;
};

class TimestampFieldWriter : public FieldWriter {
 public:
  explicit TimestampFieldWriter(FieldWriterContext& context, uint32_t nodeId)
      : FieldWriter{context, context.schemaBuilder().createTimestampMicroNanoTypeBuilder()},
        microsStream_{context.createNullableContentStreamData<int64_t>(
            typeBuilder_->asTimestampMicroNano().microsDescriptor(),
            nodeId)},
        nanosStream_{context.createContentStreamData<uint16_t>(
            typeBuilder_->asTimestampMicroNano().nanosDescriptor(),
            nodeId)},
        statisticsCollector_{context.getStatsCollector(nodeId)} {}

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    auto size = ranges.size();
    Vector<int64_t>& microsData = microsStream_.mutableData();
    Vector<uint16_t>& nanosData = nanosStream_.mutableData();
    microsData.reserve(microsData.size() + size);
    nanosData.reserve(nanosData.size() + size);

    // - Velox stores time as:
    //     seconds + nanos (0 <= nanos < 1_000_000_000)
    // - Nimble stores time as:
    //     micros         -> whole microseconds since epoch
    //     subMicrosNanos -> extra nanoseconds inside the microsecond [0, 999]
    //
    // The conversion below uses __int128_t to avoid overflow during
    // intermediate computation, then checks if the result fits in int64_t.
    auto processTimestamp = [&](velox::Timestamp ts) {
      const auto seconds = ts.getSeconds();
      const auto nanos = ts.getNanos();
      __int128_t result = static_cast<__int128_t>(seconds) * 1'000'000 +
          static_cast<int64_t>(nanos / 1'000);
      if (result < std::numeric_limits<int64_t>::min() ||
          result > std::numeric_limits<int64_t>::max()) {
        NIMBLE_USER_FAIL(
            "Could not convert Timestamp({}, {}) to microseconds",
            seconds,
            nanos);
      }
      int64_t micros = static_cast<int64_t>(result);
      uint16_t subMicrosNanos = static_cast<uint16_t>(nanos % 1'000);
      microsData.push_back(micros);
      nanosData.push_back(subMicrosNanos);
    };

    uint64_t nonNullCount;
    if (auto flat = vector->asFlatVector<velox::Timestamp>()) {
      microsStream_.ensureAdditionalNullsCapacity(flat->mayHaveNulls(), size);
      nonNullCount = iterateNonNullValues(
          ranges,
          microsStream_.mutableNonNulls(),
          FlatAdapter<velox::Timestamp>{vector},
          processTimestamp);
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      microsStream_.ensureAdditionalNullsCapacity(decoded.mayHaveNulls(), size);
      nonNullCount = iterateNonNullValues(
          ranges,
          microsStream_.mutableNonNulls(),
          DecodedAdapter<velox::Timestamp>{decoded},
          processTimestamp);
    }

    const uint64_t nullCount = size - nonNullCount;
    collectStatistics(nullCount, size);
  }

  void reset() override {
    microsStream_.reset();
    nanosStream_.reset();
  }

 private:
  void collectStatistics(uint64_t nullCount, uint64_t valueCount) {
    constexpr uint64_t kTimestampLogicalSize = 12;
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(
        (valueCount - nullCount) * kTimestampLogicalSize + nullCount);
  }

  NullableContentStreamData<int64_t>& microsStream_;
  ContentStreamData<uint16_t>& nanosStream_;
  StatisticsCollector* statisticsCollector_;
};

class RowFieldWriter : public FieldWriter {
 public:
  RowFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : FieldWriter{context, context.schemaBuilder().createRowTypeBuilder(type->size())},
        nullsStream_{context_.createNullsStreamData(
            typeBuilder_->asRow().nullsDescriptor(),
            type->id())},
        statisticsCollector_{context.getStatsCollector(type->id())},
        ignoreNulls_{type->id() == 0 && context.ignoreTopLevelNulls()} {
    auto rowType =
        std::dynamic_pointer_cast<const velox::RowType>(type->type());

    fields_.reserve(rowType->size());
    for (auto i = 0; i < rowType->size(); ++i) {
      fields_.push_back(FieldWriter::create(context, type->childAt(i)));
      typeBuilder_->asRow().addChild(
          rowType->nameOf(i), fields_.back()->typeBuilder());
    }
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    if (parallelWriteEnabled(static_cast<uint32_t>(fields_.size()))) {
      folly::coro::blockingWait(co_write(vector, ranges));
      return;
    }
    auto ctx = prepareChildRanges(vector, ranges);
    for (auto i = 0; i < fields_.size(); ++i) {
      fields_[i]->write(ctx.row->childAt(i), *ctx.childRangesPtr);
    }
    collectStatistics(ctx.nullCount, ctx.size);
  }

  folly::coro::Task<void> co_write(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges) override {
    auto ctx = prepareChildRanges(vector, ranges);

    const uint32_t numChildren = static_cast<uint32_t>(fields_.size());
    const uint32_t taskCount = computeParallelWriteTaskCount(numChildren);
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::RowFieldWriter::co_write",
        const_cast<uint32_t*>(&taskCount));

    const uint32_t childrenPerTask = numChildren / taskCount;
    const uint32_t numRemainderChildren = numChildren % taskCount;

    auto writeRange = [this, &ctx](uint32_t startIdx, uint32_t endIdx) {
      for (uint32_t i = startIdx; i < endIdx; ++i) {
        fields_[i]->write(ctx.row->childAt(i), *ctx.childRangesPtr);
      }
    };

    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    tasks.reserve(taskCount);
    uint32_t nextChildIdx = 0;
    for (uint32_t taskId = 0; taskId < taskCount; ++taskId) {
      const uint32_t endChildIdx = nextChildIdx + childrenPerTask +
          (taskId < numRemainderChildren ? 1 : 0);
      tasks.emplace_back(co_withExecutor(
          context_.encodeExecutor(),
          folly::coro::co_invoke(
              [&writeRange, nextChildIdx, endChildIdx]()
                  -> folly::coro::Task<void> {
                writeRange(nextChildIdx, endChildIdx);
                co_return;
              })));
      nextChildIdx = endChildIdx;
    }

    co_await folly::coro::collectAllRange(std::move(tasks));
    collectStatistics(ctx.nullCount, ctx.size);
  }

  void reset() override {
    nullsStream_.reset();

    for (auto& field : fields_) {
      field->reset();
    }
  }

  void close() override {
    for (auto& field : fields_) {
      field->close();
    }
  }

 private:
  // Result of null processing and child range computation, shared between
  // the synchronous write() and coroutine co_write() paths.
  struct PrepareContext {
    // Decoded row vector (either direct cast or decoded from encoded input).
    const velox::RowVector* row{};
    // Ranges for child writes: points to either the original input ranges
    // (when no nulls) or the filtered childRanges below.
    const OrderedRanges* childRangesPtr{};
    uint64_t nullCount{};
    uint64_t size{};
    // Filtered ranges containing only non-null row indices.
    OrderedRanges childRanges;
  };

  PrepareContext prepareChildRanges(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges) {
    PrepareContext ctx;
    ctx.size = ranges.size();
    ctx.nullCount = 0;
    ctx.row = vector->as<velox::RowVector>();

    if (ctx.row) {
      NIMBLE_CHECK_LE(
          fields_.size(),
          ctx.row->childrenSize(),
          "Schema mismatch: expected {} fields, but got {} fields",
          fields_.size(),
          ctx.row->childrenSize());
      nullsStream_.ensureAdditionalNullsCapacity(
          vector->mayHaveNulls(), ctx.size);
      if (ctx.row->mayHaveNulls() && !ignoreNulls_) {
        ctx.childRangesPtr = &ctx.childRanges;
        auto nonNullCount = iterateNonNullIndices<true>(
            ranges,
            nullsStream_.mutableNonNulls(),
            FlatAdapter<>{vector},
            [&](auto offset) { ctx.childRanges.add(offset, 1); });
        ctx.nullCount = ctx.size - nonNullCount;
      } else {
        ctx.childRangesPtr = &ranges;
      }
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      ctx.row = decoded.base()->as<velox::RowVector>();
      NIMBLE_CHECK_NOT_NULL(ctx.row, "Unexpected vector type");
      NIMBLE_CHECK_LE(
          fields_.size(),
          ctx.row->childrenSize(),
          "Schema mismatch: expected {} fields, but got {} fields",
          fields_.size(),
          ctx.row->childrenSize());
      ctx.childRangesPtr = &ctx.childRanges;
      nullsStream_.ensureAdditionalNullsCapacity(
          decoded.mayHaveNulls(), ctx.size);
      auto nonNullCount = ignoreNulls_
          ? iterateNonNullIndices<false>(
                ranges,
                nullsStream_.mutableNonNulls(),
                DecodedAdapter<int8_t, true>{decoded},
                [&](auto offset) { ctx.childRanges.add(offset, 1); })
          : iterateNonNullIndices<true>(
                ranges,
                nullsStream_.mutableNonNulls(),
                DecodedAdapter{decoded},
                [&](auto offset) { ctx.childRanges.add(offset, 1); });
      ctx.nullCount = ctx.size - nonNullCount;
    }

    return ctx;
  }

  void collectStatistics(uint64_t nullCount, uint64_t valueCount) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(nullCount);
  }

  std::vector<std::unique_ptr<FieldWriter>> fields_;
  NullsStreamData& nullsStream_;
  StatisticsCollector* statisticsCollector_;
  bool ignoreNulls_;
};

class MultiValueFieldWriter : public FieldWriter {
 public:
  MultiValueFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
      std::shared_ptr<LengthsTypeBuilder> typeBuilder)
      : FieldWriter{context, std::move(typeBuilder)},
        lengthsStream_{context.createNullableContentStreamData<uint32_t>(
            static_cast<LengthsTypeBuilder&>(*typeBuilder_).lengthsDescriptor(),
            type->id())},
        statisticsCollector_{context.getStatsCollector(type->id())} {}

  void reset() override {
    lengthsStream_.reset();
  }

 protected:
  void collectStatistics(uint64_t nullCount, uint64_t valueCount) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(nullCount);
  }

  template <typename T>
  const T* ingestLengths(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges,
      OrderedRanges& childRanges) {
    auto size = ranges.size();
    const T* casted = vector->as<T>();
    const velox::vector_size_t* offsets;
    const velox::vector_size_t* lengths;
    auto& data = lengthsStream_.mutableData();

    auto proc = [&](velox::vector_size_t index) {
      auto length = lengths[index];
      if (length > 0) {
        childRanges.add(offsets[index], length);
      }
      data.push_back(length);
    };

    uint64_t nullCount = 0;
    if (casted) {
      offsets = casted->rawOffsets();
      lengths = casted->rawSizes();

      lengthsStream_.ensureAdditionalNullsCapacity(
          casted->mayHaveNulls(), size);
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges,
          lengthsStream_.mutableNonNulls(),
          FlatAdapter<>{vector},
          proc);
      nullCount = size - nonNullCount;
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      casted = decoded.base()->as<T>();
      NIMBLE_CHECK_NOT_NULL(casted, "Unexpected vector type");
      offsets = casted->rawOffsets();
      lengths = casted->rawSizes();

      lengthsStream_.ensureAdditionalNullsCapacity(
          decoded.mayHaveNulls(), size);
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges,
          lengthsStream_.mutableNonNulls(),
          DecodedAdapter<>{decoded},
          proc);
      nullCount = size - nonNullCount;
    }

    collectStatistics(nullCount, size);

    return casted;
  }

  NullableContentStreamData<uint32_t>& lengthsStream_;
  StatisticsCollector* statisticsCollector_;
};

class ArrayFieldWriter : public MultiValueFieldWriter {
 public:
  ArrayFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : MultiValueFieldWriter{
            context,
            type,
            context.schemaBuilder().createArrayTypeBuilder()} {
    auto arrayType =
        std::dynamic_pointer_cast<const velox::ArrayType>(type->type());

    NIMBLE_DCHECK_EQ(type->size(), 1, "Invalid array type.");
    elements_ = FieldWriter::create(context, type->childAt(0));

    typeBuilder_->asArray().setChildren(elements_->typeBuilder());
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    OrderedRanges childRanges;
    auto array = ingestLengths<velox::ArrayVector>(vector, ranges, childRanges);
    if (childRanges.size() > 0) {
      elements_->write(array->elements(), childRanges);
    }
  }

  void reset() override {
    MultiValueFieldWriter::reset();
    elements_->reset();
  }

  void close() override {
    elements_->close();
  }

 private:
  std::unique_ptr<FieldWriter> elements_;
};

class MapFieldWriter : public MultiValueFieldWriter {
 public:
  MapFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : MultiValueFieldWriter{context, type, context.schemaBuilder().createMapTypeBuilder()},
        statisticsCollector_{context.getStatsCollector(type->id())} {
    auto mapType =
        std::dynamic_pointer_cast<const velox::MapType>(type->type());

    NIMBLE_DCHECK_EQ(type->size(), 2, "Invalid map type.");
    keys_ = FieldWriter::create(context, type->childAt(0));
    values_ = FieldWriter::create(context, type->childAt(1));
    typeBuilder_->asMap().setChildren(
        keys_->typeBuilder(), values_->typeBuilder());
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    OrderedRanges childRanges;
    auto map = ingestLengths<velox::MapVector>(vector, ranges, childRanges);
    if (childRanges.size() > 0) {
      keys_->write(map->mapKeys(), childRanges);
      values_->write(map->mapValues(), childRanges);
    }
  }

  void reset() override {
    MultiValueFieldWriter::reset();
    keys_->reset();
    values_->reset();
  }

  void close() override {
    keys_->close();
    values_->close();
  }

 private:
  std::unique_ptr<FieldWriter> keys_;
  std::unique_ptr<FieldWriter> values_;
  StatisticsCollector* statisticsCollector_;
};

class SlidingWindowMapFieldWriter : public FieldWriter {
 public:
  SlidingWindowMapFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : FieldWriter{context, context.schemaBuilder().createSlidingWindowMapTypeBuilder()},
        offsetsStream_{context.createNullableContentStreamData<uint32_t>(
            typeBuilder_->asSlidingWindowMap().offsetsDescriptor(),
            type->id())},
        lengthsStream_{context.createContentStreamData<uint32_t>(
            typeBuilder_->asSlidingWindowMap().lengthsDescriptor(),
            type->id())},
        currentOffset_(0),
        cached_{false},
        cachedLength_{0},
        statisticsCollector_{context.getStatsCollector(type->id())},
        keyStatisticsCollector_{
            context.getStatsCollector(type->childAt(0)->id())},
        valueStatisticsCollector_{
            context.getStatsCollector(type->childAt(1)->id())} {
    NIMBLE_DCHECK_EQ(type->size(), 2, "Invalid map type.");
    keys_ = FieldWriter::create(context, type->childAt(0));
    values_ = FieldWriter::create(context, type->childAt(1));
    typeBuilder_->asSlidingWindowMap().setChildren(
        keys_->typeBuilder(), values_->typeBuilder());
    cachedValue_ = velox::MapVector::create(
        type->type(), 1, context.bufferMemoryPool().get());
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    OrderedRanges childFilteredRanges;
    OrderedRanges fullChildRanges;
    // childItemCount is the (logical) total cardinality of map elements,
    // and is thus the total cardinality for both the key and value subcolumn.
    uint64_t childItemCount = 0;
    uint64_t keyNullCount = 0;
    uint64_t valueNullCount = 0;
    auto map = ingestOffsetsAndLengthsDeduplicated(
        vector,
        ranges,
        childFilteredRanges,
        fullChildRanges,
        childItemCount,
        keyNullCount,
        valueNullCount);
    if (childFilteredRanges.size() > 0) {
      keys_->write(map->mapKeys(), childFilteredRanges);
      values_->write(map->mapValues(), childFilteredRanges);
    }
    // Adjust child stats to account for deduplicated elements.
    // Child writers only see the deduplicated ranges, so their valueCount
    // is the deduplicated count. We need to add the difference to get
    // the full (non-deduplicated) valueCount. Similarly for null counts.
    auto dedupedChildItemCount = childFilteredRanges.size();
    if (childItemCount > dedupedChildItemCount) {
      // Count nulls in deduplicated child elements.
      uint64_t dedupedKeyNullCount = 0;
      uint64_t dedupedValueNullCount = 0;
      auto* mapKeys = map->mapKeys().get();
      auto* mapValues = map->mapValues().get();
      if (mapKeys->mayHaveNulls()) {
        const auto* keyNulls = mapKeys->rawNulls();
        if (keyNulls != nullptr) {
          childFilteredRanges.apply([&](auto offset, auto count) {
            dedupedKeyNullCount +=
                velox::bits::countNulls(keyNulls, offset, offset + count);
          });
        }
      }
      if (mapValues->mayHaveNulls()) {
        const auto* valueNulls = mapValues->rawNulls();
        if (valueNulls != nullptr) {
          childFilteredRanges.apply([&](auto offset, auto count) {
            dedupedValueNullCount +=
                velox::bits::countNulls(valueNulls, offset, offset + count);
          });
        }
      }
      auto additionalItemCount = childItemCount - dedupedChildItemCount;
      auto additionalKeyNullCount = keyNullCount - dedupedKeyNullCount;
      auto additionalValueNullCount = valueNullCount - dedupedValueNullCount;
      // Calculate the raw size for the additional (duplicated) elements
      // using getRawSizeFromVector, which handles variable-width and complex
      // types correctly (unlike a simple typeSize * count formula).
      RawSizeContext rawSizeContext;
      auto keyRawSize =
          getRawSizeFromVector(map->mapKeys(), fullChildRanges, rawSizeContext);
      auto dedupedKeyRawSize = getRawSizeFromVector(
          map->mapKeys(), childFilteredRanges, rawSizeContext);
      auto additionalKeyRawSize = keyRawSize - dedupedKeyRawSize;
      auto valueRawSize = getRawSizeFromVector(
          map->mapValues(), fullChildRanges, rawSizeContext);
      auto dedupedValueRawSize = getRawSizeFromVector(
          map->mapValues(), childFilteredRanges, rawSizeContext);
      auto additionalValueRawSize = valueRawSize - dedupedValueRawSize;
      amendChildStatistics(
          additionalItemCount,
          additionalKeyNullCount,
          additionalKeyRawSize,
          additionalItemCount,
          additionalValueNullCount,
          additionalValueRawSize);
    }
  }

  void reset() override {
    offsetsStream_.reset();
    lengthsStream_.reset();
    keys_->reset();
    values_->reset();
    currentOffset_ = 0;
    cached_ = false;
    cachedLength_ = 0;
  }

  void close() override {
    keys_->close();
    values_->close();
  }

 private:
  // NOTE: When supporting deduplicated stats, we need to set the
  // immediate children nodes with deduplicated stats, and then rollup from
  // there. Deduplicated nodes will record their default stats as deduplicated.
  void collectStatistics(
      uint64_t nullCount,
      uint64_t valueCount,
      uint64_t logicalSize) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(logicalSize);
  }

  // Collects additional child element statistics for deduplicated elements.
  // This is needed because child writers only see deduplicated ranges, so
  // their valueCount would only reflect the deduplicated count. We need to
  // add the difference (childItemCount- dedupedChildCount) to get the
  // full valueCount. Similarly, we need to add the null count difference
  // and the logical size for these additional elements.
  void amendChildStatistics(
      uint64_t additionalKeyItemCount,
      uint64_t additionalKeyNullCount,
      uint64_t additionalKeyRawSize,
      uint64_t additionalValueItemCount,
      uint64_t additionalValueNullCount,
      uint64_t additionalValueRawSize) {
    if (keyStatisticsCollector_) {
      keyStatisticsCollector_->addCounts(
          additionalKeyItemCount, additionalKeyNullCount);
      keyStatisticsCollector_->addLogicalSize(additionalKeyRawSize);
    }
    if (valueStatisticsCollector_) {
      valueStatisticsCollector_->addCounts(
          additionalValueItemCount, additionalValueNullCount);
      valueStatisticsCollector_->addLogicalSize(additionalValueRawSize);
    }
  }

  std::unique_ptr<FieldWriter> keys_;
  std::unique_ptr<FieldWriter> values_;
  NullableContentStreamData<uint32_t>& offsetsStream_;
  ContentStreamData<uint32_t>& lengthsStream_;
  uint32_t currentOffset_; /* Global Offset for the data */
  bool cached_;
  velox::vector_size_t cachedLength_;
  velox::VectorPtr cachedValue_;
  StatisticsCollector* statisticsCollector_;
  StatisticsCollector* keyStatisticsCollector_;
  StatisticsCollector* valueStatisticsCollector_;

  const velox::MapVector* ingestOffsetsAndLengthsDeduplicated(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges,
      OrderedRanges& filteredRanges,
      OrderedRanges& fullChildRanges,
      uint64_t& childItemCount,
      uint64_t& keyNullCount,
      uint64_t& valueNullCount) {
    const auto size = ranges.size();
    const velox::MapVector* mapVector = vector->as<velox::MapVector>();
    const velox::vector_size_t* rawOffsets;
    const velox::vector_size_t* rawLengths;
    velox::vector_size_t lastCompareIndex = -1;
    auto& offsetsData = offsetsStream_.mutableData();
    auto& lengthsData = lengthsStream_.mutableData();
    childItemCount = 0;
    keyNullCount = 0;
    valueNullCount = 0;

    // Get child null pointers for null counting.
    const uint64_t* keyNulls = nullptr;
    const uint64_t* valueNulls = nullptr;

    auto processMapIndex = [&](velox::vector_size_t index) {
      auto const length = rawLengths[index];
      // Track total child count for all map entries (including duplicates).
      childItemCount += length;
      // Track null counts and full child ranges for all elements.
      if (length > 0) {
        auto childOffset = rawOffsets[index];
        fullChildRanges.add(childOffset, length);
        if (keyNulls != nullptr) {
          keyNullCount += velox::bits::countNulls(
              keyNulls, childOffset, childOffset + length);
        }
        if (valueNulls != nullptr) {
          valueNullCount += velox::bits::countNulls(
              valueNulls, childOffset, childOffset + length);
        }
      }

      bool match = false;
      // Compare with the last element if not the first elemment
      if (lastCompareIndex >= 0) {
        match = length == rawLengths[lastCompareIndex];
        if (length > 0) {
          match = match &&
              DeduplicationUtils::CompareMapsAtIndex(
                      *mapVector, index, *mapVector, lastCompareIndex);
        }
        // If this is the first element, compare with the cached element from
        // the last batch
      } else if (cached_) {
        match =
            (length == cachedLength_ &&
             DeduplicationUtils::CompareMapsAtIndex(
                 *static_cast<velox::MapVector*>(cachedValue_.get()),
                 0,
                 *mapVector,
                 index));
      }

      if (!match && length > 0) {
        filteredRanges.add(rawOffsets[index], length);
        currentOffset_ += length;
      }
      lengthsData.push_back(length);
      offsetsData.push_back(currentOffset_ - length);
      lastCompareIndex = index;
    };

    uint64_t nullCount = 0;
    if (mapVector != nullptr) {
      rawOffsets = mapVector->rawOffsets();
      rawLengths = mapVector->rawSizes();
      // Initialize key/value null pointers for child null counting.
      auto* keysVector = mapVector->mapKeys().get();
      auto* valuesVector = mapVector->mapValues().get();
      keyNulls = keysVector->rawNulls();
      valueNulls = valuesVector->rawNulls();
      offsetsStream_.ensureAdditionalNullsCapacity(
          mapVector->mayHaveNulls(), size);
      FlatAdapter<> iterableVector{vector};
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges,
          offsetsStream_.mutableNonNulls(),
          iterableVector,
          processMapIndex);
      nullCount = size - nonNullCount;
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      mapVector = decoded.base()->template as<velox::MapVector>();
      NIMBLE_CHECK_NOT_NULL(mapVector, "Unexpected vector type");
      rawOffsets = mapVector->rawOffsets();
      rawLengths = mapVector->rawSizes();
      // Initialize key/value null pointers for child null counting.
      auto* keysVector = mapVector->mapKeys().get();
      auto* valuesVector = mapVector->mapValues().get();
      keyNulls = keysVector->rawNulls();
      valueNulls = valuesVector->rawNulls();
      offsetsStream_.ensureAdditionalNullsCapacity(
          decoded.mayHaveNulls(), size);
      DecodedAdapter<> iterableVector{decoded};
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges,
          offsetsStream_.mutableNonNulls(),
          iterableVector,
          processMapIndex);
      nullCount = size - nonNullCount;
    }

    // Copy the last valid element into the cache.
    const velox::vector_size_t idxOfLastElement = lastCompareIndex;
    if (idxOfLastElement != -1) {
      cached_ = true;
      cachedLength_ = rawLengths[idxOfLastElement];
      cachedValue_->prepareForReuse();
      velox::BaseVector::CopyRange cacheRange{
          idxOfLastElement /* source index*/,
          0 /* target index*/,
          1 /* count*/};
      cachedValue_->copyRanges(mapVector, folly::Range(&cacheRange, 1));
    }

    // Calculate non-deduplicated logical size.
    {
      RawSizeContext context;
      // This code currently uses getRawSizeFromVector to calculate the
      // non-deduplicated logical size. In the future, we should replace this
      // with a more efficient approach that passes OrderedRanges objects to
      // child FieldWriters, allowing them to directly calculate
      // non-deduplicated sizes from deduplicated vectors without needing to
      // rely on an external util.
      collectStatistics(
          nullCount, size, getRawSizeFromVector(vector, ranges, context));
    }
    return mapVector;
  }
};

class FlatMapPassthroughValueFieldWriter {
 public:
  FlatMapPassthroughValueFieldWriter(
      FieldWriterContext& context,
      const StreamDescriptorBuilder& inMapDescriptor,
      std::unique_ptr<FieldWriter> valueField,
      uint32_t nodeId)
      : valueField_{std::move(valueField)},
        inMapStream_{
            context.createContentStreamData<bool>(inMapDescriptor, nodeId)} {}

  // Write without an explicit inMaps buffer; assume all inMap bits are set.
  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges) {
    auto& data = inMapStream_.mutableData();
    data.resize(data.size() + ranges.size(), true);
    writeImpl(vector, ranges);
  }

  // Write values based on an inMaps buffer. The range needs to be shrunk
  // according to the inMaps bits first.
  void write(
      const velox::VectorPtr& vector,
      const velox::BufferPtr& inMaps,
      const OrderedRanges& ranges) {
    const auto* rawInMaps = inMaps->as<uint64_t>();
    auto& data = inMapStream_.mutableData();
    childRanges_.clear();

    ranges.applyEach([&](auto offset) {
      if (velox::bits::isBitSet(rawInMaps, offset)) {
        data.push_back(true);
        childRanges_.add(offset, 1);
      } else {
        data.push_back(false);
      }
    });
    writeImpl(vector, childRanges_);
  }

  void reset() {
    inMapStream_.reset();
    // File stats collection is done sequentially onto
    // the same underlying object.
    // We will allocate additional stat builders per value writer
    // only when we have to support feature stats.
    valueField_->reset();
  }

  void close() {
    valueField_->close();
  }

 private:
  // TODO: need to properly implement logical stats
  void writeImpl(const velox::VectorPtr& vector, const OrderedRanges& ranges) {
    valueField_->write(vector, ranges);
  }

  std::unique_ptr<FieldWriter> valueField_;
  ContentStreamData<bool>& inMapStream_;

  // Range to reuse when writing valueFields based on input inMap buffers, so we
  // don't reallocated on each write() call.
  OrderedRanges childRanges_;
};

class FlatMapValueFieldWriter {
 public:
  FlatMapValueFieldWriter(
      FieldWriterContext& context,
      const StreamDescriptorBuilder& inMapDescriptor,
      std::unique_ptr<FieldWriter> valueField,
      uint32_t nodeId)
      : valueField_{std::move(valueField)},
        inMapStream_{
            context.createContentStreamData<bool>(inMapDescriptor, nodeId)} {}

  // Clear the ranges and extend the inMapBuffer
  void prepare(uint32_t numValues) {
    auto& data = inMapStream_.mutableData();
    inMapStream_.ensureMutableDataCapacity(data.size() + numValues);
    std::fill(data.end(), data.end() + numValues, false);
  }

  // Returns whether the offset is successfully recorded.
  //
  // NOTE: this method is always called after calling prepare(), so
  // we can access the raw in map stream data without size checks.
  bool add(velox::vector_size_t offset, uint32_t mapIndex) {
    auto& data = inMapStream_.mutableData();
    const auto index = mapIndex + data.size();
    // The index being already populated means we have a key duplication.
    // In order to avoid another branching here, we perform the rest of the
    // method regardless, knowning that the whole write will be aborted
    // upon key duplication, and the rest of the states wouldn't matter.
    const bool keyDuplicated = data[index];
    ranges_.add(offset, 1);
    data[index] = true;
    return !keyDuplicated;
  }

  void write(const velox::VectorPtr& vector, uint32_t mapCount) {
    auto& data = inMapStream_.mutableData();
    data.update_size(data.size() + mapCount);

    if (ranges_.size() > 0) {
      valueField_->write(vector, ranges_);
    }

    ranges_.clear();
  }

  void backfill(uint32_t count, uint32_t reserve) {
    inMapStream_.mutableData().resize(count, false);
    prepare(reserve);
  }

  void reset() {
    inMapStream_.reset();
    // File stats collection is done sequentially onto
    // the same underlying object.
    // We will allocate additional stat builders per value writer
    // only when we have to support feature stats.
    valueField_->reset();
  }

  void close() {
    valueField_->close();
  }

 private:
  std::unique_ptr<FieldWriter> valueField_;
  ContentStreamData<bool>& inMapStream_;
  OrderedRanges ranges_;
};

template <typename T>
std::string flatMapKeyToString(T val) {
  return std::to_string(val);
}

template <>
std::string flatMapKeyToString(velox::StringView val) {
  return val.getString();
}

template <velox::TypeKind K>
class FlatMapFieldWriter : public FieldWriter {
  using KeyType = typename velox::TypeTraits<K>::NativeType;

 public:
  FlatMapFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : FieldWriter(
            context,
            context.schemaBuilder().createFlatMapTypeBuilder(
                NimbleTypeTraits<K>::scalarKind)),
        valueType_{type->childAt(1)},
        nodeId_{type->id()},
        nullsStream_{context_.createNullsStreamData(
            typeBuilder_->asFlatMap().nullsDescriptor(),
            type->id())} {
    auto* statsBuilder = context.getStatsCollector(type->id());
    if (statsBuilder) {
      // Sanity check that the stats builders are shared and thread safe.
      statisticsCollector_ =
          statsBuilder->asChecked<SharedStatisticsCollector>();
      auto keyStatsBuilder = context.getStatsCollector(type->childAt(0)->id());
      keyStatisticsCollector_ =
          keyStatsBuilder->asChecked<SharedStatisticsCollector>();
      for (auto id = valueType_->id(); id <= valueType_->maxId(); ++id) {
        NIMBLE_CHECK(context.getStatsCollector(id)->shared());
      }
    }
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    // Check the type of the received vector. Accepted types are ROW or MAP
    // (the latter either MapVector or FlatMapVector encodings).
    switch (vector->type()->kind()) {
      case velox::TypeKind::ROW:
        ingestRow(
            velox::RowVector::pushDictionaryToRowVectorLeaves(
                velox::BaseVector::loadedVectorShared(vector)),
            ranges);
        return;

      case velox::TypeKind::MAP: {
        switch (vector->wrappedVector()->encoding()) {
          case velox::VectorEncoding::Simple::FLAT_MAP:
            ingestFlatMap(vector, ranges);
            return;
          default:
            ingestMap(vector, ranges);
            return;
        }
        break;
      }

      default:
        break;
    }

    NIMBLE_UNSUPPORTED(
        "Unsupported vector type for flat map writer.", vector->toString());
  }

  // Adds all predefined keys, creating value field writers in sorted order
  // to ensure deterministic stream IDs across serializers. Must be called
  // after construction and before any writes. No-op if no predefined keys
  // are configured for this node.
  void addPredefinedKeys() {
    auto* keys = context_.getFlatMapNodeKeys(nodeId_);
    if (keys == nullptr) {
      return;
    }
    NIMBLE_CHECK(
        allValueFields_.empty(),
        "addPredefinedKeys() must be called before any writes for node {}",
        nodeId_);
    for (const auto& keyName : *keys) {
      KeyType key;
      if constexpr (
          K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
        key = KeyType(keyName);
      } else {
        key = folly::to<KeyType>(keyName);
      }
      // std::set guarantees uniqueness, so duplicates should not occur.
      NIMBLE_DCHECK(
          allValueFields_.find(key) == allValueFields_.end(),
          "Duplicate predefined key: {}",
          keyName);
      auto valueFieldWriter = FieldWriter::create(context_, valueType_);
      const auto& inMapDescriptor = typeBuilder_->asFlatMap().addChild(
          keyName, valueFieldWriter->typeBuilder());
      context_.handleFlatmapFieldAddEvent(
          *typeBuilder_, keyName, *valueFieldWriter->typeBuilder());
      auto ownedKey =
          stableKey(key, typeBuilder_->asFlatMap().childrenCount() - 1);
      allValueFields_.emplace(
          ownedKey,
          std::make_unique<FlatMapValueFieldWriter>(
              context_, inMapDescriptor, std::move(valueFieldWriter), nodeId_));
    }
  }

  FlatMapPassthroughValueFieldWriter& createPassthroughValueFieldWriter(
      const std::string& key) {
    // Creating a new passthrough value field mutates shared context state:
    // FieldWriter::create() appends to the shared streams_ and schemaBuilder_,
    // and handleFlatmapFieldAddEvent() updates shared context. Sibling flat-map
    // columns are written concurrently by RowFieldWriter::co_write when
    // parallel write is enabled, so these mutations must be serialized. This
    // mirrors getValueFieldWriter(), which already guards the identical
    // operations with flatMapSchemaMutex_; the passthrough path was missing the
    // lock.
    std::scoped_lock<std::mutex> lock{context_.flatMapSchemaMutex()};
    auto fieldWriter = FieldWriter::create(context_, valueType_);
    auto& inMapDescriptor =
        typeBuilder_->asFlatMap().addChild(key, fieldWriter->typeBuilder());
    context_.handleFlatmapFieldAddEvent(
        *typeBuilder_, key, *fieldWriter->typeBuilder());
    auto it = currentPassthroughFields_
                  .insert(
                      {key,
                       std::make_unique<FlatMapPassthroughValueFieldWriter>(
                           context_,
                           inMapDescriptor,
                           std::move(fieldWriter),
                           nodeId_)})
                  .first;
    return *it->second;
  }

  FlatMapPassthroughValueFieldWriter& findPassthroughValueFieldWriter(
      const std::string& key) {
    auto existingPair = currentPassthroughFields_.find(key);
    NIMBLE_CHECK(
        existingPair != currentPassthroughFields_.end(),
        "Field writer must already exist in map");
    return *existingPair->second;
  }

  void collectStatistics(uint64_t nullCount, uint64_t valueCount) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(nullCount);
  }

  // Collects key statistics for flatmap with string keys (VARCHAR/VARBINARY).
  // Uses actual string sizes tracked during map iteration, NOT
  // sizeof(StringView). totalKeyCount: the total number of map entries (keys)
  // across all rows. totalKeyStringSize: the sum of all key string lengths.
  // nullCount: the number of null rows.
  // valueCount: the total number of rows processed.
  void collectMapStringKeyStatistics(
      uint64_t totalKeyCount,
      uint64_t totalKeyStringSize,
      uint64_t nullCount,
      uint64_t valueCount) {
    if (!keyStatisticsCollector_) {
      return;
    }

    keyStatisticsCollector_->addCounts(totalKeyCount, nullCount);
    // For string keys, use the actual total string size, not
    // sizeof(StringView).
    keyStatisticsCollector_->addLogicalSize(totalKeyStringSize + nullCount);
  }

  // Collects key statistics for flatmap.
  // totalKeyCount: the total number of map entries (keys) across all rows.
  // nullCount: the number of null rows.
  // valueCount: the total number of rows processed.
  void collectKeyStatistics(
      uint64_t totalKeyCount,
      uint64_t nullCount,
      uint64_t valueCount) {
    if (!keyStatisticsCollector_) {
      return;
    }

    keyStatisticsCollector_->addCounts(totalKeyCount, nullCount);
    // Key logical size is: (number of keys * sizeof(KeyType)) + nullCount
    // For VARCHAR keys, this uses sizeof(velox::StringView) which is 16 bytes,
    // NOT the actual string content length. This matches FieldWriter's
    // current behavior for MAP ingestion.
    // TODO: For string keys, we should track actual string lengths instead.
    keyStatisticsCollector_->addLogicalSize(
        totalKeyCount * sizeof(KeyType) + nullCount);
  }

  // Collects key statistics for passthrough flatmap with VARCHAR keys.
  // For passthrough flatmaps, keys are ROW field names, so we use the actual
  // string lengths instead of sizeof(StringView).
  void collectPassthroughStringKeyStatistics(
      const velox::RowVector* rowVector,
      uint64_t nonNullCount,
      uint64_t nullCount,
      uint64_t valueCount) {
    if (!keyStatisticsCollector_) {
      return;
    }

    // For passthrough flatmaps, each non-null row has all keys present.
    const auto& rowType = rowVector->type()->asRow();
    const auto numKeys = rowType.size();
    const uint64_t totalKeyCount = numKeys * nonNullCount;

    keyStatisticsCollector_->addCounts(totalKeyCount, nullCount);

    // Calculate the total key string size: sum of all key name lengths
    // multiplied by the number of non-null rows.
    uint64_t totalKeyStringSize = 0;
    for (size_t i = 0; i < numKeys; ++i) {
      totalKeyStringSize += rowType.nameOf(i).size();
    }

    keyStatisticsCollector_->addLogicalSize(
        totalKeyStringSize * nonNullCount + nullCount);
  }

  void ingestFlatMap(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges) {
    // TODO: Add support for predefined flatmap key ordering with FlatMapVector
    // ingestion.
    NIMBLE_USER_CHECK(
        !hasPredefinedKeys(),
        "Predefined flatmap key ordering is not supported with FlatMapVector ingestion");
    NIMBLE_CHECK(
        currentValueFields_.empty() && allValueFields_.empty(),
        "Mixing map and flatmap vectors in the FlatMapFieldWriter is not supported");
    const auto size = ranges.size();
    const velox::FlatMapVector* flatMapVector =
        vector->as<velox::FlatMapVector>();

    OrderedRanges childRanges;
    uint64_t nonNullCount;
    if (flatMapVector) {
      nullsStream_.ensureAdditionalNullsCapacity(
          flatMapVector->mayHaveNulls(), size);
      nonNullCount = iterateNonNullIndices<true>(
          ranges,
          nullsStream_.mutableNonNulls(),
          FlatAdapter<>{vector},
          [&](auto offset) { childRanges.add(offset, 1); });
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      flatMapVector =
          decoded.base()->template asChecked<velox::FlatMapVector>();
      nullsStream_.ensureAdditionalNullsCapacity(decoded.mayHaveNulls(), size);
      nonNullCount = iterateNonNullIndices<true>(
          ranges,
          nullsStream_.mutableNonNulls(),
          DecodedAdapter<>{decoded},
          [&](auto offset) { childRanges.add(offset, 1); });
    }

    collectStatistics(size - nonNullCount, size);
    // For FlatMapVector ingestion, we need to compute the total key count by
    // summing up the number of keys present in each non-null row.
    // This matches DWRF behavior where key sizes are counted per key per row.
    // For VARCHAR keys, also track the total string size for accurate
    // statistics.
    uint64_t totalKeyCount = 0;
    uint64_t totalKeyStringSize = 0;
    const auto& inMaps = flatMapVector->inMaps();

    // Helper to compute the occurrence count for a given key index.
    const auto computeKeyOccurrences = [&](size_t keyIndex) -> uint64_t {
      if (keyIndex < inMaps.size() && inMaps[keyIndex] != nullptr) {
        uint64_t count = 0;
        const auto* rawInMaps = inMaps[keyIndex]->as<uint64_t>();
        childRanges.applyEach([&](auto offset) {
          if (velox::bits::isBitSet(rawInMaps, offset)) {
            ++count;
          }
        });
        return count;
      } else {
        return childRanges.size();
      }
    };

    // Lambda to collect key statistics for both string and non-string types.
    auto collectKeyStats = [&](const auto& keysVector) {
      for (size_t i = 0; i < flatMapVector->numDistinctKeys(); ++i) {
        const uint64_t keyOccurrences = computeKeyOccurrences(i);
        totalKeyCount += keyOccurrences;
        if constexpr (
            K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
          const velox::StringView key = keysVector.valueAt(i);
          totalKeyStringSize += key.size() * keyOccurrences;
        }
      }
    };

    if (flatMapVector->distinctKeys()->isFlatEncoding()) {
      collectKeyStats(FlatAdapter<KeyType>{flatMapVector->distinctKeys()});
    } else {
      auto decodingContext = context_.decodingContext();
      OrderedRanges keyRanges;
      keyRanges.add(0, flatMapVector->distinctKeys()->size());
      auto& decodedKeys =
          decodingContext.decode(flatMapVector->distinctKeys(), keyRanges);
      collectKeyStats(DecodedAdapter<KeyType>{decodedKeys});
    }

    if constexpr (
        K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
      collectMapStringKeyStatistics(totalKeyCount, totalKeyStringSize, 0, size);
    } else {
      collectKeyStatistics(totalKeyCount, 0, size);
    }

    // Early bail out if no ranges at the top level row vector.
    if (childRanges.size() == 0) {
      return;
    }

    const auto& values = flatMapVector->mapValues();

    // Confirm that the keys are distinct. Otherwise writing the same key
    // multiple times leads to out of bounds scan spec channels.
    std::unordered_set<std::string> distinctKeySet;
    auto processKeys = [&](const auto& keysVector) {
      for (velox::vector_size_t i = 0; i < flatMapVector->numDistinctKeys();
           ++i) {
        // Ideally we wouldn't need to convert the key to a string, but this is
        // done for backward compatibility with ingestRow().
        const auto& key = flatMapKeyToString(keysVector.valueAt(i));
        NIMBLE_CHECK(
            distinctKeySet.find(key) == distinctKeySet.end(),
            "FlatMapVector keys are not distinct.");
        distinctKeySet.insert(key);

        // Only create keys on first call to write (with valid ranges). To
        // account for key pruning, let's avoid failing on unfound subsequent
        // keys.
        auto existingPair = currentPassthroughFields_.find(key);
        auto& writer = existingPair != currentPassthroughFields_.end()
            ? *existingPair->second
            : createPassthroughValueFieldWriter(key);

        if (inMaps[i]) {
          writer.write(values[i], inMaps[i], childRanges);
        } else {
          writer.write(values[i], childRanges);
        }
      }
    };

    if (flatMapVector->distinctKeys()->isFlatEncoding()) {
      processKeys(FlatAdapter<KeyType>{flatMapVector->distinctKeys()});
    } else {
      auto decodingContext = context_.decodingContext();
      OrderedRanges keyRanges;
      keyRanges.add(0, flatMapVector->distinctKeys()->size());
      auto& decodedKeys =
          decodingContext.decode(flatMapVector->distinctKeys(), keyRanges);
      processKeys(DecodedAdapter<KeyType>{decodedKeys});
    }
  }

  void ingestRow(const velox::VectorPtr& vector, const OrderedRanges& ranges) {
    // TODO: Add support for predefined flatmap key ordering with ROW vector
    // ingestion.
    NIMBLE_USER_CHECK(
        !hasPredefinedKeys(),
        "Predefined flatmap key ordering is not supported with ROW vector ingestion");
    NIMBLE_CHECK(
        currentValueFields_.empty() && allValueFields_.empty(),
        "Mixing map and flatmap vectors in the FlatMapFieldWriter is not supported");
    const auto& rowVector = vector->as<velox::RowVector>();
    NIMBLE_CHECK_NOT_NULL(
        rowVector,
        "Unexpected vector type. Vector must be a decoded ROW vector.");
    const auto size = ranges.size();
    nullsStream_.ensureAdditionalNullsCapacity(rowVector->mayHaveNulls(), size);
    const auto& keys = rowVector->type()->asRow().names();
    const auto& values = rowVector->children();

    OrderedRanges childRanges;
    uint64_t nonNullCount = iterateNonNullIndices<true>(
        ranges,
        nullsStream_.mutableNonNulls(),
        FlatAdapter<>{vector},
        [&](auto offset) { childRanges.add(offset, 1); });

    collectStatistics(size - nonNullCount, size);
    // For ROW vector ingestion (passthrough flatmaps), keys are ROW field
    // names. For VARCHAR/VARBINARY keys, use actual string lengths instead of
    // sizeof(StringView).
    if constexpr (
        K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
      collectPassthroughStringKeyStatistics(rowVector, nonNullCount, 0, size);
    } else {
      // For non-string keys, all keys are present for all non-null rows.
      // The total key count is: numKeys * numNonNullRows
      // This matches DWRF behavior where key sizes are counted per key per row.
      collectKeyStatistics(keys.size() * nonNullCount, 0, size);
    }

    // Early bail out if no ranges at the top level row vector.
    if (childRanges.size() == 0) {
      return;
    }

    // Only create keys on first call to write (with valid ranges). Subsequent
    // calls must have the same set of keys, otherwise writer will throw.
    bool populateMap = currentPassthroughFields_.empty();

    for (velox::vector_size_t i = 0; i < keys.size(); ++i) {
      const auto& key = keys[i];
      auto& writer = populateMap ? createPassthroughValueFieldWriter(key)
                                 : findPassthroughValueFieldWriter(key);
      writer.write(values[i], childRanges);
    }
  }

  void ingestMap(const velox::VectorPtr& vector, const OrderedRanges& ranges) {
    NIMBLE_CHECK(
        currentPassthroughFields_.empty(),
        "Mixing map and flatmap vectors in the FlatMapFieldWriter is not supported");
    const auto size = ranges.size();
    const velox::vector_size_t* offsets{nullptr};
    const velox::vector_size_t* lengths{nullptr};
    uint32_t nonNullCount{0};
    uint64_t totalKeyCount{0};
    uint64_t totalKeyStringSize{
        0}; // Track actual string size for VARCHAR/VARBINARY keys
    OrderedRanges keyRanges;

    // Lambda that iterates keys of a map and records the offsets to write to
    // a particular value node.
    auto processMap = [&](velox::vector_size_t index, const auto& keysVector) {
      totalKeyCount += lengths[index];
      for (auto elementIdx = offsets[index], end = elementIdx + lengths[index];
           elementIdx < end;
           ++elementIdx) {
        // NOTE: check for the null key story here.
        const auto& keyValue = keysVector.valueAt(elementIdx);
        // Track string key sizes for VARCHAR/VARBINARY keys
        if constexpr (
            K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
          totalKeyStringSize += keyValue.size();
        }
        auto* valueField = getValueFieldWriter(keyValue, size);
        // Add the value to the buffer by recording its offset in the values
        // vector.
        const auto ret = valueField->add(elementIdx, nonNullCount);
        NIMBLE_CHECK(
            ret,
            "Duplicate key: {} at flatmap with node id {}",
            folly::to<std::string>(keyValue),
            nodeId_);
      }
      ++nonNullCount;
    };

    // Lambda that calculates child ranges
    auto computeKeyRanges = [&](velox::vector_size_t index) {
      keyRanges.add(offsets[index], lengths[index]);
    };

    // Lambda that iterates the vector
    auto processVector = [&](const auto& map, const auto& vector) {
      const auto& mapKeys = map->mapKeys();
      if (auto* flatKeys = mapKeys->template asFlatVector<KeyType>()) {
        // Keys are flat.
        FlatAdapter<KeyType> keysVector{mapKeys};
        iterateNonNullIndices<true>(
            ranges, nullsStream_.mutableNonNulls(), vector, [&](auto offset) {
              processMap(offset, keysVector);
            });
      } else {
        // Keys are encoded. Decode.
        iterateNonNullIndices<false>(
            ranges, nullsStream_.mutableNonNulls(), vector, computeKeyRanges);
        auto decodingContext = context_.decodingContext();
        auto& decodedKeys = decodingContext.decode(mapKeys, keyRanges);
        DecodedAdapter<KeyType> keysVector{decodedKeys};
        iterateNonNullIndices<true>(
            ranges, nullsStream_.mutableNonNulls(), vector, [&](auto offset) {
              processMap(offset, keysVector);
            });
      }
    };

    // Reset existing value fields for next batch
    for (auto& pair : currentValueFields_) {
      pair.second->prepare(size);
    }

    const velox::MapVector* map = vector->as<velox::MapVector>();
    if (map != nullptr) {
      // Map is flat
      offsets = map->rawOffsets();
      lengths = map->rawSizes();

      nullsStream_.ensureAdditionalNullsCapacity(map->mayHaveNulls(), size);
      processVector(map, FlatAdapter<>{vector});
    } else {
      // Map is encoded. Decode.
      auto decodingContext = context_.decodingContext();
      auto& decodedMap = decodingContext.decode(vector, ranges);
      map = decodedMap.base()->template asChecked<velox::MapVector>();
      offsets = map->rawOffsets();
      lengths = map->rawSizes();
      nullsStream_.ensureAdditionalNullsCapacity(
          decodedMap.mayHaveNulls(), size);
      processVector(map, DecodedAdapter<>{decodedMap});
    }

    // Now actually ingest the map values
    if (nonNullCount > 0) {
      auto& values = map->mapValues();

      if (parallelWriteEnabled(
              static_cast<uint32_t>(currentValueFields_.size()))) {
        folly::coro::blockingWait(co_writeMapValues(values, nonNullCount));
      } else {
        for (auto& pair : currentValueFields_) {
          pair.second->write(values, nonNullCount);
        }
      }
    }
    nonNullCount_ += nonNullCount;

    collectStatistics(size - nonNullCount, size);
    // For VARCHAR/VARBINARY keys in MAP vectors, use the actual string sizes
    // tracked during processMap iteration, not sizeof(StringView).
    if constexpr (
        K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
      collectMapStringKeyStatistics(totalKeyCount, totalKeyStringSize, 0, size);
    } else {
      // totalKeyCount is the sum of all map entry counts tracked during
      // processMap iteration, representing the actual number of keys written.
      collectKeyStatistics(totalKeyCount, 0, size);
    }
  }

  void reset() override {
    for (auto& field : currentValueFields_) {
      field.second->reset();
    }

    for (auto& field : currentPassthroughFields_) {
      field.second->reset();
    }

    nullsStream_.reset();
    nonNullCount_ = 0;
    currentValueFields_.clear();
  }

  void close() override {
    // Add dummy node so we can preserve schema of an empty flat map
    // when no fields are written
    if (allValueFields_.empty() && currentPassthroughFields_.empty()) {
      auto valueField = FieldWriter::create(context_, valueType_);
      typeBuilder_->asFlatMap().addChild("", valueField->typeBuilder());
    } else {
      for (auto& pair : allValueFields_) {
        pair.second->close();
      }
      for (auto& pair : currentPassthroughFields_) {
        pair.second->close();
      }
    }
  }

 private:
  folly::coro::Task<void> co_writeMapValues(
      const velox::VectorPtr& values,
      uint32_t nonNullCount) {
    std::vector<FlatMapValueFieldWriter*> valueWriters;
    valueWriters.reserve(currentValueFields_.size());
    for (auto& pair : currentValueFields_) {
      valueWriters.push_back(pair.second);
    }

    const uint32_t taskCount = computeParallelWriteTaskCount(
        static_cast<uint32_t>(valueWriters.size()));
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::FlatMapFieldWriter::co_writeMapValues",
        const_cast<uint32_t*>(&taskCount));

    const uint32_t writersPerTask =
        static_cast<uint32_t>(valueWriters.size()) / taskCount;
    const uint32_t numRemainderWriters =
        static_cast<uint32_t>(valueWriters.size()) % taskCount;

    auto writeRange = [&valueWriters, &values, nonNullCount](
                          uint32_t startIdx, uint32_t endIdx) {
      for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
        valueWriters[idx]->write(values, nonNullCount);
      }
    };

    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    tasks.reserve(taskCount);
    uint32_t nextIdx = 0;
    for (uint32_t taskId = 0; taskId < taskCount; ++taskId) {
      const uint32_t endIdx =
          nextIdx + writersPerTask + (taskId < numRemainderWriters ? 1 : 0);
      tasks.emplace_back(co_withExecutor(
          context_.encodeExecutor(),
          folly::coro::co_invoke(
              [&writeRange, nextIdx, endIdx]() -> folly::coro::Task<void> {
                writeRange(nextIdx, endIdx);
                co_return;
              })));
      nextIdx = endIdx;
    }

    co_await folly::coro::collectAllRange(std::move(tasks));
  }

  inline bool hasPredefinedKeys() const {
    return context_.getFlatMapNodeKeys(nodeId_) != nullptr;
  }

  // Returns a key whose underlying string data has stable lifetime. For
  // StringView keys, returns a view pointing at the owned string in the type
  // builder's names_ deque (which guarantees pointer stability on push_back).
  // For non-string keys, returns the key as-is.
  KeyType stableKey(KeyType key, size_t childIndex) const {
    if constexpr (
        K == velox::TypeKind::VARCHAR || K == velox::TypeKind::VARBINARY) {
      return KeyType(typeBuilder_->asFlatMap().nameAt(childIndex));
    } else {
      return key;
    }
  }

  FlatMapValueFieldWriter* getValueFieldWriter(KeyType key, uint32_t size) {
    auto it = currentValueFields_.find(key);
    if (it != currentValueFields_.end()) {
      return it->second;
    }

    const auto stringKey = folly::to<std::string>(key);
    NIMBLE_DCHECK(!stringKey.empty(), "String key cannot be empty for flatmap");

    // check whether the typebuilder for this key is already present
    auto flatFieldIt = allValueFields_.find(key);
    if (flatFieldIt == allValueFields_.end()) {
      NIMBLE_USER_CHECK(
          !hasPredefinedKeys(),
          "Unknown flatmap key '{}' not in predefined key list for node {}",
          key,
          nodeId_);
      std::scoped_lock<std::mutex> lock{context_.flatMapSchemaMutex()};

      // Bound the number of distinct flat-map keys to prevent unbounded native
      // memory growth. High-cardinality keys (e.g. BIGINT ids) are unsuitable
      // for flat-map encoding; use a regular map or raise
      // orc.map.flat.max.keys. A limit of 0 means unlimited. Checked under
      // flatMapSchemaMutex_ so the size check and the allValueFields_ insertion
      // below are atomic.
      const auto maxFlatMapKeys = context_.maxFlatMapKeys();
      NIMBLE_USER_CHECK(
          maxFlatMapKeys == 0 || allValueFields_.size() < maxFlatMapKeys,
          "Too many flatmap keys for node {}: limit is {}",
          nodeId_,
          maxFlatMapKeys);

      auto valueFieldWriter = FieldWriter::create(context_, valueType_);
      const auto& inMapDescriptor = typeBuilder_->asFlatMap().addChild(
          stringKey, valueFieldWriter->typeBuilder());
      context_.handleFlatmapFieldAddEvent(
          *typeBuilder_, stringKey, *valueFieldWriter->typeBuilder());
      auto flatMapValueField = std::make_unique<FlatMapValueFieldWriter>(
          context_, inMapDescriptor, std::move(valueFieldWriter), nodeId_);
      auto ownedKey =
          stableKey(key, typeBuilder_->asFlatMap().childrenCount() - 1);
      flatFieldIt =
          allValueFields_.emplace(ownedKey, std::move(flatMapValueField)).first;
    }
    // Use the key stored in allValueFields_ (which points to owned memory)
    // rather than the input key (which may point to transient vector buffers).
    it = currentValueFields_
             .emplace(flatFieldIt->first, flatFieldIt->second.get())
             .first;

    // At this point we will have at max nonNullCount_ for the field which we
    // backfill to false, later when ingest is completed (using scatter write)
    // we update the inMapBuffer to nonNullCount which represnet the actual
    // values written in file
    it->second->backfill(nonNullCount_, size);
    return it->second;
  }

  const std::shared_ptr<const velox::dwio::common::TypeWithId>& valueType_;
  const uint32_t nodeId_;

  NullsStreamData& nullsStream_;

  // This map store the FlatMapValue fields used in current flush unit.
  folly::F14FastMap<KeyType, FlatMapValueFieldWriter*> currentValueFields_;
  // This map stores the FlatMapPassthrough fields.
  folly::F14FastMap<
      std::string,
      std::unique_ptr<FlatMapPassthroughValueFieldWriter>>
      currentPassthroughFields_;
  uint64_t nonNullCount_{0};

  // This map store all FlatMapValue fields encountered by the Writer
  // across the whole file.
  folly::F14FastMap<KeyType, std::unique_ptr<FlatMapValueFieldWriter>>
      allValueFields_;
  SharedStatisticsCollector* statisticsCollector_{nullptr};
  SharedStatisticsCollector* keyStatisticsCollector_{nullptr};
};

template <velox::TypeKind K>
std::unique_ptr<FieldWriter> createTypedFlatMapFieldWriter(
    FieldWriterContext& context,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
  auto writer = std::make_unique<FlatMapFieldWriter<K>>(context, type);
  writer->addPredefinedKeys();
  return writer;
}

std::unique_ptr<FieldWriter> createFlatMapFieldWriter(
    FieldWriterContext& context,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
  NIMBLE_DCHECK_EQ(
      type->type()->kind(),
      velox::TypeKind::MAP,
      "Unexpected flat-map field type.");
  NIMBLE_DCHECK_EQ(type->size(), 2, "Invalid flat-map field type.");
  auto kind = type->childAt(0)->type()->kind();
  switch (kind) {
    case velox::TypeKind::TINYINT:
      return createTypedFlatMapFieldWriter<velox::TypeKind::TINYINT>(
          context, type);
    case velox::TypeKind::SMALLINT:
      return createTypedFlatMapFieldWriter<velox::TypeKind::SMALLINT>(
          context, type);
    case velox::TypeKind::INTEGER:
      return createTypedFlatMapFieldWriter<velox::TypeKind::INTEGER>(
          context, type);
    case velox::TypeKind::BIGINT:
      return createTypedFlatMapFieldWriter<velox::TypeKind::BIGINT>(
          context, type);
    case velox::TypeKind::VARCHAR:
      return createTypedFlatMapFieldWriter<velox::TypeKind::VARCHAR>(
          context, type);
    case velox::TypeKind::VARBINARY:
      return createTypedFlatMapFieldWriter<velox::TypeKind::VARBINARY>(
          context, type);
    default:
      NIMBLE_UNSUPPORTED(
          "Unsupported flat map key type {}.",
          type->childAt(0)->type()->toString());
  }
}

template <velox::TypeKind K>
class ArrayWithOffsetsFieldWriter : public FieldWriter {
  using SourceType = typename velox::TypeTraits<K>::NativeType;
  using OffsetType = uint32_t;

 public:
  ArrayWithOffsetsFieldWriter(
      FieldWriterContext& context,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type)
      : FieldWriter{context, context.schemaBuilder().createArrayWithOffsetsTypeBuilder()},
        offsetsStream_{context.createNullableContentStreamData<uint32_t>(
            typeBuilder_->asArrayWithOffsets().offsetsDescriptor(),
            type->id())},
        lengthsStream_{context.createContentStreamData<uint32_t>(
            typeBuilder_->asArrayWithOffsets().lengthsDescriptor(),
            type->id())},
        cached_(false),
        cachedValue_(nullptr),
        cachedSize_(0),
        statisticsCollector_{context.getStatsCollector(type->id())},
        elementStatisticsCollector_{
            context.getStatsCollector(type->childAt(0)->id())} {
    elements_ = FieldWriter::create(context, type->childAt(0));

    typeBuilder_->asArrayWithOffsets().setChildren(elements_->typeBuilder());

    cachedValue_ = velox::ArrayVector::create(
        type->type(), 1, context.bufferMemoryPool().get());
  }

  void write(const velox::VectorPtr& vector, const OrderedRanges& ranges)
      override {
    OrderedRanges childFilteredRanges;
    // childItemCount is the (logical) total cardinality of array elements.
    uint64_t childItemCount = 0;
    uint64_t elementNullCount = 0;
    const velox::ArrayVector* array;
    uint64_t nullCount = 0;
    // To unwrap the dictionaryVector we need to cast into ComplexType before
    // extracting value arrayVector
    const auto dictionaryVector =
        vector->as<velox::DictionaryVector<velox::ComplexType>>();
    if (dictionaryVector &&
        dictionaryVector->valueVector()->template as<velox::ArrayVector>() &&
        isDictionaryValidRunLengthEncoded(*dictionaryVector)) {
      array = ingestLengthsOffsetsAlreadyEncoded(
          *dictionaryVector,
          ranges,
          childFilteredRanges,
          nullCount,
          childItemCount,
          elementNullCount);
    } else {
      array = ingestLengthsOffsets(
          vector,
          ranges,
          childFilteredRanges,
          nullCount,
          childItemCount,
          elementNullCount);
    }
    if (childFilteredRanges.size() > 0) {
      elements_->write(array->elements(), childFilteredRanges);
    }
    // Adjust child stats to account for deduplicated elements.
    // Child writers only see the deduplicated ranges, so their valueCount
    // is the deduplicated count. We need to add the difference to get
    // the full (non-deduplicated) valueCount. Similarly for null counts.
    auto dedupedChildItemCount = childFilteredRanges.size();
    if (childItemCount > dedupedChildItemCount) {
      // Count nulls in deduplicated child elements.
      uint64_t dedupedElementNullCount = 0;
      auto* elements = array->elements().get();
      if (elements->mayHaveNulls()) {
        const auto* elementNulls = elements->rawNulls();
        if (elementNulls != nullptr) {
          childFilteredRanges.apply([&](auto offset, auto count) {
            dedupedElementNullCount +=
                velox::bits::countNulls(elementNulls, offset, offset + count);
          });
        }
      }
      auto additionalItemCount = childItemCount - dedupedChildItemCount;
      auto additionalNullCount = elementNullCount - dedupedElementNullCount;
      // Calculate the raw size for the additional (duplicated) elements.
      // Non-null elements take sizeof(SourceType) bytes, nulls take 1 byte.
      // NOTE: the elements are statically asserted to be fixed width types.
      auto additionalRawSize =
          (additionalItemCount - additionalNullCount) * sizeof(SourceType) +
          additionalNullCount;
      amendChildStatistics(
          additionalItemCount, additionalNullCount, additionalRawSize);
    }

    // Calculate non-deduplicated logical size.
    {
      RawSizeContext context;
      // This code currently uses getRawSizeFromVector to calculate the
      // non-deduplicated logical size. In the future, we should replace this
      // with a more efficient approach that passes OrderedRanges objects to
      // child FieldWriters, allowing them to directly calculate
      // non-deduplicated sizes from deduplicated vectors without needing to
      // rely on an external util.
      collectStatistics(
          nullCount,
          ranges.size(),
          getRawSizeFromVector(vector, ranges, context));
    }
  }

  void reset() override {
    offsetsStream_.reset();
    lengthsStream_.reset();
    elements_->reset();

    cached_ = false;
    nextOffset_ = 0;
  }

  void close() override {
    elements_->close();
  }

 private:
  // NOTE: deduplicated stats are rolled up from children nodes. So we still
  // need to record actual logical size here.
  void collectStatistics(
      uint64_t nullCount,
      uint64_t valueCount,
      uint64_t logicalSize) {
    if (!statisticsCollector_) {
      return;
    }

    statisticsCollector_->addCounts(valueCount, nullCount);
    statisticsCollector_->addLogicalSize(logicalSize);
  }

  // Collects additional child element statistics for deduplicated elements.
  // This is needed because child writers only see deduplicated ranges, so
  // their valueCount would only reflect the deduplicated count. We need to
  // add the difference (childItemCount- dedupedChildCount) to get the
  // full valueCount. Similarly, we need to add the null count difference
  // and the logical size for these additional elements.
  void amendChildStatistics(
      uint64_t additionalCount,
      uint64_t additionalNullCount,
      uint64_t additionalRawSize) {
    if (elementStatisticsCollector_) {
      elementStatisticsCollector_->addCounts(
          additionalCount, additionalNullCount);
      elementStatisticsCollector_->addLogicalSize(additionalRawSize);
    }
  }

  std::unique_ptr<FieldWriter> elements_;
  NullableContentStreamData<uint32_t>&
      offsetsStream_; /** offsets for each data after dedup */
  ContentStreamData<uint32_t>&
      lengthsStream_; /** lengths of the each deduped data */
  OffsetType nextOffset_{0}; /** next available offset for dedup storing */

  bool cached_{false};
  velox::VectorPtr cachedValue_{nullptr};
  velox::vector_size_t cachedSize_{0};
  StatisticsCollector* statisticsCollector_;
  StatisticsCollector* elementStatisticsCollector_;

  /*
   * Check if the dictionary is valid run length encoded.
   * A dictionary is valid if its offsets in order are
   * increasing or equal. Two or more offsets are equal
   * when the dictionary has been deduped (the values
   * vec will be smaller as a result)
   * The read side expects offsets to be ordered for caching,
   * so we need to ensure that they are ordered if we are going to
   * passthrough the dictionary without applying any offset dedup logic.
   * Dictionaries of 0 or size 1 are always considered dictionary length
   * encoded since there are 0 or 1 offsets to validate.
   */
  bool isDictionaryValidRunLengthEncoded(
      const velox::DictionaryVector<velox::ComplexType>& dictionaryVector) {
    const velox::vector_size_t* indices =
        dictionaryVector.indices()->template as<velox::vector_size_t>();
    for (int i = 1; i < dictionaryVector.size(); ++i) {
      if (indices[i] < indices[i - 1]) {
        return false;
      }
    }

    return true;
  }

  velox::ArrayVector* ingestLengthsOffsetsAlreadyEncoded(
      const velox::DictionaryVector<velox::ComplexType>& dictionaryVector,
      const OrderedRanges& ranges,
      OrderedRanges& filteredRanges,
      uint64_t& nullCount,
      uint64_t& childItemCount,
      uint64_t& elementNullCount) {
    auto size = ranges.size();
    offsetsStream_.ensureAdditionalNullsCapacity(
        dictionaryVector.mayHaveNulls(), size);

    auto& offsetsData = offsetsStream_.mutableData();
    auto& lengthsData = lengthsStream_.mutableData();
    auto& nonNulls = offsetsStream_.mutableNonNulls();
    nullCount = 0;
    childItemCount = 0;
    elementNullCount = 0;

    const velox::vector_size_t* offsets =
        dictionaryVector.indices()->template as<velox::vector_size_t>();
    auto valuesArrayVector =
        dictionaryVector.valueVector()->template as<velox::ArrayVector>();

    // Get element null pointers for null counting.
    auto* elementsVector = valuesArrayVector->elements().get();
    const uint64_t* elementNulls =
        elementsVector->mayHaveNulls() ? elementsVector->rawNulls() : nullptr;
    const velox::vector_size_t* valuesOffsets = valuesArrayVector->rawOffsets();
    const velox::vector_size_t* valuesSizes = valuesArrayVector->rawSizes();

    auto previousOffset = -1;
    bool newElementIngested = false;
    auto ingestDictionaryIndex = [&](auto index) {
      const auto& dictIndex = offsets[index];
      auto numElement = valuesSizes[dictIndex];
      // Track total child count for all array elements (including duplicates).
      childItemCount += numElement;
      // Track null counts in child elements.
      if (numElement > 0 && elementNulls) {
        auto childOffset = valuesOffsets[dictIndex];
        elementNullCount += velox::bits::countNulls(
            elementNulls, childOffset, childOffset + numElement);
      }

      bool match = false;
      // Only write length if first element or if consecutive offset is
      // different, meaning we have reached a new value element.
      if (previousOffset >= 0) {
        match = (offsets[index] == previousOffset);
      } else if (cached_) {
        velox::CompareFlags flags;
        match =
            (numElement == cachedSize_ &&
             valuesArrayVector
                     ->compare(cachedValue_.get(), offsets[index], 0, flags)
                     .value_or(-1) == 0);
      }

      if (!match) {
        auto arrayOffset = valuesArrayVector->offsetAt(offsets[index]);
        auto length = numElement;
        lengthsData.push_back(length);
        newElementIngested = true;
        if (length > 0) {
          filteredRanges.add(arrayOffset, length);
        }
        ++nextOffset_;
      }

      offsetsData.push_back(nextOffset_ - 1);
      previousOffset = offsets[index];
    };

    if (dictionaryVector.mayHaveNulls()) {
      ranges.applyEach([&](auto index) {
        auto notNull = !dictionaryVector.isNullAt(index);
        nonNulls.push_back(notNull);
        if (notNull) {
          ingestDictionaryIndex(index);
        } else {
          ++nullCount;
        }
      });
    } else {
      ranges.applyEach([&](auto index) { ingestDictionaryIndex(index); });
    }

    // insert last element discovered into cache
    if (newElementIngested) {
      cached_ = true;
      cachedSize_ = lengthsData[lengthsData.size() - 1];
      cachedValue_->prepareForReuse();
      velox::BaseVector::CopyRange cacheRange{
          static_cast<velox::vector_size_t>(previousOffset) /* source index*/,
          0 /* target index*/,
          1 /* count*/};
      cachedValue_->copyRanges(valuesArrayVector, folly::Range(&cacheRange, 1));
    }

    return valuesArrayVector;
  }

  template <typename Vector>
  void ingestLengthsOffsetsByElements(
      const velox::ArrayVector* array,
      const Vector& iterableVector,
      const OrderedRanges& ranges,
      const OrderedRanges& childRanges,
      OrderedRanges& filteredRanges) {
    const velox::vector_size_t* rawOffsets = array->rawOffsets();
    const velox::vector_size_t* rawLengths = array->rawSizes();
    velox::vector_size_t prevIndex = -1;
    auto& lengthsData = lengthsStream_.mutableData();
    auto& offsetsData = offsetsStream_.mutableData();
    auto& nonNulls = offsetsStream_.mutableNonNulls();

    std::function<bool(velox::vector_size_t, velox::vector_size_t)>
        compareConsecutive;

    std::function<bool(velox::vector_size_t)> compareToCache;

    /** dedup arrays by consecutive elements */
    auto dedupProc = [&](velox::vector_size_t index) {
      auto const length = rawLengths[index];

      bool match = false;
      /// Don't compare on the first run
      if (prevIndex >= 0) {
        match =
            (index == prevIndex ||
             (length == rawLengths[prevIndex] &&
              compareConsecutive(index, prevIndex)));
      } else if (cached_) { // check cache here
        match = (length == cachedSize_ && compareToCache(index));
      }

      if (!match) {
        if (length > 0) {
          filteredRanges.add(rawOffsets[index], length);
        }
        lengthsData.push_back(length);
        ++nextOffset_;
      }

      prevIndex = index;
      offsetsData.push_back(nextOffset_ - 1);
    };

    auto& vectorElements = array->elements();
    if (auto flat = vectorElements->asFlatVector<SourceType>()) {
      /** compare array at index and prevIndex to be equal */
      compareConsecutive = [&](velox::vector_size_t index,
                               velox::vector_size_t prevIndex) {
        bool match = true;
        velox::CompareFlags flags;
        for (velox::vector_size_t idx = 0; idx < rawLengths[index]; ++idx) {
          match = flat->compare(
                          flat,
                          rawOffsets[index] + idx,
                          rawOffsets[prevIndex] + idx,
                          flags)
                      .value_or(-1) == 0;
          if (!match) {
            break;
          }
        }
        return match;
      };

      compareToCache = [&](velox::vector_size_t index) {
        velox::CompareFlags flags;
        return array->compare(cachedValue_.get(), index, 0, flags)
                   .value_or(-1) == 0;
      };

      iterateNonNullIndices<false>(ranges, nonNulls, iterableVector, dedupProc);
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vectorElements, childRanges);
      /** compare array at index and prevIndex to be equal */
      compareConsecutive = [&](velox::vector_size_t index,
                               velox::vector_size_t prevIndex) {
        bool match = true;
        for (velox::vector_size_t idx = 0; idx < rawLengths[index]; ++idx) {
          match = equalDecodedVectorIndices<SourceType>(
              decoded, rawOffsets[index] + idx, rawOffsets[prevIndex] + idx);
          if (!match) {
            break;
          }
        }
        return match;
      };

      auto cachedElements =
          (cachedValue_->as<velox::ArrayVector>())->elements();
      auto cachedFlat = cachedElements->asFlatVector<SourceType>();
      compareToCache = [&](velox::vector_size_t index) {
        bool match = true;
        velox::CompareFlags flags;
        for (velox::vector_size_t idx = 0; idx < rawLengths[index]; ++idx) {
          match = compareDecodedVectorToCache<SourceType>(
              decoded, rawOffsets[index] + idx, cachedFlat, idx, flags);
          if (!match) {
            break;
          }
        }
        return match;
      };
      iterateNonNullIndices<false>(ranges, nonNulls, iterableVector, dedupProc);
    }

    // Copy the last valid element into the cache.
    // Cache is saved across calls to write(), as long as the same FieldWriter
    // object is used.
    if (prevIndex != -1 && lengthsData.size() > 0) {
      cached_ = true;
      cachedSize_ = lengthsData[lengthsData.size() - 1];
      NIMBLE_CHECK_EQ(
          lengthsData[lengthsData.size() - 1],
          rawLengths[prevIndex],
          "Unexpected index: Prev index is not the last item in the list.");
      cachedValue_->prepareForReuse();
      velox::BaseVector::CopyRange cacheRange{
          static_cast<velox::vector_size_t>(prevIndex) /* source index*/,
          0 /* target index*/,
          1 /* count*/};
      cachedValue_->copyRanges(array, folly::Range(&cacheRange, 1));
    }
  }

  const velox::ArrayVector* ingestLengthsOffsets(
      const velox::VectorPtr& vector,
      const OrderedRanges& ranges,
      OrderedRanges& filteredRanges,
      uint64_t& nullCount,
      uint64_t& childItemCount,
      uint64_t& elementNullCount) {
    auto size = ranges.size();
    const velox::ArrayVector* arrayVector = vector->as<velox::ArrayVector>();
    const velox::vector_size_t* rawOffsets;
    const velox::vector_size_t* rawLengths;
    OrderedRanges childRanges;
    childItemCount = 0;
    elementNullCount = 0;

    // Get element null pointer for null counting.
    const uint64_t* elementNulls = nullptr;

    auto proc = [&](velox::vector_size_t index) {
      auto length = rawLengths[index];
      // Track total child count for all array elements (including duplicates).
      childItemCount += length;
      // Track null counts in child elements.
      if (length > 0) {
        if (elementNulls != nullptr) {
          auto childOffset = rawOffsets[index];
          elementNullCount += velox::bits::countNulls(
              elementNulls, childOffset, childOffset + length);
        }
        childRanges.add(rawOffsets[index], length);
      }
    };

    if (arrayVector) {
      rawOffsets = arrayVector->rawOffsets();
      rawLengths = arrayVector->rawSizes();
      // Initialize element null pointer for child null counting.
      auto* elementsVector = arrayVector->elements().get();
      elementNulls = elementsVector->rawNulls();

      offsetsStream_.ensureAdditionalNullsCapacity(
          arrayVector->mayHaveNulls(), size);
      FlatAdapter<> iterableVector{vector};
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges, offsetsStream_.mutableNonNulls(), iterableVector, proc);
      nullCount = size - nonNullCount;
      ingestLengthsOffsetsByElements(
          arrayVector, iterableVector, ranges, childRanges, filteredRanges);
    } else {
      auto decodingContext = context_.decodingContext();
      auto& decoded = decodingContext.decode(vector, ranges);
      arrayVector = decoded.base()->template as<velox::ArrayVector>();
      NIMBLE_CHECK_NOT_NULL(arrayVector, "Unexpected vector type");
      rawOffsets = arrayVector->rawOffsets();
      rawLengths = arrayVector->rawSizes();
      // Initialize element null pointer for child null counting.
      auto* elementsVector = arrayVector->elements().get();
      elementNulls = elementsVector->rawNulls();

      offsetsStream_.ensureAdditionalNullsCapacity(
          decoded.mayHaveNulls(), size);
      DecodedAdapter<> iterableVector{decoded};
      auto nonNullCount = iterateNonNullIndices<true>(
          ranges, offsetsStream_.mutableNonNulls(), iterableVector, proc);
      nullCount = size - nonNullCount;
      ingestLengthsOffsetsByElements(
          arrayVector, iterableVector, ranges, childRanges, filteredRanges);
    }
    return arrayVector;
  }
};

std::unique_ptr<FieldWriter> createArrayWithOffsetsFieldWriter(
    FieldWriterContext& context,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
  NIMBLE_DCHECK_EQ(
      type->type()->kind(),
      velox::TypeKind::ARRAY,
      "Unexpected offset-array field type.");
  NIMBLE_DCHECK_EQ(type->size(), 1, "Invalid offset-array field type.");
  auto kind = type->childAt(0)->type()->kind();
  switch (kind) {
    case velox::TypeKind::TINYINT:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::TINYINT>>(context, type);
    case velox::TypeKind::SMALLINT:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::SMALLINT>>(
          context, type);
    case velox::TypeKind::INTEGER:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::INTEGER>>(context, type);
    case velox::TypeKind::BIGINT:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::BIGINT>>(context, type);
    case velox::TypeKind::REAL:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::REAL>>(context, type);
    case velox::TypeKind::DOUBLE:
      return std::make_unique<
          ArrayWithOffsetsFieldWriter<velox::TypeKind::DOUBLE>>(context, type);
    default:
      NIMBLE_UNSUPPORTED(
          "Unsupported dedup array element type {}.",
          type->childAt(0)->type()->toString());
  }
}

} // namespace

DecodingContextPool::DecodingContext::DecodingContext(
    DecodingContextPool& pool,
    std::unique_ptr<velox::DecodedVector> decodedVector,
    std::unique_ptr<velox::SelectivityVector> selectivityVector)
    : pool_{pool},
      decodedVector_{std::move(decodedVector)},
      selectivityVector_{std::move(selectivityVector)} {}

DecodingContextPool::DecodingContext::~DecodingContext() {
  pool_.addContext(std::move(decodedVector_), std::move(selectivityVector_));
}

velox::DecodedVector& DecodingContextPool::DecodingContext::decode(
    const velox::VectorPtr& vector,
    const OrderedRanges& ranges) {
  selectivityVector_->resize(vector->size());
  selectivityVector_->clearAll();
  ranges.apply([&](auto offset, auto size) {
    selectivityVector_->setValidRange(offset, offset + size, true);
  });
  selectivityVector_->updateBounds();

  decodedVector_->decode(*vector, *selectivityVector_);
  return *decodedVector_;
}

DecodingContextPool::DecodingContextPool(
    std::function<void(void)> vectorDecoderVisitor)
    : vectorDecoderVisitor_{std::move(vectorDecoderVisitor)} {
  NIMBLE_CHECK(vectorDecoderVisitor_, "vectorDecoderVisitor must be set");
  pool_.reserve(folly::available_concurrency());
}

void DecodingContextPool::addContext(
    std::unique_ptr<velox::DecodedVector> decodedVector,
    std::unique_ptr<velox::SelectivityVector> selectivityVector) {
  std::scoped_lock<std::mutex> lock{mutex_};
  pool_.emplace_back(std::move(decodedVector), std::move(selectivityVector));
}

DecodingContextPool::DecodingContext DecodingContextPool::reserveContext() {
  vectorDecoderVisitor_();

  std::scoped_lock<std::mutex> lock{mutex_};
  if (pool_.empty()) {
    return DecodingContext{
        *this,
        std::make_unique<velox::DecodedVector>(),
        std::make_unique<velox::SelectivityVector>()};
  }

  auto pair = std::move(pool_.back());
  pool_.pop_back();
  return DecodingContext{*this, std::move(pair.first), std::move(pair.second)};
}

size_t DecodingContextPool::size() const {
  return pool_.size();
}

std::unique_ptr<FieldWriter> FieldWriter::create(
    FieldWriterContext& context,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type,
    std::function<void(TypeBuilder&, uint32_t)> typeAddedHandler) {
  context.setTypeAddedHandler(std::move(typeAddedHandler));
  return create(context, type);
}

std::unique_ptr<FieldWriter> FieldWriter::create(
    FieldWriterContext& context,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& type) {
  std::unique_ptr<FieldWriter> field;
  switch (type->type()->kind()) {
    case velox::TypeKind::BOOLEAN: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::BOOLEAN>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::TINYINT: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::TINYINT>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::SMALLINT: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::SMALLINT>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::INTEGER: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::INTEGER>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::BIGINT: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::BIGINT>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::REAL: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::REAL>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::DOUBLE: {
      field = std::make_unique<SimpleFieldWriter<velox::TypeKind::DOUBLE>>(
          context, type->id());
      break;
    }
    case velox::TypeKind::VARCHAR: {
      if (context.disableSharedStringBuffers()) {
        field = std::make_unique<StringFieldWriter<velox::TypeKind::VARCHAR>>(
            context, type->id());
      } else {
        field = std::make_unique<
            SimpleFieldWriter<velox::TypeKind::VARCHAR, StringConverter>>(
            context, type->id());
      }
      break;
    }
    case velox::TypeKind::VARBINARY: {
      if (context.disableSharedStringBuffers()) {
        field = std::make_unique<StringFieldWriter<velox::TypeKind::VARBINARY>>(
            context, type->id());
      } else {
        field = std::make_unique<
            SimpleFieldWriter<velox::TypeKind::VARBINARY, StringConverter>>(
            context, type->id());
      }
      break;
    }
    case velox::TypeKind::TIMESTAMP: {
      field = std::make_unique<TimestampFieldWriter>(context, type->id());
      break;
    }
    case velox::TypeKind::ROW: {
      field = std::make_unique<RowFieldWriter>(context, type);
      break;
    }
    case velox::TypeKind::ARRAY: {
      field = context.hasDictionaryArrayNodeId(type->id())
          ? createArrayWithOffsetsFieldWriter(context, type)
          : std::make_unique<ArrayFieldWriter>(context, type);
      break;
    }
    case velox::TypeKind::MAP: {
      // A map can both be a flat map and a deduplicated map.
      // Flat map takes precedence over deduplicated map, i.e. the outer map
      // will be a flat map whereas the child maps will be deduplicated.
      if (context.hasFlatMapNodeId(type->id())) {
        field = createFlatMapFieldWriter(context, type);
      } else if (context.hasDeduplicatedMapNodeId(type->id())) {
        field = std::make_unique<SlidingWindowMapFieldWriter>(context, type);
      } else {
        field = std::make_unique<MapFieldWriter>(context, type);
      }
      break;
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported kind: {}.", type->type()->kind());
  }
  context.handleAddedType(*field->typeBuilder(), type->id());
  return field;
}

uint32_t FieldWriter::computeParallelWriteTaskCount(
    uint32_t numChildren) const {
  if (numChildren == 0) {
    return 0;
  }
  const uint32_t maxByStreams =
      numChildren / std::max(1u, context_.minStreamsPerEncodeUnit());
  return std::clamp(
      std::min(context_.maxEncodeParallelism(), maxByStreams), 1u, numChildren);
}

bool FieldWriter::parallelWriteEnabled(uint32_t numChildren) const {
  return context_.encodeExecutor() != nullptr &&
      computeParallelWriteTaskCount(numChildren) > 1;
}

} // namespace facebook::nimble
