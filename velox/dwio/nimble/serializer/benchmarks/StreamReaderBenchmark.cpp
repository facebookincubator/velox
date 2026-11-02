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

#include <array>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/coro/BlockingWait.h>
#include <folly/init/Init.h>

#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/StreamReader.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble {
namespace {

constexpr uint32_t kNumRows{1'024};

// Bounds the string buffers that repeated point lookups retain on one output.
constexpr uint32_t kOutputResetInterval{256};

// Models a wide MySQL table of nullable scalar columns, held in memory as
// several stripes with plain (uncompressed) encodings. Stripes may omit
// columns, as schema changes do, so stripes can differ in which streams they
// carry.
struct Table {
  // Nimble schema of the table.
  std::shared_ptr<const Type> type;
  // Velox counterpart of 'type', used to allocate outputs.
  velox::TypePtr outputType;
  // Encoded stream of each column, per stripe.
  std::vector<std::vector<std::string>> columnStreams;
  // Streams of each stripe in projected order. Stream 0 is the absent row
  // nulls stream.
  std::vector<std::vector<std::string_view>> stripeStreams;
};

ScalarKind columnKind(uint32_t column) {
  switch (column % 3) {
    case 0:
      return ScalarKind::Int64;
    case 1:
      return ScalarKind::Int32;
    default:
      return ScalarKind::String;
  }
}

// Stripes s and s + numPatterns share a stream-presence pattern. Pattern p
// omits the columns whose index is p modulo numPatterns, and a single pattern
// omits none.
bool isAbsent(uint32_t column, uint32_t stripe, uint32_t numPatterns) {
  return numPatterns > 1 && column % numPatterns == stripe % numPatterns;
}

velox::TypePtr columnType(uint32_t column) {
  switch (column % 3) {
    case 0:
      return velox::BIGINT();
    case 1:
      return velox::INTEGER();
    default:
      return velox::VARCHAR();
  }
}

template <typename T>
std::string encodeNullableColumn(
    const std::vector<std::optional<T>>& values,
    velox::memory::MemoryPool& pool) {
  std::vector<T> nonNullValues;
  Vector<bool> isNonNull{&pool, values.size()};
  for (size_t i{0}; i < values.size(); ++i) {
    isNonNull[i] = values[i].has_value();
    if (values[i].has_value()) {
      nonNullValues.emplace_back(*values[i]);
    }
  }

  Buffer buffer{pool};
  auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::vector<std::pair<EncodingType, float>>{{EncodingType::Trivial, 1.0}},
      /*compressionOptions=*/std::nullopt,
      /*identifier=*/std::nullopt);
  const auto encoded = EncodingFactory::encodeNullable<T>(
      std::move(policy), nonNullValues, isNonNull, buffer);
  ChunkedStreamWriter writer{buffer};
  std::string stream;
  for (const auto segment : writer.encode(encoded)) {
    stream.append(segment);
  }
  return stream;
}

std::string encodeColumn(
    uint32_t column,
    uint32_t stripe,
    velox::memory::MemoryPool& pool) {
  const auto isNullRow = [](uint32_t row) { return row % 10 == 0; };
  const uint32_t firstRow{stripe * kNumRows};
  const auto kind = columnKind(column);
  if (kind == ScalarKind::Int64) {
    std::vector<std::optional<int64_t>> values;
    for (uint32_t row{0}; row < kNumRows; ++row) {
      values.emplace_back(
          isNullRow(row)
              ? std::nullopt
              : std::optional<int64_t>{
                    static_cast<int64_t>(column) * kNumRows + firstRow + row});
    }
    return encodeNullableColumn<int64_t>(values, pool);
  }
  if (kind == ScalarKind::Int32) {
    std::vector<std::optional<int32_t>> values;
    for (uint32_t row{0}; row < kNumRows; ++row) {
      values.emplace_back(
          isNullRow(row) ? std::nullopt
                         : std::optional<int32_t>{
                               static_cast<int32_t>(column + firstRow + row)});
    }
    return encodeNullableColumn<int32_t>(values, pool);
  }

  std::vector<std::string> storage;
  for (uint32_t row{0}; row < kNumRows; ++row) {
    storage.emplace_back(
        "mysql_value_" + std::to_string(column) + "_" +
        std::to_string(firstRow + row));
  }
  std::vector<std::optional<std::string_view>> values;
  for (uint32_t row{0}; row < kNumRows; ++row) {
    values.emplace_back(
        isNullRow(row) ? std::nullopt
                       : std::optional<std::string_view>{storage.at(row)});
  }
  return encodeNullableColumn<std::string_view>(values, pool);
}

void fillTable(
    uint32_t numColumns,
    uint32_t numStripes,
    uint32_t numPatterns,
    Table& table) {
  auto pool = velox::memory::memoryManager()->addLeafPool("mysql_table");
  std::vector<std::string> names;
  std::vector<std::shared_ptr<const Type>> children;
  std::vector<velox::TypePtr> outputTypes;
  for (uint32_t column{0}; column < numColumns; ++column) {
    names.emplace_back("column_" + std::to_string(column));
    children.emplace_back(
        std::make_shared<const ScalarType>(
            StreamDescriptor{column + 1, columnKind(column)}));
    outputTypes.emplace_back(columnType(column));
  }
  table.type = std::make_shared<const RowType>(
      StreamDescriptor{0, ScalarKind::Bool}, names, children);
  table.outputType = velox::ROW(std::move(names), std::move(outputTypes));

  for (uint32_t stripe{0}; stripe < numStripes; ++stripe) {
    std::vector<std::string> columnStreams;
    columnStreams.reserve(numColumns);
    std::vector<std::string_view> streams;
    streams.reserve(numColumns + 1);
    streams.emplace_back();
    for (uint32_t column{0}; column < numColumns; ++column) {
      if (isAbsent(column, stripe, numPatterns)) {
        streams.emplace_back();
        continue;
      }
      columnStreams.emplace_back(encodeColumn(column, stripe, *pool));
      streams.emplace_back(columnStreams.back());
    }
    table.columnStreams.emplace_back(std::move(columnStreams));
    table.stripeStreams.emplace_back(std::move(streams));
  }
}

const Table&
getTable(uint32_t numColumns, uint32_t numStripes, uint32_t numPatterns) {
  static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Table> tables;
  const auto key = std::make_tuple(numColumns, numStripes, numPatterns);
  const auto it = tables.find(key);
  if (it != tables.end()) {
    return it->second;
  }
  auto& table = tables[key];
  fillTable(numColumns, numStripes, numPatterns, table);
  return table;
}

// Owns a reader over one table together with the pools it decodes into.
struct ReaderFixture {
  ReaderFixture(
      uint32_t numColumns,
      uint32_t numStripes,
      uint32_t numPatterns,
      const char* name)
      : table{getTable(numColumns, numStripes, numPatterns)},
        pool{velox::memory::memoryManager()->addLeafPool(name)},
        reader{
            table.type,
            pool.get(),
            Encoding::Options{.bufferPool = &bufferPool}} {}

  void resetOutput() {
    output = velox::BaseVector::create(table.outputType, 1, pool.get());
  }

  // Table that 'reader' decodes. Cached for the life of the process.
  const Table& table;
  // Allocates the reader's buffers and the output.
  const std::shared_ptr<velox::memory::MemoryPool> pool;
  // Reuses the scratch buffers of the reader's encoding views.
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  // Decodes the stripes of 'table'.
  StreamReader reader;
  // Receives the decoded rows.
  velox::VectorPtr output;
};

uint32_t lookupRow(uint32_t lookup) {
  return (lookup * 7'919) % kNumRows;
}

// Runs 'lookup' over every stripe on warm caches before timing 'iters' calls.
template <typename Lookup>
void measureLookups(
    uint32_t iters,
    uint32_t numWarmupLookups,
    ReaderFixture& fixture,
    folly::BenchmarkSuspender& suspender,
    const Lookup& lookup) {
  fixture.resetOutput();
  for (uint32_t i{0}; i < numWarmupLookups; ++i) {
    lookup(i);
  }
  suspender.dismiss();

  for (uint32_t i{0}; i < iters; ++i) {
    if (i % kOutputResetInterval == 0) {
      folly::BenchmarkSuspender outputSuspender;
      fixture.resetOutput();
    }
    lookup(i);
  }
  suspender.rehire();
  folly::doNotOptimizeAway(fixture.output);
}

// Binds a different stripe on every call, cycling through the stripes.
void bindStreams(
    uint32_t iters,
    uint32_t numColumns,
    uint32_t numStripes,
    uint32_t numPatterns) {
  folly::BenchmarkSuspender suspender;
  ReaderFixture fixture{numColumns, numStripes, numPatterns, "bind_streams"};
  for (uint32_t i{0}; i < numStripes; ++i) {
    fixture.reader.setStreams(fixture.table.stripeStreams[i]);
  }
  suspender.dismiss();

  for (uint32_t i{0}; i < iters; ++i) {
    fixture.reader.setStreams(fixture.table.stripeStreams[i % numStripes]);
  }
  suspender.rehire();
}

// Reads one row per call and moves to the next stripe after 'lookupsPerStripe'
// calls. Every call passes the stripe's streams, so each call rebinds them.
void lookupRebindEveryCall(
    uint32_t iters,
    uint32_t numColumns,
    uint32_t numStripes,
    uint32_t lookupsPerStripe,
    uint32_t numPatterns) {
  folly::BenchmarkSuspender suspender;
  ReaderFixture fixture{numColumns, numStripes, numPatterns, "lookup_rebind"};
  const auto lookup = [&](uint32_t i) {
    const uint32_t row{lookupRow(i)};
    const std::array<RowRange, 1> ranges{{{row, row + 1}}};
    const auto stripe = (i / lookupsPerStripe) % numStripes;
    folly::coro::blockingWait(fixture.reader.co_read(
        fixture.table.stripeStreams[stripe],
        ranges,
        /*outputOffset=*/0,
        fixture.output));
  };
  measureLookups(
      iters, numStripes * lookupsPerStripe, fixture, suspender, lookup);
}

// Same lookups as lookupRebindEveryCall, but binds a stripe only when the
// lookups move to it.
void lookupBindPerStripe(
    uint32_t iters,
    uint32_t numColumns,
    uint32_t numStripes,
    uint32_t lookupsPerStripe,
    uint32_t numPatterns) {
  folly::BenchmarkSuspender suspender;
  ReaderFixture fixture{
      numColumns, numStripes, numPatterns, "lookup_bind_per_stripe"};
  std::optional<uint32_t> boundStripe;
  const auto lookup = [&](uint32_t i) {
    const uint32_t row{lookupRow(i)};
    const std::array<RowRange, 1> ranges{{{row, row + 1}}};
    const auto stripe = (i / lookupsPerStripe) % numStripes;
    if (boundStripe != stripe) {
      fixture.reader.setStreams(fixture.table.stripeStreams[stripe]);
      boundStripe = stripe;
    }
    folly::coro::blockingWait(
        fixture.reader.co_read(ranges, /*outputOffset=*/0, fixture.output));
  };
  measureLookups(
      iters, numStripes * lookupsPerStripe, fixture, suspender, lookup);
}

// Every stripe carries every column, so all stripes share one presence pattern.
BENCHMARK_NAMED_PARAM(bindStreams, cols16_stripes4, 16, 4, 1)
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes4, 200, 4, 1)

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_perStripe1, 200, 4, 1, 1)
BENCHMARK_RELATIVE_NAMED_PARAM(
    lookupBindPerStripe,
    cols200_perStripe1,
    200,
    4,
    1,
    1)

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_perStripe8, 200, 4, 8, 1)
BENCHMARK_RELATIVE_NAMED_PARAM(
    lookupBindPerStripe,
    cols200_perStripe8,
    200,
    4,
    8,
    1)

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_perStripe64, 200, 4, 64, 1)
BENCHMARK_RELATIVE_NAMED_PARAM(
    lookupBindPerStripe,
    cols200_perStripe64,
    200,
    4,
    64,
    1)

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols16_perStripe1, 16, 4, 1, 1)
BENCHMARK_RELATIVE_NAMED_PARAM(
    lookupBindPerStripe,
    cols16_perStripe1,
    16,
    4,
    1,
    1)

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols16_perStripe64, 16, 4, 64, 1)
BENCHMARK_RELATIVE_NAMED_PARAM(
    lookupBindPerStripe,
    cols16_perStripe64,
    16,
    4,
    64,
    1)

BENCHMARK_DRAW_LINE();

// Stripes with different presence patterns. A reader keeps the reader of the
// last bound pattern, so a pattern that differs from the previous stripe's
// rebuilds it. With patterns cycling through the stripes, every bind rebuilds.
// The 16-stripe, 1-pattern rows separate that cost from the cost of touching
// 16 stripes.
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes4_patterns4, 200, 4, 4)
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes8_patterns8, 200, 8, 8)
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes9_patterns9, 200, 9, 9)
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes16_patterns16, 200, 16, 16)
BENCHMARK_NAMED_PARAM(bindStreams, cols200_stripes16_patterns1, 200, 16, 1)
BENCHMARK_NAMED_PARAM(bindStreams, cols16_stripes16_patterns16, 16, 16, 16)
BENCHMARK_NAMED_PARAM(bindStreams, cols16_stripes16_patterns1, 16, 16, 1)

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_s4_p4, 200, 4, 1, 4)
BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_s16_p16, 200, 16, 1, 16)
BENCHMARK_NAMED_PARAM(lookupRebindEveryCall, cols200_s16_p1, 200, 16, 1, 1)

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
  return 0;
}
