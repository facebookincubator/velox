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

/// Measures selective reads that keep a few rows spread over many chunks, on
/// the same data written without and with chunk stats. Without chunk stats, a
/// read batch loads, and so decompresses, the chunk where it starts and every
/// chunk between the kept rows it reads. With them, reads go straight to the
/// chunks of kept rows with the stream index.
///
/// The first part reads ROW(key BIGINT, s VARCHAR, a ARRAY(VARCHAR)). 'key'
/// holds the row numbers in random order, so chunk stats cannot prune chunks
/// for the filter on it, which keeps evenly spaced rows. 's' is a 1 KB string,
/// and 'a' holds 0 to 7 strings of 256 bytes. The writer cuts chunks of 512 KB
/// to 1 MB raw, and stripes of 32 MB raw. A batch never spans stripes, so a
/// batch of all rows reads a whole stripe.
///
/// The second part reads ROW(key BIGINT, v), where 'v' is a VARCHAR of 256
/// bytes or a BIGINT with 64 or 1,024 distinct values, so the writer encodes it
/// as a dictionary, in about 100 chunks. It reads with zero-copy strings,
/// preserved dictionaries and dictionary-aware reads, so VARCHAR values come
/// out as dictionaries.
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/selective/tests:sparse_read_chunk_skip_benchmark
///
/// Read wall time in ms (opt mode, 100K rows, ~100 chunks per column), no
/// chunk stats -> chunk stats. Each cell takes the fastest of kNumRuns reads.
/// With 400 kept rows every chunk holds one, so all chunks are decoded either
/// way. A batch still opens the chunk where it starts, which shows with 40 kept
/// rows in batches of 1,024.
///  kept rows  batch 1024   batch 10000  stripe
///  4          121 -> 40    121 -> 43    122 -> 39
///  40         132 -> 96    134 -> 86    134 -> 77
///  400        133 -> 131   130 -> 132   129 -> 130
///  all        132 -> 131   134 -> 133   139 -> 135
///
/// Dictionary-encoded 'v' (400K rows, ~100 chunks), read wall time in ms, no
/// chunk stats -> chunk stats. Reading VARCHAR as a dictionary still loads
/// every chunk between the kept rows of a batch.
///  type     distinct  kept rows  batch 1024    batch 10000   stripe
///  VARCHAR  64        4          6.1 -> 4.0    6.0 -> 3.8    6.0 -> 5.9
///  VARCHAR  64        40         6.9 -> 5.4    6.8 -> 6.5    6.7 -> 6.7
///  VARCHAR  1024      4          19.1 -> 5.9   18.8 -> 5.6   20.0 -> 19.8
///  VARCHAR  1024      40         21.2 -> 12.5  21.5 -> 19.8  22.1 -> 22.0
///  BIGINT   64        4          4.8 -> 4.3    4.6 -> 4.1    4.6 -> 4.0
///  BIGINT   64        40         5.4 -> 5.0    5.2 -> 5.1    5.0 -> 4.7
///  BIGINT   1024      4          5.4 -> 4.4    5.2 -> 4.2    5.1 -> 4.2
///  BIGINT   1024      40         6.0 -> 5.3    5.8 -> 5.7    5.8 -> 5.0

#include <algorithm>
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <folly/init/Init.h>

#include "velox/common/time/Timer.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr vector_size_t kNumRows{100'000};
constexpr vector_size_t kNumDictionaryRows{400'000};
constexpr vector_size_t kWriteBatchRows{4'096};
constexpr int kNumRuns{5};

// Times of one read.
struct ReadTimes {
  // Wall time of the next() loop.
  uint64_t readNanos;
  // Decode CPU time of each measured column, from the reader's decoding stats.
  std::vector<uint64_t> decodeNanos;
};

// Settings of one read.
struct ReadOptions {
  // Rows per next() call; a call never reads past the end of a stripe.
  int batchSize;
  // Reads string dictionaries as dictionaries.
  bool dictionaryReads;
};

uint64_t decodeNanos(const dwio::common::RuntimeStats& stats, uint32_t node) {
  const auto nodeStats = stats.columnStats.find(node);
  NIMBLE_CHECK(nodeStats != stats.columnStats.end(), "Missing column stats");
  const auto format = nodeStats->second.find(dwio::common::FileFormat::NIMBLE);
  NIMBLE_CHECK(
      format != nodeStats->second.end() &&
          format->second.decodingStats.has_value(),
      "Missing decoding stats");
  return format->second.decodingStats->decodeCPUTimeNanos.sum();
}

// Returns the numbers 0 to 'numRows' - 1 in random order.
std::vector<int64_t> shuffledRowNumbers(vector_size_t numRows) {
  std::vector<int64_t> rowNumbers(numRows);
  std::iota(rowNumbers.begin(), rowNumbers.end(), 0);
  std::shuffle(rowNumbers.begin(), rowNumbers.end(), std::mt19937(7));
  return rowNumbers;
}

// Returns a string of 'length' random lowercase letters.
std::string randomString(std::mt19937& rng, int length) {
  std::uniform_int_distribution<int> letter('a', 'z');
  std::string result(length, ' ');
  for (auto& character : result) {
    character = static_cast<char>(letter(rng));
  }
  return result;
}

// Writes 'batches' to an in-memory file with chunks of half to all of
// 'maxChunkRawSize' raw bytes and stripes of 32 MB raw.
std::string writeFile(
    memory::MemoryPool& pool,
    const std::vector<VectorPtr>& batches,
    bool chunkStats,
    uint64_t maxChunkRawSize) {
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.eagerChunking = true;
  writerOptions.minStreamChunkRawSize = maxChunkRawSize / 2;
  writerOptions.maxStreamChunkRawSize = maxChunkRawSize;
  writerOptions.enableChunkStats = chunkStats;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<StripeRawSizeFlushPolicy>(32 << 20);
  };
  return test::createNimbleFile(
      pool, batches, writerOptions, /*flushAfterWrite=*/false);
}

std::string batchName(int batchSize, vector_size_t numRows) {
  return batchSize == numRows ? std::string("stripe")
                              : fmt::format("{}", batchSize);
}

// Writes the files of each part and prints the fastest read times of each
// query on the file without and the file with chunk stats.
class SparseReadChunkSkipBenchmark : public velox::test::VectorTestBase {
 public:
  SparseReadChunkSkipBenchmark() {
    registerSelectiveNimbleReaderFactory();
  }

  ~SparseReadChunkSkipBenchmark() {
    unregisterSelectiveNimbleReaderFactory();
  }

  // Reads long strings.
  void runLongStrings() {
    fmt::print(
        "{:<11}{:<8}{:>30}{:>30}\n",
        "keptRows",
        "batch",
        "no chunk stats: read / s / a",
        "chunk stats: read / s / a");
    const auto rowType =
        ROW({{"key", BIGINT()}, {"s", VARCHAR()}, {"a", ARRAY(VARCHAR())}});
    // Node ids of 's' and of the elements of 'a'. The root is node 0.
    const std::vector<uint32_t> nodes{2, 4};
    const auto batches = makeLongStrings();
    const auto withoutChunkStats =
        writeFile(*pool()->parent(), batches, /*chunkStats=*/false, 1 << 20);
    const auto withChunkStats =
        writeFile(*pool()->parent(), batches, /*chunkStats=*/true, 1 << 20);
    for (const vector_size_t numKept : {4, 40, 400, kNumRows}) {
      for (const int batchSize : {1'024, 10'000, kNumRows}) {
        const ReadOptions options{batchSize, /*dictionaryReads=*/false};
        fmt::print(
            "{:<11}{:<8}{:>30}{:>30}\n",
            numKept == kNumRows ? std::string("all")
                                : fmt::format("{}", numKept),
            batchName(batchSize, kNumRows),
            format(bestRead(
                withoutChunkStats, rowType, nodes, keys_, numKept, options)),
            format(bestRead(
                withChunkStats, rowType, nodes, keys_, numKept, options)));
      }
    }
  }

  // Reads a dictionary-encoded VARCHAR and BIGINT column.
  void runDictionaries() {
    fmt::print(
        "\n{:<9}{:<10}{:<10}{:<8}{:>22}{:>22}\n",
        "type",
        "distinct",
        "keptRows",
        "batch",
        "no chunk stats: read / v",
        "chunk stats: read / v");
    for (const auto& type : std::vector<TypePtr>{VARCHAR(), BIGINT()}) {
      const auto rowType = ROW({{"key", BIGINT()}, {"v", type}});
      // Node id of 'v'.
      const std::vector<uint32_t> nodes{2};
      // 256-byte strings or 8-byte integers in about 100 chunks.
      const uint64_t maxChunkRawSize =
          (type->isVarchar() ? 256 : 8) * kNumDictionaryRows / 100;
      for (const int numDistinct : {64, 1'024}) {
        const auto batches = makeDictionaryColumn(type, numDistinct);
        const auto withoutChunkStats = writeFile(
            *pool()->parent(), batches, /*chunkStats=*/false, maxChunkRawSize);
        const auto withChunkStats = writeFile(
            *pool()->parent(), batches, /*chunkStats=*/true, maxChunkRawSize);
        for (const vector_size_t numKept : {4, 40}) {
          for (const int batchSize : {1'024, 10'000, kNumDictionaryRows}) {
            const ReadOptions options{batchSize, /*dictionaryReads=*/true};
            fmt::print(
                "{:<9}{:<10}{:<10}{:<8}{:>22}{:>22}\n",
                type->toString(),
                numDistinct,
                numKept,
                batchName(batchSize, kNumDictionaryRows),
                format(bestRead(
                    withoutChunkStats,
                    rowType,
                    nodes,
                    dictionaryKeys_,
                    numKept,
                    options)),
                format(bestRead(
                    withChunkStats,
                    rowType,
                    nodes,
                    dictionaryKeys_,
                    numKept,
                    options)));
          }
        }
      }
    }
  }

 private:
  static std::string format(const ReadTimes& times) {
    auto result =
        fmt::format("{:.1f}", static_cast<double>(times.readNanos) / 1e6);
    for (const auto nanos : times.decodeNanos) {
      result += fmt::format(" / {:.1f}", static_cast<double>(nanos) / 1e6);
    }
    return result;
  }

  // Returns kNumRows rows in batches of kWriteBatchRows: 'key', a 1 KB string
  // 's', and 'a' with 0 to 7 strings of 256 bytes.
  std::vector<VectorPtr> makeLongStrings() {
    std::mt19937 rng(42);
    std::vector<VectorPtr> batches;
    for (vector_size_t firstRow = 0; firstRow < kNumRows;
         firstRow += kWriteBatchRows) {
      const auto numRows = std::min(kWriteBatchRows, kNumRows - firstRow);
      std::vector<vector_size_t> offsets;
      std::vector<std::string> elements;
      for (vector_size_t row = 0; row < numRows; ++row) {
        offsets.push_back(static_cast<vector_size_t>(elements.size()));
        const int numElements = (firstRow + row) % 8;
        for (int element = 0; element < numElements; ++element) {
          elements.push_back(randomString(rng, 256));
        }
      }
      batches.push_back(makeRowVector(
          {"key", "s", "a"},
          {makeFlatVector<int64_t>(
               numRows, [&](auto row) { return keys_[firstRow + row]; }),
           makeFlatVector<std::string>(
               numRows, [&](auto /*row*/) { return randomString(rng, 1'024); }),
           makeArrayVector(offsets, makeFlatVector<std::string>(elements))}));
    }
    return batches;
  }

  // Returns kNumDictionaryRows rows in batches of kWriteBatchRows: 'key' and
  // 'v' of 'type', which picks one of 'numDistinct' values at random.
  std::vector<VectorPtr> makeDictionaryColumn(
      const TypePtr& type,
      int numDistinct) {
    std::mt19937 rng(42);
    std::vector<std::string> strings;
    std::vector<int64_t> integers;
    for (int index = 0; index < numDistinct; ++index) {
      strings.push_back(randomString(rng, 256));
      integers.push_back(
          static_cast<int64_t>((static_cast<uint64_t>(rng()) << 32) | rng()));
    }
    std::uniform_int_distribution<int> pick(0, numDistinct - 1);
    std::vector<VectorPtr> batches;
    for (vector_size_t firstRow = 0; firstRow < kNumDictionaryRows;
         firstRow += kWriteBatchRows) {
      const auto numRows =
          std::min(kWriteBatchRows, kNumDictionaryRows - firstRow);
      const auto keys = makeFlatVector<int64_t>(
          numRows, [&](auto row) { return dictionaryKeys_[firstRow + row]; });
      VectorPtr values;
      if (type->isVarchar()) {
        values = makeFlatVector<std::string>(
            numRows, [&](auto /*row*/) { return strings[pick(rng)]; });
      } else {
        values = makeFlatVector<int64_t>(
            numRows, [&](auto /*row*/) { return integers[pick(rng)]; });
      }
      batches.push_back(makeRowVector({"key", "v"}, {keys, values}));
    }
    return batches;
  }

  ReadTimes bestRead(
      const std::string& file,
      const RowTypePtr& rowType,
      const std::vector<uint32_t>& nodes,
      const std::vector<int64_t>& keys,
      vector_size_t numKept,
      const ReadOptions& options) {
    ReadTimes best{UINT64_MAX, std::vector<uint64_t>(nodes.size(), UINT64_MAX)};
    for (int run = 0; run < kNumRuns; ++run) {
      const auto times = read(file, rowType, nodes, keys, numKept, options);
      best.readNanos = std::min(best.readNanos, times.readNanos);
      for (size_t index = 0; index < nodes.size(); ++index) {
        best.decodeNanos[index] =
            std::min(best.decodeNanos[index], times.decodeNanos[index]);
      }
    }
    return best;
  }

  // Reads 'file' once, keeping 'numKept' evenly spaced rows whose 'key' is in
  // 'keys', and returns the read time and the decode CPU of each of 'nodes'.
  ReadTimes read(
      const std::string& file,
      const RowTypePtr& rowType,
      const std::vector<uint32_t>& nodes,
      const std::vector<int64_t>& keys,
      vector_size_t numKept,
      const ReadOptions& options) {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    const auto numRows = static_cast<vector_size_t>(keys.size());
    if (numKept < numRows) {
      std::vector<int64_t> keptKeys;
      const auto step = numRows / numKept;
      for (vector_size_t index = 0; index < numKept; ++index) {
        keptKeys.push_back(keys[index * step + step / 2]);
      }
      scanSpec->childByName("key")->setFilter(
          common::createBigintValues(keptKeys, /*nullAllowed=*/false));
    }

    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setScanSpec(scanSpec);
    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE)
            ->createReader(
                std::make_unique<dwio::common::BufferedInput>(
                    std::make_shared<InMemoryReadFile>(file), *pool()),
                readerOptions);

    // The stripe loads in the timed loop rather than in createRowReader.
    dwio::common::RowReaderOptions rowReaderOptions;
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.setRequestedType(rowType);
    rowReaderOptions.setEagerFirstStripeLoad(false);
    rowReaderOptions.setStringDecoderZeroCopy(true);
    rowReaderOptions.setNimblePreserveDictionaryEncoding(
        options.dictionaryReads);
    rowReaderOptions.setNimbleDictionaryAwareReads(options.dictionaryReads);
    rowReaderOptions.setCollectColumnStats(true);
    auto rowReader = reader->createRowReader(rowReaderOptions);

    auto result = BaseVector::create(rowType, 0, pool());
    uint64_t readNanos{0};
    {
      NanosecondWallTimer timer(&readNanos);
      while (rowReader->next(options.batchSize, result) > 0) {
        // Projected columns load lazily; load them like a consumer would.
        for (auto& child : result->asUnchecked<RowVector>()->children()) {
          child = BaseVector::loadedVectorShared(child);
        }
      }
    }
    dwio::common::RuntimeStats stats;
    rowReader->updateRuntimeStats(stats);
    ReadTimes times{readNanos, {}};
    for (const auto node : nodes) {
      times.decodeNanos.push_back(decodeNanos(stats, node));
    }
    return times;
  }

  // The 'key' of each row of the long-string and the dictionary files.
  const std::vector<int64_t> keys_{shuffledRowNumbers(kNumRows)};
  const std::vector<int64_t> dictionaryKeys_{
      shuffledRowNumbers(kNumDictionaryRows)};
};

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::initializeMemoryManager(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::SparseReadChunkSkipBenchmark benchmark;
  benchmark.runLongStrings();
  benchmark.runDictionaries();
  return 0;
}
