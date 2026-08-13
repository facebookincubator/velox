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

/// E2E benchmark: measures total column decode CPU time through the full
/// Velox selective reader pipeline. Writes a 10M-row Nimble file, reads it
/// back with filters at various selectivities. Includes all reader overhead:
/// encoding construction, row selection, lazy materialization, filter
/// evaluation — not just the raw decode step.
///
/// For isolated decode-only microbenchmarks, see BlockBitPackingBenchmark.cpp.
///
/// Two-column schema: ROW(c0: INTEGER, c1: INTEGER).
///   - c0: filter column — random values in [0, 499]. A BigintRange filter on
///     c0 determines which rows pass. c0 is always scanned densely.
///   - c1: data column — the column under test with varying encoding. When c0
///     has a filter, c1 is read SPARSELY (only rows passing c0's filter),
///     exercising the sparse bulkScan path.
///
/// Encodings under test (applied globally to both columns):
///   - Trivial + MetaInternal compression
///   - FixedBitWidth + MetaInternal compression
///   - BlockBitPacking, uncompressed
///
/// Data patterns for c1:
///   - NarrowBlocks: per-block 4 bits, global ~20 bits (BBP sweet spot)
///   - Uniform: ~20 bits everywhere (all encodings comparable)
///
/// Run:
///   buck2 run @//mode/opt \
///     fbcode//velox/dwio/nimble/velox/selective/tests:int_encoding_benchmark
///
/// Results (opt mode, 10M rows, c1 decode time in ms):
///
/// NarrowBlocks (int32, per-block 4-bit, decode_ms(decompress_ms)):
///              Dense     50%      20%      10%       8%       1%     0.2%
///  Trivial+MI  40(34)   38(35)   39(36)   39(36)   38(35)   38(35)   38(35)
///  FBW+MI      71(42)   69(41)   56(42)   49(42)   48(41)   48(42)   46(42)
///  BBP          5(0)     4(0)     9(0)     5(0)     5(0)     4(0)     4(0)
///
/// Uniform (int32, ~20-bit uniform, decode_ms(decompress_ms)):
///              Dense     50%      20%      10%       8%       1%     0.2%
///  Trivial+MI  46(44)   46(44)   47(44)   46(44)   48(44)   48(44)   47(44)
///  FBW+MI      58(28)   56(29)   55(28)   37(29)   35(28)   34(29)   32(28)
///  BBP          8(0)     6(0)    12(0)     8(0)     7(0)     5(0)     4(0)
///
/// BBP multi-type (decode_ms(decompress_ms)):
///              Dense     50%      20%      10%       8%      *5%*      1%
///  int8         8(0)     7(0)     7(0)     7(0)     9(0)     6(0)     4(0)
///  int16        7(0)     5(0)    11(0)     9(0)     8(0)     6(0)     4(0)
///  int32        8(0)     6(0)    12(0)    10(0)    10(0)     7(0)     5(0)
///  int64        9(0)     7(0)    13(0)    10(0)    10(0)     8(0)     5(0)

#include <random>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

constexpr int kTotalRows = 10'000'000;
constexpr int kBatchSize = 1'024;

auto kType = ROW({"c0", "c1"}, {INTEGER(), INTEGER()});

/// Builds writer options that force a single encoding type with the given
/// compression. Pass std::nullopt for comprOpts to disable compression.
WriterOptions makeWriterOptions(
    EncodingType encoding,
    std::optional<CompressionOptions> comprOpts) {
  WriterOptions writerOptions;
  writerOptions.encodingSelectionPolicyCreator =
      [encodingFactory =
           ManualEncodingSelectionPolicyFactory{{{encoding, 1.0}}, comprOpts}](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };
  return writerOptions;
}

struct ColumnStats {
  uint64_t decodeNs;
  uint64_t decompressNs;
};

struct BenchResult {
  int64_t constructorUs;
  int64_t decodeUs;
  int64_t nextUs;
  int64_t lazyLoadUs;
  ColumnStats c0;
  ColumnStats c1;
};

struct BenchmarkState : public velox::test::VectorTestBase {
  BenchmarkState() {
    registerSelectiveNimbleReaderFactory();
  }

  ~BenchmarkState() {
    unregisterSelectiveNimbleReaderFactory();
  }

  template <typename T>
  VectorPtr makeVector(const std::vector<T>& values) {
    return makeFlatVector<T>(values);
  }

  /// Filter column: random values in [0, 499].
  /// BigintRange(0, 0) → ~0.2% selectivity.
  /// BigintRange(0, 4) → ~1% selectivity.
  /// BigintRange(0, 99) → ~20% selectivity.
  VectorPtr generateFilterColumn() {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> dist(0, 499);
    std::vector<int32_t> values(kTotalRows);
    for (int i = 0; i < kTotalRows; ++i) {
      values[i] = dist(rng);
    }
    return makeFlatVector<int32_t>(values);
  }

  /// Each 1024-row block has random values in a narrow range
  /// [blockIdx*1000, blockIdx*1000+15] (4 bits per block), but the global
  /// range spans ~1M (~20 bits).
  VectorPtr generateNarrowBlocks() {
    std::mt19937 rng(42);
    std::vector<int32_t> values(kTotalRows);
    for (int i = 0; i < kTotalRows; ++i) {
      auto blockIdx = i / 1024;
      values[i] = blockIdx * 1000 + static_cast<int32_t>(rng() % 16);
    }
    return makeFlatVector<int32_t>(values);
  }

  /// Random values uniformly distributed in [0, 1M). Per-block and global
  /// bit widths are both ~20 bits.
  VectorPtr generateUniform() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 999'999);
    std::vector<int32_t> values(kTotalRows);
    for (int i = 0; i < kTotalRows; ++i) {
      values[i] = dist(rng);
    }
    return makeFlatVector<int32_t>(values);
  }

  /// Writes a two-column file: c0 = filterCol, c1 = dataCol.
  /// Logs the write time and file size.
  std::string writeFile(
      const std::string& label,
      const VectorPtr& filterCol,
      const VectorPtr& dataCol,
      EncodingType encoding,
      std::optional<CompressionOptions> comprOpts) {
    auto opts = makeWriterOptions(encoding, comprOpts);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto file = test::createNimbleFile(
        *pool()->parent(),
        makeRowVector({"c0", "c1"}, {filterCol, dataCol}),
        std::move(opts));
    auto t1 = std::chrono::high_resolution_clock::now();
    auto writeUs =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    LOG(INFO) << "Write " << label << ": fileSize=" << file.size()
              << " bytes, writeTime=" << writeUs << "us";
    return file;
  }

  void readNoFilter(const std::string& file, const TypePtr& rowType = kType) {
    auto scanSpec = makeScanSpec(rowType);
    auto readers = createReaders(file, scanSpec, rowType);
    drainReader(*readers.rowReader, rowType);
  }

  void readSparse(
      const std::string& file,
      int64_t lower,
      int64_t upper,
      const TypePtr& rowType = kType) {
    auto scanSpec = makeScanSpec(rowType);
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(lower, upper, false));
    auto readers = createReaders(file, scanSpec, rowType);
    drainReader(*readers.rowReader, rowType);
  }

  BenchResult timedRead(
      const std::string& file,
      std::optional<std::pair<int64_t, int64_t>> filter,
      const TypePtr& rowType = kType) {
    auto scanSpec = makeScanSpec(rowType);
    if (filter.has_value()) {
      scanSpec->childByName("c0")->setFilter(
          std::make_unique<common::BigintRange>(
              filter->first, filter->second, false));
      // Force c1 to be read eagerly (not lazy) so it receives the
      // sparse RowSet from c0's filter and goes through the sparse
      // bulkScan path (V::dense = false).
      scanSpec->childByName("c1")->setFilter(
          std::make_unique<common::AlwaysTrue>());
    }

    auto us = [](auto d) {
      return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
    };

    // Constructor: createReader + createRowReader. Eager stripe load
    // disabled so encoding construction moves into the decode phase.
    auto t0 = std::chrono::high_resolution_clock::now();
    auto readers = createReaders(file, scanSpec, rowType);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Decode via next() loop. For dense (no filter), both columns
    // are lazy — loadedVectorShared forces materialization. For
    // sparse (filter on c0), c0 is read eagerly in next(), c1 is
    // lazy-loaded after.
    auto drain = drainReader(*readers.rowReader, rowType);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Extract per-column decode and decompression stats (CPU time).
    // nodeId: root=0, c0=1, c1=2.
    dwio::common::RuntimeStatistics stats;
    readers.rowReader->updateRuntimeStats(stats);
    ColumnStats c0Stats{};
    ColumnStats c1Stats{};
    if (stats.columnReaderStats.decodingStatsSet.has_value()) {
      auto* c0m = stats.columnReaderStats.decodingStatsSet->getOrCreate(1);
      c0Stats = {
          c0m->decodeCPUTimeNanos.sum(), c0m->decompressCPUTimeNanos.sum()};
      auto* c1m = stats.columnReaderStats.decodingStatsSet->getOrCreate(2);
      c1Stats = {
          c1m->decodeCPUTimeNanos.sum(), c1m->decompressCPUTimeNanos.sum()};
    }

    return {
        us(t1 - t0),
        us(t2 - t1),
        drain.nextUs,
        drain.lazyLoadUs,
        c0Stats,
        c1Stats};
  }

 private:
  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  std::shared_ptr<common::ScanSpec> makeScanSpec(const TypePtr& rowType) {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    return scanSpec;
  }

  Readers createReaders(
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const TypePtr& rowType) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setScanSpec(scanSpec);
    options.setDataIoStats(ioStats_);
    options.setMetadataIoStats(ioStats_);

    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);

    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(
        std::dynamic_pointer_cast<const RowType>(rowType));
    rowOptions.setEagerFirstStripeLoad(false);
    // zeroCopy=true uses nimble::EncodingFactory (not legacy), which forwards
    // Encoding::Options including decodingStats to encoding constructors.
    rowOptions.setStringDecoderZeroCopy(true);
    rowOptions.setCollectColumnStats(true);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  struct DrainStats {
    int64_t nextUs{0};
    int64_t lazyLoadUs{0};
    int64_t totalUs{0};
  };

  DrainStats drainReader(
      dwio::common::RowReader& rowReader,
      const TypePtr& rowType = kType) {
    auto result = BaseVector::create(rowType, 0, pool());
    int64_t sum = 0;
    int64_t nextUs = 0;
    int64_t lazyUs = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
      auto tn0 = std::chrono::high_resolution_clock::now();
      auto rows = rowReader.next(kBatchSize, result);
      auto tn1 = std::chrono::high_resolution_clock::now();
      nextUs += std::chrono::duration_cast<std::chrono::microseconds>(tn1 - tn0)
                    .count();
      if (rows == 0) {
        break;
      }
      auto tl0 = std::chrono::high_resolution_clock::now();
      auto* row = result->as<RowVector>();
      for (auto i = 0; i < row->childrenSize(); ++i) {
        auto loaded = BaseVector::loadedVectorShared(row->childAt(i));
        if (loaded->size() > 0) {
          sum += loaded->hashValueAt(0);
        }
      }
      auto tl1 = std::chrono::high_resolution_clock::now();
      lazyUs += std::chrono::duration_cast<std::chrono::microseconds>(tl1 - tl0)
                    .count();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    folly::doNotOptimizeAway(sum);
    return {
        nextUs,
        lazyUs,
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
  }

  std::shared_ptr<velox::io::IoStatistics> ioStats_ =
      std::make_shared<velox::io::IoStatistics>();
};

BenchmarkState& state() {
  static BenchmarkState s;
  return s;
}

const CompressionOptions kMetaInternal = CompressionOptions{};

// --- Narrow-block files (c0=filter, c1=narrow-block data) ---

std::string& narrowTrivial() {
  static auto f = state().writeFile(
      "NarrowBlocks Trivial+MI",
      state().generateFilterColumn(),
      state().generateNarrowBlocks(),
      EncodingType::Trivial,
      kMetaInternal);
  return f;
}

std::string& narrowFBW() {
  static auto f = state().writeFile(
      "NarrowBlocks FBW+MI    ",
      state().generateFilterColumn(),
      state().generateNarrowBlocks(),
      EncodingType::FixedBitWidth,
      kMetaInternal);
  return f;
}

std::string& narrowBBP() {
  static auto f = state().writeFile(
      "NarrowBlocks BBP       ",
      state().generateFilterColumn(),
      state().generateNarrowBlocks(),
      EncodingType::BlockBitPacking,
      std::nullopt);
  return f;
}

// --- Uniform files (c0=filter, c1=uniform data) ---

std::string& uniformTrivial() {
  static auto f = state().writeFile(
      "Uniform      Trivial+MI",
      state().generateFilterColumn(),
      state().generateUniform(),
      EncodingType::Trivial,
      kMetaInternal);
  return f;
}

std::string& uniformFBW() {
  static auto f = state().writeFile(
      "Uniform      FBW+MI    ",
      state().generateFilterColumn(),
      state().generateUniform(),
      EncodingType::FixedBitWidth,
      kMetaInternal);
  return f;
}

std::string& uniformBBP() {
  static auto f = state().writeFile(
      "Uniform      BBP       ",
      state().generateFilterColumn(),
      state().generateUniform(),
      EncodingType::BlockBitPacking,
      std::nullopt);
  return f;
}

/// Runs N iterations, reports min constructor, decode, and per-column stats.
void benchmark(
    const std::string& label,
    const std::string& file,
    std::optional<std::pair<int64_t, int64_t>> filter,
    const TypePtr& rowType = kType,
    int runs = 10) {
  int64_t minCtor = std::numeric_limits<int64_t>::max();
  int64_t minDecode = std::numeric_limits<int64_t>::max();
  BenchResult best{};
  for (int r = 0; r < runs; ++r) {
    auto res = state().timedRead(file, filter, rowType);
    auto msNs2 = [](double ns) { return ns / 1e6; };
    if (res.c0.decodeNs < 100000) { // < 0.1ms — outlier
      LOG(INFO) << label << " run" << r << " OUTLIER:"
                << " decode=" << res.decodeUs / 1000.0
                << "ms, next=" << res.nextUs / 1000.0
                << "ms, lazy=" << res.lazyLoadUs / 1000.0
                << "ms, c0_dec=" << msNs2(res.c0.decodeNs)
                << "ms, c1_dec=" << msNs2(res.c1.decodeNs) << "ms";
    }
    if (res.decodeUs < minDecode) {
      minDecode = res.decodeUs;
      best = res;
    }
    minCtor = std::min(minCtor, res.constructorUs);
  }
  auto ms = [](double us) { return us / 1000.0; };
  auto msNs = [](double ns) { return ns / 1e6; };
  LOG(INFO) << label << ": total=" << ms(minCtor + minDecode)
            << "ms [ctor=" << ms(minCtor) << ", next=" << ms(best.nextUs)
            << ", lazy=" << ms(best.lazyLoadUs)
            << "] | c0[cpu_dec=" << msNs(best.c0.decodeNs)
            << ", cpu_decomp=" << msNs(best.c0.decompressNs)
            << "] c1[cpu_dec=" << msNs(best.c1.decodeNs)
            << ", cpu_decomp=" << msNs(best.c1.decompressNs) << "]";
}

void runBenchmarks() {
  struct FileSet {
    const char* name;
    std::string& (*trivial)();
    std::string& (*fbw)();
    std::string& (*bbp)();
  };
  FileSet files[] = {
      {"NarrowBlocks", narrowTrivial, narrowFBW, narrowBBP},
      {"Uniform     ", uniformTrivial, uniformFBW, uniformBBP},
  };

  struct Filter {
    const char* name;
    std::optional<std::pair<int64_t, int64_t>> range;
  };
  Filter filters[] = {
      {"Dense     ", std::nullopt},
      {"Sparse50% ", {{0, 249}}},
      {"Sparse20% ", {{0, 99}}},
      {"Sparse10% ", {{0, 49}}},
      {"Sparse8%  ", {{0, 39}}},
      {"Sparse1%  ", {{0, 4}}},
      {"Sparse0.2%", {{0, 0}}},
  };

  for (auto& fs : files) {
    LOG(INFO) << "=== " << fs.name << " ===";
    for (auto& flt : filters) {
      LOG(INFO) << "--- " << flt.name << " ---";
      benchmark(std::string("  Trivial+MI"), fs.trivial(), flt.range);
      benchmark(std::string("  FBW+MI    "), fs.fbw(), flt.range);
      benchmark(std::string("  BBP       "), fs.bbp(), flt.range);
    }
  }
}

void runMultiTypeBenchmarks() {
  auto& s = state();

  struct TypeConfig {
    const char* name;
    TypePtr veloxType;
    std::function<VectorPtr()> generate;
  };

  auto makeUniform = [&](auto maxVal) {
    using T = decltype(maxVal);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(
        0, static_cast<int64_t>(maxVal));
    std::vector<T> values(kTotalRows);
    for (int i = 0; i < kTotalRows; ++i) {
      values[i] = static_cast<T>(dist(rng));
    }
    return s.makeVector<T>(values);
  };

  TypeConfig types[] = {
      {"int8", TINYINT(), [&]() { return makeUniform(int8_t{63}); }},
      {"int16", SMALLINT(), [&]() { return makeUniform(int16_t{16383}); }},
      {"int32", INTEGER(), [&]() { return makeUniform(int32_t{999999}); }},
      {"int64", BIGINT(), [&]() { return makeUniform(int64_t{999999999}); }},
  };

  struct Filter {
    const char* name;
    std::optional<std::pair<int64_t, int64_t>> range;
  };
  Filter filters[] = {
      {"Dense     ", std::nullopt},
      {"Sparse50% ", {{0, 249}}},
      {"Sparse20% ", {{0, 99}}},
      {"Sparse15% ", {{0, 74}}},
      {"Sparse10% ", {{0, 49}}},
      {"Sparse8%  ", {{0, 39}}},
      {"Sparse5%  ", {{0, 24}}},
      {"Sparse1%  ", {{0, 4}}},
      {"Sparse0.4%", {{0, 1}}},
  };

  auto filterCol = s.generateFilterColumn();

  for (auto& tc : types) {
    auto dataCol = tc.generate();
    auto rowType = ROW({"c0", "c1"}, {INTEGER(), tc.veloxType});
    auto file = s.writeFile(
        std::string("BBP ") + tc.name,
        filterCol,
        dataCol,
        EncodingType::BlockBitPacking,
        std::nullopt);

    LOG(INFO) << "=== BBP " << tc.name << " ===";
    for (auto& flt : filters) {
      LOG(INFO) << "--- " << flt.name << " ---";
      benchmark(std::string("  BBP ") + tc.name, file, flt.range, rowType);
    }
  }
}

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::testingSetInstance(
      facebook::velox::memory::MemoryManager::Options{});

  facebook::nimble::runBenchmarks();
  facebook::nimble::runMultiTypeBenchmarks();
  return 0;
}
