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

// Times loading one Iceberg deletion vector and applying it to every batch of
// a 1M-row file, the work DeletionVectorReader does for one split.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <map>
#include <numeric>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/FileSink.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;

namespace {

constexpr int32_t kRows = 1'000'000;
constexpr int32_t kBatchSize = 10'000;

// java.util.Random, so the deleted rows match those an Iceberg Java benchmark
// draws with `new Random(seed)`, and both sides decode identical bitmaps.
class JavaRandom {
 public:
  explicit JavaRandom(int64_t seed) : seed_((seed ^ kMultiplier) & kMask) {}

  int32_t nextInt(int32_t bound) {
    if ((bound & -bound) == bound) {
      return static_cast<int32_t>(
          (static_cast<int64_t>(bound) * next(31)) >> 31);
    }
    int32_t bits;
    int32_t value;
    do {
      bits = next(31);
      value = bits % bound;
    } while (static_cast<int32_t>(
                 static_cast<uint32_t>(bits) - static_cast<uint32_t>(value) +
                 static_cast<uint32_t>(bound - 1)) < 0);
    return value;
  }

 private:
  static constexpr int64_t kMultiplier = 0x5DEECE66DLL;
  static constexpr int64_t kMask = (int64_t{1} << 48) - 1;

  int32_t next(int bits) {
    seed_ = (seed_ * kMultiplier + 0xBLL) & kMask;
    return static_cast<int32_t>(seed_ >> (48 - bits));
  }

  int64_t seed_;
};

// Picks 'pct' percent of the rows with a partial Fisher-Yates shuffle.
std::vector<int64_t> deletedRows(int32_t pct) {
  const int32_t count = static_cast<int32_t>(int64_t{kRows} * pct / 100);
  std::vector<int32_t> rows(kRows);
  std::iota(rows.begin(), rows.end(), 0);
  JavaRandom random(42 + pct);
  for (int32_t i = 0; i < count; ++i) {
    std::swap(rows[i], rows[i + random.nextInt(kRows - i)]);
  }
  return std::vector<int64_t>(rows.begin(), rows.begin() + count);
}

struct Fixture {
  std::shared_ptr<common::testutil::TempDirectoryPath> directory;
  std::shared_ptr<memory::MemoryPool> pool;
  std::map<int32_t, IcebergDeleteFile> files;
};

Fixture& fixture() {
  static Fixture instance;
  return instance;
}

IcebergDeleteFile writeDeletionVector(int32_t pct) {
  auto& state = fixture();
  const auto rows = deletedRows(pct);
  DeletionVectorWriter writer;
  writer.addDeletedPositions(rows);

  const auto path =
      fmt::format("{}/dv_{}.puffin", state.directory->getPath(), pct);
  auto sink = dwio::common::FileSink::create(
      "file:" + path, {.pool = state.pool.get()});
  const auto [offset, length] = writePuffinFile(
      *sink,
      *state.pool,
      writer.serialize(),
      "data.parquet",
      static_cast<int64_t>(rows.size()));
  const auto size = sink->size();
  sink->close();

  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      path,
      dwio::common::FileFormat::PUFFIN,
      rows.size(),
      size,
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      static_cast<int64_t>(offset),
      static_cast<int64_t>(length),
      "data.parquet");
}

// Loads the deletion vector, then fills one delete bitmap per batch when
// 'apply' is set.
void run(uint32_t iterations, int32_t pct, bool apply) {
  auto& state = fixture();
  const auto& file = state.files.at(pct);
  BufferPtr bitmap;
  BENCHMARK_SUSPEND {
    bitmap = AlignedBuffer::allocate<uint8_t>(
        bits::nbytes(kBatchSize), state.pool.get(), 0);
  }
  for (uint32_t i = 0; i < iterations; ++i) {
    DeletionVectorReader reader(file, 0, state.pool.get(), nullptr);
    reader.readDeletePositions(0, 0, bitmap);
    if (!apply) {
      continue;
    }
    uint64_t deleted = 0;
    for (int32_t base = 0; base < kRows; base += kBatchSize) {
      std::memset(bitmap->asMutable<uint8_t>(), 0, bitmap->size());
      reader.readDeletePositions(base, kBatchSize, bitmap);
      deleted += bits::countBits(bitmap->as<uint64_t>(), 0, kBatchSize);
    }
    VELOX_CHECK_EQ(deleted, file.recordCount);
  }
}

void load(uint32_t iterations, int32_t pct) {
  run(iterations, pct, false);
}

void loadAndApply(uint32_t iterations, int32_t pct) {
  run(iterations, pct, true);
}

} // namespace

BENCHMARK_NAMED_PARAM(load, 1_pct, 1)
BENCHMARK_NAMED_PARAM(load, 5_pct, 5)
BENCHMARK_NAMED_PARAM(load, 20_pct, 20)
BENCHMARK_NAMED_PARAM(load, 50_pct, 50)
BENCHMARK_DRAW_LINE();
BENCHMARK_NAMED_PARAM(loadAndApply, 1_pct, 1)
BENCHMARK_NAMED_PARAM(loadAndApply, 5_pct, 5)
BENCHMARK_NAMED_PARAM(loadAndApply, 20_pct, 20)
BENCHMARK_NAMED_PARAM(loadAndApply, 50_pct, 50)

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  filesystems::registerLocalFileSystem();
  dwio::common::registerFileSinks();

  auto& state = fixture();
  state.directory = common::testutil::TempDirectoryPath::create();
  state.pool =
      memory::memoryManager()->addLeafPool("DeletionVectorReaderBenchmark");
  for (int32_t pct : {1, 5, 20, 50}) {
    state.files.emplace(pct, writeDeletionVector(pct));
  }

  folly::runBenchmarks();
  state.files.clear();
  state.pool.reset();
  state.directory.reset();
  return 0;
}
