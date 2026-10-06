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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <folly/String.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/Validation.h"

// The drivers only publish a timing for a cell whose reads returned the input,
// so the checks are pinned here against a target that is wrong on purpose: it
// corrupts one row, optionally only once it has served some clean reads, the
// way an arm that goes wrong once warm would.

namespace facebook::nimble::mlidc {
namespace {

Vector<int64_t> makeData(uint32_t n) {
  Vector<int64_t> data{benchmarks::benchmarkPool().get()};
  data.resize(n);
  uint64_t state = 0x2545F4914F6CDD1DULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<int64_t>((state >> 40) ^ (i % 13));
  }
  return data;
}

// Holds the column in the clear, like CopyTarget in ChunkedTargetTest.cpp,
// but returns a wrong value for one row once it has served numCleanReads
// reads. Can also be told to reverse the column, which models an arm that
// returns the multiset in another order, or to throw.
class CorruptingTarget : public NimbleBenchTargetBase<int64_t> {
 public:
  CorruptingTarget(
      const Vector<int64_t>& data,
      std::optional<uint32_t> corruptRow,
      size_t numCleanReads)
      : values_(data.data(), data.data() + data.size()),
        corruptRow_{corruptRow},
        numCleanReads_{numCleanReads} {}

  void encode(
      const Vector<int64_t>& data,
      const Encoding::Options&,
      const subintsplit::TuningConfig&) override {
    values_.assign(data.data(), data.data() + data.size());
  }

  void materializeAll(int64_t* dst, uint32_t n) override {
    copyRows(0, n, dst);
  }

  void materializeRange(uint32_t begin, uint32_t count, int64_t* dst) override {
    copyRows(begin, count, dst);
  }

  void skipThenMaterialize(
      std::span<const nimble::RowRange> ranges,
      int64_t* dst) override {
    const bool corrupt = nextReadCorrupts();
    for (const auto& range : ranges) {
      copyRowsAs(range.startRow, range.numRows(), dst, corrupt);
      dst += range.numRows();
    }
  }

  size_t payloadSize() const override {
    return values_.size() * sizeof(int64_t);
  }

  size_t residentBytes() const override {
    return payloadSize();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {};
  }

  ReadPath readPath() const override {
    return ReadPath::kIndexed;
  }

  void setReversed(bool reversed) {
    reversed_ = reversed;
  }

  void setThrows(bool throws) {
    throws_ = throws;
  }

 private:
  bool nextReadCorrupts() {
    if (throws_) {
      throw std::runtime_error("decode failed");
    }
    return numReads_++ >= numCleanReads_;
  }

  void copyRows(uint32_t begin, uint32_t count, int64_t* dst) {
    copyRowsAs(begin, count, dst, nextReadCorrupts());
  }

  void copyRowsAs(uint32_t begin, uint32_t count, int64_t* dst, bool corrupt) {
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t row = begin + i;
      const size_t source = reversed_ ? values_.size() - 1 - row : row;
      dst[i] = values_[source];
      if (corrupt && corruptRow_ == row) {
        dst[i] ^= 1;
      }
    }
  }

  std::vector<int64_t> values_;
  std::optional<uint32_t> corruptRow_;
  size_t numCleanReads_;
  size_t numReads_{0};
  bool reversed_{false};
  bool throws_{false};
};

constexpr uint32_t kRows = 1'000;
constexpr uint32_t kCorruptRow = 417;

EncoderEntry<int64_t> makeEntry(bool preservesRowOrder) {
  EncoderEntry<int64_t> entry;
  entry.name = "Corrupting";
  entry.preservesRowOrder = preservesRowOrder;
  return entry;
}

TEST(ValidationTest, correctTargetPassesEveryRead) {
  const auto data = makeData(kRows);
  CorruptingTarget target{data, std::nullopt, 0};
  const auto entry = makeEntry(true);
  ValidationReference<int64_t> reference;
  ASSERT_EQ(
      ValidationReference<int64_t>::build(target, entry, data, reference),
      std::nullopt);

  const std::vector<size_t> probes = {0, kCorruptRow, kRows - 1};
  const std::vector<nimble::RowRange> ranges = {{10, 20}, {400, 500}};
  ValidationLedger ledger;
  EXPECT_TRUE(ledger.check("Corrupting", "d", "all", [&] {
    return validateMaterializeAll<int64_t>(target, reference.values());
  }));
  EXPECT_TRUE(ledger.check("Corrupting", "d", "points", [&] {
    return validatePoints<int64_t>(target, reference.values(), probes);
  }));
  EXPECT_TRUE(ledger.check("Corrupting", "d", "range", [&] {
    return validateRange<int64_t>(target, reference.values(), 400, 100);
  }));
  EXPECT_TRUE(ledger.check("Corrupting", "d", "gather", [&] {
    return validateGather<int64_t>(target, reference.values(), ranges);
  }));
  EXPECT_EQ(ledger.failures(), 0);
  EXPECT_EQ(ledger.exitCode(), 0);
}

// Each read shape catches the corrupt row when, and only when, it reads it,
// and the message names the row.
TEST(ValidationTest, oneCorruptRowFailsEveryReadThatReachesIt) {
  const auto data = makeData(kRows);
  CorruptingTarget target{data, kCorruptRow, 0};
  const std::span<const int64_t> input(data.data(), data.size());

  EXPECT_THAT(
      validateMaterializeAll<int64_t>(target, input),
      testing::Optional(testing::HasSubstr("row 417")));
  EXPECT_THAT(
      validatePoints<int64_t>(
          target, input, std::vector<size_t>{3, kCorruptRow, 9}),
      testing::Optional(testing::HasSubstr("row 417")));
  EXPECT_THAT(
      validateRange<int64_t>(target, input, 400, 100),
      testing::Optional(testing::HasSubstr("row 417")));
  EXPECT_THAT(
      validateGather<int64_t>(
          target, input, std::vector<nimble::RowRange>{{0, 5}, {410, 420}}),
      testing::Optional(testing::HasSubstr("row 417")));

  EXPECT_EQ(
      validatePoints<int64_t>(target, input, std::vector<size_t>{3, 9}),
      std::nullopt);
  EXPECT_EQ(validateRange<int64_t>(target, input, 0, 400), std::nullopt);
  EXPECT_EQ(
      validateGather<int64_t>(
          target, input, std::vector<nimble::RowRange>{{0, 5}, {418, 420}}),
      std::nullopt);
}

// A target that is right on its first read and wrong once warm passes the
// check made before timing; the check on the last timed output catches it.
TEST(ValidationTest, corruptionOnceWarmIsCaughtOnTheTimedOutput) {
  const auto data = makeData(kRows);
  CorruptingTarget target{data, kCorruptRow, 1};
  const std::span<const int64_t> input(data.data(), data.size());
  const std::vector<nimble::RowRange> ranges = {{100, 200}, {400, 450}};

  ValidationLedger ledger;
  EXPECT_TRUE(ledger.check("Corrupting", "d", "before timing", [&] {
    return validateGather<int64_t>(target, input, ranges);
  }));

  std::vector<int64_t> sink(kRows);
  poisonOutput<int64_t>(sink);
  for (int iteration = 0; iteration < 3; ++iteration) {
    target.skipThenMaterialize(ranges, sink.data());
  }
  EXPECT_FALSE(ledger.check("Corrupting", "d", "timed output", [&] {
    return checkGatherOutput<int64_t>(
        std::span<const int64_t>(sink.data(), 150), input, ranges);
  }));
  EXPECT_EQ(ledger.failures(), 1);
  EXPECT_EQ(ledger.exitCode(), 2);
}

// A read that writes nothing must not pass on what the buffer already held.
TEST(ValidationTest, poisonedOutputFailsAReadThatWritesNothing) {
  const auto data = makeData(kRows);
  const std::span<const int64_t> input(data.data(), data.size());
  std::vector<int64_t> sink(input.begin(), input.end());
  poisonOutput<int64_t>(sink);
  EXPECT_NE(firstMismatch<int64_t>(sink, input), std::nullopt);
}

TEST(ValidationTest, throwingReadCountsAsAFailure) {
  const auto data = makeData(kRows);
  CorruptingTarget target{data, std::nullopt, 0};
  target.setThrows(true);
  ValidationLedger ledger;
  EXPECT_FALSE(ledger.check("Corrupting", "d", "all", [&] {
    return validateMaterializeAll<int64_t>(
        target, std::span<const int64_t>(data.data(), data.size()));
  }));
  EXPECT_EQ(ledger.exitCode(), 2);
}

// An arm that does not preserve row order is held to a permutation of the
// input, and its partial reads to its own full decode.
TEST(ValidationTest, orderFreeArmIsCheckedAsAPermutation) {
  const auto data = makeData(kRows);
  const auto entry = makeEntry(false);

  CorruptingTarget reversed{data, std::nullopt, 0};
  reversed.setReversed(true);
  ValidationReference<int64_t> reference;
  ASSERT_EQ(
      ValidationReference<int64_t>::build(reversed, entry, data, reference),
      std::nullopt);
  EXPECT_EQ(
      validatePoints<int64_t>(
          reversed, reference.values(), std::vector<size_t>{0, kCorruptRow}),
      std::nullopt);

  CorruptingTarget corrupt{data, kCorruptRow, 0};
  corrupt.setReversed(true);
  ValidationReference<int64_t> corruptReference;
  EXPECT_THAT(
      ValidationReference<int64_t>::build(
          corrupt, entry, data, corruptReference),
      testing::Optional(testing::HasSubstr("not a permutation")));
}

// What a sweep reads back: a failed cell is a row marked validated=0 and
// skipped=1 with no timing, and a passing one says validated=1.
TEST(ValidationTest, failedCellIsWrittenAsNotValidated) {
  const auto path =
      (std::filesystem::temp_directory_path() / "mlidc_validation_test.csv")
          .string();
  const auto entry = makeEntry(true);
  {
    CsvResultWriter csv(
        path, {"driver", "encoding", "A", "time_ns", "skipped", "validated"});
    writeValidationFailureRow<int64_t>(
        csv, "test", "d", entry, [&] { csv.set("A", int64_t{kCorruptRow}); });
    csv.flush();
  }
  std::ifstream file(path);
  std::string header;
  std::string row;
  ASSERT_TRUE(std::getline(file, header));
  ASSERT_TRUE(std::getline(file, row));
  std::vector<std::string> fields;
  folly::split(',', row, fields);
  EXPECT_THAT(
      fields, testing::ElementsAre("test", "Corrupting", "417", "", "1", "0"));
  std::filesystem::remove(path);
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  facebook::velox::memory::MemoryManager::initialize({});
  return RUN_ALL_TESTS();
}

#else
int main() {
  return 0;
}
#endif
