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

#include <memory>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BlockCodecTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"

// The benchmark drivers report numbers rather than assert them, so the two
// properties the amortisation measurement rests on are pinned here: that a
// target answers honestly when the harness asks how its reads are served, and
// that a structure built once is built once.

namespace facebook::nimble::mlidc {
namespace {

Vector<int64_t> makeData(uint32_t n) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<int64_t> data{pool.get()};
  data.resize(n);
  uint64_t state = 0x2545F4914F6CDD1DULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<int64_t>((state >> 40) ^ (i % 13));
  }
  return data;
}

// Counts how often it is asked to decode the column, so that "decoded once,
// not once per read" is assertable without timing anything.
//
// Models a blackbox codec: every read, however small, decodes everything.
class CountingTarget : public NimbleBenchTargetBase<int64_t> {
 public:
  explicit CountingTarget(std::vector<int64_t> values)
      : values_{std::move(values)} {}

  void encode(const Vector<int64_t>&, const Encoding::Options&) override {}

  void materializeAll(int64_t* dst, uint32_t n) override {
    ++numFullDecodes_;
    std::copy_n(values_.data(), n, dst);
  }

  void materializeRange(uint32_t begin, uint32_t count, int64_t* dst) override {
    ++numFullDecodes_;
    std::copy_n(values_.data() + begin, count, dst);
  }

  void skipThenMaterialize(
      std::span<const nimble::RowRange> ranges,
      int64_t* dst) override {
    ++numFullDecodes_;
    for (const auto& range : ranges) {
      std::copy_n(values_.data() + range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
  }

  size_t payloadSize() const override {
    return values_.size() * sizeof(int64_t);
  }

  // Holds only what it was given. A decorator wrapping this one adds whatever
  // it keeps decoded on top, which is what the resident-bytes tests below
  // measure the difference of.
  size_t residentBytes() const override {
    return values_.size() * sizeof(int64_t);
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(values_.data()),
         values_.size() * sizeof(int64_t)}};
  }

  ReadPath readPath() const override {
    return ReadPath::kWholePayload;
  }

  size_t numFullDecodes() const {
    return numFullDecodes_;
  }

 private:
  std::vector<int64_t> values_;
  size_t numFullDecodes_{0};
};

constexpr uint32_t kRows = 3'000;

std::vector<int64_t> toVector(const Vector<int64_t>& data) {
  return std::vector<int64_t>(data.data(), data.data() + data.size());
}

// The claim the +materialize arms make. A blackbox codec answers every probe by
// decoding the column; wrapping it decodes once and serves the rest from the
// buffer, and that difference is the whole reason the arm exists. Asserted on a
// decode counter rather than on elapsed time, so it holds on any machine.
TEST(MaterializingTargetTest, decodesOnceForManyReads) {
  const auto data = makeData(kRows);
  constexpr size_t kProbes = 64;

  auto bare = std::make_unique<CountingTarget>(toVector(data));
  auto* bareRaw = bare.get();
  int64_t value{};
  for (size_t i = 0; i < kProbes; ++i) {
    bareRaw->materializeRange(static_cast<uint32_t>(i * 7), 1, &value);
  }
  EXPECT_EQ(bareRaw->numFullDecodes(), kProbes);

  auto inner = std::make_unique<CountingTarget>(toVector(data));
  auto* innerRaw = inner.get();
  MaterializingTarget<int64_t> materializing{std::move(inner), kRows};
  for (size_t i = 0; i < kProbes; ++i) {
    const auto index = static_cast<uint32_t>(i * 7);
    materializing.materializeRange(index, 1, &value);
    ASSERT_EQ(value, data[index]) << "probe " << i;
  }
  EXPECT_EQ(innerRaw->numFullDecodes(), 1u);
  EXPECT_EQ(materializing.numBuilds(), 1u);
}

// The other half of what +materialize costs. decodesOnceForManyReads says it
// stops decoding; this says what it holds in order to stop, which is a whole
// decoded column on top of the payload. The pair is the point of the
// resident-bytes axis: the arm buys its per-probe time with memory, and a
// chart showing only time cannot see the price.
TEST(MaterializingTargetTest, residentBytesGrowsByADecodedColumn) {
  const auto data = makeData(kRows);
  auto inner = std::make_unique<CountingTarget>(toVector(data));
  const size_t innerBytes = inner->residentBytes();
  MaterializingTarget<int64_t> target{std::move(inner), kRows};

  // Nothing decoded yet, so it holds what the inner codec holds.
  EXPECT_EQ(target.residentBytes(), innerBytes);

  int64_t value{};
  target.materializeRange(11, 1, &value);
  EXPECT_GE(target.residentBytes(), innerBytes + kRows * sizeof(int64_t));

  // A second read adds nothing: the column is already held.
  const size_t afterFirst = target.residentBytes();
  target.materializeRange(12, 1, &value);
  EXPECT_EQ(target.residentBytes(), afterFirst);

  // Discarding gives it back, or a driver's discard/rebuild cycle would leak.
  target.discardAccessStructure();
  EXPECT_EQ(target.residentBytes(), innerBytes);
}

// Every target answers residentBytes, and a compressed-only arm must answer
// with something close to its payload rather than with a decoded column. This
// is the baseline the frontier is drawn against, so a cursor arm quietly
// holding more than it stores would be visible here.
TEST(CursorTargetTest, residentBytesIsAboutThePayload) {
  const auto data = makeData(kRows);
  NimbleBenchTargetImpl<FixedBitWidthEncoding<int64_t>> target;
  target.encode(data, Encoding::Options{});

  const size_t raw = static_cast<size_t>(kRows) * sizeof(int64_t);
  int64_t value{};
  target.materializeRange(23, 1, &value);
  ASSERT_EQ(value, data[23]);

  EXPECT_GE(target.residentBytes(), target.payloadSize());
  // FixedBitWidth keeps no decoded copy, so a read must not have left one.
  EXPECT_LT(target.residentBytes(), raw);
}

// Discarding has to actually discard, or a driver measuring construction would
// measure a no-op and report a build cost of nothing.
TEST(MaterializingTargetTest, discardForcesTheNextReadToRebuild) {
  const auto data = makeData(kRows);
  auto inner = std::make_unique<CountingTarget>(toVector(data));
  auto* innerRaw = inner.get();
  MaterializingTarget<int64_t> target{std::move(inner), kRows};

  int64_t value{};
  target.materializeRange(11, 1, &value);
  EXPECT_EQ(innerRaw->numFullDecodes(), 1u);

  target.discardAccessStructure();
  target.materializeRange(11, 1, &value);
  EXPECT_EQ(innerRaw->numFullDecodes(), 2u);
  EXPECT_EQ(value, data[11]);
}

// A view target must survive the discard/rebuild cycle the drivers put it
// through, on the encoded bytes rather than on a stub.
TEST(ViewTargetTest, rebuildsTheViewAfterDiscard) {
  const auto data = makeData(kRows);
  NimbleViewBenchTargetImpl<FixedBitWidthEncoding<int64_t>> target;
  target.encode(data, Encoding::Options{});

  EXPECT_TRUE(target.buildsAccessStructure());
  EXPECT_EQ(target.readPath(), ReadPath::kIndexed);

  int64_t value{};
  target.materializeRange(17, 1, &value);
  ASSERT_EQ(value, data[17]);

  target.discardAccessStructure();
  value = 0;
  target.materializeRange(17, 1, &value);
  EXPECT_EQ(value, data[17]);

  target.discardAccessStructure();
  target.buildAccessStructure();
  std::vector<int64_t> all(kRows);
  target.materializeAll(all.data(), kRows);
  EXPECT_THAT(all, testing::ElementsAreArray(data.data(), kRows));
}

// A cursor target has nothing to build, and must say so rather than report a
// build cost that is really a first read.
TEST(CursorTargetTest, hasNoBuildPhase) {
  const auto data = makeData(kRows);
  NimbleBenchTargetImpl<FixedBitWidthEncoding<int64_t>> target;
  target.encode(data, Encoding::Options{});

  EXPECT_FALSE(target.buildsAccessStructure());
  EXPECT_EQ(target.readPath(), ReadPath::kCursor);

  // No-ops, and reads keep working across them.
  target.discardAccessStructure();
  target.buildAccessStructure();
  int64_t value{};
  target.materializeRange(23, 1, &value);
  EXPECT_EQ(value, data[23]);
}

// The column the point driver writes as emulated_point_read. Only an indexed
// read answers a one-row probe with one row's work; the other three paths
// decode more than they were asked for.
TEST(ReadPathTest, onlyIndexedServesAPointReadDirectly) {
  EXPECT_TRUE(servesPointReadDirectly(ReadPath::kIndexed));
  EXPECT_FALSE(servesPointReadDirectly(ReadPath::kCursor));
  EXPECT_FALSE(servesPointReadDirectly(ReadPath::kBlock));
  EXPECT_FALSE(servesPointReadDirectly(ReadPath::kWholePayload));
}

// A block arm is addressable only while the column spans more than one block.
// Configured block size does not settle that; how the column actually landed
// does, which is why the target derives it instead of the entry declaring it.
TEST(ReadPathTest, blockTargetReportsWholePayloadWhenItHoldsOneBlock) {
  const auto data = makeData(kRows);

  BlockCompressedTarget<int64_t> manyBlocks{
      std::make_unique<NimbleBlockCodec<int64_t>>(CompressionType::Zstd),
      /*blockSize=*/64,
      "test"};
  manyBlocks.encode(data, Encoding::Options{});
  EXPECT_GT(manyBlocks.numBlocks(), 1u);
  EXPECT_EQ(manyBlocks.readPath(), ReadPath::kBlock);

  BlockCompressedTarget<int64_t> oneBlock{
      std::make_unique<NimbleBlockCodec<int64_t>>(CompressionType::Zstd),
      /*blockSize=*/kRows * 2,
      "test"};
  oneBlock.encode(data, Encoding::Options{});
  EXPECT_EQ(oneBlock.numBlocks(), 1u);
  EXPECT_EQ(oneBlock.readPath(), ReadPath::kWholePayload);
}

// The blackbox arms the sweep registers, and their materialised twins. Both
// halves must round-trip, and the pair must differ in read path and in nothing
// else, or the amortisation curve is comparing two different things.
TEST(BlackboxArmTest, wholeColumnArmsAndTheirMaterializedTwins) {
  const auto data = makeData(kRows);

  const std::vector<EncoderEntry<int64_t>> bare{
      buildZstdWholeEncoder<int64_t>(), buildOpenZLEncoder<int64_t>()};
  for (const auto& entry : bare) {
    auto twin = withMaterializedAccess<int64_t>(entry);
    EXPECT_EQ(twin.name, entry.name + "+materialize");

    auto bareTarget = entry.factory(data, Encoding::Options{});
    auto twinTarget = twin.factory(data, Encoding::Options{});
    EXPECT_EQ(bareTarget->readPath(), ReadPath::kWholePayload) << entry.name;
    EXPECT_FALSE(bareTarget->buildsAccessStructure()) << entry.name;
    EXPECT_EQ(twinTarget->readPath(), ReadPath::kIndexed) << twin.name;
    EXPECT_TRUE(twinTarget->buildsAccessStructure()) << twin.name;

    // Materialising changes what a read costs, not what the column occupies.
    EXPECT_EQ(twinTarget->payloadSize(), bareTarget->payloadSize())
        << entry.name;

    std::vector<int64_t> out(kRows);
    twinTarget->materializeAll(out.data(), kRows);
    EXPECT_THAT(out, testing::ElementsAreArray(data.data(), kRows))
        << twin.name;

    int64_t value{};
    twinTarget->materializeRange(kRows - 1, 1, &value);
    EXPECT_EQ(value, data[kRows - 1]) << twin.name;
  }
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
