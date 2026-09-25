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

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/SubIntSplitEncodingView.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

class TransformedEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  // Values shaped like a packed identifier: a low counter, a small worker
  // field and a slowly-varying high field, which is the structure SubIntSplit
  // is built to find. Seed defaults to the value every existing caller here
  // relies on; only callers that need two streams to disagree pass a
  // different one.
  Vector<uint64_t> packedIdentifiers(uint32_t count, uint64_t seed = 1234) {
    Vector<uint64_t> values{pool_.get()};
    values.resize(count);
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < count; ++i) {
      const uint64_t sequence = i & 0xFFF;
      const uint64_t worker = rng() % 24;
      const uint64_t timestamp = 1'700'000'000'000ULL + i / 8;
      values[i] = (timestamp << 22) | (worker << 12) | sequence;
    }
    return values;
  }

  // Decodes `encoded` through the encoding's bulk path and checks every row.
  template <typename Values>
  void expectMaterializes(
      std::string_view encoded,
      const Values& values,
      const Encoding::Options& options) {
    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, encoded, nullptr, options);
    std::vector<uint64_t> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(decoded[i], values[i]) << "row " << i;
    }
  }

  // Reads `encoded` through the view's bulk read and checks every row.
  template <typename Values>
  static void expectViewReads(
      const SubIntSplitEncodingView<uint64_t>& view,
      const Values& values) {
    std::vector<uint64_t> bulk(values.size());
    view.read(0, values.size(), bulk.data());
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(bulk[i], values[i]) << "bulk row " << i;
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

std::vector<TransformId> transformsUnderTest() {
  return {TransformId::KeyDerived};
}

} // namespace

// The layer is only worth anything if the reader hands back exactly what the
// writer was given, so this is the gate on every measurement that follows.
TEST_F(TransformedEncodingTest, roundTripsThroughTheEncoding) {
  const auto values = packedIdentifiers(4096);
  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    // Without this, selection is free to decide id does not pay on this
    // data and this test round-trips an untransformed stream while
    // believing it exercised id -- which is exactly what it did for six
    // rounds before anything here checked anyTransform() at all.
    options.subIntSplitForceApply = true;

    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    subintsplit::TransformInfo info;
    subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
    ASSERT_TRUE(info.anyTransform()) << toString(id) << " was not applied";
    SCOPED_TRACE(toString(id));
    expectMaterializes(encoded, values, options);
  }
}

// The forced ablation arm leaves the key to the search: every candidate key
// forces the transform on the other sections, and the smallest of those is
// kept, so the stream is transformed whatever the cost comparison would have
// said, and still reads back.
TEST_F(TransformedEncodingTest, forcedWithASearchedKeyTransforms) {
  const auto values = packedIdentifiers(4096);
  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform =
      static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
  options.subIntSplitKeySection = 0xFF;
  options.subIntSplitForceApply = true;
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  subintsplit::TransformInfo info;
  const auto sections =
      subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
  ASSERT_GT(sections.size(), 1u);
  ASSERT_TRUE(info.anyTransform());
  expectMaterializes(encoded, values, options);
}

// Layout capture is a third reader of this header, after the encoding and the
// view. It used to walk the header itself and treat the byte after splitCount
// as reserved, which is the byte that now says whether a transform block
// follows, so a reordered stream was captured from the wrong offset. The type
// was not in its switch either, and the switch has no default, so the capture
// came back childless and carrying no boundaries, reporting nothing wrong.
TEST_F(TransformedEncodingTest, capturesTheLayoutOfAReorderedStream) {
  const auto values = packedIdentifiers(4096);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});
  const auto plainLayout =
      EncodingLayoutCapture::capture(plain, Encoding::Options{});

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform =
      static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
  options.subIntSplitKeySection = 1;
  options.subIntSplitForceApply = true;
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  subintsplit::TransformInfo info;
  const auto sections =
      subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
  ASSERT_TRUE(info.anyTransform());

  const auto captured = EncodingLayoutCapture::capture(encoded, options);

  EXPECT_EQ(captured.encodingType(), EncodingType::SubIntSplitReordered);
  // One child per section, which is what a childless capture failed to give.
  EXPECT_EQ(captured.childrenCount(), sections.size());
  EXPECT_GT(captured.childrenCount(), 0u);
  // The split is a property of the values, and the permutation reorders rows
  // rather than changing which bits a section covers, so the boundaries match
  // the untransformed capture of the same data.
  EXPECT_EQ(captured.childrenCount(), plainLayout.childrenCount());
  EXPECT_EQ(
      captured.config().get(
          std::string(subintsplit::kSplitBoundariesConfigKey)),
      plainLayout.config().get(
          std::string(subintsplit::kSplitBoundariesConfigKey)));
}

// A stream that chose no transform must be byte-identical to one written
// before transforms existed, so existing data and readers are untouched.
TEST_F(TransformedEncodingTest, noTransformIsUnchangedOnTheWire) {
  const auto values = packedIdentifiers(2048);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  Buffer explicitBuffer{*pool_};
  Encoding::Options none;
  none.subIntSplitTransform = static_cast<uint8_t>(TransformId::None);
  const auto alsoPlain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      explicitBuffer, values, CompressionType::Uncompressed, none);

  EXPECT_EQ(plain.size(), alsoPlain.size());
  EXPECT_EQ(std::string(plain), std::string(alsoPlain));
}

// The encoding type is what tells a reader whether an inverse has to be
// applied, so it has to follow what selection actually chose rather than what
// was offered. Asserting on a particular transform being applied would only be
// asserting that it happened to pay on this data.
TEST_F(TransformedEncodingTest, announcesTheTypeThatMatchesWhatItChose) {
  const auto values = packedIdentifiers(8192);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});
  ASSERT_EQ(static_cast<EncodingType>(plain[0]), EncodingType::SubIntSplit);

  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    subintsplit::TransformInfo info;
    subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
    const auto type = static_cast<EncodingType>(encoded[0]);
    if (info.anyTransform()) {
      EXPECT_EQ(type, EncodingType::SubIntSplitReordered)
          << toString(id)
          << " was applied but the stream reads as untransformed";
    } else {
      EXPECT_EQ(type, EncodingType::SubIntSplit)
          << toString(id)
          << " was declined but the stream reads as transformed";
      EXPECT_EQ(std::string(encoded), std::string(plain))
          << toString(id) << " declined but did not leave the bytes alone";
    }
  }
}

// The key section rebuilds the order of the sections keyed on it, so it must
// reach the decoder untouched. Stated as an invariant over whatever selection
// chose, since a key is only held back when something was actually keyed on it.
TEST_F(TransformedEncodingTest, neverTransformsTheKeySection) {
  const auto values = packedIdentifiers(8192);
  for (uint8_t keySection : {uint8_t{0}, uint8_t{1}}) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform =
        static_cast<uint8_t>(TransformId::KeyDerived);
    options.subIntSplitKeySection = keySection;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    subintsplit::TransformInfo info;
    const auto sections =
        subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
    ASSERT_FALSE(sections.empty());
    if (info.anyTransform()) {
      ASSERT_EQ(info.keySection, keySection);
      EXPECT_EQ(info.transformIds[keySection], 0)
          << "the key section must not carry a transform";
    } else {
      EXPECT_EQ(info.keySection, subintsplit::TransformInfo::kNoKeySection)
          << "no section was keyed, so none should be held back as a key";
    }

    expectMaterializes(encoded, values, options);
  }
}

// The sequential decoder serves point and gather reads in the drivers by
// resetting, skipping and materialising, so a read that starts inside a
// transform block has to work rather than be refused.
TEST_F(TransformedEncodingTest, readsRangesThatStartInsideABlock) {
  const auto values = packedIdentifiers(9000);
  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, encoded, nullptr, options);

    // Offsets that fall inside a block, span a boundary, and reach the ragged
    // last block.
    const std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {1, 1}, {4095, 3}, {4096, 1}, {1234, 5000}, {8999, 1}, {0, 9000}};
    for (auto [offset, count] : ranges) {
      encoding->reset();
      if (offset > 0) {
        encoding->skip(offset);
      }
      std::vector<uint64_t> got(count);
      encoding->materialize(count, got.data());
      for (uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(got[i], values[offset + i])
            << toString(id) << " at offset " << offset << " row " << i;
      }
    }

    // A gather: several ranges in ascending order without an intervening
    // reset, which is how the gather driver drives it.
    encoding->reset();
    uint32_t position = 0;
    for (uint32_t start : {10u, 4100u, 4200u, 8000u}) {
      encoding->skip(start - position);
      uint64_t got = 0;
      encoding->materialize(1, &got);
      ASSERT_EQ(got, values[start]) << toString(id) << " gather at " << start;
      position = start + 1;
    }
  }
}

// A transform is applied to a section only where it pays for itself, so
// offering one can never produce a larger stream than not offering it. Without
// this, a transform forced onto every section reorders sections that had
// nothing to gain, which on a high-cardinality column costs more than the
// whole encoding.
TEST_F(TransformedEncodingTest, neverChoosesATransformThatCosts) {
  const auto values = packedIdentifiers(16384);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto offered = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    EXPECT_LE(offered.size(), plain.size())
        << toString(id) << " was applied where it did not pay";

    // Whatever it chose still has to read back.
    SCOPED_TRACE(toString(id));
    expectMaterializes(offered, values, options);
  }
}

// Selection has to be at least as good as declining, or it is not selection.
// Every candidate is priced against the untransformed encoding and plain is
// the incumbent, so a chosen transform is one that was strictly smaller --
// which makes "never larger than plain" the property, not an aspiration.
TEST_F(TransformedEncodingTest, autoSelectionNeverCostsMoreThanNoTransform) {
  const auto values = packedIdentifiers(16384);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitAutoTransform = true;
  const auto chosen = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  EXPECT_LE(chosen.size(), plain.size());

  expectMaterializes(chosen, values, options);
}

// Sections encoded concurrently are the sections encoded one after another:
// each is encoded exactly as it would be alone and written in section order.
TEST_F(TransformedEncodingTest, sectionExecutorEncodesTheSameBytes) {
  const auto values = packedIdentifiers(65'536);

  Buffer serialBuffer{*pool_};
  const auto serial = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      serialBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  folly::CPUThreadPoolExecutor executor{4};
  Buffer concurrentBuffer{*pool_};
  Encoding::Options options;
  options.subIntSplitSectionExecutor = &executor;
  const auto concurrent = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      concurrentBuffer, values, CompressionType::Uncompressed, options);

  EXPECT_EQ(serial, concurrent);
}

// Leaving the option off must change nothing at all. This is the check that
// the wider search is opt-in rather than a behaviour change smuggled into
// every column that already encodes with SubIntSplit, and it is the same
// standard the Burrows-Wheeler removal was held to.
TEST_F(TransformedEncodingTest, autoSelectionOffIsByteIdenticalToBefore) {
  const auto values = packedIdentifiers(16384);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  Buffer offBuffer{*pool_};
  Encoding::Options off;
  off.subIntSplitAutoTransform = false;
  const auto unchanged = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      offBuffer, values, CompressionType::Uncompressed, off);

  EXPECT_EQ(unchanged, plain);
}

// Asking the encoder to choose and ordering it to obey are contradictory, and
// silently honouring one would make the other's result a lie. A test that
// pinned a transform alongside auto would otherwise pass while measuring
// something nobody asked for.
//
// Internal rather than user: these options are set by nimble code, not by
// whoever wrote the file being encoded, which is also what the neighbouring
// subIntSplitForceApply validation checks against.
TEST_F(TransformedEncodingTest, autoSelectionRefusesToAlsoBeForced) {
  const auto values = packedIdentifiers(2048);

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitAutoTransform = true;
  options.subIntSplitForceApply = true;
  options.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);
  options.subIntSplitKeySection = 1;

  EXPECT_THROW(
      test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
          buffer, values, CompressionType::Uncompressed, options),
      NimbleInternalError);
}

// Every other test here pins the key section, so none of them runs the search
// that production actually uses: with subIntSplitKeySection unset the encoder
// prices one attempt per candidate key and keeps the smallest. That loop is the
// only place where work belonging to one candidate can reach another, and the
// permutation a key-derived transform gathers by is built once per attempt.
//
// A permutation left over from an earlier candidate would still decode without
// error, since the reader rebuilds the order from whichever key the header
// names; it would simply hand back rows in an order nobody asked for. So the
// check is not that the stream reads back, but that the search returned one of
// the attempts it was choosing between: whatever key it settled on, pinning
// that key has to reproduce the same bytes.
TEST_F(TransformedEncodingTest, keySearchReturnsOneOfTheAttemptsItPriced) {
  const auto values = packedIdentifiers(16384);

  Encoding::Options searchOptions;
  searchOptions.subIntSplitTransform =
      static_cast<uint8_t>(TransformId::KeyDerived);
  // 0xFF is the default, and means: try every section and keep the best.
  // subIntSplitForceApply is deliberately not set, since it requires a pinned
  // key and would therefore skip the very loop under test.
  searchOptions.subIntSplitKeySection = 0xFF;

  Buffer searchBuffer{*pool_};
  const auto searched = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      searchBuffer, values, CompressionType::Uncompressed, searchOptions);

  subintsplit::TransformInfo info;
  const auto sections =
      subintsplit::parseSections(searched, Encoding::kPrefixSize, &info);
  ASSERT_GT(sections.size(), 1u) << "a key search needs more than one section";

  expectMaterializes(searched, values, searchOptions);

  bool matchedSomeCandidate = false;
  for (uint8_t candidate = 0; candidate < sections.size(); ++candidate) {
    Buffer pinnedBuffer{*pool_};
    Encoding::Options pinnedOptions = searchOptions;
    pinnedOptions.subIntSplitKeySection = candidate;
    const auto pinned = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        pinnedBuffer, values, CompressionType::Uncompressed, pinnedOptions);
    if (pinned == searched) {
      matchedSomeCandidate = true;
      break;
    }
  }
  EXPECT_TRUE(matchedSomeCandidate)
      << "the search produced a stream no single candidate key reproduces";
}

// A key-derived permutation spans the whole section, and a probe follows the
// position map rather than rebuilding anything. This is the property that lets
// it go unblocked: if the map and a full decode ever disagreed, the gain from
// not blocking would be bought with wrong answers.
TEST_F(TransformedEncodingTest, keyDerivedProbesAgreeWithAFullDecode) {
  const auto values = packedIdentifiers(9000);
  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);
  options.subIntSplitKeySection = 1;
  options.subIntSplitForceApply = true;
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  subintsplit::TransformInfo info;
  subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
  ASSERT_TRUE(info.anyTransform()) << "KeyDerived was not applied";

  SubIntSplitEncodingView<uint64_t> view{encoded, pool_.get(), options};
  expectViewReads(view, values);
  // Probed out of order, so a map that only worked when walked forwards would
  // be caught.
  for (uint32_t i = 8999; i < values.size(); i -= 331) {
    ASSERT_EQ(view.readAt(i), values[i]) << "probe row " << i;
    if (i < 331) {
      break;
    }
  }
}

// A range-list read on a transformed stream chooses once, for the whole list,
// between reading each range the way a single-range read would and decoding
// the column once and copying the ranges out. Both choices have to return the
// source rows, KeyDerived through the position map. The lists include
// sparse ones that stay per range and dense ones that decode the column, and
// the untransformed stream is read the same way as the baseline.
TEST_F(TransformedEncodingTest, rangeListsAgreeWithTheSourceValues) {
  const auto values = packedIdentifiers(20'011);
  std::vector<TransformId> transforms{TransformId::None};
  const auto underTest = transformsUnderTest();
  transforms.insert(transforms.end(), underTest.begin(), underTest.end());
  for (auto id : transforms) {
    SCOPED_TRACE(toString(id));
    Buffer buffer{*pool_};
    Encoding::Options options;
    if (id != TransformId::None) {
      options.subIntSplitTransform = static_cast<uint8_t>(id);
      options.subIntSplitKeySection = 1;
      options.subIntSplitForceApply = true;
    }
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    subintsplit::TransformInfo info;
    subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
    ASSERT_EQ(info.anyTransform(), id != TransformId::None)
        << toString(id) << " was not applied as requested";

    SubIntSplitEncodingView<uint64_t> view{encoded, pool_.get(), options};
    test::expectRangeListReads(view, values);
  }
}

// A permuted section is moved at whatever width it was stored at, and the
// narrow widths are a different code path from the 64-bit one. Every other test
// here uses values whose sections land at eight bytes, so this shapes them to
// split into a narrow low section under a low-cardinality high one -- the shape
// Medicare1.NPI has, and the one that first exercised the narrow path only when
// a benchmark crashed on it.
//
// Needs three fields, not two: a bare (low, high) pair fits in 16 bits and the
// splitter keeps it as one section, which makes subIntSplitKeySection = 1
// (and subIntSplitForceApply's precondition) invalid -- that is what threw
// once this test was forced to actually apply the transform instead of
// silently declining it. Adding packedIdentifiers' third, wide, slowly-varying
// field is what reliably produces a multi-section split elsewhere in this
// file; it does not touch the low field, so the section under test stays
// exactly as narrow.
TEST_F(TransformedEncodingTest, permutesNarrowSectionsToo) {
  Vector<uint64_t> values{pool_.get()};
  values.resize(20000);
  std::mt19937_64 rng(31);
  for (uint32_t i = 0; i < values.size(); ++i) {
    // Ten low bits, so the low section is far narrower than a word, under a
    // low-cardinality middle field for the sort to group by, under a wide,
    // slowly-varying high field that is what makes the splitter cut this
    // into more than one section at all.
    const uint64_t low = rng() % 1024;
    const uint64_t mid = rng() % 40;
    const uint64_t high = 1'000'000ULL + i / 8;
    values[i] = (high << 16) | (mid << 10) | low;
  }

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);

  // Read before pinning, same reasoning as the cache tests: keySection = 1
  // throws rather than declining if the column does not actually split into
  // at least two sections.
  Buffer probeBuffer{*pool_};
  const auto probeEncoded =
      test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
          probeBuffer,
          values,
          CompressionType::Uncompressed,
          Encoding::Options{});
  subintsplit::TransformInfo probeInfo;
  const auto probeSections = subintsplit::parseSections(
      probeEncoded, Encoding::kPrefixSize, &probeInfo);
  ASSERT_GE(probeSections.size(), 2u)
      << "test needs a multi-section column so keySection=1 is valid; "
      << "this shape produced only one section";
  options.subIntSplitKeySection = 1;
  options.subIntSplitForceApply = true;

  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  subintsplit::TransformInfo info;
  subintsplit::parseSections(encoded, Encoding::kPrefixSize, &info);
  ASSERT_TRUE(info.anyTransform()) << "KeyDerived was not applied";

  SubIntSplitEncodingView<uint64_t> view{encoded, pool_.get(), options};
  expectViewReads(view, values);
  for (uint32_t i = 0; i < values.size(); i += 97) {
    ASSERT_EQ(view.readAt(i), values[i]) << "probe row " << i;
  }
  expectMaterializes(encoded, values, options);
}

// A retired transform id must be refused by every reader rather than decoded
// without its inverse, which would hand back plausible wrong values.
TEST_F(TransformedEncodingTest, readersRejectRetiredTransformIds) {
  const auto values = packedIdentifiers(4096);
  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);
  options.subIntSplitKeySection = 1;
  options.subIntSplitForceApply = true;
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  TransformInfo info;
  parseSections(encoded, Encoding::kPrefixSize, &info);
  const auto transformed = std::find_if(
      info.transformIds.begin(), info.transformIds.end(), [](uint8_t id) {
        return id != 0;
      });
  ASSERT_NE(transformed, info.transformIds.end());
  // The ids follow the flags byte, the row frame block when present, and the
  // key section byte.
  const uint8_t flags =
      static_cast<uint8_t>(encoded[Encoding::kPrefixSize + 1]);
  const size_t idOffset = Encoding::kPrefixSize + 2 +
      ((flags & kFlagRowFrame) != 0 ? kRowFrameHeaderSize : 0) + 1 +
      (transformed - info.transformIds.begin());
  ASSERT_EQ(
      static_cast<uint8_t>(encoded[idOffset]),
      static_cast<uint8_t>(TransformId::KeyDerived));

  for (const uint8_t retired : {2, 3, 4, 5, 6, 7}) {
    SCOPED_TRACE(static_cast<int>(retired));
    std::string corrupt{encoded.data(), encoded.size()};
    corrupt[idOffset] = static_cast<char>(retired);
    EXPECT_THROW(
        SubIntSplitEncoding<uint64_t>(*pool_, corrupt, nullptr, options),
        NimbleUserError);
    EXPECT_THROW(
        (SubIntSplitEncodingView<uint64_t>{corrupt, pool_.get(), options}),
        NimbleUserError);
  }
}

// PositionCache and BlockCache (SubIntSplitEncodingView.h) are thread_local,
// keyed only on the raw `this` pointer, with nothing to invalidate them.
// Every other test here builds exactly one view, so address reuse -- a
// second view constructed where an earlier, destroyed one lived -- has never
// been exercised. Placement-new forces that reuse deterministically, the way
// a stack slot in a loop or an allocator size class could produce it in
// production without any help.
TEST_F(
    TransformedEncodingTest,
    positionCacheDoesNotLeakAcrossViewsAtTheSameAddress) {
  using ViewType = SubIntSplitEncodingView<uint64_t>;

  // packedIdentifiers with keySection=1 is the shape every passing transform
  // test in this file already uses to get KeyDerived selected -- two earlier
  // synthetic generators here (a random key/payload pair, then a
  // mergeirr-shaped one with no explicit field boundary) both failed for
  // reasons specific to how the automatic splitter and the opt-in cost gate
  // work, not because the general idea was wrong. Different seeds so the two
  // streams' sorted key orders, and therefore their position maps, differ --
  // and different row counts, not just different seeds, so a leak cannot
  // hide behind cache.positions already happening to be the right size: the
  // block-cache leak this same trick found was only visible once the two
  // streams' cached array sizes actually disagreed.
  Buffer bufferA{*pool_};
  const auto valuesA = packedIdentifiers(4096, /*seed=*/111);
  Encoding::Options optionsA;
  optionsA.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);

  // Read before pinning: subIntSplitKeySection = 1 throws
  // "key section is outside the split" if the column happens to produce
  // fewer than two sections, which an uncaught NimbleInternalError from
  // inside encode() would make confusing to diagnose. The split does not
  // depend on subIntSplitTransform or subIntSplitKeySection, so probing with
  // default options finds the same boundaries the real encode below will.
  Buffer probeBufferA{*pool_};
  const auto probeEncodedA =
      test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
          probeBufferA,
          valuesA,
          CompressionType::Uncompressed,
          Encoding::Options{});
  subintsplit::TransformInfo probeInfoA;
  const auto sectionsA = subintsplit::parseSections(
      probeEncodedA, Encoding::kPrefixSize, &probeInfoA);
  ASSERT_GE(sectionsA.size(), 2u)
      << "test needs a multi-section column so keySection=1 is valid; "
      << "packedIdentifiers produced only one section";
  optionsA.subIntSplitKeySection = 1;
  optionsA.subIntSplitForceApply = true;

  const auto encodedA = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      bufferA, valuesA, CompressionType::Uncompressed, optionsA);
  subintsplit::TransformInfo infoA;
  subintsplit::parseSections(encodedA, Encoding::kPrefixSize, &infoA);
  ASSERT_TRUE(infoA.anyTransform())
      << "test precondition: KeyDerived must actually be selected for "
      << "stream A, or this test exercises nothing";

  Buffer bufferB{*pool_};
  const auto valuesB = packedIdentifiers(9000, /*seed=*/222);
  Encoding::Options optionsB;
  optionsB.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);

  Buffer probeBufferB{*pool_};
  const auto probeEncodedB =
      test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
          probeBufferB,
          valuesB,
          CompressionType::Uncompressed,
          Encoding::Options{});
  subintsplit::TransformInfo probeInfoB;
  const auto sectionsB = subintsplit::parseSections(
      probeEncodedB, Encoding::kPrefixSize, &probeInfoB);
  ASSERT_GE(sectionsB.size(), 2u)
      << "test needs a multi-section column so keySection=1 is valid; "
      << "packedIdentifiers produced only one section";
  optionsB.subIntSplitKeySection = 1;
  optionsB.subIntSplitForceApply = true;

  const auto encodedB = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      bufferB, valuesB, CompressionType::Uncompressed, optionsB);
  subintsplit::TransformInfo infoB;
  subintsplit::parseSections(encodedB, Encoding::kPrefixSize, &infoB);
  ASSERT_TRUE(infoB.anyTransform())
      << "test precondition: KeyDerived must actually be selected for "
      << "stream B, or this test exercises nothing";

  // Self-check that a leaked cache would actually be visible: if every row's
  // value happened to agree between the streams at the positions a leaked
  // permutation would touch, this test would pass whether or not the bug is
  // present. Uniform-random low bits over independent seeds make an
  // across-the-board coincidence astronomically unlikely; assert it directly
  // rather than trust that.
  int differing = 0;
  for (size_t i = 0; i < valuesA.size(); ++i) {
    differing += (valuesA[i] != valuesB[i]);
  }
  ASSERT_GT(differing, static_cast<int>(valuesA.size()) / 2)
      << "test precondition: streams A and B must mostly disagree row by "
      << "row, or a leaked cache would not be distinguishable from a correct "
      << "one";

  alignas(ViewType) unsigned char storage[sizeof(ViewType)];

  auto* viewA = new (storage) ViewType(encodedA, pool_.get(), optionsA);
  {
    SCOPED_TRACE("stream A");
    expectViewReads(*viewA, valuesA);
  }
  const void* addressA = static_cast<void*>(viewA);
  viewA->~ViewType();

  // A different key-derived stream, constructed at the exact address the
  // first view just vacated. If PositionCache's `cache.owner == this` check
  // is fooled by the reused address, this bulk read comes back permuted by
  // stream A's position map instead of stream B's own.
  auto* viewB = new (storage) ViewType(encodedB, pool_.get(), optionsB);
  ASSERT_EQ(static_cast<void*>(viewB), addressA)
      << "test precondition: placement-new must reuse the same address, or "
      << "this test exercises nothing";
  {
    SCOPED_TRACE("stream B");
    expectViewReads(*viewB, valuesB);
  }
  viewB->~ViewType();
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
