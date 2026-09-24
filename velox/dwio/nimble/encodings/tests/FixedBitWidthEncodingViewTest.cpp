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

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <gtest/gtest.h>

#include <unordered_map>

#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FixedBitWidthEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsFixedBitWidthEncoding) {
  expectReads<nimble::FixedBitWidthEncoding<int32_t>>(
      makeVector({10, 12, 15, 31, 33, 63}), {5, 0, 4, 2, 1});
}

TEST_F(FixedBitWidthEncodingViewTest, readsInternallyCompressedPayload) {
  const auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/36);
  const auto positions = randomizedPositions(/*seed=*/37);
  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::FixedBitWidthEncoding<uint32_t>>(
        values, positions, {}, compressionType);
  }
}

TEST_F(FixedBitWidthEncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::FixedBitWidthEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/6),
      randomizedPositions(/*seed=*/7));
}

// KeyDerivedTransform falls back to hashing every row into an F14FastMap
// whenever denseRunIds declines, so this pins the direct-mapped table's
// output identical to that fallback's first-occurrence numbering -- the
// equivalence nothing else exercises, since before this the fast path never
// fired for FixedBitWidth at all.
TEST_F(FixedBitWidthEncodingViewTest, denseRunIdsMatchesHashFallback) {
  auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/13);
  auto serialized =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, {});
  auto view = nimble::createEncodingView(serialized, pool_.get(), {});
  ASSERT_NE(view, nullptr);

  std::vector<uint32_t> ids;
  std::vector<uint64_t> table;
  ASSERT_TRUE(
      view->denseRunIds(0, static_cast<uint32_t>(values.size()), ids, table));
  ASSERT_EQ(ids.size(), values.size());

  // The same first-occurrence numbering KeyDerivedTransform's fallback uses,
  // computed independently here so the fast path is checked against it
  // rather than against itself.
  std::unordered_map<uint32_t, uint32_t> runOf;
  std::vector<uint32_t> expectedIds(values.size());
  std::vector<uint32_t> expectedValues;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto inserted =
        runOf.try_emplace(values[i], static_cast<uint32_t>(runOf.size()));
    if (inserted.second) {
      expectedValues.push_back(values[i]);
    }
    expectedIds[i] = inserted.first->second;
  }

  EXPECT_EQ(ids, expectedIds);
  ASSERT_EQ(table.size(), expectedValues.size());
  for (size_t id = 0; id < table.size(); ++id) {
    uint32_t decoded;
    __builtin_memcpy(&decoded, &table[id], sizeof(decoded));
    EXPECT_EQ(decoded, expectedValues[id]) << "run id " << id;
  }
}
