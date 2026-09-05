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
#include "velox/exec/RowLayout.h"

#include <gtest/gtest.h>

#include "velox/exec/RowContainer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {
namespace {

// Tests for RowLayout. Two flavors:
//
//  1. 'verifyLayoutMatches' cases build a real RowContainer with a parameter
//     set, call RowLayout::compute() with the equivalent options, and assert
//     the publicly observable layout (fixedRowSize, nextOffset, rowSizeOffset,
//     probedFlagOffset, countOffset and per-column RowColumn) is identical.
//     RowContainer's constructor consumes RowLayout, so these assertions
//     guard the public layout accessors against regressions.
//
//  2. 'goldenXxx' cases compare RowLayout::compute() against hand-traced
//     expected values. They lock in absolute correctness independent of
//     RowContainer.
class RowLayoutTest : public testing::Test, public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void verifyLayoutMatches(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<Accumulator>& accumulators,
      const std::vector<TypePtr>& dependentTypes,
      const RowLayoutOptions& options) {
    RowContainer container(
        keyTypes,
        options.nullableKeys,
        accumulators,
        dependentTypes,
        options.hasNext,
        /*isJoinBuild=*/false,
        options.hasProbedFlag,
        options.hasCountFlag,
        options.hasNormalizedKeys,
        /*useListRowIndex=*/false,
        pool_.get());

    auto layout =
        RowLayout::compute(keyTypes, accumulators, dependentTypes, options);

    EXPECT_EQ(layout.fixedRowSize, container.fixedRowSize());
    EXPECT_EQ(layout.nextOffset, container.nextOffset());
    EXPECT_EQ(layout.rowSizeOffset, container.rowSizeOffset());
    EXPECT_EQ(layout.probedFlagOffset, container.probedFlagOffset());
    EXPECT_EQ(layout.countOffset, container.countOffset());

    const size_t expectedColumns =
        keyTypes.size() + accumulators.size() + dependentTypes.size();
    ASSERT_EQ(layout.rowColumns.size(), expectedColumns);
    ASSERT_EQ(layout.offsets.size(), expectedColumns);
    ASSERT_EQ(layout.nullOffsets.size(), expectedColumns);

    for (size_t i = 0; i < expectedColumns; ++i) {
      const auto fromContainer = container.columnAt(i);
      const auto fromLayout = layout.rowColumns[i];
      EXPECT_EQ(fromLayout.offset(), fromContainer.offset()) << "column " << i;
      EXPECT_EQ(fromLayout.nullByte(), fromContainer.nullByte())
          << "column " << i;
      EXPECT_EQ(fromLayout.nullMask(), fromContainer.nullMask())
          << "column " << i;
    }
  }

  // Returns a fixed-size accumulator with the given byte size and alignment.
  // The destroy and spill-extract callbacks are inert; the layout computation
  // does not invoke them.
  static Accumulator makeAccumulator(int32_t size, int32_t alignment) {
    return Accumulator(
        /*isFixedSize=*/true,
        size,
        /*usesExternalMemory=*/false,
        alignment,
        /*spillType=*/nullptr,
        [](folly::Range<char**>, VectorPtr&) { VELOX_UNREACHABLE(); },
        [](folly::Range<char**>) {});
  }

  // Returns a variable-size accumulator. Layout treats it as variable-width,
  // forcing a row-size counter.
  static Accumulator makeVariableAccumulator(int32_t alignment) {
    return Accumulator(
        /*isFixedSize=*/false,
        /*fixedSize=*/16,
        /*usesExternalMemory=*/false,
        alignment,
        /*spillType=*/nullptr,
        [](folly::Range<char**>, VectorPtr&) { VELOX_UNREACHABLE(); },
        [](folly::Range<char**>) {});
  }
};

TEST_F(RowLayoutTest, keysOnlyNullable) {
  verifyLayoutMatches({BIGINT(), INTEGER(), SMALLINT()}, {}, {}, {});
}

TEST_F(RowLayoutTest, keysOnlyNotNullable) {
  verifyLayoutMatches({BIGINT(), INTEGER()}, {}, {}, {.nullableKeys = false});
}

TEST_F(RowLayoutTest, keysWithDependents) {
  verifyLayoutMatches({BIGINT()}, {}, {VARCHAR(), DOUBLE(), BOOLEAN()}, {});
}

TEST_F(RowLayoutTest, withProbedFlag) {
  verifyLayoutMatches(
      {BIGINT(), VARCHAR()},
      {},
      {INTEGER()},
      {.hasNext = true, .hasProbedFlag = true});
}

TEST_F(RowLayoutTest, withCountFlag) {
  verifyLayoutMatches(
      {BIGINT()}, {}, {}, {.nullableKeys = false, .hasCountFlag = true});
}

TEST_F(RowLayoutTest, withNextPointer) {
  verifyLayoutMatches({BIGINT()}, {}, {DOUBLE()}, {.hasNext = true});
}

TEST_F(RowLayoutTest, withFixedAccumulator) {
  verifyLayoutMatches(
      {BIGINT()}, {makeAccumulator(/*size=*/8, /*alignment=*/8)}, {}, {});
}

TEST_F(RowLayoutTest, withAlignedAccumulator) {
  // Mirrors the alignment exercise inside RowContainerTest.alignment.
  verifyLayoutMatches(
      {SMALLINT()},
      {makeAccumulator(/*size=*/42, /*alignment=*/64)},
      {},
      {.hasNormalizedKeys = true});
}

TEST_F(RowLayoutTest, withVariableWidthAccumulator) {
  verifyLayoutMatches(
      {BIGINT()}, {makeVariableAccumulator(/*alignment=*/8)}, {}, {});
}

TEST_F(RowLayoutTest, withNormalizedKey) {
  verifyLayoutMatches(
      {BIGINT(), INTEGER()}, {}, {}, {.hasNormalizedKeys = true});
}

TEST_F(RowLayoutTest, variableWidthKey) {
  verifyLayoutMatches({VARCHAR()}, {}, {}, {});
}

TEST_F(RowLayoutTest, complexTypes) {
  verifyLayoutMatches(
      {BIGINT()},
      {},
      {ARRAY(INTEGER()), MAP(BIGINT(), VARCHAR()), ROW({BIGINT(), VARCHAR()})},
      {});
}

TEST_F(RowLayoutTest, allFlagsOn) {
  verifyLayoutMatches(
      {BIGINT(), VARCHAR()},
      {makeAccumulator(/*size=*/16, /*alignment=*/8),
       makeAccumulator(/*size=*/24, /*alignment=*/16)},
      {DOUBLE(), ARRAY(INTEGER())},
      {.hasNext = true,
       .hasProbedFlag = true,
       .hasCountFlag = true,
       .hasNormalizedKeys = true});
}

TEST_F(RowLayoutTest, manyKeysForceFlagOverflow) {
  // With many nullable keys plus an accumulator, the flag region is forced to
  // straddle the next byte boundary at the start of the accumulator flags.
  // This exercises the flagOffset = (flagOffset + 7) & -8 alignment step.
  verifyLayoutMatches(
      {BIGINT(),
       INTEGER(),
       SMALLINT(),
       TINYINT(),
       BIGINT(),
       INTEGER(),
       SMALLINT(),
       TINYINT(),
       BIGINT(),
       INTEGER(),
       SMALLINT(),
       TINYINT()},
      {makeAccumulator(/*size=*/8, /*alignment=*/8)},
      {},
      {});
}

// Hand-traced layout for a minimal nullable-keys case.
// Layout:
//   bytes  0..7  : key 0 (BIGINT)
//   bytes  8..11 : key 1 (INTEGER)
//   bytes 12     : flag byte (bit 0 = key0 null, bit 1 = key1 null,
//                              bit 2 = free flag)
TEST_F(RowLayoutTest, goldenSimpleKeys) {
  auto layout = RowLayout::compute(
      {BIGINT(), INTEGER()}, /*accumulators=*/{}, /*dependentTypes=*/{}, {});

  EXPECT_EQ(layout.fixedRowSize, 13);
  EXPECT_EQ(layout.flagBytes, 1);
  EXPECT_EQ(layout.alignment, 1);
  EXPECT_EQ(layout.freeFlagOffset, 12 * 8 + 2);
  EXPECT_EQ(layout.probedFlagOffset, 0);
  EXPECT_EQ(layout.nextOffset, 0);
  EXPECT_EQ(layout.countOffset, 0);
  EXPECT_EQ(layout.rowSizeOffset, 0);
  EXPECT_EQ(layout.originalNormalizedKeySize, 0);
  EXPECT_FALSE(layout.usesExternalMemory);

  ASSERT_EQ(layout.offsets.size(), 2);
  EXPECT_EQ(layout.offsets[0], 0);
  EXPECT_EQ(layout.offsets[1], 8);

  ASSERT_EQ(layout.nullOffsets.size(), 2);
  EXPECT_EQ(layout.nullOffsets[0], 12 * 8 + 0);
  EXPECT_EQ(layout.nullOffsets[1], 12 * 8 + 1);

  ASSERT_EQ(layout.rowColumns.size(), 2);
  EXPECT_EQ(layout.rowColumns[0].offset(), 0);
  EXPECT_EQ(layout.rowColumns[0].nullByte(), 12);
  EXPECT_EQ(layout.rowColumns[0].nullMask(), 0x1);
  EXPECT_EQ(layout.rowColumns[1].offset(), 8);
  EXPECT_EQ(layout.rowColumns[1].nullByte(), 12);
  EXPECT_EQ(layout.rowColumns[1].nullMask(), 0x2);
}

// Hand-traced layout for a hash-join build side with a fixed-size accumulator,
// a probed flag, a count slot, a next pointer and a normalized-key prefix.
// Layout:
//   bytes  0..7  : key 0 (BIGINT)
//   bytes  8..9  : flag bytes
//                  byte 8: bits 0..6 unused (flag region starts at byte 8 so
//                          flag bit numbering restarts here),
//                  bit 64        = key 0 null
//                  bits 65..71   = unused (forced to next byte boundary
//                                  because there is an accumulator),
//                  bits 72,73    = accumulator null + initialized,
//                  bit 74        = dependent (DOUBLE) null,
//                  bit 75        = probed,
//                  bit 76        = free
//   bytes 10..15 : padding to align accumulator to 8
//   bytes 16..23 : accumulator payload (size 8, alignment 8)
//   bytes 24..31 : dependent (DOUBLE)
//   bytes 32..39 : next-row pointer
//   bytes 40..43 : count int32_t
//   bytes 44..47 : padding to align row size to 8
//   plus 8 bytes reserved before the row pointer for the normalized key
TEST_F(RowLayoutTest, goldenHashJoinWithAccumulator) {
  auto layout = RowLayout::compute(
      {BIGINT()},
      {makeAccumulator(/*size=*/8, /*alignment=*/8)},
      {DOUBLE()},
      {.hasNext = true,
       .hasProbedFlag = true,
       .hasCountFlag = true,
       .hasNormalizedKeys = true});

  EXPECT_EQ(layout.fixedRowSize, 48);
  EXPECT_EQ(layout.flagBytes, 2);
  EXPECT_EQ(layout.alignment, 8);
  EXPECT_EQ(layout.probedFlagOffset, 8 * 8 + 11);
  EXPECT_EQ(layout.freeFlagOffset, 8 * 8 + 12);
  EXPECT_EQ(layout.nextOffset, 32);
  EXPECT_EQ(layout.countOffset, 40);
  EXPECT_EQ(layout.rowSizeOffset, 0);
  EXPECT_EQ(layout.originalNormalizedKeySize, 8);
  EXPECT_FALSE(layout.usesExternalMemory);

  ASSERT_EQ(layout.offsets.size(), 3);
  EXPECT_EQ(layout.offsets[0], 0);
  EXPECT_EQ(layout.offsets[1], 16);
  EXPECT_EQ(layout.offsets[2], 24);

  ASSERT_EQ(layout.nullOffsets.size(), 3);
  EXPECT_EQ(layout.nullOffsets[0], 8 * 8 + 0);
  EXPECT_EQ(layout.nullOffsets[1], 8 * 8 + 8);
  EXPECT_EQ(layout.nullOffsets[2], 8 * 8 + 10);

  ASSERT_EQ(layout.rowColumns.size(), 3);
  EXPECT_EQ(layout.rowColumns[0].offset(), 0);
  EXPECT_EQ(layout.rowColumns[0].nullByte(), 8);
  EXPECT_EQ(layout.rowColumns[0].nullMask(), 0x1);
  EXPECT_EQ(layout.rowColumns[1].offset(), 16);
  EXPECT_EQ(layout.rowColumns[1].nullByte(), 9);
  EXPECT_EQ(layout.rowColumns[1].nullMask(), 0x1);
  EXPECT_EQ(layout.rowColumns[2].offset(), 24);
  EXPECT_EQ(layout.rowColumns[2].nullByte(), 9);
  EXPECT_EQ(layout.rowColumns[2].nullMask(), 0x4);
}

} // namespace
} // namespace facebook::velox::exec::test
