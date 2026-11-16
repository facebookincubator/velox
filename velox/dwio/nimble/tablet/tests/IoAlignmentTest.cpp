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

#include "velox/dwio/nimble/tablet/IoAlignment.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/file/File.h"
#include "velox/common/file/Region.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble {
namespace {

constexpr uint64_t kAlignment{4'096};
constexpr uint64_t kFileSize{10'000};

// Offset and length of a region, which gtest can compare and print.
using OffsetAndLength = std::pair<uint64_t, uint64_t>;

OffsetAndLength offsetAndLength(const velox::common::Region& region) {
  return {region.offset, region.length};
}

// In-memory file that reports direct I/O with kAlignment.
class DirectIoReadFile final : public velox::InMemoryReadFile {
 public:
  using velox::InMemoryReadFile::InMemoryReadFile;

  bool directIo(uint64_t& alignment) const override {
    alignment = kAlignment;
    return true;
  }
};

TEST(IoAlignmentTest, readAlignment) {
  EXPECT_EQ(readAlignment(DirectIoReadFile{std::string_view{}}), kAlignment);
  EXPECT_EQ(readAlignment(velox::InMemoryReadFile{std::string_view{}}), 1);
}

TEST(IoAlignmentTest, alignedRegion) {
  // Widens a region to the blocks that hold it.
  EXPECT_EQ(
      offsetAndLength(alignedRegion({4'200, 100}, kAlignment, kFileSize)),
      OffsetAndLength(4'096, 4'096));
  EXPECT_EQ(
      offsetAndLength(alignedRegion({4'000, 200}, kAlignment, kFileSize)),
      OffsetAndLength(0, 8'192));

  // Keeps an aligned region.
  EXPECT_EQ(
      offsetAndLength(alignedRegion({4'096, 4'096}, kAlignment, kFileSize)),
      OffsetAndLength(4'096, 4'096));

  // Accepts a region that ends at the end of the file, or is empty there. Its
  // read runs past the end of the file, to the end of the last block.
  EXPECT_EQ(
      offsetAndLength(alignedRegion({9'000, 1'000}, kAlignment, kFileSize)),
      OffsetAndLength(8'192, 4'096));
  EXPECT_EQ(
      offsetAndLength(alignedRegion({kFileSize, 0}, kAlignment, kFileSize)),
      OffsetAndLength(8'192, 4'096));

  // Keeps any region at alignment 1.
  EXPECT_EQ(
      offsetAndLength(alignedRegion({9'000, 500}, /*alignment=*/1, kFileSize)),
      OffsetAndLength(9'000, 500));
}

TEST(IoAlignmentTest, alignedRegionPastEndOfFileThrows) {
  {
    SCOPED_TRACE("Offset past the end, with offset + length wrapping to 16");
    NIMBLE_ASSERT_FILE_THROW(
        alignedRegion(
            {std::numeric_limits<uint64_t>::max() - 15, 32},
            kAlignment,
            kFileSize),
        "Read region starts past the end of the file");
  }
  {
    SCOPED_TRACE("Length past the end");
    NIMBLE_ASSERT_FILE_THROW(
        alignedRegion({9'000, 1'001}, kAlignment, kFileSize),
        "Read region extends past the end of the file");
  }
  {
    SCOPED_TRACE("Length past the end, with offset + length wrapping to 8'999");
    NIMBLE_ASSERT_FILE_THROW(
        alignedRegion(
            {9'000, std::numeric_limits<uint64_t>::max()},
            kAlignment,
            kFileSize),
        "Read region extends past the end of the file");
  }
}

TEST(IoAlignmentTest, fileReadBytes) {
  EXPECT_EQ(fileReadBytes({}, kFileSize), 0);

  const std::vector<velox::common::Region> insideFile{
      {0, 4'096}, {4'096, 4'096}};
  EXPECT_EQ(fileReadBytes(insideFile, kFileSize), 8'192);

  // Counts only the bytes of the last block before the end of the file.
  const std::vector<velox::common::Region> pastEndOfFile{
      {0, 4'096}, {8'192, 4'096}};
  EXPECT_EQ(
      fileReadBytes(pastEndOfFile, kFileSize), 4'096 + (kFileSize - 8'192));
}

} // namespace
} // namespace facebook::nimble
