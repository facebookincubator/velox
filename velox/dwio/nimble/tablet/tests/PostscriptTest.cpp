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

#include "velox/dwio/nimble/tablet/Postscript.h"

#include <gtest/gtest.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletWriter.h"

using namespace facebook::nimble;

TEST(PostscriptTest, defaultConstruction) {
  Postscript ps;
  EXPECT_EQ(ps.footerSize(), 0);
  EXPECT_EQ(ps.footerCompressionType(), CompressionType::Uncompressed);
  EXPECT_EQ(ps.checksum(), 0);
  EXPECT_EQ(ps.majorVersion(), 0);
  EXPECT_EQ(ps.minorVersion(), 0);
}

TEST(PostscriptTest, parameterizedConstruction) {
  Postscript ps{100, CompressionType::Zstd, ChecksumType::XXH3_64, 2, 3};
  EXPECT_EQ(ps.footerSize(), 100);
  EXPECT_EQ(ps.footerCompressionType(), CompressionType::Zstd);
  EXPECT_EQ(ps.checksum(), 0);
  EXPECT_EQ(ps.checksumType(), ChecksumType::XXH3_64);
  EXPECT_EQ(ps.majorVersion(), 2);
  EXPECT_EQ(ps.minorVersion(), 3);
}

TEST(PostscriptTest, serializeRoundTrip) {
  struct TestParam {
    uint32_t footerSize;
    CompressionType compressionType;
    ChecksumType checksumType;
    uint32_t majorVersion;
    uint32_t minorVersion;

    std::string debugString() const {
      return fmt::format(
          "footerSize {}, compressionType {}, checksumType {}, version {}.{}",
          footerSize,
          static_cast<int>(compressionType),
          static_cast<int>(checksumType),
          majorVersion,
          minorVersion);
    }
  };

  std::vector<TestParam> testSettings = {
      {0, CompressionType::Uncompressed, ChecksumType::XXH3_64, 0, 0},
      {100, CompressionType::Zstd, ChecksumType::XXH3_64, kVersionMajor, 0},
      {8 * 1024 * 1024,
       CompressionType::Zstd,
       ChecksumType::XXH3_64,
       kVersionMajor,
       kVersionMinor},
      {std::numeric_limits<uint32_t>::max(),
       CompressionType::Uncompressed,
       ChecksumType::XXH3_64,
       kVersionMajor,
       0xFFFF},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    Postscript original{
        testData.footerSize,
        testData.compressionType,
        testData.checksumType,
        testData.majorVersion,
        testData.minorVersion};

    auto serialized = original.serialize();
    EXPECT_EQ(serialized.size(), Postscript::kSize);

    auto parsed = Postscript::parse(serialized);
    EXPECT_EQ(parsed.footerSize(), original.footerSize());
    EXPECT_EQ(parsed.footerCompressionType(), original.footerCompressionType());
    EXPECT_EQ(parsed.checksum(), original.checksum());
    EXPECT_EQ(parsed.checksumType(), original.checksumType());
    EXPECT_EQ(parsed.majorVersion(), original.majorVersion());
    EXPECT_EQ(parsed.minorVersion(), original.minorVersion());
  }
}

TEST(PostscriptTest, serializeSize) {
  Postscript ps{100, CompressionType::Zstd, ChecksumType::XXH3_64, 1, 0};
  auto buf = ps.serialize();
  EXPECT_EQ(buf.size(), Postscript::kSize);
}

TEST(PostscriptTest, parseTooSmall) {
  std::string buf(Postscript::kSize - 1, '\0');
  NIMBLE_ASSERT_THROW(Postscript::parse(buf), "Invalid postscript length");
}

TEST(PostscriptTest, parseBadMagic) {
  Postscript ps{100, CompressionType::Zstd, ChecksumType::XXH3_64, 1, 0};
  auto buf = ps.serialize();
  buf[Postscript::kSize - 1] = 'X';
  NIMBLE_ASSERT_THROW(Postscript::parse(buf), "Magic number mismatch");
}

TEST(PostscriptTest, identifiesNimbleFile) {
  facebook::velox::memory::MemoryManager memoryManager;
  auto pool = memoryManager.addLeafPool("PostscriptTest");

  std::string data;
  facebook::velox::InMemoryWriteFile writeFile{&data};
  auto writer = TabletWriter::create(&writeFile, *pool, {});
  writer->close();
  writeFile.close();

  facebook::velox::InMemoryReadFile file{std::string_view{data}};

  EXPECT_TRUE(isNimbleFile(file));

  data.back() = '\0';
  EXPECT_FALSE(isNimbleFile(file));
}
