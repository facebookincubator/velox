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

#include "folly/Random.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/encryption/TestProvider.h"
#include "velox/dwio/dwrf/common/Common.h"
#include "velox/dwio/dwrf/reader/BinaryStreamReader.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/dwio/dwrf/test/utils/E2EWriterTestUtil.h"
#include "velox/dwio/dwrf/utils/ProtoUtils.h"
#include "velox/type/Type.h"
#include "velox/type/fbhive/HiveTypeParser.h"

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::encryption;
using namespace facebook::velox::dwio::common::encryption::test;
using namespace facebook::velox::dwrf;
using namespace facebook::velox::dwio::common::exception;
using folly::Random;

using facebook::velox::RowType;
using facebook::velox::common::Region;
using facebook::velox::dwrf::detail::BinaryStreamReader;
using facebook::velox::dwrf::detail::BinaryStripeStreams;
using facebook::velox::type::fbhive::HiveTypeParser;

class RecordingInputStream : public facebook::velox::InMemoryReadFile {
 public:
  RecordingInputStream() : InMemoryReadFile(std::string()) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const facebook::velox::FileIoContext& context) const override {
    reads_.emplace_back(offset, length);
    return {static_cast<char*>(buf), length};
  }

  const std::vector<Region>& getReads() const {
    return reads_;
  }

 private:
  mutable std::vector<Region> reads_;
};

class BinaryStreamReaderTest : public ::testing::Test {
 protected:
  std::shared_ptr<facebook::velox::io::IoStatistics> dataIoStats_{
      std::make_shared<facebook::velox::io::IoStatistics>()};
  std::shared_ptr<facebook::velox::io::IoStatistics> metadataIoStats_{
      std::make_shared<facebook::velox::io::IoStatistics>()};
};

auto createFooter(
    google::protobuf::Arena& arena,
    uint32_t strideLen,
    const std::string& schema) {
  auto footer = google::protobuf::Arena::CreateMessage<proto::Footer>(&arena);
  footer->set_rowindexstride(strideLen);
  auto type = HiveTypeParser().parse(schema);
  FooterWriteWrapper wrapper{footer};
  ProtoUtils::writeType(*type, wrapper);
  return footer;
}

TEST_F(BinaryStreamReaderTest, encryptionNotSupported) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();

  google::protobuf::Arena arena;
  auto footer = createFooter(arena, 100, "struct<a:int,b:float>");
  footer->add_statistics();
  auto encryption = footer->mutable_encryption();
  encryption->set_keyprovider(
      facebook::velox::dwrf::encryption::toProto(EncryptionProvider::Unknown));
  encryption->add_encryptiongroups();
  encryption->mutable_encryptiongroups(0)->add_nodes(0);
  encryption->mutable_encryptiongroups(0)->add_statistics();
  auto stripe = footer->add_stripes();
  stripe->add_keymetadata();

  auto is = std::make_shared<RecordingInputStream>();
  auto readerBase = std::make_shared<ReaderBase>(
      *pool,
      std::make_unique<BufferedInput>(is, *pool),
      std::make_unique<PostScript>(proto::PostScript{}),
      footer,
      nullptr,
      /*handler=*/nullptr,
      dataIoStats_,
      metadataIoStats_);

  std::vector<uint64_t> columnIds{1};
  VELOX_ASSERT_THROW(
      std::make_unique<BinaryStreamReader>(readerBase, columnIds),
      "encryption not supported");
}

TEST_F(BinaryStreamReaderTest, columnIdsEmpty) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();

  google::protobuf::Arena arena;
  auto footer = createFooter(arena, 100, "struct<a:int,b:float>");

  auto is = std::make_shared<RecordingInputStream>();
  auto readerBase = std::make_shared<ReaderBase>(
      *pool,
      std::make_unique<BufferedInput>(is, *pool),
      std::make_unique<PostScript>(proto::PostScript{}),
      footer,
      nullptr,
      /*handler=*/nullptr,
      dataIoStats_,
      metadataIoStats_);

  std::vector<uint64_t> columnIds;
  VELOX_ASSERT_THROW(
      std::make_unique<BinaryStreamReader>(readerBase, columnIds),
      "At least one column expected to be read");
}

TEST_F(BinaryStreamReaderTest, EmptyFile) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();

  constexpr uint32_t STRIDE_LEN = 100;
  google::protobuf::Arena arena;
  auto footer = createFooter(arena, STRIDE_LEN, "struct<a:int,b:float>");

  auto is = std::make_shared<RecordingInputStream>();
  auto readerBase = std::make_shared<ReaderBase>(
      *pool,
      std::make_unique<BufferedInput>(is, *pool),
      std::make_unique<PostScript>(proto::PostScript{}),
      footer,
      nullptr,
      /*handler=*/nullptr,
      dataIoStats_,
      metadataIoStats_);

  std::vector<uint64_t> columnIds{1};
  BinaryStreamReader binaryReader(readerBase, columnIds);

  EXPECT_EQ(binaryReader.getStrideLen(), STRIDE_LEN);
  EXPECT_EQ(binaryReader.getCurrentStripeIndex(), 0);
  auto statistics = binaryReader.getStatistics();
  EXPECT_EQ(statistics.size(), 2);
  // Node 0 is always selected. Column 1 is node 2.
  EXPECT_TRUE(statistics.contains(0));
  EXPECT_TRUE(statistics.contains(2));

  // No Stripes should return 0
  EXPECT_EQ(binaryReader.next(), nullptr);
}

void verifyEncoding(
    const proto::ColumnEncoding& encoding,
    uint32_t nodeId,
    uint32_t sequence,
    proto::ColumnEncoding_Kind kind) {
  EXPECT_EQ(encoding.kind(), kind);
  EXPECT_EQ(encoding.node(), nodeId);
  EXPECT_EQ(encoding.sequence(), sequence);
}

void verifyStream(
    const BinaryStripeStreams& binaryStripeStream,
    uint32_t nodeId,
    uint32_t sequence,
    const std::vector<proto::Stream_Kind>& streamKinds) {
  auto streamIdentifiers = binaryStripeStream.getStreamIdentifiers(nodeId);
  ASSERT_EQ(streamIdentifiers.size(), streamKinds.size());
  for (auto& kind : streamKinds) {
    bool isFound = false;
    DwrfStreamIdentifier expectedStreamId(nodeId, sequence, 0, kind);
    for (auto& streamId : streamIdentifiers) {
      if (streamId == expectedStreamId) {
        isFound = true;
        break;
      }
    }
    EXPECT_TRUE(isFound) << expectedStreamId.toString();

    EXPECT_GT(binaryStripeStream.getStreamLength(expectedStreamId), 0);
    EXPECT_NE(binaryStripeStream.getStream(expectedStreamId, {}), nullptr);
  }
}

TEST_F(BinaryStreamReaderTest, BasicFlow) {
  auto type = HiveTypeParser().parse("struct<a:int,b:float>");

  auto config = std::make_shared<Config>();

  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024, FileSink::Options{.pool = pool.get()});
  auto sinkPtr = sink.get();

  constexpr uint32_t NUM_STRIPES = 5;
  constexpr uint32_t ROWS_PER_STRIPE = 150;

  auto batches = E2EWriterTestUtil::generateBatches(
      type, NUM_STRIPES, ROWS_PER_STRIPE, Random::rand32(), *pool);

  auto writer = E2EWriterTestUtil::writeData(
      std::move(sink),
      type,
      batches,
      config,
      E2EWriterTestUtil::simpleFlushPolicyFactory(true));

  auto input = std::make_unique<BufferedInput>(
      std::make_shared<facebook::velox::InMemoryReadFile>(
          std::string(sinkPtr->data(), sinkPtr->size())),
      *pool);

  facebook::velox::dwio::common::ReaderOptions readerOpts(pool.get());
  readerOpts.setFileFormat(FileFormat::DWRF);
  readerOpts.setDataIoStats(dataIoStats_);
  readerOpts.setMetadataIoStats(metadataIoStats_);
  auto readerBase = std::make_shared<ReaderBase>(readerOpts, std::move(input));
  std::vector<uint64_t> columnIds{1};
  BinaryStreamReader binaryReader(readerBase, columnIds);

  EXPECT_EQ(binaryReader.getStrideLen(), readerBase->footer().rowIndexStride());
  EXPECT_EQ(binaryReader.getCurrentStripeIndex(), 0);
  auto stats = binaryReader.getStatistics();
  EXPECT_EQ(stats.size(), 2);
  // ColumnId 1 - maps to NodeId 2, NodeId 0 (root node) is always selected.
  EXPECT_TRUE(stats.contains(0));
  EXPECT_TRUE(stats.contains(2));

  for (auto i = 0; i < NUM_STRIPES; ++i) {
    auto binaryStripeStream = binaryReader.next();
    EXPECT_NE(binaryStripeStream, nullptr);
    EXPECT_EQ(
        binaryStripeStream->getStripeInfo().numberOfRows(), ROWS_PER_STRIPE);

    // Node 0 - DIRECT
    EXPECT_EQ(binaryStripeStream->getEncodings(0).size(), 1);
    verifyEncoding(
        binaryStripeStream->getEncodings(0).at(0),
        0 /*nodeId*/,
        0 /*sequence*/,
        proto::ColumnEncoding::DIRECT);

    EXPECT_EQ(binaryStripeStream->getEncodings(0).size(), 1);
    verifyStream(
        *binaryStripeStream, 0, 0, {proto::Stream_Kind::Stream_Kind_ROW_INDEX});
    // Node 1 - Not selected, hence 0 results returned.
    EXPECT_EQ(binaryStripeStream->getEncodings(1).size(), 0);

    // Node 2 - Float has only direct encoding.
    EXPECT_EQ(binaryStripeStream->getEncodings(2).size(), 1);
    verifyEncoding(
        binaryStripeStream->getEncodings(2).at(0),
        2,
        0,
        proto::ColumnEncoding::DIRECT);
    verifyStream(
        *binaryStripeStream,
        2 /*nodeId*/,
        0 /*sequence*/,
        {proto::Stream_Kind::Stream_Kind_ROW_INDEX,
         proto::Stream_Kind::Stream_Kind_PRESENT,
         proto::Stream_Kind::Stream_Kind_DATA});
  }
  EXPECT_EQ(binaryReader.next(), nullptr);
}
