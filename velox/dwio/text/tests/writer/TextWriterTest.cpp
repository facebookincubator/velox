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

/// TODO: use TextReader

#include "velox/dwio/text/writer/TextWriter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/text/tests/writer/FileReaderUtil.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::velox::text {
/// TODO: add fuzzer test once text reader is move in OSS
class TextWriterTest : public testing::Test,
                       public velox::test::VectorTestBase {
 public:
  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
    dwio::common::LocalFileSink::registerFactory();
    rootPool_ = memory::memoryManager()->addRootPool("TextWriterTests");
    leafPool_ = rootPool_->addLeafChild("TextWriterTests");
    tempPath_ = exec::test::TempDirectoryPath::create();
  }

 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  constexpr static float kInf = std::numeric_limits<float>::infinity();
  constexpr static double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::shared_ptr<exec::test::TempDirectoryPath> tempPath_;
};

TEST_F(TextWriterTest, write) {
  auto schema =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"},
          {BOOLEAN(),
           TINYINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           TIMESTAMP(),
           VARCHAR(),
           VARBINARY()});
  auto data = makeRowVector(
      {"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"},
      {
          makeConstant(true, 3),
          makeFlatVector<int8_t>({1, 2, 3}),
          makeFlatVector<int16_t>({1, 2, 3}), // TODO null
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<float>({1.1, kInf, 3.1}),
          makeFlatVector<double>({1.1, kNaN, 3.1}),
          makeFlatVector<Timestamp>(
              3, [](auto i) { return Timestamp(i, i * 1'000'000); }),
          makeFlatVector<StringView>({"hello", "world", "cpp"}, VARCHAR()),
          makeFlatVector<StringView>({"hello", "world", "cpp"}, VARBINARY()),
      });

  WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  auto filePath =
      fs::path(fmt::format("{}/test_text_writer.txt", tempPath_->getPath()));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto writer = std::make_unique<TextWriter>(
      schema,
      std::move(sink),
      std::make_shared<text::WriterOptions>(writerOptions));
  writer->write(data);
  writer->close();

  std::vector<std::vector<std::string>> result = parseTextFile(filePath);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0].size(), 10);
  // bool type
  EXPECT_EQ(result[0][0], "true");
  EXPECT_EQ(result[1][0], "true");
  EXPECT_EQ(result[2][0], "true");

  // tinyint
  EXPECT_EQ(result[0][1], "1");
  EXPECT_EQ(result[1][1], "2");
  EXPECT_EQ(result[2][1], "3");

  // smallint
  EXPECT_EQ(result[0][2], "1");
  EXPECT_EQ(result[1][2], "2");
  EXPECT_EQ(result[2][2], "3");

  // int
  EXPECT_EQ(result[0][3], "1");
  EXPECT_EQ(result[1][3], "2");
  EXPECT_EQ(result[2][3], "3");

  // bigint
  EXPECT_EQ(result[0][4], "1");
  EXPECT_EQ(result[1][4], "2");
  EXPECT_EQ(result[2][4], "3");

  // float
  EXPECT_EQ(result[0][5], "1.100000");
  EXPECT_EQ(result[1][5], "Infinity");
  EXPECT_EQ(result[2][5], "3.100000");

  // double
  EXPECT_EQ(result[0][6], "1.100000");
  EXPECT_EQ(result[1][6], "NaN");
  EXPECT_EQ(result[2][6], "3.100000");

  // timestamp
  EXPECT_EQ(result[0][7], "1970-01-01 00:00:00.000");
  EXPECT_EQ(result[1][7], "1970-01-01 00:00:01.001");
  EXPECT_EQ(result[2][7], "1970-01-01 00:00:02.002");

  // varchar
  EXPECT_EQ(result[0][8], "hello");
  EXPECT_EQ(result[1][8], "world");
  EXPECT_EQ(result[2][8], "cpp");

  // varbinary
  EXPECT_EQ(result[0][9], "aGVsbG8=");
  EXPECT_EQ(result[1][9], "d29ybGQ=");
  EXPECT_EQ(result[2][9], "Y3Bw");
}

TEST_F(TextWriterTest, complexTypes) {
  const vector_size_t length = 13;
  const auto keyVector = makeFlatVector<int64_t>(
      {1,   111,  22, 22222, 333, 33, 44,    5,   555, 66, 7777,     7,
       777, 8888, 88, 9,     99,  10, 10000, 111, 1,   11, 11122222, 123142});
  const auto valueVector = makeFlatVector<bool>(
      {false, true, true,  false, false, true,  false, true,
       false, true, false, true,  false, false, true,  true,
       false, true, false, true,  false, true,  true,  true});
  BufferPtr sizes = facebook::velox::allocateOffsets(length, pool());
  BufferPtr offsets = facebook::velox::allocateOffsets(length, pool());
  auto rawSizes = sizes->asMutable<vector_size_t>();
  auto rawOffsets = offsets->asMutable<vector_size_t>();
  rawSizes[0] = 2;
  rawSizes[1] = 2;
  rawSizes[2] = 2;
  rawSizes[3] = 1;
  rawSizes[4] = 2;
  rawSizes[5] = 1;
  rawSizes[6] = 3;
  rawSizes[7] = 2;
  rawSizes[8] = 2;
  rawSizes[9] = 2;
  rawSizes[10] = 3;
  rawSizes[11] = 1;
  rawSizes[12] = 1;
  for (int i = 1; i < length; i++) {
    rawOffsets[i] = rawOffsets[i - 1] + rawSizes[i - 1];
  }

  const auto data = makeRowVector(
      {makeArrayVector<int64_t>(
           {{1, 11, 111},
            {22, 22222},
            {333, 33},
            {4444, 44},
            {5, 555},
            {666, 66, 66},
            {7777, 7, 777},
            {8888, 88},
            {9, 99},
            {10, 10000},
            {111, 1, 111},
            {12, 11122222, 222},
            {13, 11133333, 333}}),
       makeArrayVector<double>(
           {{1.123, 1.3123},
            {2.333, -5512, 1.23},
            {-6.1, 65.777},
            {4.2, 24, 324.11},
            {47.2, 213.23},
            {79.5, -44.11},
            {3.1415926, 441.124},
            {-221.145, 878.43, -11},
            {93.12, 632},
            {-4123.11, -177.1},
            {950.2, -4412},
            {43.66, 33121.43},
            {-42.11, -123.43}}),
       std::make_shared<MapVector>(
           pool(),
           MAP(keyVector->type(), valueVector->type()),
           nullptr,
           length,
           offsets,
           sizes,
           keyVector,
           valueVector)});

  const auto schema = ROW(
      {{"col_bigint_arr", ARRAY(BIGINT())},
       {"col_double_arr", ARRAY(DOUBLE())},
       {"col_map", MAP(BIGINT(), BOOLEAN())}});

  WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  auto filePath =
      fs::path(fmt::format("{}/test_text_writer.txt", tempPath_->getPath()));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});

  // use traits to specify delimiters when it is not nested
  const auto serDeOptions = dwio::common::SerDeOptions('\x01', '|', '#');
  auto writer = std::make_unique<TextWriter>(
      schema,
      std::move(sink),
      std::make_shared<text::WriterOptions>(writerOptions),
      serDeOptions);
  writer->write(data);
  writer->close();

  std::vector<std::vector<std::string>> result = parseTextFile(filePath);
  EXPECT_EQ(result.size(), 13);
  EXPECT_EQ(result[0].size(), 3);

  // Row 0: [1,11,111], [1.123,1.3123], {1:false,111:true}
  EXPECT_EQ(result[0][0], "1|11|111");
  EXPECT_EQ(result[0][1], "1.123000|1.312300");
  EXPECT_EQ(result[0][2], "1#false|111#true");

  // Row 1: [22,22222], [2.333,-5512,1.23], {22:true,22222:false}
  EXPECT_EQ(result[1][0], "22|22222");
  EXPECT_EQ(result[1][1], "2.333000|-5512.000000|1.230000");
  EXPECT_EQ(result[1][2], "22#true|22222#false");

  // Row 2: [333,33], [-6.1,65.777], {333:false,33:true}
  EXPECT_EQ(result[2][0], "333|33");
  EXPECT_EQ(result[2][1], "-6.100000|65.777000");
  EXPECT_EQ(result[2][2], "333#false|33#true");

  // Row 3: [4444,44], [4.2,24,324.11], {44:false}
  EXPECT_EQ(result[3][0], "4444|44");
  EXPECT_EQ(result[3][1], "4.200000|24.000000|324.110000");
  EXPECT_EQ(result[3][2], "44#false");

  // Row 4: [5,555], [47.2,213.23], {5:true,555:false}
  EXPECT_EQ(result[4][0], "5|555");
  EXPECT_EQ(result[4][1], "47.200000|213.230000");
  EXPECT_EQ(result[4][2], "5#true|555#false");

  // Row 5: [666,66,66], [79.5,-44.11], {66:true}
  EXPECT_EQ(result[5][0], "666|66|66");
  EXPECT_EQ(result[5][1], "79.500000|-44.110000");
  EXPECT_EQ(result[5][2], "66#true");

  // Row 6: [7777,7,777], [3.1415926,441.124], {7777:false,7:true,777:false}
  EXPECT_EQ(result[6][0], "7777|7|777");
  EXPECT_EQ(result[6][1], "3.141593|441.124000");
  EXPECT_EQ(result[6][2], "7777#false|7#true|777#false");

  // Row 7: [8888,88], [-221.145,878.43,-11], {8888:false,88:true}
  EXPECT_EQ(result[7][0], "8888|88");
  EXPECT_EQ(result[7][1], "-221.145000|878.430000|-11.000000");
  EXPECT_EQ(result[7][2], "8888#false|88#true");

  // Row 8: [9,99], [93.12,632], {9:true,99:false}
  EXPECT_EQ(result[8][0], "9|99");
  EXPECT_EQ(result[8][1], "93.120000|632.000000");
  EXPECT_EQ(result[8][2], "9#true|99#false");

  // Row 9: [10,10000], [-4123.11,-177.1], {10:true,10000:false}
  EXPECT_EQ(result[9][0], "10|10000");
  EXPECT_EQ(result[9][1], "-4123.110000|-177.100000");
  EXPECT_EQ(result[9][2], "10#true|10000#false");

  // Row 10: [111,1,111], [950.2,-4412], {111:true,1:false,11:true}
  EXPECT_EQ(result[10][0], "111|1|111");
  EXPECT_EQ(result[10][1], "950.200000|-4412.000000");
  EXPECT_EQ(result[10][2], "111#true|1#false|11#true");

  // Row 11: [12,11122222,222], [43.66,33121.43], {11122222:true}
  EXPECT_EQ(result[11][0], "12|11122222|222");
  EXPECT_EQ(result[11][1], "43.660000|33121.430000");
  EXPECT_EQ(result[11][2], "11122222#true");

  // Row 12: [13,11133333,333], [-42.11,-123.43], {123142:true}
  EXPECT_EQ(result[12][0], "13|11133333|333");
  EXPECT_EQ(result[12][1], "-42.110000|-123.430000");
  EXPECT_EQ(result[12][2], "123142#true");
}
TEST_F(TextWriterTest, abort) {
  auto schema = ROW({"c0", "c1"}, {BIGINT(), BOOLEAN()});
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeConstant(true, 3),
      });

  WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  writerOptions.defaultFlushCount = 10;
  auto filePath = fs::path(
      fmt::format("{}/test_text_writer_abort.txt", tempPath_->getPath()));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto writer = std::make_unique<TextWriter>(
      schema,
      std::move(sink),
      std::make_shared<text::WriterOptions>(writerOptions));
  writer->write(data);
  writer->abort();

  std::string result = readFile(filePath);
  // With defaultFlushCount as 10, it will trigger two times of flushes before
  // abort, and abort will discard the remaining 5 characters in buffer. The
  // written file would have:
  // 1^Atrue
  // 2^Atrue
  // 3^A
  EXPECT_EQ(result.size(), 14);
}
} // namespace facebook::velox::text
