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

#include "velox/dwio/text/reader/TextReader.h"
#include "velox/dwio/text/writer/TextWriter.h"

#include <gtest/gtest.h>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/text/tests/writer/FileReaderUtil.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::text {

class TextReaderTest : public testing::Test,
                       public velox::test::VectorTestBase {
 public:
  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
    rootPool_ = memory::memoryManager()->addRootPool("TextReaderTests");
    leafPool_ = rootPool_->addLeafChild("TextReaderTests");
  }

 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManagerOptions{});
  }

  static std::shared_ptr<LocalReadFile> getFile(const std::string& fileName) {
    return std::make_shared<LocalReadFile>(test::getDataFilePath(
        "velox/dwio/text/tests/reader", "../examples/" + fileName));
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const std::shared_ptr<LocalReadFile>& file,
      const RowTypePtr& fileSchema,
      const RowTypePtr& scanSchema,
      const dwio::common::SerDeOptions& serdeOptions =
          dwio::common::SerDeOptions(),
      const uint64_t offset = 0,
      const uint64_t length = std::numeric_limits<uint64_t>::max()) {
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setFileFormat(dwio::common::FileFormat::TEXT);
    readerOptions.setFileSchema(fileSchema);
    readerOptions.setSerDeOptions(serdeOptions);
    auto reader = std::make_unique<facebook::velox::text::TextReader>(
        std::make_unique<dwio::common::BufferedInput>(file, *leafPool_),
        std::move(readerOptions));

    dwio::common::RowReaderOptions rowReaderOpts;
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
    scanSpec->addAllChildFields(*scanSchema);
    rowReaderOpts.setScanSpec(scanSpec);
    auto rowReader =
        reader->createRowReader(rowReaderOpts.range(offset, length));
    return rowReader;
  }

  constexpr static float kInf = std::numeric_limits<float>::infinity();
  constexpr static double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;

  RowTypePtr simpleSchema_ = ROW(
      {
          "col_boolean",
          "col_tinyint",
          "col_smallint",
          "col_integer",
          "col_bigint",
          "col_real",
          "col_double",
          "col_timestamp",
          "col_varchar",
          "col_varbinary",
      },
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
  RowTypePtr complexSchema_ =
      ROW({"col_boolean",
           "col_tinyint",
           "col_smallint",
           "col_integer",
           "col_date",
           "col_bigint",
           "col_real",
           "col_double",
           "col_timestamp",
           "col_varchar",
           "col_varbinary",
           "col_array",
           "col_row",
           "col_map"},
          {BOOLEAN(),
           TINYINT(),
           SMALLINT(),
           INTEGER(),
           DATE(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           TIMESTAMP(),
           VARCHAR(),
           VARBINARY(),
           ARRAY(INTEGER()),
           ROW({"c0", "c1", "c2"}, {BOOLEAN(), INTEGER(), VARCHAR()}),
           MAP(VARCHAR(), INTEGER())});
};

TEST_F(TextReaderTest, simpleCase) {
  auto rowReader =
      createRowReader(getFile("simple.txt"), simpleSchema_, simpleSchema_);

  auto expected = makeRowVector(
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

  auto actual = BaseVector::create(simpleSchema_, 10, leafPool_.get());
  auto result = rowReader->next(3, actual);
  ASSERT_EQ(result, 3);
  ASSERT_EQ(actual->size(), 3);
  ASSERT_EQ(actual->type(), simpleSchema_);
  test::assertEqualVectors(expected, actual);
}

TEST_F(TextReaderTest, complexCase) {
  auto expected = makeRowVector(
      {"col_boolean",
       "col_tinyint",
       "col_smallint",
       "col_integer",
       "col_date",
       "col_bigint",
       "col_real",
       "col_double",
       "col_timestamp",
       "col_varchar",
       "col_varbinary",
       "col_array",
       "col_row",
       "col_map"},
      {// BOOLEAN with some nulls
       makeNullableFlatVector<bool>(
           {true,
            false,
            true,
            false,
            true,
            std::nullopt,
            true,
            false,
            true,
            false}),

       // TINYINT
       makeFlatVector<int8_t>({127, -128, 0, 100, 50, -50, 25, -25, 64, -64}),

       // SMALLINT with one null (example)
       makeNullableFlatVector<int16_t>(
           {32767,
            std::nullopt,
            0,
            1000,
            500,
            -500,
            250,
            std::nullopt,
            1024,
            -1024}),

       // INTEGER (int32), no nulls here for example
       makeNullableFlatVector<int32_t>(
           {1, 2, 3, 4, 5, 6, std::nullopt, 8, 9, 10}),

       // DATE stored as int32 days since epoch
       makeNullableFlatVector<int32_t>(
           {DATE()->toDays("2024-06-10"),
            DATE()->toDays("2024-06-11"),
            std::nullopt,
            DATE()->toDays("2024-06-13"),
            DATE()->toDays("2024-06-14"),
            DATE()->toDays("2024-06-15"),
            DATE()->toDays("2024-06-16"),
            DATE()->toDays("2024-06-17"),
            DATE()->toDays("2024-06-18"),
            DATE()->toDays("2024-06-19")},
           DATE()),

       // BIGINT
       makeNullableFlatVector<int64_t>(
           {1000000000000,
            2000000000000,
            3000000000000,
            std::nullopt,
            5000000000000,
            6000000000000,
            7000000000000,
            8000000000000,
            9000000000000,
            10000000000000}),

       // REAL (float) with Infinity and null
       makeNullableFlatVector<float>(
           {3.14f,
            kInf,
            1.41f,
            2.71f,
            0.5772f,
            9.81f,
            6.63f,
            1.62f,
            std::nullopt,
            7.77f}),

       // DOUBLE with NaN and null
       makeNullableFlatVector<double>(
           {2.71828,
            3.14159,
            kNaN,
            0.5772,
            std::nullopt,
            3.333,
            2.222,
            4.444,
            5.555,
            6.666}),

       // TIMESTAMP (int64_t nanoseconds)
       makeFlatVector<Timestamp>(
           10,
           [](auto i) {
             static const int64_t baseSeconds =
                 1'718'013'600; // 2024-06-10 10:00:00 UTC in epoch seconds
             return Timestamp(
                 baseSeconds + i * 86400 + i * 3600 + i * 660 + i * 11, 0);
           }),

       // VARCHAR
       makeNullableFlatVector<StringView>(
           {"hello",
            "world",
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "corge",
            "grault",
            std::nullopt},
           VARCHAR()),

       // VARBINARY (here just simple string as placeholder)
       makeNullableFlatVector<StringView>(
           {"hello",
            std::nullopt,
            "1234",
            "4321",
            "abcd1234",
            "dcba4321",
            "12345678",
            "87654321",
            "deadbeef",
            "beefdead"},
           VARBINARY()),

       // ARRAY<int32_t>
       makeArrayVector<int32_t>(
           {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12},
            {13, 14, 15},
            {16, 17, 18},
            {19, 20, 21},
            {22, 23, 24},
            {25, 26, 27},
            {28, 29, 30}}),

       // ROW(struct) with three fields: boolean, int32, string
       makeRowVector(
           {makeFlatVector<bool>(
                {true,
                 false,
                 true,
                 false,
                 true,
                 false,
                 true,
                 false,
                 true,
                 false}),
            makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
            makeNullableFlatVector<StringView>(
                {"a", "b", "c", "d", "e", "f", "g", "h", "i", std::nullopt},
                VARCHAR())}),

       // MAP<string,int32>
       makeMapVector<StringView, int32_t>({
           {{"k1", 10}},
           {{"k2", 20}},
           {{"k3", 30}},
           {{"k4", 40}},
           {{"k5", 50}},
           {{"k6", 60}},
           {{"k7", 70}},
           {{"k8", 80}},
           {{"k9", 90}},
           {{"k10", 100}},
       })});

  auto rowReader = createRowReader(
      getFile("complex.txt"),
      complexSchema_,
      complexSchema_,
      dwio::common::SerDeOptions('^', '|', ':'));
  auto actual = BaseVector::create(complexSchema_, 20, leafPool_.get());
  auto result = rowReader->next(10, actual);
  ASSERT_EQ(result, 10);
  ASSERT_EQ(actual->size(), 10);
  ASSERT_EQ(actual->type(), complexSchema_);
  test::assertEqualVectors(expected, actual);
}

TEST_F(TextReaderTest, complexCaseMultipleSplits) {
  auto file = getFile("complex.txt");
  auto fileLength = file->size();

  auto scanSchema = ROW(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {DATE(),
       TIMESTAMP(),
       VARBINARY(),
       ARRAY(INTEGER()),
       ROW({"c0", "c1", "c2"}, {BOOLEAN(), INTEGER(), VARCHAR()})});

  std::vector<std::unique_ptr<dwio::common::RowReader>> rowReaders;
  for (int i = 0; i < 5; ++i) {
    auto offset = i * (fileLength / 5);
    auto length = (i == 4) ? fileLength - offset : fileLength / 5;
    rowReaders.push_back(createRowReader(
        file,
        complexSchema_,
        scanSchema,
        dwio::common::SerDeOptions('^', '|', ':'),
        offset,
        length));
  }

  // Check the results of the first row reader
  auto actual = BaseVector::create(scanSchema, 10, leafPool_.get());
  auto result = rowReaders[0]->next(10, actual);
  ASSERT_EQ(result, 2);
  ASSERT_EQ(actual->size(), 2);
  ASSERT_EQ(actual->type(), scanSchema);
  auto expected = makeRowVector(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {makeFlatVector<int32_t>(
           {DATE()->toDays("2024-06-10"), DATE()->toDays("2024-06-11")},
           DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1718013600, 0), Timestamp(1718104271, 0)}),
       makeNullableFlatVector<StringView>({"hello", std::nullopt}, VARBINARY()),
       makeArrayVector<int32_t>({{1, 2, 3}, {4, 5, 6}}),
       makeRowVector(
           {makeFlatVector<bool>({true, false}),
            makeFlatVector<int32_t>({1, 2}),
            makeFlatVector<StringView>({"a", "b"}, VARCHAR())})});
  test::assertEqualVectors(expected, actual);

  // Check the results of the second row reader
  result = rowReaders[1]->next(10, actual);
  ASSERT_EQ(result, 3);
  ASSERT_EQ(actual->size(), 3);
  ASSERT_EQ(actual->type(), scanSchema);
  expected = makeRowVector(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {makeNullableFlatVector<int32_t>(
           {std::nullopt,
            DATE()->toDays("2024-06-13"),
            DATE()->toDays("2024-06-14")},
           DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1718194942, 0),
            Timestamp(1718285613, 0),
            Timestamp(1718376284, 0)}),
       makeFlatVector<StringView>({"1234", "4321", "abcd1234"}, VARBINARY()),
       makeArrayVector<int32_t>({{7, 8, 9}, {10, 11, 12}, {13, 14, 15}}),
       makeRowVector(
           {makeFlatVector<bool>({true, false, true}),
            makeFlatVector<int32_t>({3, 4, 5}),
            makeFlatVector<StringView>({"c", "d", "e"}, VARCHAR())})});
  test::assertEqualVectors(expected, actual);

  // Check the results of the third row reader
  result = rowReaders[2]->next(10, actual);
  ASSERT_EQ(result, 2);
  ASSERT_EQ(actual->size(), 2);
  ASSERT_EQ(actual->type(), scanSchema);
  expected = makeRowVector(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {makeFlatVector<int32_t>(
           {DATE()->toDays("2024-06-15"), DATE()->toDays("2024-06-16")},
           DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1718466955, 0), Timestamp(1718557626, 0)}),
       makeFlatVector<StringView>({"dcba4321", "12345678"}, VARBINARY()),
       makeArrayVector<int32_t>({{16, 17, 18}, {19, 20, 21}}),
       makeRowVector(
           {makeFlatVector<bool>({false, true}),
            makeFlatVector<int32_t>({6, 7}),
            makeFlatVector<StringView>({"f", "g"}, VARCHAR())})});
  test::assertEqualVectors(expected, actual);

  // Check the results of the fourth row reader
  result = rowReaders[3]->next(10, actual);
  ASSERT_EQ(result, 2);
  ASSERT_EQ(actual->size(), 2);
  ASSERT_EQ(actual->type(), scanSchema);
  expected = makeRowVector(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {makeFlatVector<int32_t>(
           {DATE()->toDays("2024-06-17"), DATE()->toDays("2024-06-18")},
           DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1718648297, 0), Timestamp(1718738968, 0)}),
       makeFlatVector<StringView>({"87654321", "deadbeef"}, VARBINARY()),
       makeArrayVector<int32_t>({{22, 23, 24}, {25, 26, 27}}),
       makeRowVector(
           {makeFlatVector<bool>({false, true}),
            makeFlatVector<int32_t>({8, 9}),
            makeFlatVector<StringView>({"h", "i"}, VARCHAR())})});
  test::assertEqualVectors(expected, actual);

  // Check the results of the fifth row reader
  result = rowReaders[4]->next(10, actual);
  ASSERT_EQ(result, 1);
  ASSERT_EQ(actual->size(), 1);
  ASSERT_EQ(actual->type(), scanSchema);
  expected = makeRowVector(
      {"col_date", "col_timestamp", "col_varbinary", "col_array", "col_row"},
      {makeFlatVector<int32_t>(
           std::vector<int32_t>{DATE()->toDays("2024-06-19")}, DATE()),
       makeFlatVector<Timestamp>(
           {Timestamp(1718829639, 0), Timestamp(1718791200, 0)}),
       makeFlatVector<StringView>({"beefdead"}, VARBINARY()),
       makeArrayVector<int32_t>({{28, 29, 30}}),
       makeRowVector(
           {makeFlatVector<bool>(std::vector<bool>{false}),
            makeFlatVector<int32_t>(std::vector<int32_t>{10}),
            makeNullableFlatVector<StringView>({std::nullopt}, VARCHAR())})});
  test::assertEqualVectors(expected, actual);
}

} // namespace facebook::velox::text
