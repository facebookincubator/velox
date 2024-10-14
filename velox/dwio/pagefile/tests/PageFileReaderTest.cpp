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

#include "velox/dwio/pagefile/tests/PageFileTestBase.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/vector/tests/utils/VectorMaker.h"


using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::pagefile;

class PageFileReaderTest : public PageFileTestBase {
 public:
  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const std::string& fileName,
      const RowTypePtr& rowType) {
    const std::string sample(getExampleFilePath(fileName));

    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    auto reader = createReader(sample, readerOptions);

    RowReaderOptions rowReaderOpts;
    rowReaderOpts.select(
        std::make_shared<facebook::velox::dwio::common::ColumnSelector>(
            rowType, rowType->names()));
    rowReaderOpts.setScanSpec(makeScanSpec(rowType));
    auto rowReader = reader->createRowReader(rowReaderOpts);
    return rowReader;
  }

  void assertReadWithExpected(
      const std::string& fileName,
      const RowTypePtr& rowType,
      const RowVectorPtr& expected) {
    auto rowReader = createRowReader(fileName, rowType);
    assertReadWithReaderAndExpected(rowType, *rowReader, expected, *pool_);
  }

  void assertReadWithFilters(
      const std::string& fileName,
      const RowTypePtr& fileSchema,
      FilterMap filters,
      const RowVectorPtr& expected) {
    const auto filePath(getExampleFilePath(fileName));
    dwio::common::ReaderOptions readerOpts{leafPool_.get()};
    auto reader = createReader(filePath, readerOpts);
    assertReadWithReaderAndFilters(
        std::move(reader), fileName, fileSchema, std::move(filters), expected);
  }
};

TEST_F(PageFileReaderTest, parseSample) {
  // sample.pagefile holds two columns (a: BIGINT, b: DOUBLE) and
  // 20 rows (10 rows per group). Group offsets are 153 and 614.
  // Data is in plain uncompressed format:
  //   a: [1..20]
  //   b: [1.0..20.0]
  const std::string sample(getExampleFilePath("test.feather"));

  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  auto reader = createReader(sample, readerOptions);
  EXPECT_EQ(reader->numberOfRows(), 20ULL);

  auto type = reader->typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type()->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type()->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);

  auto rowReaderOpts = getReaderOpts(sampleSchema());
  auto scanSpec = makeScanSpec(sampleSchema());
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
      makeFlatVector<double>(20, [](auto row) { return row + 1; }),
  });

  assertReadWithReaderAndExpected(
      sampleSchema(), *rowReader, expected, *leafPool_);
}

//TEST_F(PageFileReaderTest, uberParseComplex1) {
//  const std::string sample(getExampleFilePath("uber_marmaray_schema_issue.pagefile"));
//  facebook::velox::dwio::common::ReaderOptions readerOptions{leafPool_.get()};
//  auto reader = createReader(sample, readerOptions);
//  auto numRows = reader->numberOfRows();
//  std::cout << "Num of rows:" << (numRows ? *numRows : -1) << std::endl;
//  auto type = reader->typeWithId();
//  RowReaderOptions rowReaderOpts;
//  auto columnName = "country";
//  auto rowType = ROW({columnName}, {VARCHAR()});
//  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
//  auto rowReader = reader->createRowReader(rowReaderOpts);
//  auto result = BaseVector::create(rowType, 10, leafPool_.get());
//  constexpr int kBatchSize = 100;
//  auto output_rows = 0;
//  while (rowReader->next(kBatchSize, result)) {
//    output_rows = output_rows + result->size();
//  }
//  std::cout << "total rows in result:" << output_rows << std::endl;
//  EXPECT_EQ(numRows, output_rows);
//}
//
//TEST_F(PageFileReaderTest, uberParseComplex) {
//  // This file was crashing earlier and it no longer crashes
//  const std::string sample(getExampleFilePath("uber_bitpacked_encoding_issue.pagefile"));
//
//  facebook::velox::dwio::common::ReaderOptions readerOptions{leafPool_.get()};
//  auto reader = createReader(sample, readerOptions);
//  auto numRows = reader->numberOfRows();
//  std::cout << "Num of rows:" << (numRows ? *numRows : -1) << std::endl;
//  auto type = reader->typeWithId();
//  RowReaderOptions rowReaderOpts;
//  auto rowType1 = ROW({"msg.event.category"}, {ARRAY(VARCHAR())});
//  rowReaderOpts.setScanSpec(makeScanSpec(rowType1));
//  auto rowReader = reader->createRowReader(rowReaderOpts);
//  auto result = BaseVector::create(rowType1, 10, leafPool_.get());
//  constexpr int kBatchSize = 100;
//  while (rowReader->next(kBatchSize, result)) {
//  }

