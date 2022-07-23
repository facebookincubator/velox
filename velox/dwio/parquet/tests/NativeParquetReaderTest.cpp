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

//
// Created by Ying Su on 2/27/22.
//
#include "velox/dwio/parquet/reader/NativeParquetReader.h"
#include "ParquetReaderTest.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/dwrf/test/utils/FilterGenerator.h"

using namespace facebook::velox::parquet;
using namespace facebook::velox;
using namespace connector::hive;

std::unique_ptr<common::ScanSpec> makeScanSpec(
    const SubfieldFilters& filters,
    const std::shared_ptr<const RowType>& rowType);

class NativeParquetReaderTest : public ParquetReaderTest {};

TEST_F(NativeParquetReaderTest, readBigint) {
  // sample.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 20 rows (10 rows per group). Group offsets are 153 and 614.
  // Data is in plain uncompressed format:
  //   a: [1..20]
  //   b: [1.0..20.0]
  const std::string sample(getExampleFilePath("sample.parquet"));

  ReaderOptions readerOptions;
  NativeParquetReader reader(
      std::make_unique<FileInputStream>(sample), readerOptions);
  //
  EXPECT_EQ(reader.numberOfRows(), 20ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);

  RowTypePtr rowType = ROW({"a"}, {BIGINT()});
  auto filterGenerator =
      std::make_unique<dwio::dwrf::FilterGenerator>(rowType, 1);
  auto scanSpec = filterGenerator->makeScanSpec(SubfieldFilters{});
  RowReaderOptions rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader.createRowReader(rowReaderOpts);
  auto expected = vectorMaker_->rowVector({rangeVector<int64_t>(20, 1)});
  assertReadExpected(
      rowType, *rowReader, expected, readerOptions.getMemoryPool());
}

TEST_F(NativeParquetReaderTest, readRowMap) {
  const std::string sample(getExampleFilePath("tmp_row.parquet"));

  ReaderOptions readerOptions;
  NativeParquetReader reader(
      std::make_unique<FileInputStream>(sample), readerOptions);

  EXPECT_EQ(reader.numberOfRows(), 1ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 1ULL);

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("c"), col0);

  auto col0_0 = col0->childAt(0);
  EXPECT_EQ(col0_0->type->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col0->childByName("c0"), col0_0);

  auto col0_1 = col0->childAt(1);
  EXPECT_EQ(col0_1->type->kind(), TypeKind::MAP);
  EXPECT_EQ(col0->childByName("c1"), col0_1);

  auto col0_1_0 = col0_1->childAt(0);
  EXPECT_EQ(col0_1_0->type->kind(), TypeKind::VARCHAR);
}
