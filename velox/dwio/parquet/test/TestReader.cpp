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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/common/base/test_utils/GTestUtils.h"
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/type/fbhive/HiveTypeParser.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <fmt/core.h>
#include <array>
#include <numeric>

using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::type::fbhive;
using namespace facebook::velox;
using namespace facebook::velox::parquet;

inline std::string getExampleFilePath(const std::string& fileName) {
  return test::getDataFilePath(
      "velox/dwio/parquet/test", "examples/" + fileName);
}

static RowReaderOptions getSample1ReaderOpts() {
  RowReaderOptions rowReaderOpts;
  std::shared_ptr<const RowType> rowType =
      std::dynamic_pointer_cast<const RowType>(HiveTypeParser().parse("struct<\
          a:bigint,\
          b:double>"));
  auto cs = std::make_shared<ColumnSelector>(
      rowType, std::vector<std::string>{"a", "b"});
  rowReaderOpts.select(cs);
  return rowReaderOpts;
}

TEST(TestReader, testReadSample1_Full) {
  // sample1.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 20 rows (10 rows per group). Group offsets are 153 and 614.
  // Data is in plain uncompressed format:
  //   a: [1..20]
  //   b: [1.0..20.0]
  const std::string sample1(getExampleFilePath("sample1.parquet"));

  ReaderOptions readerOptions;
  ParquetReader reader(
      std::make_unique<FileInputStream>(sample1), readerOptions);

  EXPECT_EQ(reader.numberOfRows(), 20ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);

  RowReaderOptions rowReaderOpts = getSample1ReaderOpts();
  auto rowReader = reader.createRowReader(rowReaderOpts);

  uint64_t total = 0;
  VectorPtr result;
  while (total < 20) {
    auto part = rowReader->next(1000, result);
    EXPECT_GT(part, 0);
    auto rowVector = result->as<RowVector>();
    EXPECT_EQ(rowVector->childrenSize(), 2ULL);
    auto colA = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
    auto colB = rowVector->childAt(1)->as<SimpleVector<double>>();
    for (uint64_t i = 0; i < part; ++i) {
      EXPECT_EQ(colA->valueAt(i), total + i + 1);
      EXPECT_EQ(colB->valueAt(i), double(total + i + 1));
    }
    total += part;
  }
  EXPECT_EQ(total, 20);
  EXPECT_EQ(rowReader->next(1000, result), 0);
}

TEST(TestReader, testReadSample1_Range1) {
  const std::string sample1(getExampleFilePath("sample1.parquet"));

  ReaderOptions readerOptions;
  ParquetReader reader(
      std::make_unique<FileInputStream>(sample1), readerOptions);

  RowReaderOptions rowReaderOpts = getSample1ReaderOpts();
  rowReaderOpts.range(0, 200);
  auto rowReader = reader.createRowReader(rowReaderOpts);

  uint64_t total = 0;
  VectorPtr result;
  while (total < 10) {
    auto part = rowReader->next(1000, result);
    EXPECT_GT(part, 0);
    auto rowVector = result->as<RowVector>();
    EXPECT_EQ(rowVector->childrenSize(), 2ULL);
    auto colA = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
    auto colB = rowVector->childAt(1)->as<SimpleVector<double>>();
    for (uint64_t i = 0; i < part; ++i) {
      EXPECT_EQ(colA->valueAt(i), total + i + 1);
      EXPECT_EQ(colB->valueAt(i), double(total + i + 1));
    }
    total += part;
  }
  EXPECT_EQ(total, 10);
  EXPECT_EQ(rowReader->next(1000, result), 0);
}

TEST(TestReader, testReadSample1_Range2) {
  const std::string sample1(getExampleFilePath("sample1.parquet"));

  ReaderOptions readerOptions;
  ParquetReader reader(
      std::make_unique<FileInputStream>(sample1), readerOptions);

  RowReaderOptions rowReaderOpts = getSample1ReaderOpts();
  rowReaderOpts.range(200, 500);
  auto rowReader = reader.createRowReader(rowReaderOpts);

  uint64_t total = 0;
  VectorPtr result;
  while (total < 10) {
    auto part = rowReader->next(1000, result);
    EXPECT_GT(part, 0);
    auto rowVector = result->as<RowVector>();
    EXPECT_EQ(rowVector->childrenSize(), 2ULL);
    auto colA = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
    auto colB = rowVector->childAt(1)->as<SimpleVector<double>>();
    for (uint64_t i = 0; i < part; ++i) {
      EXPECT_EQ(colA->valueAt(i), total + i + 11);
      EXPECT_EQ(colB->valueAt(i), double(total + i + 11));
    }
    total += part;
  }
  EXPECT_EQ(total, 10);
  EXPECT_EQ(rowReader->next(1000, result), 0);
}

TEST(TestReader, testReadSample1_EmptyRange) {
  const std::string sample1(getExampleFilePath("sample1.parquet"));

  ReaderOptions readerOptions;
  ParquetReader reader(
      std::make_unique<FileInputStream>(sample1), readerOptions);

  RowReaderOptions rowReaderOpts = getSample1ReaderOpts();
  rowReaderOpts.range(300, 10);
  auto rowReader = reader.createRowReader(rowReaderOpts);

  VectorPtr result;
  EXPECT_EQ(rowReader->next(1000, result), 0);
}

TEST(TestReader, testReadSample1_BigintRangeFilter) {
  const std::string sample1(getExampleFilePath("sample1.parquet"));

  ReaderOptions readerOptions;
  ParquetReader reader(
      std::make_unique<FileInputStream>(sample1), readerOptions);

  RowReaderOptions rowReaderOpts = getSample1ReaderOpts();
  common::ScanSpec scanSpec("");
  scanSpec.getOrCreateChild(common::Subfield("a"))
      ->setFilter(std::make_unique<common::BigintRange>(16, 20, false));
  scanSpec.getOrCreateChild(common::Subfield("b"));
  rowReaderOpts.setScanSpec(&scanSpec);
  auto rowReader = reader.createRowReader(rowReaderOpts);

  uint64_t total = 0;
  VectorPtr result;
  while (auto part = rowReader->next(1000, result)) {
    EXPECT_GT(part, 0);
    auto rowVector = result->as<RowVector>();
    EXPECT_EQ(rowVector->childrenSize(), 2ULL);
    auto colA = rowVector->childAt(0)->as<SimpleVector<int64_t>>();
    auto colB = rowVector->childAt(1)->as<SimpleVector<double>>();
    for (uint64_t i = 0; i < part; ++i) {
      EXPECT_EQ(colA->valueAt(i), total + i + 16);
      EXPECT_EQ(colB->valueAt(i), double(total + i + 16));
    }
    total += part;
  }
  EXPECT_EQ(total, 5);
}