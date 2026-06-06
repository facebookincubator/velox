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
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Options.h"

using namespace ::testing;
using namespace facebook::velox::dwio::common;

TEST(OptionsTests, defaultRowNumberColumnInfoTest) {
  // appendRowNumberColumn flag should be false by default
  RowReaderOptions rowReaderOptions;
  ASSERT_EQ(std::nullopt, rowReaderOptions.rowNumberColumnInfo());
}

TEST(OptionsTests, fluxFileFormatRoundTrip) {
  ASSERT_EQ(FileFormat::FLUX, toFileFormat("flux"));
  ASSERT_EQ("flux", toString(FileFormat::FLUX));
}

TEST(OptionsTests, setRowNumberColumnInfoTest) {
  RowReaderOptions rowReaderOptions;
  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 0;
  rowNumberColumnInfo.name = "test";
  rowReaderOptions.setRowNumberColumnInfo(rowNumberColumnInfo);
  auto rowNumberColumn = rowReaderOptions.rowNumberColumnInfo().value();
  ASSERT_EQ(rowNumberColumnInfo.insertPosition, rowNumberColumn.insertPosition);
  ASSERT_EQ(rowNumberColumnInfo.name, rowNumberColumn.name);
}

TEST(OptionsTests, testRowNumberColumnInfoInCopy) {
  RowReaderOptions rowReaderOptions;
  RowReaderOptions rowReaderOptionsCopy{rowReaderOptions};
  ASSERT_EQ(std::nullopt, rowReaderOptionsCopy.rowNumberColumnInfo());

  RowNumberColumnInfo rowNumberColumnInfo;
  rowNumberColumnInfo.insertPosition = 0;
  rowNumberColumnInfo.name = "test";
  rowReaderOptions.setRowNumberColumnInfo(rowNumberColumnInfo);
  RowReaderOptions rowReaderOptionsSecondCopy{rowReaderOptions};
  auto rowNumberColumn =
      rowReaderOptionsSecondCopy.rowNumberColumnInfo().value();
  ASSERT_EQ(rowNumberColumnInfo.insertPosition, rowNumberColumn.insertPosition);
  ASSERT_EQ(rowNumberColumnInfo.name, rowNumberColumn.name);
}

TEST(OptionsTests, cacheData) {
  facebook::velox::memory::MemoryManager::testingSetInstance({});
  auto pool =
      facebook::velox::memory::memoryManager()->addRootPool("cacheDataTest");
  facebook::velox::dwio::common::ReaderOptions options(pool.get());
  EXPECT_TRUE(options.cacheData());

  options.setCacheData(false);
  EXPECT_FALSE(options.cacheData());

  options.setCacheData(true);
  EXPECT_TRUE(options.cacheData());
}
