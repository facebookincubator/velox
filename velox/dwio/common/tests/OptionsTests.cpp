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

#include <typeinfo>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Options.h"

using namespace ::testing;
using namespace facebook::velox::dwio::common;

namespace {

// Format options that support cloning, used to verify that
// WriterOptions::clone() copies the format options instead of sharing them.
class CloneableFormatOptions : public FormatSpecificOptions {
 public:
  explicit CloneableFormatOptions(int32_t value) : value{value} {}

  std::shared_ptr<FormatSpecificOptions> clone() const override {
    return std::make_shared<CloneableFormatOptions>(*this);
  }

  int32_t value;
};

// A WriterOptions subclass that fails to override clone(), which must be
// rejected rather than silently sliced.
struct SlicingWriterOptions : public WriterOptions {
  int32_t extraField{0};
};

// A WriterOptions subclass that overrides clone() as required.
struct CloneableWriterOptions : public WriterOptions {
  int32_t extraField{0};

  std::shared_ptr<WriterOptions> clone() const override {
    return deepCopyInto(std::make_shared<CloneableWriterOptions>(*this));
  }
};

} // namespace

TEST(OptionsTests, defaultRowNumberColumnInfoTest) {
  // appendRowNumberColumn flag should be false by default
  RowReaderOptions rowReaderOptions;
  ASSERT_EQ(std::nullopt, rowReaderOptions.rowNumberColumnInfo());
}

TEST(OptionsTests, fluxFileFormatRoundTrip) {
  ASSERT_EQ(FileFormat::FLUX, toFileFormat("flux"));
  ASSERT_EQ("flux", FileFormatName::toName(FileFormat::FLUX));
}

TEST(OptionsTests, formatConfigPrefix) {
  EXPECT_EQ("parquet.", formatConfigPrefix(FileFormat::PARQUET, "."));
  EXPECT_EQ("parquet_", formatConfigPrefix(FileFormat::PARQUET, "_"));
  EXPECT_EQ("orc.", formatConfigPrefix(FileFormat::DWRF, "."));
  EXPECT_EQ("orc_", formatConfigPrefix(FileFormat::DWRF, "_"));
  EXPECT_EQ("", formatConfigPrefix(FileFormat::UNKNOWN, "."));
}

TEST(OptionsTests, commonReaderOptions) {
  facebook::velox::memory::MemoryManager::testingSetInstance({});
  auto pool = facebook::velox::memory::memoryManager()->addRootPool(
      "commonReaderOptionsTest");
  facebook::velox::dwio::common::ReaderOptions options(pool.get());

  options.setColumnMappingMode(ColumnMappingMode::kName);
  options.setFooterSpeculativeIoSize(1234);

  EXPECT_EQ(options.columnMappingMode(), ColumnMappingMode::kName);
  EXPECT_EQ(options.footerSpeculativeIoSize(), 1234);
  EXPECT_EQ(
      makeColumnReaderOptions(options).columnMappingMode_,
      ColumnMappingMode::kName);
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

TEST(OptionsTests, writerOptionsCloneIsDeep) {
  WriterOptions options;
  options.sessionTimezoneName = "America/Los_Angeles";
  options.serdeParameters = {{"key", "value"}};
  options.formatSpecificOptions = std::make_shared<CloneableFormatOptions>(11);

  const auto cloned = options.clone();
  ASSERT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->sessionTimezoneName, "America/Los_Angeles");
  EXPECT_EQ(cloned->serdeParameters, options.serdeParameters);

  // The format options are copied, so merging into them leaves the source
  // object unchanged.
  ASSERT_NE(cloned->formatSpecificOptions, nullptr);
  EXPECT_NE(cloned->formatSpecificOptions, options.formatSpecificOptions);
  const auto clonedFormatOptions =
      std::dynamic_pointer_cast<CloneableFormatOptions>(
          cloned->formatSpecificOptions);
  ASSERT_NE(clonedFormatOptions, nullptr);
  EXPECT_EQ(clonedFormatOptions->value, 11);

  // Mutating the copy must leave the source untouched.
  cloned->sessionTimezoneName = "UTC";
  clonedFormatOptions->value = 22;
  EXPECT_EQ(options.sessionTimezoneName, "America/Los_Angeles");
  EXPECT_EQ(
      std::dynamic_pointer_cast<CloneableFormatOptions>(
          options.formatSpecificOptions)
          ->value,
      11);
}

TEST(OptionsTests, writerOptionsCloneKeepsSubclassTypeAndDeepCopies) {
  CloneableWriterOptions options;
  options.extraField = 7;
  options.formatSpecificOptions = std::make_shared<CloneableFormatOptions>(11);

  const auto cloned = options.clone();
  ASSERT_NE(cloned, nullptr);
  // The copy keeps the concrete subclass type and its fields.
  const auto typedClone =
      std::dynamic_pointer_cast<CloneableWriterOptions>(cloned);
  ASSERT_NE(typedClone, nullptr);
  EXPECT_EQ(typedClone->extraField, 7);

  // Subclasses must deep-copy the format options as well.
  ASSERT_NE(cloned->formatSpecificOptions, nullptr);
  EXPECT_NE(cloned->formatSpecificOptions, options.formatSpecificOptions);
}

TEST(OptionsTests, writerOptionsCloneRejectsSubclassWithoutOverride) {
  SlicingWriterOptions options;
  VELOX_ASSERT_THROW(
      options.clone(), "WriterOptions subclass must override clone()");
}

TEST(OptionsTests, formatSpecificOptionsCloneUnsupportedByDefault) {
  FormatSpecificOptions options;
  VELOX_ASSERT_THROW(
      options.clone(),
      "Cloning format-specific options is not supported for these options.");
}

TEST(OptionsTests, writerOptionsCloneKeepsFormatOptionsSubclassType) {
  WriterOptions options;
  options.formatSpecificOptions = std::make_shared<CloneableFormatOptions>(11);

  const auto cloned = options.clone();
  ASSERT_NE(cloned->formatSpecificOptions, nullptr);
  EXPECT_TRUE(
      typeid(*cloned->formatSpecificOptions) ==
      typeid(*options.formatSpecificOptions));
}
