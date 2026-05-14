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

#include "velox/connectors/hive/iceberg/WriterOptionsAdapter.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Verifies the dispatch table in createWriterOptionsAdapter():
// PARQUET, DWRF, NIMBLE return non-null adapters; everything else returns
// null. This is the single source of truth for which file formats the
// Iceberg DataSink supports on the write path.
TEST(WriterOptionsAdapterTest, createWriterOptionsAdapterDispatch) {
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::PARQUET), nullptr);
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::DWRF), nullptr);
  EXPECT_NE(
      createWriterOptionsAdapter(dwio::common::FileFormat::NIMBLE), nullptr);

  // ORC, TEXT, JSON, ALPHA, etc. are intentionally unsupported on the
  // write path until each gets its own end-to-end coverage.
  EXPECT_EQ(createWriterOptionsAdapter(dwio::common::FileFormat::ORC), nullptr);
  EXPECT_EQ(
      createWriterOptionsAdapter(dwio::common::FileFormat::TEXT), nullptr);
  EXPECT_EQ(
      createWriterOptionsAdapter(dwio::common::FileFormat::JSON), nullptr);
}

// Verifies isSupportedFileFormat() agrees with createWriterOptionsAdapter().
TEST(WriterOptionsAdapterTest, isSupportedFileFormatMatchesDispatch) {
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::PARQUET));
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::DWRF));
  EXPECT_TRUE(isSupportedFileFormat(dwio::common::FileFormat::NIMBLE));

  EXPECT_FALSE(isSupportedFileFormat(dwio::common::FileFormat::ORC));
  EXPECT_FALSE(isSupportedFileFormat(dwio::common::FileFormat::TEXT));
  EXPECT_FALSE(isSupportedFileFormat(dwio::common::FileFormat::JSON));
}

// Verifies the manifest format string written into Iceberg commit messages
// matches the cross-engine convention. Iceberg's manifest vocabulary has no
// DWRF/NIMBLE enum, so both report "ORC" (matching the Java planner's
// FileFormat.{DWRF,NIMBLE}.toIceberg()) so downstream Iceberg consumers can
// interpret the message. Parquet reports "PARQUET" because Iceberg has a
// native enum for it. The on-disk format is identified at read time via
// the file extension and on-disk magic bytes, so writing "ORC" for NIMBLE
// is safe.
TEST(WriterOptionsAdapterTest, toManifestFormatString) {
  EXPECT_EQ(
      toManifestFormatString(dwio::common::FileFormat::PARQUET), "PARQUET");
  EXPECT_EQ(toManifestFormatString(dwio::common::FileFormat::DWRF), "ORC");
  EXPECT_EQ(toManifestFormatString(dwio::common::FileFormat::NIMBLE), "ORC");
}

// Verifies toManifestFormatString() throws for unsupported formats rather
// than silently returning an incorrect string.
TEST(WriterOptionsAdapterTest, toManifestFormatStringThrowsForUnsupported) {
  VELOX_ASSERT_THROW(
      toManifestFormatString(dwio::common::FileFormat::ORC),
      "Unsupported file format for Iceberg manifest");
  VELOX_ASSERT_THROW(
      toManifestFormatString(dwio::common::FileFormat::TEXT),
      "Unsupported file format for Iceberg manifest");
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
