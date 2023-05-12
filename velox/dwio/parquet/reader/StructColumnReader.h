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

#pragma once

#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"

namespace facebook::velox::parquet {

class StructColumnReader : public dwio::common::SelectiveStructColumnReader {
 public:
  StructColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void seekToRowGroup(uint32_t index) override;

  /// Creates the streams for 'rowGroup in 'input'. Does not load yet.
  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  // No-op in Parquet. All readers switch row groups at the same time, there is
  // no on-demand skipping to a new row group.
  void advanceFieldReader(
      dwio::common::SelectiveColumnReader* FOLLY_NONNULL /*reader*/,
      vector_size_t /*offset*/) override {}

  void setNullsFromRepDefs(PageReader& pageReader);

  dwio::common::SelectiveColumnReader* FOLLY_NULLABLE childForRepDefs() const {
    return childForRepDefs_;
  }

  /// Nested struct readers all get null flags and lengths for
  /// contained repeated readers for each range of top level rows. At
  /// the end of a read() with filters in different members, some of
  /// wich are structs themselves, different inner structs may be left
  /// on different rows. Before receiving the next set of
  /// nulls/lengths, the contained complex readers need to be
  /// positioned at the end of the last set of nulls/lengths.
  void seekToEndOfPresetNulls();

  void filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsContext&,
      dwio::common::FormatData::FilterRowGroupsResult&) const override;

 private:
  bool filterMatches(const thrift::RowGroup& rowGroup);
  dwio::common::SelectiveColumnReader* findBestLeaf();

  // Leaf column reader used for getting nullability information for
  // 'this'. This is nullptr for the root of a table.
  dwio::common::SelectiveColumnReader* FOLLY_NULLABLE childForRepDefs_{nullptr};

  // Mode for getting nulls from repdefs. kStructOverLists if 'this'
  // only has list children.
  LevelMode levelMode_;

  // The level information for extracting nulls for 'this' from the
  // repdefs in a leaf PageReader.
  ::parquet::internal::LevelInfo levelInfo_;
};

} // namespace facebook::velox::parquet
