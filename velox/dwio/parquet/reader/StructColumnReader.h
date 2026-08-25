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

#include <memory>

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/parquet/common/LevelConversion.h"

namespace facebook::velox::dwio::common {
class BufferedInput;
}

namespace facebook::velox::parquet {

enum class LevelMode;
class PageReader;
class ParquetParams;

class StructColumnReader : public dwio::common::SelectiveStructColumnReader {
 public:
  StructColumnReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  ~StructColumnReader() override;

  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override;

  void seekToRowGroup(int64_t index) override;

  /// Creates the streams for 'rowGroup'. Checks whether row 'rowGroup'
  /// has been buffered in 'input'. If true, return the input. Or else creates
  /// the streams in a new input and loads.
  std::shared_ptr<dwio::common::BufferedInput> loadRowGroup(
      uint32_t index,
      const std::shared_ptr<dwio::common::BufferedInput>& input);

  // No-op in Parquet. All readers switch row groups at the same time, there is
  // no on-demand skipping to a new row group.
  void advanceFieldReader(
      dwio::common::SelectiveColumnReader* /*reader*/,
      int64_t /*offset*/) override {}

  void setNullsFromRepDefs(PageReader& pageReader);

  /// Returns the reader that supplies repetition and definition levels for
  /// this struct. This is null exactly for the root struct. For a nested
  /// struct, the reader may be a logical child or a synthetic physical leaf
  /// and must not be advanced using the enclosing struct's row count.
  dwio::common::SelectiveColumnReader* repDefSourceReader() const {
    return repDefSourceReader_;
  }

  /// Nested struct readers all get null flags and lengths for
  /// contained repeated readers for each range of top level rows. At
  /// the end of a read() with filters in different members, some of
  /// which are structs themselves, different inner structs may be left
  /// on different rows. Before receiving the next set of
  /// nulls/lengths, the contained complex readers need to be
  /// positioned at the end of the last set of nulls/lengths.
  void seekToEndOfPresetNulls();

  void filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsContext&,
      dwio::common::FormatData::FilterRowGroupsResult&) const override;

 private:
  struct SyntheticRepDefSource;

  // Creates a non-projected physical leaf reader to source repetition and
  // definition levels when no logical child reader is available.
  void ensureSyntheticRepDefSource(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      ParquetParams& params);

  void applyMissingFieldPolicy(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      bool nullStructIfAllFieldsMissing);

  dwio::common::SelectiveColumnReader* FOLLY_NONNULL findBestLeaf();

  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  bool isRowGroupBuffered(uint32_t index, dwio::common::BufferedInput& input);

  // Reader subtree used for getting nullability information for 'this'.
  dwio::common::SelectiveColumnReader* repDefSourceReader_{nullptr};

  // Mode for getting nulls from repdefs. kStructOverLists if the source is
  // below an ARRAY or MAP.
  LevelMode levelMode_;

  // The level information for extracting nulls for 'this' from the
  // repdefs in a leaf PageReader.
  LevelInfo levelInfo_;

  // Owns the synthetic non-projected reader and its ScanSpec.
  std::unique_ptr<SyntheticRepDefSource> syntheticRepDefSource_;
};

} // namespace facebook::velox::parquet
