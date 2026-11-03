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
#include <unordered_map>
#include <unordered_set>

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/parquet/common/LevelConversion.h"

namespace facebook::velox::dwio::common {
class BufferedInput;
}

namespace facebook::velox::parquet {

enum class LevelMode;
class PageReader;
class ParquetData;
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

  /// Root reader only. If set, the lazy columns (top-level children produced
  /// as LazyVectors: projected, no filter) are not enqueued with their row
  /// group until the scan shows they are needed: markAllLazyColumnsNeeded()
  /// marks all of them, a first read of one of them marks that column only. A
  /// needed lazy column is enqueued with every following row group; for the
  /// row groups already buffered its chunks are enqueued into a separate
  /// input and loaded right away.
  void setDeferLazyColumnPrefetch(bool defer) {
    deferLazyColumnPrefetch_ = defer;
  }

  /// A row passed the filters, so all lazy columns will be read.
  void markAllLazyColumnsNeeded();

  /// Drops the inputs kept for deferred chunks of 'index'th row group.
  void releaseRowGroup(uint32_t index);

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

  // Resolves lazyColumns_ and lazyColumnLeaves_ on first use (isTopLevel is
  // set after construction).
  void resolveLazyColumns();

  // Marks 'column' as needed and enqueues + loads the chunks of all needed
  // lazy columns that are missing in the buffered row groups.
  void markLazyColumnNeeded(dwio::common::SelectiveColumnReader* column);

  void prefetchNeededLazyColumns();

  bool deferLazyColumnPrefetch_{false};
  bool lazyColumnsResolved_{false};
  std::unordered_set<dwio::common::SelectiveColumnReader*> lazyColumns_;
  std::unordered_set<dwio::common::SelectiveColumnReader*> neededLazyColumns_;
  // Physical columns (leaves) of the lazy columns, with their lazy column.
  std::vector<std::pair<ParquetData*, dwio::common::SelectiveColumnReader*>>
      lazyColumnLeaves_;
  // BufferedInput of each buffered row group.
  std::unordered_map<uint32_t, std::weak_ptr<dwio::common::BufferedInput>>
      rowGroupInputs_;
  // Inputs of the deferred loads, per row group. Each time more lazy columns
  // become needed for a buffered row group, their chunks are enqueued into a
  // new input that is loaded once (a BufferedInput must not be loaded twice).
  // All are kept until the row group is released.
  std::unordered_map<
      uint32_t,
      std::vector<std::shared_ptr<dwio::common::BufferedInput>>>
      deferredInputs_;

  // Mode for getting nulls from repdefs. kStructOverLists if the source is
  // below an ARRAY or MAP.
  LevelMode levelMode_;

  // The level information for extracting nulls for 'this' from the
  // repdefs in a leaf PageReader.
  LevelInfo levelInfo_;

  // True when all requested fields are missing from this file and the
  // configured policy requires the struct itself to be null.
  bool nullStructForMissingFields_{false};

  // Owns the synthetic non-projected reader and its ScanSpec.
  std::unique_ptr<SyntheticRepDefSource> syntheticRepDefSource_;
};

} // namespace facebook::velox::parquet
