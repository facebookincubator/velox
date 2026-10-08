/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/common/SelectiveRepeatedColumnReader.h"
#include "velox/dwio/nimble/velox/selective/ChunkedDecoder.h"
#include "velox/dwio/nimble/velox/selective/NimbleData.h"

namespace facebook::nimble {

class ListColumnReader : public velox::dwio::common::SelectiveListColumnReader {
 public:
  ListColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec);

  void readLengths(int32_t* lengths, int32_t numLengths, const uint64_t* nulls)
      override;

  uint64_t skip(uint64_t numValues) override;

  void resetFilterCaches() override {
    if (child_) {
      child_->resetFilterCaches();
    }
  }

  void seekToRowGroup(int64_t /*index*/) override {
    VELOX_UNREACHABLE();
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const override;
};

class MapColumnReader : public velox::dwio::common::SelectiveMapColumnReader {
 public:
  MapColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec);

  void readLengths(int32_t* lengths, int32_t numLengths, const uint64_t* nulls)
      override;

  uint64_t skip(uint64_t numValues) override;

  void seekToRowGroup(int64_t /*index*/) override {
    VELOX_UNREACHABLE();
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const override;
};

class MapAsStructColumnReader
    : public velox::dwio::common::SelectiveMapAsStructColumnReader {
 public:
  MapAsStructColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec);

  void readLengths(int32_t* lengths, int32_t numLengths, const uint64_t* nulls)
      override;

  uint64_t skip(uint64_t numValues) override;

  void seekToRowGroup(int64_t /*index*/) override {
    NIMBLE_UNREACHABLE();
  }
};

class DeduplicatedArrayColumnReader;
class DeduplicatedMapColumnReader;

class DeduplicatedReadHelper {
 public:
  DeduplicatedReadHelper(
      velox::dwio::common::SelectiveRepeatedColumnReader* columnReader,
      bool deduplicatedLength);

  size_t prepareDeduplicatedStates(int32_t numLengths, const uint64_t* nulls);

  vector_size_t getSelectedAlphabetLength(
      velox::RowSet rows,
      size_t alphabetSize);

  void skip(uint64_t numValues, size_t alphabetSize);

  void makeNestedRowSet(
      const velox::RowSet& rows,
      const uint64_t* nulls,
      int32_t maxRow);

  void makeAlphabetOffsetsAndSizes(
      velox::ArrayVectorBase& alphabet,
      vector_size_t lastRunLength);

  void populateSelectedIndices(
      const uint64_t* nulls,
      velox::RowSet rows,
      velox::vector_size_t* indices);

 private:
  velox::vector_size_t& lengthAt(velox::vector_size_t i) {
    return columnReader_->allLengths_[i];
  }

  velox::dwio::common::SelectiveRepeatedColumnReader* const columnReader_;
  const bool deduplicatedLength_;

  // Decoder of the deduplicated lengths.
  std::unique_ptr<ChunkedDecoder> const lengthsDecoder_;

  velox::BufferPtr runStartRowsHolder_;
  int32_t* runStartRows_;
  std::vector<vector_size_t> selectedIndices_;

  velox::vector_size_t latestOffset_ = -1;
  velox::vector_size_t latestLength_ = -1;
  velox::vector_size_t lastRunLength_ = 0;
  velox::vector_size_t skippedLastRunlength_;
  bool copyLastRun_ = false;
  bool loadLastRun_ = true;
  // TODO: is there a way to get rid of this? Maybe we can similate this by
  // augmenting the 0th of runStartRows and allLengths somehow, and just don't
  // add 0 to selectedIndices_.
  bool startFromLastRun_ = false;
  bool lastRunLoaded_ = false;

  friend class DeduplicatedArrayColumnReader;
  friend class DeduplicatedMapColumnReader;
};

class DeduplicatedArrayColumnReader
    : public velox::dwio::common::SelectiveListColumnReader {
 public:
  DeduplicatedArrayColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec);

  void readLengths(int32_t* lengths, int32_t numLengths, const uint64_t* nulls)
      override;

  uint64_t skip(uint64_t numValues) override;

  void makeNestedRowSet(const velox::RowSet& rows, int32_t maxRow) override;

  void resetFilterCaches() override {
    if (child_) {
      child_->resetFilterCaches();
    }
  }

  void seekToRowGroup(int64_t /*index*/) override {
    VELOX_UNREACHABLE();
  }

  void read(
      int64_t offset,
      const velox::RowSet& rows,
      const uint64_t* incomingNulls) override;

  void getValues(const velox::RowSet& rows, velox::VectorPtr* result) override;

 private:
  bool copyLastRun() const {
    return deduplicatedReadHelper_.copyLastRun_ && lastRunValue_;
  }

  // Because we don't know if we ended the last batch in the middle of a run,
  // unless it's at the chunk boundaries. However, if we load all runs up to the
  // latest chunk we risk having a much higher memory footprint than necessary.
  // The current approach is to cache the last run value in case the run
  // continues at the start of the next batch. (And even when at chunk
  // boundaries, we can get some small memory saving for output vector if we
  // reuse the last run.)
  velox::VectorPtr lastRunValue_;

  DeduplicatedReadHelper deduplicatedReadHelper_;
  friend class DeduplicatedReadHelper;
};

class DeduplicatedMapColumnReader
    : public velox::dwio::common::SelectiveMapColumnReader {
 public:
  DeduplicatedMapColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec);

  void readLengths(int32_t* lengths, int32_t numLengths, const uint64_t* nulls)
      override;

  uint64_t skip(uint64_t numValues) override;

  void makeNestedRowSet(const velox::RowSet& rows, int32_t maxRow) override;

  void seekToRowGroup(int64_t /*index*/) override {
    VELOX_UNREACHABLE();
  }

  void read(
      int64_t offset,
      const velox::RowSet& rows,
      const uint64_t* incomingNulls) override;

  void getValues(const velox::RowSet& rows, velox::VectorPtr* result) override;

 private:
  bool copyLastRun() const {
    return deduplicatedReadHelper_.copyLastRun_ && lastRunKeys_;
  }

  // Because we don't know if we ended the last batch in the middle of a run,
  // unless it's at the chunk boundaries. However, if we load all runs up to the
  // latest chunk we risk having a much higher memory footprint than necessary.
  // The current approach is to cache the last run value in case the run
  // continues at the start of the next batch. (And even when at chunk
  // boundaries, we can get some small memory saving for output vector if we
  // reuse the last run.)
  velox::VectorPtr lastRunKeys_;
  velox::VectorPtr lastRunValues_;

  DeduplicatedReadHelper deduplicatedReadHelper_;
  friend class DeduplicatedReadHelper;
};

} // namespace facebook::nimble
