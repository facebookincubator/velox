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
#include "velox/connectors/paimon/PaimonMergeSplitReader.h"

#include <algorithm>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/TreeOfLosers.h"
#include "velox/connectors/paimon/PaimonMergeStream.h"

namespace facebook::velox::connector::paimon {
namespace {

// Returns true if row 'row' of 'batch' and row 'otherRow' of 'otherBatch' have
// the same key, which is in the first 'numKeys' columns of both.
bool sameKey(
    const RowVector& batch,
    vector_size_t row,
    const RowVector& otherBatch,
    vector_size_t otherRow,
    column_index_t numKeys) {
  for (column_index_t i = 0; i < numKeys; ++i) {
    if (!batch.childAt(i)->equalValueAt(
            otherBatch.childAt(i).get(), row, otherRow)) {
      return false;
    }
  }
  return true;
}

// Position of a row among the data files of a split.
struct FileRow {
  size_t fileIndex;
  uint64_t row;
};

// Picks the row a merge engine returns for a key from the rows of the key in
// merge order, as Paimon Java's DeduplicateMergeFunction and
// FirstRowMergeFunction do. Like Paimon Java, returns the row of a single-row
// key without merging, and drops a picked DELETE or UPDATE_BEFORE row. When
// compaction looks up keys, deduplicate merges only the rows of level 0 and
// the first row of the lowest level above 0, as Paimon Java's
// LookupMergeFunction does.
class MergeFunction {
 public:
  MergeFunction(
      const std::vector<PaimonDataFile>& dataFiles,
      const PaimonTableHandle::MergeOptions& options)
      : dataFiles_(dataFiles),
        firstRow_(options.mergeEngine == PaimonMergeEngine::kFirstRow),
        ignoreDelete_(options.ignoreDelete),
        // Paimon Java does not wrap the first-row merge function.
        lookup_(options.needLookup && !firstRow_) {}

  void add(const PaimonMergeStream& stream) {
    ++numRows_;
    const bool retract = stream.isRetract();
    if (lookup_) {
      const auto level = stream.level();
      if (level > 0) {
        if (!highLevelRow_.has_value() || level < highLevelRow_->level) {
          highLevelRow_ = HighLevelRow{
              FileRow{stream.fileIndex(), stream.fileRow()},
              level,
              retract,
              numRows_};
        }
        return;
      }
    }
    if (retract && ignoreDelete_) {
      return;
    }
    if (!firstRow_) {
      picked_ = FileRow{stream.fileIndex(), stream.fileRow()};
      pickedRetract_ = retract;
      pickedPosition_ = numRows_;
      return;
    }
    if (retract) {
      retractFileIndex_ = stream.fileIndex();
    } else if (!picked_.has_value()) {
      picked_ = FileRow{stream.fileIndex(), stream.fileRow()};
    }
  }

  // Returns the row the key returns, if any, and resets for the next key.
  std::optional<FileRow> finish() {
    if (numRows_ > 1 && retractFileIndex_.has_value()) {
      VELOX_USER_FAIL(
          "Paimon first-row merge engine cannot merge a DELETE or "
          "UPDATE_BEFORE row unless 'ignore-delete' is true: {}",
          dataFiles_[*retractFileIndex_].path);
    }
    // Paimon Java passes the high-level row to deduplicate at its position in
    // merge order among the level-0 rows, so the row wins if it sorts after
    // the last kept level-0 row or no level-0 row is kept.
    if (highLevelRow_.has_value() &&
        !(highLevelRow_->retract && ignoreDelete_) &&
        (!picked_.has_value() || highLevelRow_->position > pickedPosition_)) {
      picked_ = highLevelRow_->row;
      pickedRetract_ = highLevelRow_->retract;
    }
    std::optional<FileRow> result;
    if (!pickedRetract_) {
      result = picked_;
    }
    numRows_ = 0;
    picked_.reset();
    pickedRetract_ = false;
    pickedPosition_ = 0;
    retractFileIndex_.reset();
    highLevelRow_.reset();
    return result;
  }

 private:
  // The first row of the lowest level above 0 of a key.
  struct HighLevelRow {
    FileRow row;
    int32_t level;
    bool retract;
    // 1-based position of the row among the rows of the key in merge order.
    // E.g. with 'sequence.field' = 'ts' and the rows (level 0, ts 5),
    // (level 1, ts 25) and (level 0, ts 30), the level-1 row is at 2, so
    // finish() returns the level-0 row at 3. Without that row, it would
    // return the level-1 row, which sorts after the level-0 row at 1.
    uint64_t position;
  };

  // Data files of the split, whose levels lookup merges read and whose paths
  // error messages name.
  const std::vector<PaimonDataFile>& dataFiles_;
  // True for the first-row merge engine, false for deduplicate.
  const bool firstRow_;
  // True to skip DELETE and UPDATE_BEFORE rows, as 'ignore-delete' asks.
  const bool ignoreDelete_;
  // True to merge only the rows of level 0 and the first row of the lowest
  // level above 0, as Paimon Java's LookupMergeFunction does.
  const bool lookup_;
  // Number of rows of the current key added so far.
  uint64_t numRows_{0};
  // Row the current key returns so far: the last row for deduplicate, the
  // first row that is not a retract for first-row.
  std::optional<FileRow> picked_;
  // True if picked_ is a DELETE or UPDATE_BEFORE row, so the key returns
  // nothing.
  bool pickedRetract_{false};
  // 1-based position of the picked row among the rows of the key.
  uint64_t pickedPosition_{0};
  // File of a first-row retract, which Paimon Java rejects when it merges.
  std::optional<size_t> retractFileIndex_;
  // Set only for lookup merges.
  std::optional<HighLevelRow> highLevelRow_;
};

} // namespace

PaimonMergeSplitReader::PaimonMergeSplitReader(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
        partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const connector::FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : PaimonSplitReader(
          fileSplit,
          paimonSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(std::move(paimonSplit)),
      paimonTableHandle_(tableHandle) {}

uint64_t PaimonMergeSplitReader::next(uint64_t size, VectorPtr& output) {
  if (rowSelections_.empty()) {
    mergeKeys(size);
  }
  return PaimonSplitReader::next(size, output);
}

const uint64_t* PaimonMergeSplitReader::rowSelection() const {
  return rowSelections_[currentDataFileIndex()]->as<uint64_t>();
}

void PaimonMergeSplitReader::mergeKeys(uint64_t batchSize) {
  const auto& dataFiles = paimonSplit_->dataFiles();
  auto streams = openMergeStreams(*paimonTableHandle_, batchSize);
  std::vector<BufferPtr> rowSelections;
  rowSelections.reserve(streams.size());
  for (const auto& stream : streams) {
    // A bit per row the metadata of the file counts, which the merge stream of
    // the file checks.
    rowSelections.emplace_back(
        AlignedBuffer::allocate<uint64_t>(
            std::max<uint64_t>(
                bits::nwords(dataFiles[stream->fileIndex()].rowCount), 1),
            pool_,
            0));
  }

  const auto numKeys = static_cast<column_index_t>(
      paimonTableHandle_->nonPartitionPrimaryKeys().size());
  TreeOfLosers<PaimonMergeStream> merge(std::move(streams));
  MergeFunction mergeFunction(dataFiles, paimonTableHandle_->mergeOptions());
  const auto finishKey = [&]() {
    if (const auto picked = mergeFunction.finish()) {
      bits::setBit(
          rowSelections[picked->fileIndex]->asMutable<uint64_t>(), picked->row);
    }
  };
  // First row of the current key. Holding its batch keeps the row readable
  // after its stream moves on.
  RowVectorPtr keyBatch;
  vector_size_t keyRow{0};
  while (auto* stream = merge.next()) {
    if (keyBatch == nullptr ||
        !sameKey(*keyBatch, keyRow, *stream->batch(), stream->row(), numKeys)) {
      finishKey();
      keyBatch = stream->batch();
      keyRow = stream->row();
    }
    mergeFunction.add(*stream);
    stream->pop();
  }
  finishKey();
  rowSelections_ = std::move(rowSelections);
}

} // namespace facebook::velox::connector::paimon
