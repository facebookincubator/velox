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

#include <folly/String.h>
#include <algorithm>
#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/TreeOfLosers.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/connectors/common/FileDataSource.h"
#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::paimon {
namespace {

// Columns the key reads return: the primary keys that are not partition
// columns, the sequence fields, _SEQUENCE_NUMBER and _VALUE_KIND. All but
// _VALUE_KIND order the rows of a split, since Paimon Java orders the rows of a
// key by sequence fields and then by sequence number.
struct MergeColumns {
  // Names and types of the columns, in the order above.
  RowTypePtr type;

  // Number of leading key columns, which group the rows of a key.
  column_index_t numKeys{0};

  // Compare flags of the leading columns that order rows.
  std::vector<CompareFlags> compareFlags;
};

// Returns 'name' in lowercase if 'lowerCase' is true.
std::string systemColumnName(std::string_view name, bool lowerCase) {
  std::string result{name};
  if (lowerCase) {
    folly::toLowerAscii(result);
  }
  return result;
}

// Builds the merge columns. 'lowerCaseSystemColumns' requests _SEQUENCE_NUMBER
// and _VALUE_KIND in lowercase, for file readers that read file column names
// as lowercase.
MergeColumns makeMergeColumns(
    const PaimonTableHandle& tableHandle,
    bool lowerCaseSystemColumns) {
  const auto& rowType = tableHandle.rowType();
  const auto& keys = tableHandle.nonPartitionPrimaryKeys();
  const auto& partitionKeys = tableHandle.partitionKeys();
  const auto& options = tableHandle.mergeOptions();
  const auto contains = [](const std::vector<std::string>& names,
                           const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
  };

  MergeColumns columns;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (const auto& key : keys) {
    names.push_back(key);
    types.push_back(rowType->findChild(key));
    columns.compareFlags.push_back({.nullsFirst = true, .ascending = true});
  }
  columns.numKeys = static_cast<column_index_t>(keys.size());
  // Keys compare equal within a key, and partition columns within a split.
  for (const auto& field : options.sequenceFields) {
    if (contains(keys, field) || contains(partitionKeys, field)) {
      continue;
    }
    names.push_back(field);
    types.push_back(rowType->findChild(field));
    // Paimon Java puts nulls first in either order.
    columns.compareFlags.push_back(
        {.nullsFirst = true, .ascending = options.sequenceFieldsAscending});
  }
  names.push_back(
      systemColumnName(kSequenceNumberColumn, lowerCaseSystemColumns));
  types.push_back(BIGINT());
  columns.compareFlags.push_back({.nullsFirst = true, .ascending = true});
  names.push_back(systemColumnName(kValueKindColumn, lowerCaseSystemColumns));
  types.push_back(TINYINT());
  columns.type = ROW(std::move(names), std::move(types));
  return columns;
}

// Table handle of the key reads, which read the merge columns of a file with
// no filters. Columns missing from the file read as nulls.
//
// TODO: Pass the subfield filters and the remaining filter conjuncts on primary
// key columns to the key reads, so they skip rows and files as Paimon Java's
// merge reads do when files overlap. Filters on other columns must not apply
// before the merge, since dropping the latest row of a key would return an
// older row instead.
class KeyTableHandle final : public connector::FileTableHandle {
 public:
  KeyTableHandle(
      const connector::FileTableHandle& tableHandle,
      RowTypePtr keyType)
      : connector::FileTableHandle(
            tableHandle.connectorId(),
            tableHandle.tableName(),
            /*subfieldFilters=*/{},
            /*remainingFilter=*/nullptr,
            std::move(keyType),
            tableHandle.tableParameters(),
            /*sampleRate=*/1.0,
            tableHandle.dbName()) {}

  std::vector<connector::FileColumnHandlePtr> filterColumnHandles()
      const override {
    return {};
  }
};

// Returns true if a row of 'vector' is null.
bool hasNulls(const BaseVector& vector) {
  DecodedVector decoded(vector);
  const auto* nulls = decoded.nulls();
  return nulls != nullptr &&
      !bits::isAllSet(nulls, 0, vector.size(), bits::kNotNull);
}

// Checks that the key, _SEQUENCE_NUMBER and _VALUE_KIND columns of a key read
// have no nulls. Columns missing from the file read as nulls. Paimon primary
// keys are not null, and null keys would merge every row into one key.
void checkMergeColumnsNotNull(
    const RowVector& batch,
    const MergeColumns& mergeColumns,
    std::string_view path) {
  for (column_index_t i = 0; i < mergeColumns.numKeys; ++i) {
    VELOX_USER_CHECK(
        !hasNulls(*batch.childAt(i)),
        "Paimon primary-key file must have a non-null {} column: {}",
        mergeColumns.type->nameOf(i),
        path);
  }
  const auto numColumns = mergeColumns.type->size();
  VELOX_USER_CHECK(
      !hasNulls(*batch.childAt(numColumns - 2)),
      "Paimon primary-key file must have a non-null _SEQUENCE_NUMBER "
      "column: {}",
      path);
  VELOX_USER_CHECK(
      !hasNulls(*batch.childAt(numColumns - 1)),
      "Paimon primary-key file must have a non-null _VALUE_KIND column: {}",
      path);
}

// Rows of one data file in merge order, since Paimon sorts a file by key and
// the rows of a key by sequence. Skips the rows deleted by the deletion vector
// of the file.
class KeyStream final : public MergeStream {
 public:
  // 'deletionVectorReader' is null if the file has no deletion vector.
  KeyStream(
      size_t fileIndex,
      const PaimonDataFile& dataFile,
      std::unique_ptr<connector::FileSplitReader> reader,
      std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader,
      const MergeColumns& mergeColumns,
      uint64_t batchSize,
      memory::MemoryPool* pool)
      : fileIndex_(fileIndex),
        dataFile_(dataFile),
        reader_(std::move(reader)),
        mergeColumns_(mergeColumns),
        batchSize_(batchSize),
        pool_(pool),
        deletionVectorReader_(std::move(deletionVectorReader)),
        deletedRows_(
            deletionVectorReader_ != nullptr
                ? AlignedBuffer::allocate<uint64_t>(
                      bits::nwords(batchSize),
                      pool)
                : nullptr) {
    // An empty split, such as a missing file the session ignores, has no row
    // reader and adds no rows.
    atEnd_ = reader_->emptySplit();
    if (!atEnd_) {
      skipDeletedRows();
    }
  }

  bool hasData() const override {
    return !atEnd_;
  }

  int32_t compare(const MergeStream& other) const override {
    const auto& otherStream = static_cast<const KeyStream&>(other);
    for (column_index_t i = 0; i < mergeColumns_.compareFlags.size(); ++i) {
      const auto result = orderColumns_[i]
                              ->compare(
                                  otherStream.orderColumns_[i],
                                  row_,
                                  otherStream.row_,
                                  mergeColumns_.compareFlags[i])
                              .value();
      if (result != 0) {
        return result;
      }
    }
    // Rows of a key tie on sequence number when two writers of a bucket
    // restore the same snapshot, since each numbers its rows from the largest
    // sequence number in the bucket. Paimon Java leaves their order
    // unspecified, and the file order keeps the result deterministic. Each
    // file has one stream, and the merge never compares a stream with itself.
    VELOX_DCHECK_NE(fileIndex_, otherStream.fileIndex_);
    return fileIndex_ < otherStream.fileIndex_ ? -1 : 1;
  }

  // Moves to the next row that is not deleted.
  void pop() {
    ++row_;
    skipDeletedRows();
  }

  size_t fileIndex() const {
    return fileIndex_;
  }

  // Position of the current row in the file.
  uint64_t fileRow() const {
    return batchStartRow_ + row_;
  }

  const RowVectorPtr& batch() const {
    return batch_;
  }

  // Index of the current row in batch().
  vector_size_t row() const {
    return row_;
  }

  // Returns true if the current row is a DELETE or UPDATE_BEFORE.
  bool isRetract() const {
    const auto kind = paimonRowKindFromValue(rowKinds_->valueAt(row_));
    return kind == PaimonRowKind::kDelete ||
        kind == PaimonRowKind::kUpdateBefore;
  }

 private:
  // Moves from row_ past deleted rows, reading batches as needed. Sets atEnd_
  // past the last row.
  void skipDeletedRows() {
    for (;;) {
      if (row_ == numBatchRows_ && !loadBatch()) {
        atEnd_ = true;
        return;
      }
      if (!hasDeletedRows_ ||
          !bits::isBitSet(deletedRows_->as<uint64_t>(), row_)) {
        return;
      }
      ++row_;
    }
  }

  // Reads the next batch. Returns false at the end of the file.
  bool loadBatch() {
    batchStartRow_ += numBatchRows_;
    row_ = 0;
    // Reuses the previous batch unless the current key still refers to it.
    VectorPtr output = std::move(batch_);
    if (output == nullptr || output.use_count() > 1) {
      output = BaseVector::create(mergeColumns_.type, 0, pool_);
    }
    numBatchRows_ =
        static_cast<vector_size_t>(reader_->next(batchSize_, output));
    // The row count in the metadata of the file sizes its row selection, so the
    // file must have exactly that many rows. Checking each batch fails before a
    // row past the count can be selected.
    if (numBatchRows_ == 0) {
      VELOX_USER_CHECK_EQ(
          batchStartRow_,
          dataFile_.rowCount,
          "Paimon data file row count must match its metadata: {}",
          dataFile_.path);
      return false;
    }
    VELOX_USER_CHECK_LE(
        batchStartRow_ + numBatchRows_,
        dataFile_.rowCount,
        "Paimon data file row count must match its metadata: {}",
        dataFile_.path);
    batch_ = std::static_pointer_cast<RowVector>(output);
    VELOX_CHECK_EQ(batch_->size(), numBatchRows_);
    for (auto& child : batch_->children()) {
      child = BaseVector::loadedVectorShared(child);
    }
    checkMergeColumnsNotNull(*batch_, mergeColumns_, dataFile_.path);
    rowKinds_ = batch_->childAt(mergeColumns_.type->size() - 1)
                    ->asChecked<SimpleVector<int8_t>>();
    orderColumns_.clear();
    for (column_index_t i = 0; i < mergeColumns_.compareFlags.size(); ++i) {
      orderColumns_.push_back(batch_->childAt(i).get());
    }

    hasDeletedRows_ = deletionVectorReader_ != nullptr &&
        !deletionVectorReader_->noMoreData();
    if (hasDeletedRows_) {
      std::memset(
          deletedRows_->asMutable<uint64_t>(),
          0,
          bits::nwords(numBatchRows_) * sizeof(uint64_t));
      deletionVectorReader_->readDeletePositions(
          batchStartRow_, numBatchRows_, deletedRows_);
    }
    return true;
  }

  // Position of the file in the split, which orders rows that tie.
  const size_t fileIndex_;
  // File the stream reads, named in error messages.
  const PaimonDataFile& dataFile_;
  // Reads the merge columns of the file.
  const std::unique_ptr<connector::FileSplitReader> reader_;
  // Merge columns, shared by the streams of a split.
  const MergeColumns& mergeColumns_;
  // Maximum number of rows a read returns.
  const uint64_t batchSize_;
  // Pool of the batches the reads return.
  memory::MemoryPool* const pool_;
  // Reads the deletion vector of the file, or null if the file has none.
  const std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader_;
  // Bitmap of the rows of batch_ the deletion vector deletes, valid if
  // hasDeletedRows_.
  const BufferPtr deletedRows_;

  // True if the deletion vector may delete rows of batch_.
  bool hasDeletedRows_{false};

  // Rows of the last read, or null past the end of the file.
  RowVectorPtr batch_;
  // File position of the first row of batch_.
  uint64_t batchStartRow_{0};
  // Number of rows in batch_.
  vector_size_t numBatchRows_{0};
  // Index of the current row in batch_.
  vector_size_t row_{0};
  // Columns of batch_ that order rows, and its row kinds.
  std::vector<const BaseVector*> orderColumns_;
  const SimpleVector<int8_t>* rowKinds_{nullptr};
  // True once the file has no row left that is not deleted.
  bool atEnd_{false};
};

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

  void add(const KeyStream& stream) {
    ++numRows_;
    const bool retract = stream.isRetract();
    if (lookup_) {
      const auto level = dataFiles_[stream.fileIndex()].level;
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
      paimonTableHandle_(tableHandle),
      lowerCaseSystemColumns_(
          fileConfig->isFileColumnNamesReadAsLowerCase(
              connectorQueryCtx->sessionProperties()) &&
          paimonSplit_->dataFiles().front().fileFormat !=
              dwio::common::FileFormat::NIMBLE) {}

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
  VELOX_CHECK_GT(batchSize, 0);
  const auto& dataFiles = paimonSplit_->dataFiles();
  const auto mergeColumns =
      makeMergeColumns(*paimonTableHandle_, lowerCaseSystemColumns_);
  const auto keyTableHandle =
      std::make_shared<KeyTableHandle>(*paimonTableHandle_, mergeColumns.type);
  // Key reads have no filters, so they skip no file.
  dwio::common::RuntimeStats keyRuntimeStats;

  std::vector<std::unique_ptr<KeyStream>> streams;
  std::vector<BufferPtr> rowSelections;
  streams.reserve(dataFiles.size());
  rowSelections.reserve(dataFiles.size());
  for (size_t fileIndex = 0; fileIndex < dataFiles.size(); ++fileIndex) {
    const auto& dataFile = dataFiles[fileIndex];
    // Each reader adapts its ScanSpec to its file, so the files read at once
    // cannot share one.
    //
    // TODO: Pass the filters on primary key columns as 'subfieldFilters', as
    // the KeyTableHandle TODO describes. 'outputSubfields' stays empty, since
    // the merge compares whole keys and sequence fields.
    const auto scanSpec = connector::FileDataSource::makeScanSpec(
        mergeColumns.type,
        /*outputSubfields=*/{},
        /*subfieldFilters=*/{},
        mergeColumns.type,
        /*partitionKeys=*/{},
        /*infoColumns=*/{},
        /*specialColumns=*/{},
        /*disableStatsBasedFilterReorder=*/false,
        pool_);
    auto reader = connector::FileSplitReader::create(
        makeFileConnectorSplit(*paimonSplit_, dataFile),
        keyTableHandle,
        partitionKeys_,
        connectorQueryCtx_,
        fileConfig_,
        mergeColumns.type,
        dataIoStats_,
        metadataIoStats_,
        ioStats_,
        fileHandleFactory_,
        ioExecutor_,
        scanSpec);
    reader->initialize(
        /*randomSkip=*/nullptr, /*remainingFilterColumns=*/{});
    reader->prepareSplit(/*metadataFilter=*/nullptr, keyRuntimeStats);
    auto deletionVectorReader = dataFile.deletionFile.has_value()
        ? std::make_unique<PaimonDeletionVectorReader>(
              *dataFile.deletionFile,
              pool_,
              fileConfig_->config(),
              connectorQueryCtx_->fsTokenProvider())
        : nullptr;
    streams.emplace_back(
        std::make_unique<KeyStream>(
            fileIndex,
            dataFile,
            std::move(reader),
            std::move(deletionVectorReader),
            mergeColumns,
            batchSize,
            pool_));
    // A bit per row the metadata of the file counts, which the key stream of
    // the file checks.
    rowSelections.emplace_back(
        AlignedBuffer::allocate<uint64_t>(
            std::max<uint64_t>(bits::nwords(dataFile.rowCount), 1), pool_, 0));
  }

  TreeOfLosers<KeyStream> merge(std::move(streams));
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
        !sameKey(
            *keyBatch,
            keyRow,
            *stream->batch(),
            stream->row(),
            mergeColumns.numKeys)) {
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
