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
#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/TreeOfLosers.h"
#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::paimon {
namespace {

// Columns the key reads return: the trimmed primary keys, the sequence fields,
// _SEQUENCE_NUMBER and _VALUE_KIND. All but _VALUE_KIND order the rows of a
// split, since Paimon Java orders the rows of a key by sequence fields and
// then by sequence number.
struct MergeColumns {
  RowTypePtr type;

  // Number of leading key columns, which group the rows of a key.
  column_index_t numKeys{0};

  // Compare flags of the leading columns that order rows.
  std::vector<CompareFlags> compareFlags;
};

MergeColumns makeMergeColumns(const PaimonTableHandle& tableHandle) {
  const auto& rowType = tableHandle.rowType();
  const auto& keys = tableHandle.trimmedPrimaryKeys();
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
  names.emplace_back(kSequenceNumberColumn);
  types.push_back(BIGINT());
  columns.compareFlags.push_back({.nullsFirst = true, .ascending = true});
  names.emplace_back(kValueKindColumn);
  types.push_back(TINYINT());
  columns.type = ROW(std::move(names), std::move(types));
  return columns;
}

// Table handle of the key reads, which read the merge columns of a file with
// no filters. Columns missing from the file read as nulls.
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

bool hasNulls(const BaseVector& vector) {
  if (!vector.mayHaveNulls()) {
    return false;
  }
  for (vector_size_t row = 0; row < vector.size(); ++row) {
    if (vector.isNullAt(row)) {
      return true;
    }
  }
  return false;
}

// Rows of one data file in merge order, since Paimon sorts a file by key and
// the rows of a key by sequence. Skips the rows deleted by the deletion vector
// of the file.
class KeyStream final : public MergeStream {
 public:
  KeyStream(
      size_t fileIndex,
      const PaimonDataFile& dataFile,
      std::unique_ptr<connector::FileSplitReader> reader,
      const MergeColumns& columns,
      uint64_t batchSize,
      memory::MemoryPool* pool)
      : fileIndex_(fileIndex),
        dataFile_(dataFile),
        reader_(std::move(reader)),
        columns_(columns),
        batchSize_(batchSize),
        pool_(pool),
        deletionVectorReader_(
            dataFile.deletionFile.has_value()
                ? std::make_unique<PaimonDeletionVectorReader>(
                      *dataFile.deletionFile)
                : nullptr),
        deletedRows_(
            deletionVectorReader_ != nullptr
                ? AlignedBuffer::allocate<uint64_t>(
                      bits::nwords(batchSize),
                      pool)
                : nullptr) {
    // An empty split has no row reader.
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
    for (column_index_t i = 0; i < columns_.compareFlags.size(); ++i) {
      const auto result = orderColumns_[i]
                              ->compare(
                                  otherStream.orderColumns_[i],
                                  row_,
                                  otherStream.row_,
                                  columns_.compareFlags[i])
                              .value();
      if (result != 0) {
        return result;
      }
    }
    // Sequence numbers are unique within a bucket, so only a file listed twice
    // ties here. The file order keeps the result deterministic.
    if (fileIndex_ != otherStream.fileIndex_) {
      return fileIndex_ < otherStream.fileIndex_ ? -1 : 1;
    }
    return 0;
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

  // Number of rows in the file, known once the stream is at end.
  uint64_t numFileRows() const {
    VELOX_CHECK(atEnd_);
    return batchStartRow_;
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
    // Reads into a new vector, since the current key may still refer to the
    // previous batch.
    VectorPtr output = BaseVector::create(columns_.type, 0, pool_);
    numBatchRows_ =
        static_cast<vector_size_t>(reader_->next(batchSize_, output));
    if (numBatchRows_ == 0) {
      batch_ = nullptr;
      return false;
    }
    batch_ = std::static_pointer_cast<RowVector>(output);
    VELOX_CHECK_EQ(batch_->size(), numBatchRows_);
    for (auto& child : batch_->children()) {
      child = BaseVector::loadedVectorShared(child);
    }

    // Columns missing from the file read as nulls.
    const auto numColumns = columns_.type->size();
    VELOX_USER_CHECK(
        !hasNulls(*batch_->childAt(numColumns - 2)),
        "Paimon primary-key file must have a non-null _SEQUENCE_NUMBER "
        "column: {}",
        dataFile_.path);
    const auto& valueKinds = batch_->childAt(numColumns - 1);
    VELOX_USER_CHECK(
        !hasNulls(*valueKinds),
        "Paimon primary-key file must have a non-null _VALUE_KIND column: {}",
        dataFile_.path);
    rowKinds_ = valueKinds->asChecked<SimpleVector<int8_t>>();
    orderColumns_.clear();
    for (column_index_t i = 0; i < columns_.compareFlags.size(); ++i) {
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

  const size_t fileIndex_;
  const PaimonDataFile& dataFile_;
  const std::unique_ptr<connector::FileSplitReader> reader_;
  const MergeColumns& columns_;
  const uint64_t batchSize_;
  memory::MemoryPool* const pool_;
  const std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader_;
  // Bitmap of the rows of batch_ the deletion vector deletes, valid if
  // hasDeletedRows_.
  const BufferPtr deletedRows_;

  bool hasDeletedRows_{false};

  RowVectorPtr batch_;
  // File position of the first row of batch_.
  uint64_t batchStartRow_{0};
  vector_size_t numBatchRows_{0};
  vector_size_t row_{0};
  // Columns of batch_ that order rows, and its row kinds.
  std::vector<const BaseVector*> orderColumns_;
  const SimpleVector<int8_t>* rowKinds_{nullptr};
  bool atEnd_{false};
};

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
// key without merging, and drops a picked DELETE or UPDATE_BEFORE row.
class MergeFunction {
 public:
  MergeFunction(
      const PaimonTableHandle::MergeOptions& options,
      const std::vector<PaimonDataFile>& dataFiles)
      : firstRow_(options.mergeEngine == PaimonMergeEngine::kFirstRow),
        ignoreDelete_(options.ignoreDelete),
        dataFiles_(dataFiles) {}

  void add(const KeyStream& stream) {
    ++numRows_;
    const bool retract = stream.isRetract();
    if (retract && ignoreDelete_) {
      return;
    }
    if (!firstRow_) {
      picked_ = FileRow{stream.fileIndex(), stream.fileRow()};
      pickedRetract_ = retract;
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
    std::optional<FileRow> result;
    if (!pickedRetract_) {
      result = picked_;
    }
    numRows_ = 0;
    picked_.reset();
    pickedRetract_ = false;
    retractFileIndex_.reset();
    return result;
  }

 private:
  const bool firstRow_;
  const bool ignoreDelete_;
  const std::vector<PaimonDataFile>& dataFiles_;
  uint64_t numRows_{0};
  std::optional<FileRow> picked_;
  bool pickedRetract_{false};
  // File of a first-row retract, which Paimon Java rejects when it merges.
  std::optional<size_t> retractFileIndex_;
};

// Sets the bit of 'row', growing 'bitmap' geometrically with cleared words.
void setKeptRow(BufferPtr& bitmap, uint64_t row) {
  const auto numWords = bitmap->size() / sizeof(uint64_t);
  if (row >= numWords * 64) {
    AlignedBuffer::reallocate<uint64_t>(
        &bitmap, std::max(bits::nwords(row + 1), 2 * numWords), 0);
  }
  bits::setBit(bitmap->asMutable<uint64_t>(), row);
}

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
  if (keptRows_.empty()) {
    mergeKeys(size);
  }
  return PaimonSplitReader::next(size, output);
}

const uint64_t* PaimonMergeSplitReader::keptRows() const {
  return keptRows_[currentDataFileIndex()]->as<uint64_t>();
}

void PaimonMergeSplitReader::mergeKeys(uint64_t batchSize) {
  VELOX_CHECK_GT(batchSize, 0);
  const auto& dataFiles = paimonSplit_->dataFiles();
  const auto mergeColumns = makeMergeColumns(*paimonTableHandle_);
  const auto keyTableHandle =
      std::make_shared<KeyTableHandle>(*paimonTableHandle_, mergeColumns.type);
  // Key reads have no filters, so they skip no file.
  dwio::common::RuntimeStats keyRuntimeStats;

  std::vector<std::unique_ptr<KeyStream>> streams;
  std::vector<KeyStream*> fileStreams;
  std::vector<BufferPtr> keptRows;
  streams.reserve(dataFiles.size());
  fileStreams.reserve(dataFiles.size());
  keptRows.reserve(dataFiles.size());
  for (size_t fileIndex = 0; fileIndex < dataFiles.size(); ++fileIndex) {
    const auto& dataFile = dataFiles[fileIndex];
    // Each reader adapts its ScanSpec to its file, so the files read at once
    // cannot share one.
    const auto scanSpec = makeScanSpec(
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
    reader->configureReaderOptions(/*randomSkip=*/nullptr);
    reader->prepareSplit(/*metadataFilter=*/nullptr, keyRuntimeStats);
    streams.emplace_back(
        std::make_unique<KeyStream>(
            fileIndex,
            dataFile,
            std::move(reader),
            mergeColumns,
            batchSize,
            pool_));
    fileStreams.push_back(streams.back().get());
    keptRows.emplace_back(
        AlignedBuffer::allocate<uint64_t>(
            std::max<uint64_t>(bits::nwords(dataFile.rowCount), 1), pool_, 0));
  }

  TreeOfLosers<KeyStream> merge(std::move(streams));
  MergeFunction mergeFunction(paimonTableHandle_->mergeOptions(), dataFiles);
  const auto finishKey = [&]() {
    if (const auto picked = mergeFunction.finish()) {
      setKeptRow(keptRows[picked->fileIndex], picked->row);
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

  // The reads of a file test the bits of all its rows.
  for (size_t fileIndex = 0; fileIndex < fileStreams.size(); ++fileIndex) {
    auto& bitmap = keptRows[fileIndex];
    const auto numWords = bits::nwords(fileStreams[fileIndex]->numFileRows());
    if (bitmap->size() < numWords * sizeof(uint64_t)) {
      AlignedBuffer::reallocate<uint64_t>(&bitmap, numWords, 0);
    }
  }
  keptRows_ = std::move(keptRows);
}

} // namespace facebook::velox::connector::paimon
