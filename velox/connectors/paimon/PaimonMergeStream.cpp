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
#include "velox/connectors/paimon/PaimonMergeStream.h"

#include <folly/String.h>
#include <algorithm>
#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::paimon {
namespace {

// Returns 'name' in lowercase if 'lowerCase' is true.
std::string systemColumnName(std::string_view name, bool lowerCase) {
  std::string result{name};
  if (lowerCase) {
    folly::toLowerAscii(result);
  }
  return result;
}

// Returns true if a row of 'vector' is null.
bool hasNulls(const BaseVector& vector) {
  DecodedVector decoded(vector);
  const auto* nulls = decoded.nulls();
  return nulls != nullptr &&
      !bits::isAllSet(nulls, 0, vector.size(), bits::kNotNull);
}

// Checks that the key, _SEQUENCE_NUMBER and _VALUE_KIND columns of a merge
// read have no nulls. Columns missing from the file read as nulls. Paimon
// primary keys are not null, and null keys would merge every row into one key.
void checkMergeColumnsNotNull(
    const RowVector& batch,
    const PaimonMergeColumns& mergeColumns,
    std::string_view path) {
  for (column_index_t i = 0; i < mergeColumns.numKeys; ++i) {
    VELOX_USER_CHECK(
        !hasNulls(*batch.childAt(i)),
        "Paimon primary-key file must have a non-null {} column: {}",
        mergeColumns.type->nameOf(i),
        path);
  }
  VELOX_USER_CHECK(
      !hasNulls(*batch.childAt(mergeColumns.sequenceNumberChannel())),
      "Paimon primary-key file must have a non-null _SEQUENCE_NUMBER "
      "column: {}",
      path);
  VELOX_USER_CHECK(
      !hasNulls(*batch.childAt(mergeColumns.valueKindChannel())),
      "Paimon primary-key file must have a non-null _VALUE_KIND column: {}",
      path);
}

} // namespace

// static
PaimonMergeColumns PaimonMergeColumns::create(
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

  PaimonMergeColumns columns;
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

PaimonMergeStream::PaimonMergeStream(
    size_t fileIndex,
    const PaimonDataFile& dataFile,
    std::unique_ptr<connector::FileSplitReader> reader,
    std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader,
    std::shared_ptr<const PaimonMergeColumns> mergeColumns,
    uint64_t batchSize,
    memory::MemoryPool* pool)
    : fileIndex_(fileIndex),
      dataFile_(dataFile),
      reader_(std::move(reader)),
      mergeColumns_(std::move(mergeColumns)),
      batchSize_(batchSize),
      pool_(pool),
      deletionVectorReader_(std::move(deletionVectorReader)),
      deletedRows_(
          deletionVectorReader_ != nullptr
              ? AlignedBuffer::allocate<uint64_t>(bits::nwords(batchSize), pool)
              : nullptr) {
  // An empty split, such as a missing file the session ignores, has no row
  // reader and adds no rows.
  atEnd_ = reader_->emptySplit();
  if (!atEnd_) {
    skipDeletedRows();
  }
}

int32_t PaimonMergeStream::compare(const MergeStream& other) const {
  const auto& otherStream = static_cast<const PaimonMergeStream&>(other);
  const auto& compareFlags = mergeColumns_->compareFlags;
  for (column_index_t i = 0; i < compareFlags.size(); ++i) {
    const auto result = orderColumns_[i]
                            ->compare(
                                otherStream.orderColumns_[i],
                                row_,
                                otherStream.row_,
                                compareFlags[i])
                            .value();
    if (result != 0) {
      return result;
    }
  }
  // Rows of a key tie on sequence number when two writers of a bucket restore
  // the same snapshot, since each numbers its rows from the largest sequence
  // number in the bucket. Paimon Java leaves their order unspecified, and the
  // file order keeps the result deterministic. Each file has one stream, and
  // the merge never compares a stream with itself.
  VELOX_DCHECK_NE(fileIndex_, otherStream.fileIndex_);
  return fileIndex_ < otherStream.fileIndex_ ? -1 : 1;
}

bool PaimonMergeStream::isRetract() const {
  const auto kind = paimonRowKindFromValue(rowKinds_->valueAt(row_));
  return kind == PaimonRowKind::kDelete || kind == PaimonRowKind::kUpdateBefore;
}

void PaimonMergeStream::skipDeletedRows() {
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

bool PaimonMergeStream::loadBatch() {
  batchStartRow_ += numBatchRows_;
  row_ = 0;
  // Reuses the previous batch unless the current key still refers to it.
  VectorPtr output = std::move(batch_);
  if (output == nullptr || output.use_count() > 1) {
    output = BaseVector::create(mergeColumns_->type, 0, pool_);
  }
  numBatchRows_ = static_cast<vector_size_t>(reader_->next(batchSize_, output));
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
  checkMergeColumnsNotNull(*batch_, *mergeColumns_, dataFile_.path);
  rowKinds_ = batch_->childAt(mergeColumns_->valueKindChannel())
                  ->asChecked<SimpleVector<int8_t>>();
  orderColumns_.clear();
  for (column_index_t i = 0; i < mergeColumns_->compareFlags.size(); ++i) {
    orderColumns_.push_back(batch_->childAt(i).get());
  }

  hasDeletedRows_ =
      deletionVectorReader_ != nullptr && !deletionVectorReader_->noMoreData();
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

} // namespace facebook::velox::connector::paimon
