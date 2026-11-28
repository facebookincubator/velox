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
#include "velox/connectors/paimon/PaimonFieldMergeSplitReader.h"

#include <algorithm>
#include <limits>

#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/common/Reader.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::paimon {
namespace {

// Returns the columns of 'scanSpec' that the merge reads after the merge
// columns of 'tableHandle': all but the partition columns of 'split', which
// take their values from the split, and the primary keys and sequence fields,
// which are merge columns. Columns 'readerOutputType' does not have, such as
// columns only filters read, take the table type.
RowTypePtr makeValueColumns(
    const common::ScanSpec& scanSpec,
    const PaimonConnectorSplit& split,
    const PaimonTableHandle& tableHandle,
    const RowType& readerOutputType) {
  const auto& keys = tableHandle.nonPartitionPrimaryKeys();
  const auto& sequenceFields = tableHandle.mergeOptions().sequenceFields;
  const auto contains = [](const std::vector<std::string>& names,
                           const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
  };
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (const auto& childSpec : scanSpec.children()) {
    const auto& name = childSpec->fieldName();
    if (split.partitionKeys().contains(name)) {
      continue;
    }
    VELOX_USER_CHECK_EQ(
        childSpec->columnType(),
        common::ScanSpec::ColumnType::kRegular,
        "Paimon merge-on-read does not support the column: {}",
        name);
    if (contains(keys, name) || contains(sequenceFields, name)) {
      continue;
    }
    names.push_back(name);
    types.push_back(
        readerOutputType.containsChild(name)
            ? readerOutputType.findChild(name)
            : tableHandle.rowType()->findChild(name));
  }
  return ROW(std::move(names), std::move(types));
}

// Creates the merge function of 'options' for every column of 'layout' but
// _SEQUENCE_NUMBER and _VALUE_KIND, the primary keys and sequence fields
// included.
std::unique_ptr<PaimonMergeFunction> makeMergeFunction(
    const PaimonMergeLayout& layout,
    const PaimonTableHandle::MergeOptions& options) {
  std::vector<column_index_t> mergeChannels;
  for (column_index_t channel = 0; channel < layout.type->size(); ++channel) {
    if (channel != layout.sequenceNumberChannel() &&
        channel != layout.valueKindChannel()) {
      mergeChannels.push_back(channel);
    }
  }
  return PaimonMergeFunction::create(layout.type, mergeChannels, options);
}

} // namespace

PaimonFieldMergeSplitReader::PaimonFieldMergeSplitReader(
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
      valueColumns_(makeValueColumns(
          *scanSpec,
          *paimonSplit_,
          *paimonTableHandle_,
          *readerOutputType)),
      layout_(
          PaimonMergeLayout::create(
              *paimonTableHandle_,
              valueColumns_,
              /*lowerCaseSystemColumns=*/false)),
      mergeFunction_(
          makeMergeFunction(layout_, paimonTableHandle_->mergeOptions())) {}

bool PaimonFieldMergeSplitReader::open(uint64_t batchSize) {
  // The merged rows take partition values from the split, as raw reads do,
  // and every other column of the scan from the files.
  const auto& partitionValues = paimonSplit_->partitionKeys();
  for (const auto& childSpec : scanSpec_->children()) {
    const auto& name = childSpec->fieldName();
    if (auto it = partitionValues.find(name); it != partitionValues.end()) {
      setPartitionValue(childSpec.get(), name, it->second);
      // Merged rows apply no filter to constants.
      if (childSpec->filter() != nullptr) {
        uint64_t passed{1};
        childSpec->applyFilter(*childSpec->constantValue(), 1, &passed);
        if (passed == 0) {
          return false;
        }
      }
      continue;
    }
    // A raw split read before may have left the column missing from its file.
    childSpec->setConstantValue(nullptr);
  }
  scanSpec_->resetCachedValues(false);
  merge_ = std::make_unique<TreeOfLosers<PaimonMergeStream>>(
      openMergeStreams(*paimonTableHandle_, valueColumns_, batchSize));
  return true;
}

uint64_t PaimonFieldMergeSplitReader::next(uint64_t size, VectorPtr& output) {
  VELOX_CHECK_GT(size, 0);
  if (merge_ == nullptr && !atEnd_) {
    atEnd_ = !open(size);
  }
  if (atEnd_) {
    return 0;
  }

  const auto maxRows = static_cast<vector_size_t>(
      std::min<uint64_t>(size, std::numeric_limits<vector_size_t>::max()));
  auto merged = BaseVector::create<RowVector>(layout_.type, maxRows, pool_);
  numOutputRows_ = 0;
  uint64_t numFileRows{0};
  for (;;) {
    auto* stream = merge_->next();
    if (stream == nullptr) {
      finishKey(*merged);
      atEnd_ = true;
      break;
    }
    if (!keyRows_.empty() &&
        !stream->hasSameKey(*keyRows_.front().batch, keyRows_.front().row)) {
      finishKey(*merged);
      // The current row starts the first key of the next call.
      if (numOutputRows_ == maxRows) {
        break;
      }
    }
    keyRows_.push_back(
        {stream->batch(), stream->row(), stream->level(), stream->rowKind()});
    stream->pop();
    ++numFileRows;
  }
  flushCopies(*merged);
  if (numFileRows == 0) {
    return 0;
  }
  merged->resize(numOutputRows_);

  dwio::common::Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  output =
      dwio::common::RowReader::projectColumns(merged, *scanSpec_, &mutation);
  return numFileRows;
}

namespace {

// Writes _SEQUENCE_NUMBER and _VALUE_KIND of a merged row of kind 'kind' to row
// 'row' of 'output'. The merged row takes the sequence number of row
// 'sourceRow' of 'source', the last row it merges.
void writeSystemColumns(
    const PaimonMergeLayout& layout,
    const RowVector& source,
    vector_size_t sourceRow,
    PaimonRowKind kind,
    vector_size_t row,
    RowVector& output) {
  const auto sequenceNumberChannel = layout.sequenceNumberChannel();
  output.childAt(sequenceNumberChannel)
      ->copy(source.childAt(sequenceNumberChannel).get(), row, sourceRow, 1);
  output.childAt(layout.valueKindChannel())
      ->asChecked<FlatVector<int8_t>>()
      ->set(row, static_cast<int8_t>(kind));
}

} // namespace

void PaimonFieldMergeSplitReader::finishKey(RowVector& output) {
  if (keyRows_.empty()) {
    return;
  }
  // Paimon Java's ReducerMergeFunctionWrapper returns the row of a single-row
  // key unmerged, before any lookup.
  if (keyRows_.size() == 1) {
    const auto& keyRow = keyRows_.front();
    if (!isRetract(keyRow.kind)) {
      copyRow(keyRow.batch, keyRow.row, output);
    }
    keyRows_.clear();
    return;
  }

  // With lookup compaction, a row above level 0 already merges the rows of its
  // key in higher levels. The merge function then takes the rows of level 0
  // and only the row of the lowest level above 0, as Paimon Java's
  // LookupMergeFunction does, so that no row counts twice.
  const bool lookup = paimonTableHandle_->mergeOptions().needLookup;
  const auto* highLevelRow = lookup ? pickHighLevelRow() : nullptr;
  const KeyRow* lastRow{nullptr};
  mergeFunction_->reset();
  for (const auto& keyRow : keyRows_) {
    if (lookup && keyRow.level > 0 && &keyRow != highLevelRow) {
      continue;
    }
    mergeFunction_->add(*keyRow.batch, keyRow.row, keyRow.kind);
    lastRow = &keyRow;
  }
  VELOX_CHECK_NOT_NULL(lastRow);
  const auto kind = mergeFunction_->resultKind();
  if (!isRetract(kind)) {
    mergeFunction_->write(numOutputRows_, output);
    writeSystemColumns(
        layout_, *lastRow->batch, lastRow->row, kind, numOutputRows_, output);
    ++numOutputRows_;
  }
  keyRows_.clear();
}

const PaimonFieldMergeSplitReader::KeyRow*
PaimonFieldMergeSplitReader::pickHighLevelRow() const {
  const KeyRow* highLevelRow{nullptr};
  for (const auto& keyRow : keyRows_) {
    if (keyRow.level > 0 &&
        (highLevelRow == nullptr || keyRow.level < highLevelRow->level)) {
      highLevelRow = &keyRow;
    }
  }
  return highLevelRow;
}

void PaimonFieldMergeSplitReader::copyRow(
    const RowVectorPtr& batch,
    vector_size_t row,
    RowVector& output) {
  if (batch != copySource_) {
    flushCopies(output);
    copySource_ = batch;
  }
  if (!copyRanges_.empty()) {
    auto& range = copyRanges_.back();
    // finishKey() writes merged rows to 'output' directly, so the last range
    // can end before row numOutputRows_ even when the source rows continue it.
    if (range.sourceIndex + range.count == row &&
        range.targetIndex + range.count == numOutputRows_) {
      ++range.count;
      ++numOutputRows_;
      return;
    }
  }
  copyRanges_.push_back({row, numOutputRows_, 1});
  ++numOutputRows_;
}

void PaimonFieldMergeSplitReader::flushCopies(RowVector& output) {
  if (!copyRanges_.empty()) {
    output.copyRanges(copySource_.get(), copyRanges_);
    copyRanges_.clear();
  }
  copySource_ = nullptr;
}

} // namespace facebook::velox::connector::paimon
