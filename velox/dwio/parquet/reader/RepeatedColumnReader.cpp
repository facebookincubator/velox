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

#include "velox/dwio/parquet/reader/RepeatedColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/parquet/reader/StructColumnReader.h"

namespace facebook::velox::parquet {

class ParquetTypeWithId;

namespace {
PageReader* readLeafRepDefs(
    dwio::common::SelectiveColumnReader* reader,
    int32_t numTop,
    bool mustRead) {
  const auto& children = reader->children();
  const auto kind = reader->fileType().type()->kind();
  PageReader* pageReader = nullptr;
  if (kind == TypeKind::ROW) {
    auto* structReader = static_cast<StructColumnReader*>(reader);
    // A struct can have no logical children and use a separate physical leaf
    // solely as its rep/def source.
    auto* repDefSourceReader = structReader->repDefSourceReader();
    pageReader = readLeafRepDefs(repDefSourceReader, numTop, true);
    structReader->setNullsFromRepDefs(*pageReader);
    for (auto* child : children) {
      if (child != repDefSourceReader) {
        readLeafRepDefs(child, numTop, false);
      }
    }
    return pageReader;
  }
  if (children.empty()) {
    if (!mustRead) {
      return nullptr;
    }
    auto* leafPageReader = reader->formatData().as<ParquetData>().reader();
    leafPageReader->decodeRepDefs(numTop);
    return leafPageReader;
  }
  if (kind == TypeKind::ARRAY) {
    pageReader = readLeafRepDefs(children[0], numTop, true);
    auto* list = static_cast<ListColumnReader*>(reader);
    list->setLengthsFromRepDefs(*pageReader);
    return pageReader;
  }
  if (kind == TypeKind::MAP) {
    pageReader = readLeafRepDefs(children[0], numTop, true);
    readLeafRepDefs(children[1], numTop, false);
    auto* map = static_cast<MapColumnReader*>(reader);
    map->setLengthsFromRepDefs(*pageReader);
    return pageReader;
  }
  return pageReader;
}

void skipUnreadLengthsAndNulls(dwio::common::SelectiveColumnReader& reader) {
  // A struct can have no logical children but still hold preset nulls decoded
  // from its synthetic rep/def source. Advance these nulls before the empty-
  // children check below.
  const auto kind = reader.fileType().type()->kind();
  if (kind == TypeKind::ROW) {
    static_cast<StructColumnReader*>(&reader)->seekToEndOfPresetNulls();
    return;
  }
  const auto& children = reader.children();
  if (children.empty()) {
    return;
  }
  if (kind == TypeKind::ARRAY) {
    static_cast<ListColumnReader*>(&reader)->skipUnreadLengths();
  } else if (kind == TypeKind::MAP) {
    static_cast<MapColumnReader*>(&reader)->skipUnreadLengths();
  } else {
    VELOX_UNREACHABLE();
  }
}

} // namespace

void enqueueRowGroupRecursive(
    dwio::common::SelectiveColumnReader& reader,
    uint32_t index,
    dwio::common::BufferedInput& input) {
  const auto& children = reader.children();
  if (children.empty()) {
    if (reader.fileType().type()->kind() == TypeKind::ROW) {
      auto* structReader = static_cast<StructColumnReader*>(&reader);
      // Only the root struct has no rep/def source.
      if (auto* repDefSourceReader = structReader->repDefSourceReader()) {
        enqueueRowGroupRecursive(*repDefSourceReader, index, input);
      }
      return;
    }
    return reader.formatData().as<ParquetData>().enqueueRowGroup(index, input);
  }
  for (auto* child : children) {
    enqueueRowGroupRecursive(*child, index, input);
  }
}

void prepareRepDefsAndOffset(
    dwio::common::SelectiveColumnReader& reader,
    int64_t offset,
    const RowSet& rows) {
  const auto previousOffset = reader.readOffset();
  const auto* parent = reader.fileType().parent();
  // Only a direct child of the root struct owns the repdefs for its subtree.
  if (parent && !parent->parent()) {
    const int32_t numTop =
        static_cast<int32_t>(offset + rows.back() + 1 - previousOffset);
    skipUnreadLengthsAndNulls(reader);
    readLeafRepDefs(&reader, numTop, true);

    if (offset > previousOffset) {
      // There is no page reader on this level so cannot call skipNullsOnly on
      // it.
      reader.skip(offset - previousOffset);
    }
  }

  if (offset > previousOffset) {
    reader.setReadOffset(offset);
  }
}

MapColumnReader::MapColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    ParquetParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveMapColumnReader(
          requestedType,
          fileType,
          params,
          scanSpec) {
  DWIO_ENSURE_EQ(fileType_->id(), fileType->id(), "working on the same node");
  auto& keyChildType = requestedType->childAt(0);
  auto& elementChildType = requestedType->childAt(1);
  keyReader_ = ParquetColumnReader::build(
      columnReaderOptions,
      keyChildType,
      fileType_->childAt(0),
      params,
      *scanSpec.children()[0]);
  elementReader_ = ParquetColumnReader::build(
      columnReaderOptions,
      elementChildType,
      fileType_->childAt(1),
      params,
      *scanSpec.children()[1]);
  reinterpret_cast<const ParquetTypeWithId*>(fileType.get())
      ->makeLevelInfo(levelInfo_);
  children_ = {keyReader_.get(), elementReader_.get()};
}

void MapColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  enqueueRowGroupRecursive(*this, index, input);
}

void MapColumnReader::seekToRowGroup(int64_t index) {
  SelectiveMapColumnReader::seekToRowGroup(index);
  readOffset_ = 0;
  childTargetReadOffset_ = 0;
  BufferPtr noBuffer;
  formatData_->as<ParquetData>().setNulls(noBuffer, 0);
  lengths_.setLengths(nullptr);
}

void MapColumnReader::skipUnreadLengths() {
  auto& previousLengths = lengths_.lengths();
  if (previousLengths) {
    auto numPreviousLengths =
        (previousLengths->size() / sizeof(vector_size_t)) -
        lengths_.nextLengthIndex();
    if (numPreviousLengths) {
      skip(numPreviousLengths);
    }
  }
}

void MapColumnReader::setLengthsFromRepDefs(PageReader& pageReader) {
  auto repDefRange = pageReader.repDefRange();
  int32_t numRepDefs = repDefRange.second - repDefRange.first;
  BufferPtr lengths = std::move(lengths_.lengths());
  dwio::common::ensureCapacity<int32_t>(lengths, numRepDefs, pool_);
  memset(lengths->asMutable<uint64_t>(), 0, lengths->size());
  dwio::common::ensureCapacity<uint64_t>(
      nullsInReadRange_, bits::nwords(numRepDefs), pool_);
  auto numLists = pageReader.getLengthsAndNulls(
      LevelMode::kList,
      levelInfo_,
      repDefRange.first,
      repDefRange.second,
      numRepDefs,
      lengths->asMutable<int32_t>(),
      nullsInReadRange()->asMutable<uint64_t>(),
      0);
  lengths->setSize(numLists * sizeof(int32_t));
  formatData_->as<ParquetData>().setNulls(nullsInReadRange(), numLists);
  setLengths(std::move(lengths));
}

void MapColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  prepareRepDefsAndOffset(*this, offset, rows);
  SelectiveMapColumnReader::read(offset, rows, incomingNulls);

  // The child should be at the end of the range provided to this
  // read() so that it can receive new repdefs for the next set of top
  // level rows. The end of the range is not the end of unused lengths
  // because all lengths maty have been used but the last one might
  // have been 0.  If the last list was 0 and the previous one was not
  // in 'rows' we will be at the end of the last non-zero list in
  // 'rows', which is not the end of the lengths. ORC can seek to this
  // point on next read, Parquet needs to seek here because new
  // repdefs will be scanned and new lengths provided, overwriting the
  // previous ones before the next read().
  keyReader_->seekTo(childTargetReadOffset_, false);
  elementReader_->seekTo(childTargetReadOffset_, false);
}

void MapColumnReader::filterRowGroups(
    uint64_t rowGroupSize,
    const dwio::common::StatsContext& context,
    dwio::common::FormatData::FilterRowGroupsResult& result) const {
  // empty placeholder to avoid incorrect calling on parent's impl
}

ListColumnReader::ListColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    ParquetParams& params,
    common::ScanSpec& scanSpec)
    : dwio::common::SelectiveListColumnReader(
          requestedType,
          fileType,
          params,
          scanSpec) {
  auto& childType = requestedType->childAt(0);
  child_ = ParquetColumnReader::build(
      columnReaderOptions,
      childType,
      fileType_->childAt(0),
      params,
      *scanSpec.children()[0]);
  reinterpret_cast<const ParquetTypeWithId*>(fileType.get())
      ->makeLevelInfo(levelInfo_);
  children_ = {child_.get()};
}

void ListColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  enqueueRowGroupRecursive(*this, index, input);
}

void ListColumnReader::seekToRowGroup(int64_t index) {
  SelectiveListColumnReader::seekToRowGroup(index);
  readOffset_ = 0;
  childTargetReadOffset_ = 0;
  BufferPtr noBuffer;
  formatData_->as<ParquetData>().setNulls(noBuffer, 0);
  lengths_.setLengths(nullptr);
  child_->seekToRowGroup(index);
}

void ListColumnReader::skipUnreadLengths() {
  auto& previousLengths = lengths_.lengths();
  if (previousLengths) {
    auto numPreviousLengths =
        (previousLengths->size() / sizeof(vector_size_t)) -
        lengths_.nextLengthIndex();
    if (numPreviousLengths) {
      skip(numPreviousLengths);
    }
  }
}

void ListColumnReader::setLengthsFromRepDefs(PageReader& pageReader) {
  auto repDefRange = pageReader.repDefRange();
  int32_t numRepDefs = repDefRange.second - repDefRange.first;
  BufferPtr lengths = std::move(lengths_.lengths());
  dwio::common::ensureCapacity<int32_t>(lengths, numRepDefs + 1, pool_);
  memset(lengths->asMutable<uint64_t>(), 0, lengths->size());
  dwio::common::ensureCapacity<uint64_t>(
      nullsInReadRange_, bits::nwords(numRepDefs + 1), pool_);
  auto numLists = pageReader.getLengthsAndNulls(
      LevelMode::kList,
      levelInfo_,
      repDefRange.first,
      repDefRange.second,
      numRepDefs,
      lengths->asMutable<int32_t>(),
      nullsInReadRange()->asMutable<uint64_t>(),
      0);
  lengths->setSize(numLists * sizeof(int32_t));
  formatData_->as<ParquetData>().setNulls(nullsInReadRange(), numLists);
  setLengths(std::move(lengths));
}
void ListColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  prepareRepDefsAndOffset(*this, offset, rows);
  SelectiveListColumnReader::read(offset, rows, incomingNulls);

  // The child should be at the end of the range provided to this
  // read() so that it can receive new repdefs for the next set of top
  // level rows. The end of the range is not the end of unused lengths
  // because all lengths maty have been used but the last one might
  // have been 0.  If the last list was 0 and the previous one was not
  // in 'rows' we will be at the end of the last non-zero list in
  // 'rows', which is not the end of the lengths. ORC can seek to this
  // point on next read, Parquet needs to seek here because new
  // repdefs will be scanned and new lengths provided, overwriting the
  // previous ones before the next read().
  child_->seekTo(childTargetReadOffset_, false);
}

void ListColumnReader::filterRowGroups(
    uint64_t rowGroupSize,
    const dwio::common::StatsContext& context,
    dwio::common::FormatData::FilterRowGroupsResult& result) const {
  // empty placeholder to avoid incorrect calling on parent's impl
}

} // namespace facebook::velox::parquet
