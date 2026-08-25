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

#include "velox/dwio/parquet/reader/StructColumnReader.h"

#include <algorithm>
#include <optional>
#include <tuple>

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetData.h"
#include "velox/dwio/parquet/reader/RepeatedColumnReader.h"

namespace facebook::velox::common {
class ScanSpec;
}

namespace facebook::velox::parquet {
namespace {

struct PhysicalLeafCost {
  const ParquetTypeWithId* type;
  bool hasCompleteMetadata{true};
  uint64_t readBytes{0};
  uint64_t uncompressedBytes{0};
  uint64_t numValues{0};

  bool operator<(const PhysicalLeafCost& other) const {
    if (hasCompleteMetadata != other.hasCompleteMetadata) {
      // Prefer a leaf with metadata for every non-empty row group. Partial
      // metadata can underestimate its cost and may make the leaf unreadable.
      return hasCompleteMetadata;
    }
    if (!hasCompleteMetadata) {
      // Neither cost is reliable. Use schema order as a deterministic fallback.
      return type->column() < other.type->column();
    }
    // Minimize bytes fetched first, then bytes decompressed and levels decoded.
    // Use schema order to make costs deterministic.
    return std::tuple{readBytes, uncompressedBytes, numValues, type->column()} <
        std::tuple{
            other.readBytes,
            other.uncompressedBytes,
            other.numValues,
            other.type->column()};
  }
};

PhysicalLeafCost physicalLeafCost(
    const ParquetTypeWithId& type,
    const FileMetaDataPtr& fileMetaData) {
  PhysicalLeafCost cost{.type = &type};
  // The source reader is fixed before row-group filtering. Use all non-empty
  // row groups as a whole-file cost estimate; an individual split may have a
  // different cheapest leaf for its subset of row groups.
  for (auto i = 0; i < fileMetaData.numRowGroups(); ++i) {
    auto rowGroup = fileMetaData.rowGroup(i);
    if (rowGroup.numRows() == 0) {
      continue;
    }
    if (type.column() >= static_cast<uint32_t>(rowGroup.numColumns())) {
      cost.hasCompleteMetadata = false;
      return cost;
    }
    auto chunk = rowGroup.columnChunk(type.column());
    if (!chunk.hasMetadata()) {
      cost.hasCompleteMetadata = false;
      return cost;
    }
    const auto readSize = chunk.readSize();
    if (readSize == 0) {
      // Size metadata is optional. A non-empty row group cannot have a
      // zero-byte column chunk, so zero means the read cost is unknown.
      cost.hasCompleteMetadata = false;
      return cost;
    }
    cost.readBytes += readSize;
    cost.uncompressedBytes +=
        static_cast<uint64_t>(chunk.totalUncompressedSize());
    cost.numValues += static_cast<uint64_t>(chunk.numValues());
  }
  return cost;
}

std::optional<PhysicalLeafCost> findCheapestPhysicalLeafImpl(
    const ParquetTypeWithId& type,
    const FileMetaDataPtr& fileMetaData) {
  if (type.getChildren().empty()) {
    if (type.isLeaf()) {
      return physicalLeafCost(type, fileMetaData);
    }
    return std::nullopt;
  }

  std::optional<PhysicalLeafCost> best;
  for (auto i = 0; i < type.getChildren().size(); ++i) {
    auto candidate =
        findCheapestPhysicalLeafImpl(type.parquetChildAt(i), fileMetaData);
    if (candidate && (!best || *candidate < *best)) {
      best = std::move(candidate);
    }
  }
  return best;
}

const ParquetTypeWithId& findCheapestPhysicalLeaf(
    const ParquetTypeWithId& type,
    const FileMetaDataPtr& fileMetaData) {
  auto best = findCheapestPhysicalLeafImpl(type, fileMetaData);
  VELOX_CHECK(
      best.has_value(),
      "Cannot source repetition/definition levels for nested struct: {}",
      type.fullName());
  return *best->type;
}

LevelMode repDefSourceLevelMode(
    const ParquetTypeWithId& structType,
    const ParquetTypeWithId& sourceType) {
  const auto* node = &sourceType;
  while (node != &structType) {
    if (node->type()->kind() == TypeKind::ARRAY ||
        node->type()->kind() == TypeKind::MAP) {
      return LevelMode::kStructOverLists;
    }
    node = node->parquetParent();
  }
  return LevelMode::kNulls;
}

const ParquetTypeWithId& repDefSourceType(
    const dwio::common::SelectiveColumnReader& reader) {
  const auto* source = &reader;
  while (source->fileType().type()->kind() == TypeKind::ROW) {
    source =
        static_cast<const StructColumnReader*>(source)->repDefSourceReader();
  }
  return *reinterpret_cast<const ParquetTypeWithId*>(&source->fileType());
}

} // namespace

struct StructColumnReader::SyntheticRepDefSource {
  // Members are destroyed in reverse declaration order. Keep the ScanSpec
  // alive until after the reader that references it is destroyed.
  std::unique_ptr<common::ScanSpec> scanSpec;

  // Reads repetition and definition levels without producing values.
  std::unique_ptr<dwio::common::SelectiveColumnReader> reader;
};

StructColumnReader::~StructColumnReader() = default;

StructColumnReader::StructColumnReader(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    ParquetParams& params,
    common::ScanSpec& scanSpec)
    : SelectiveStructColumnReader(
          columnReaderOptions,
          requestedType,
          fileType,
          params,
          scanSpec) {
  auto& childSpecs = scanSpec_->stableChildren();
  for (auto i = 0; i < childSpecs.size(); ++i) {
    auto childSpec = childSpecs[i];
    if (childSpec->isConstant() || isChildMissing(*childSpec)) {
      childSpec->setSubscript(kConstantChildSpecSubscript);
      continue;
    }
    if (!childSpecs[i]->readFromFile()) {
      continue;
    }
    auto childFileType = fileType_->childByName(childSpec->fieldName());
    auto childRequestedType =
        requestedType_->asRow().findChild(childSpec->fieldName());
    addChild(
        ParquetColumnReader::build(
            columnReaderOptions,
            childRequestedType,
            childFileType,
            params,
            *childSpec));

    childSpecs[i]->setSubscript(children_.size() - 1);
  }
  applyMissingFieldPolicy(
      columnReaderOptions, params.nullStructIfAllFieldsMissing());
  ensureSyntheticRepDefSource(columnReaderOptions, params);
  auto type = reinterpret_cast<const ParquetTypeWithId*>(fileType_.get());
  if (type->parent()) {
    type->makeLevelInfo(levelInfo_);
    repDefSourceReader_ = findBestLeaf();
    levelMode_ =
        repDefSourceLevelMode(*type, repDefSourceType(*repDefSourceReader_));
  }
  VELOX_DCHECK_EQ(type->parent() == nullptr, repDefSourceReader_ == nullptr);
}

void StructColumnReader::applyMissingFieldPolicy(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    bool nullStructIfAllFieldsMissing) {
  if (!nullStructIfAllFieldsMissing) {
    return;
  }

  auto& childSpecs = scanSpec_->stableChildren();
  if (childSpecs.empty()) {
    scanSpec_->setConstantValue(
        BaseVector::createNullConstant(requestedType_, 1, pool_));
    return;
  }

  const bool useColumnNames = columnReaderOptions.columnMappingMode_ ==
      dwio::common::ColumnMappingMode::kName;
  if (useColumnNames &&
      std::all_of(childSpecs.begin(), childSpecs.end(), [&](auto* childSpec) {
        return childSpec->columnType() ==
            common::ScanSpec::ColumnType::kRegular &&
            isChildMissing(*childSpec);
      })) {
    scanSpec_->setConstantValue(
        BaseVector::createNullConstant(requestedType_, 1, pool_));
    return;
  }

  scanSpec_->setConstantValue(nullptr);
}

void StructColumnReader::ensureSyntheticRepDefSource(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    ParquetParams& params) {
  auto type = reinterpret_cast<const ParquetTypeWithId*>(fileType_.get());
  if (!type->parent() || !children_.empty()) {
    return;
  }

  const auto* leafType =
      &findCheapestPhysicalLeaf(*type, params.fileMetaData());
  std::shared_ptr<const dwio::common::TypeWithId> leafFileType(
      fileType_, leafType);

  // Struct nullness can be derived from any leaf under the struct. Keep the
  // least expensive physical leaf solely as the repetition/definition source.
  auto repDefSource = std::make_unique<SyntheticRepDefSource>();
  repDefSource->scanSpec = std::make_unique<common::ScanSpec>(leafType->name_);
  repDefSource->scanSpec->setProjectOut(false);
  repDefSource->reader = ParquetColumnReader::build(
      columnReaderOptions,
      leafFileType->type(),
      leafFileType,
      params,
      *repDefSource->scanSpec);
  syntheticRepDefSource_ = std::move(repDefSource);
}

dwio::common::SelectiveColumnReader* FOLLY_NONNULL
StructColumnReader::findBestLeaf() {
  if (children_.empty()) {
    return syntheticRepDefSource_->reader.get();
  }

  SelectiveColumnReader* best = nullptr;
  for (auto i = 0; i < children_.size(); ++i) {
    auto child = children_[i];
    auto kind = child->fileType().type()->kind();
    // Complex type child repdefs must be read in any case.
    if (kind == TypeKind::ROW || kind == TypeKind::ARRAY) {
      return child;
    }
    if (!best) {
      best = child;
    } else if (best->scanSpec()->filter() && !child->scanSpec()->filter()) {
      continue;
    } else if (!best->scanSpec()->filter() && child->scanSpec()->filter()) {
      best = child;
      continue;
    } else if (kind < best->fileType().type()->kind()) {
      best = child;
    }
  }
  return best;
}

void StructColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* /*incomingNulls*/) {
  prepareRepDefsAndOffset(*this, offset, rows);
  SelectiveStructColumnReader::read(offset, rows, nullptr);
}

std::shared_ptr<dwio::common::BufferedInput> StructColumnReader::loadRowGroup(
    uint32_t index,
    const std::shared_ptr<dwio::common::BufferedInput>& input) {
  if (isRowGroupBuffered(index, *input)) {
    enqueueRowGroup(index, *input);
    return input;
  }
  auto newInput = input->clone();
  enqueueRowGroup(index, *newInput);
  newInput->load(dwio::common::LogType::STRIPE);
  return newInput;
}

bool StructColumnReader::isRowGroupBuffered(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  auto [offset, length] =
      formatData().as<ParquetData>().getRowGroupRegion(index);
  return input.isBuffered(offset, length);
}

void StructColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  enqueueRowGroupRecursive(*this, index, input);
}

void StructColumnReader::seekToRowGroup(int64_t index) {
  SelectiveStructColumnReader::seekToRowGroup(index);
  BufferPtr noBuffer;
  formatData_->as<ParquetData>().setNulls(noBuffer, 0);
  readOffset_ = 0;
  for (auto& child : children_) {
    child->seekToRowGroup(index);
  }
  // Keep the rep/def source in sync when switching row groups.
  if (syntheticRepDefSource_) {
    syntheticRepDefSource_->reader->seekToRowGroup(index);
  }
}

void StructColumnReader::seekToEndOfPresetNulls() {
  auto numUnread = formatData_->as<ParquetData>().presetNullsLeft();
  for (auto i = 0; i < children_.size(); ++i) {
    auto child = children_[i];
    if (!child) {
      continue;
    }

    if (child->fileType().type()->kind() != TypeKind::ROW) {
      child->seekTo(readOffset_ + numUnread, false);
    } else if (child->fileType().type()->kind() == TypeKind::ROW) {
      reinterpret_cast<StructColumnReader*>(child)->seekToEndOfPresetNulls();
    }
  }
  readOffset_ += numUnread;
  formatData_->as<ParquetData>().skipNulls(numUnread, false);
}

void StructColumnReader::setNullsFromRepDefs(PageReader& pageReader) {
  if (levelInfo_.defLevel == 0) {
    return;
  }
  auto repDefRange = pageReader.repDefRange();
  int32_t numRepDefs = repDefRange.second - repDefRange.first;
  dwio::common::ensureCapacity<uint64_t>(
      nullsInReadRange_, bits::nwords(numRepDefs), pool_);
  auto numStructs = pageReader.getLengthsAndNulls(
      levelMode_,
      levelInfo_,
      repDefRange.first,
      repDefRange.second,
      numRepDefs,
      nullptr,
      nullsInReadRange()->asMutable<uint64_t>(),
      0);
  // Repeated parents still need rep/def levels to determine the number of
  // structs. Preserve that count, but mark every struct null when schema
  // evolution synthesized this field as a null constant.
  if (scanSpec_->isConstant() && scanSpec_->constantValue()->isNullAt(0)) {
    bits::fillBits(
        nullsInReadRange()->asMutable<uint64_t>(), 0, numStructs, bits::kNull);
  }
  formatData_->as<ParquetData>().setNulls(nullsInReadRange(), numStructs);
}

void StructColumnReader::filterRowGroups(
    uint64_t rowGroupSize,
    const dwio::common::StatsContext& context,
    dwio::common::FormatData::FilterRowGroupsResult& result) const {
  for (const auto& child : children_) {
    child->filterRowGroups(rowGroupSize, context, result);
  }
}

} // namespace facebook::velox::parquet
