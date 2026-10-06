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

#include "velox/dwio/nimble/index/ClusterIndexBase.h"

#include <algorithm>
#include <numeric>
#include <shared_mutex>

#include "folly/json/json.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble::index {

namespace {

const serialization::ClusterIndex* getIndexRoot(const Section& rootSection) {
  const auto* indexRoot = flatbuffers::GetRoot<serialization::ClusterIndex>(
      rootSection.content().data());
  NIMBLE_CHECK_NOT_NULL(indexRoot);
  return indexRoot;
}

uint32_t getIndexPartitionCount(const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* indexPartitions = root.index_partitions();
  NIMBLE_CHECK_NOT_NULL(indexPartitions);
  NIMBLE_CHECK_GT(indexPartitions->size(), 0, "ClusterIndex cannot be empty");
  return indexPartitions->size();
}

std::string_view getFirstKey(const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* keys = root.partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(
      keys->size(),
      1,
      "partition_keys must have at least 2 entries (firstKey + one lastKey per partition)");
  return velox::checkedNotNull(keys->Get(0))->string_view();
}

std::string_view getLastKey(const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* keys = root.partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(keys->size(), 1);
  return velox::checkedNotNull(keys->Get(keys->size() - 1))->string_view();
}

std::vector<std::string_view> getPartitionLastKeys(
    const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* keys = root.partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(keys->size(), 1);
  std::vector<std::string_view> result;
  result.reserve(keys->size() - 1);
  for (uint32_t i = 1; i < keys->size(); ++i) {
    result.emplace_back(velox::checkedNotNull(keys->Get(i))->string_view());
  }
  return result;
}

std::vector<std::string> getIndexColumns(
    const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* indexColumns = root.index_columns();
  NIMBLE_CHECK_NOT_NULL(indexColumns);
  std::vector<std::string> result;
  result.reserve(indexColumns->size());
  for (const auto* column : *indexColumns) {
    result.emplace_back(velox::checkedNotNull(column)->string_view());
  }
  return result;
}

std::vector<SortOrder> getSortOrders(
    const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* sortOrders = root.sort_orders();
  NIMBLE_CHECK_NOT_NULL(sortOrders);
  std::vector<SortOrder> result;
  result.reserve(sortOrders->size());
  for (const auto* sortOrder : *sortOrders) {
    const auto sv = velox::checkedNotNull(sortOrder)->string_view();
    try {
      result.emplace_back(SortOrder::deserialize(folly::parseJson(sv)));
    } catch (const folly::json::parse_error&) {
      // Backward compatibility: old files stored sort orders as raw strings
      // (e.g., "ASC NULLS FIRST") instead of JSON. Default to ascending.
      result.emplace_back(SortOrder{.ascending = true});
    }
  }
  return result;
}

// Builds cumulative row offsets from partition_row_counts in the FlatBuffer.
// Returns a prefix-sum vector of size numPartitions + 1.
std::vector<uint32_t> getPartitionRows(
    const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* rowCounts = root.partition_row_counts();
  NIMBLE_CHECK_NOT_NULL(rowCounts);
  NIMBLE_CHECK_GT(rowCounts->size(), 0, "partition_row_counts cannot be empty");

  std::vector<uint32_t> offsets;
  offsets.reserve(rowCounts->size() + 1);
  offsets.emplace_back(0);
  for (uint32_t i = 0; i < rowCounts->size(); ++i) {
    const auto count = static_cast<uint32_t>(rowCounts->Get(i));
    NIMBLE_CHECK_GT(count, 0, "Partition {} has zero rows", i);
    offsets.emplace_back(offsets.back() + count);
  }
  return offsets;
}

} // namespace

// ---------------------------------------------------------------------------
// ClusterIndexBase
// ---------------------------------------------------------------------------

ClusterIndexBase::~ClusterIndexBase() = default;

ClusterIndexBase::ClusterIndexBase(
    Section rootSection,
    std::shared_ptr<MetadataInput> metadataInput,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
    KeyReaderFactory keyReaderFactory,
    bool pinIndex,
    bool preloadIndex,
    velox::memory::MemoryPool* pool)
    : IndexLookup{IndexType::Cluster},
      rootSection_{std::move(rootSection)},
      indexRoot_{getIndexRoot(rootSection_)},
      numPartitions_{getIndexPartitionCount(indexRoot_)},
      indexColumns_{getIndexColumns(indexRoot_)},
      sortOrders_{getSortOrders(indexRoot_)},
      firstKey_{getFirstKey(indexRoot_)},
      lastKey_{getLastKey(indexRoot_)},
      partitionKeys_{getPartitionLastKeys(indexRoot_)},
      partitionRows_{getPartitionRows(indexRoot_)},
      numRows_{partitionRows_.at(partitionRows_.size() - 1)},
      metadataInput_{std::move(metadataInput)},
      dataInput_{std::move(dataInput)},
      keyReaderFactory_{std::move(keyReaderFactory)},
      pool_{pool},
      pinIndex_{pinIndex},
      partitions_(numPartitions_) {
  NIMBLE_CHECK(!indexColumns_.empty());
  NIMBLE_CHECK_EQ(indexColumns_.size(), sortOrders_.size());
  NIMBLE_CHECK_EQ(partitionRows_.size(), numPartitions_ + 1);
  NIMBLE_CHECK_EQ(partitionRows_.at(0), 0);
  NIMBLE_CHECK_EQ(partitionKeys_.size(), numPartitions_);
  NIMBLE_CHECK_NOT_NULL(metadataInput_);
  NIMBLE_CHECK_NOT_NULL(dataInput_);
  NIMBLE_CHECK_NOT_NULL(pool_);
  NIMBLE_CHECK(static_cast<bool>(keyReaderFactory_));
  if (!preloadIndex) {
    return;
  }
  NIMBLE_CHECK(
      pinIndex_,
      "preloadIndex requires pinIndex=true to retain preloaded chunks");
  this->preloadIndex();
}

void ClusterIndexBase::preloadIndex() {
  NIMBLE_CHECK_GT(numPartitions_, 0);
  std::vector<uint32_t> partitionIds(numPartitions_);
  std::iota(partitionIds.begin(), partitionIds.end(), 0);
  loadPartitions(partitionIds);

  for (uint32_t partitionId = 0; partitionId < numPartitions_; ++partitionId) {
    loadPartitionKeyStream(partitions_.at(partitionId).get());
  }
}

void ClusterIndexBase::loadPartitionKeyStream(IndexPartition* partition) const {
  auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t numChunks = loadedPartition.numChunks();
  NIMBLE_CHECK_GT(
      numChunks, 0, "Partition {} has no chunks", loadedPartition.id);
  std::vector<std::unique_ptr<velox::dwio::common::SeekableInputStream>>
      streams;
  streams.reserve(numChunks);
  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    const ChunkLocation chunkLocation{
        chunkIdx,
        loadedPartition.chunkOffset(chunkIdx),
        loadedPartition.chunkSize(chunkIdx),
        loadedPartition.rowOffset(chunkIdx)};
    streams.push_back(
        dataInput_->enqueue(loadedPartition.chunkStreamRegion(chunkLocation)));
  }
  dataInput_->load(velox::dwio::common::LogType::GROUP_INDEX);

  auto stream = streams.begin();
  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx, ++stream) {
    auto& decodedChunk = loadedPartition.decodedChunks.at(chunkIdx);
    decodedChunk.chunkOffset = loadedPartition.chunkOffset(chunkIdx);
    // Per-iteration scratch so the previous chunk's buffer isn't aliased
    // and overwritten if the next chunk fits in its capacity.
    velox::BufferPtr scratch;
    decodedChunk.data =
        decodeKeyChunk(std::move(*stream), keyReaderFactory_, scratch, pool_);
  }
}

void ClusterIndexBase::loadPartitions(
    std::span<const uint32_t> partitionIds) const {
  NIMBLE_CHECK(!partitionIds.empty(), "partitionIds must not be empty");
  std::unique_lock l(mutex_);
  std::vector<MetadataSection> sections;
  sections.reserve(partitionIds.size());
  for (uint32_t id : partitionIds) {
    NIMBLE_CHECK_LT(id, numPartitions_);
    NIMBLE_CHECK_NULL(partitions_[id]);
    sections.push_back(partitionSection(id));
  }
  auto buffers = metadataInput_->load(
      std::span<const MetadataSection>{sections.data(), sections.size()});
  NIMBLE_CHECK_EQ(buffers.size(), partitionIds.size());
  auto partitionId = partitionIds.begin();
  for (auto& buffer : buffers) {
    initializePartition(*partitionId, std::move(*buffer));
    ++partitionId;
  }
}

std::optional<uint32_t> ClusterIndexBase::lookupPartition(
    std::string_view key) const {
  if (key < firstKey_) {
    return 0;
  }
  const auto it =
      std::lower_bound(partitionKeys_.begin(), partitionKeys_.end(), key);
  if (it == partitionKeys_.end()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(it - partitionKeys_.begin());
}

const ClusterIndexBase::IndexPartition* ClusterIndexBase::lookupPartition(
    uint32_t row) const {
  NIMBLE_CHECK_LT(
      row, numRows_, "Row {} is beyond file total rows {}", row, numRows_);
  const auto it =
      std::upper_bound(partitionRows_.begin(), partitionRows_.end(), row);
  NIMBLE_CHECK(it != partitionRows_.begin());
  const uint32_t partitionId =
      static_cast<uint32_t>(it - partitionRows_.begin()) - 1;
  return loadPartition(partitionId);
}

uint32_t ClusterIndexBase::partitionRow(uint32_t partitionId, uint32_t row)
    const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  NIMBLE_CHECK_GE(row, partitionRows_.at(partitionId));
  NIMBLE_CHECK_LT(row, partitionRows_.at(partitionId + 1));
  return row - partitionRows_.at(partitionId);
}

void ClusterIndexBase::resolvePartitionBounds(
    std::vector<PartitionLookup>& partitionLookups) const {
  std::sort(
      partitionLookups.begin(),
      partitionLookups.end(),
      [](const auto& a, const auto& b) {
        return a.partitionId < b.partitionId;
      });

  for (size_t i = 0; i < partitionLookups.size();) {
    const auto* partition = loadPartition(partitionLookups.at(i).partitionId);
    const auto& loadedPartition = *velox::checkedNotNull(partition);

    while (i < partitionLookups.size() &&
           partitionLookups.at(i).partitionId == loadedPartition.id) {
      const auto& entry = partitionLookups.at(i);
      *velox::checkedNotNull(entry.targetRow) =
          resolvePartitionRow(partition, entry.encodedKey, entry.inclusive);
      ++i;
    }
  }
}

IndexLookup::LookupResult ClusterIndexBase::buildLookupResult(
    const LookupOptions& options,
    const std::vector<RowRange>& partitionRowRanges) const {
  std::vector<RowRange> locations;
  locations.reserve(partitionRowRanges.size());
  std::vector<uint32_t> offsets;
  offsets.reserve(partitionRowRanges.size() + 1);
  offsets.emplace_back(0);

  for (const auto& lookupRange : partitionRowRanges) {
    auto range = lookupRange;
    if (options.rowRange.has_value()) {
      range = range.intersect(options.rowRange.value());
    }
    if (!range.empty()) {
      locations.emplace_back(range);
    }
    offsets.emplace_back(locations.size());
  }

  return LookupResult{std::move(locations), std::move(offsets)};
}

void ClusterIndexBase::lookupPartitions(
    const LookupRequest& request,
    std::vector<PartitionLookup>& partitionLookups,
    std::vector<RowRange>& partitionRowRanges) const {
  partitionLookups.reserve(2 * request.size());
  partitionRowRanges.assign(request.size(), RowRange{});

  if (request.mode() == LookupRequest::Mode::PointLookup) {
    partitionPointLookups(request, partitionLookups, partitionRowRanges);
  } else {
    partitionRangeLookups(request, partitionLookups, partitionRowRanges);
  }
}

void ClusterIndexBase::partitionPointLookups(
    const LookupRequest& request,
    std::vector<PartitionLookup>& partitionLookups,
    std::vector<RowRange>& partitionRowRanges) const {
  for (uint32_t i = 0; i < request.size(); ++i) {
    const auto encodedKey = request.pointKey(i);
    const auto partitionId = lookupPartition(encodedKey);
    if (!partitionId.has_value()) {
      continue;
    }
    // Lower bound: first row >= key (inclusive).
    partitionLookups.emplace_back(
        partitionId.value(),
        encodedKey,
        /*inclusive=*/true,
        &partitionRowRanges.at(i).startRow);
    // Upper bound: first row > key (exclusive).
    partitionLookups.emplace_back(
        partitionId.value(),
        encodedKey,
        /*inclusive=*/false,
        &partitionRowRanges.at(i).endRow);
  }
}

void ClusterIndexBase::partitionRangeLookups(
    const LookupRequest& request,
    std::vector<PartitionLookup>& partitionLookups,
    std::vector<RowRange>& partitionRowRanges) const {
  for (uint32_t i = 0; i < request.size(); ++i) {
    const auto& keyBound = request.rangeBound(i);
    NIMBLE_CHECK(
        keyBound.lowerKey.has_value() || keyBound.upperKey.has_value(),
        "At least one of lowerKey or upperKey must be set");

    // Skip empty ranges where upper bound <= lower bound.
    if (keyBound.lowerKey.has_value() && keyBound.upperKey.has_value() &&
        keyBound.upperKey.value() <= keyBound.lowerKey.value()) {
      continue;
    }

    if (keyBound.lowerKey.has_value()) {
      const auto partitionId = lookupPartition(keyBound.lowerKey.value());
      if (!partitionId.has_value()) {
        continue;
      }
      partitionLookups.emplace_back(
          partitionId.value(),
          keyBound.lowerKey.value(),
          /*inclusive=*/true,
          &partitionRowRanges.at(i).startRow);
    }

    if (keyBound.upperKey.has_value()) {
      const auto partitionId = lookupPartition(keyBound.upperKey.value());
      if (partitionId.has_value()) {
        // Upper bound is exclusive: seekAtOrAfter gives the correct cutoff.
        partitionLookups.emplace_back(
            partitionId.value(),
            keyBound.upperKey.value(),
            /*inclusive=*/true,
            &partitionRowRanges.at(i).endRow);
      } else {
        partitionRowRanges.at(i).endRow = numRows_;
      }
    } else {
      partitionRowRanges.at(i).endRow = numRows_;
    }
  }
}

IndexLookup::LookupResult ClusterIndexBase::lookup(
    const LookupRequest& request) const {
  std::vector<PartitionLookup> partitionLookups;
  std::vector<RowRange> partitionRowRanges;
  lookupPartitions(request, partitionLookups, partitionRowRanges);
  NIMBLE_CHECK_EQ(partitionRowRanges.size(), request.size());
  NIMBLE_CHECK_LE(partitionLookups.size(), 2 * request.size());
  resolvePartitionBounds(partitionLookups);

  return buildLookupResult(request.options(), partitionRowRanges);
}

uint32_t ClusterIndexBase::resolvePartitionRow(
    const IndexPartition* partition,
    std::string_view encodedKey,
    bool inclusive) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto chunkLocation = lookupChunk(partition, encodedKey);
  const auto partitionRow =
      seekInChunk(partition, chunkLocation, encodedKey, inclusive);
  return partitionRows_.at(loadedPartition.id) + partitionRow;
}

folly::Range<const uint32_t*> ClusterIndexBase::partitionChunkRows(
    uint32_t partitionId) const {
  const auto* partition = loadPartition(partitionId);
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto& partitionIndex = *velox::checkedNotNull(loadedPartition.index);
  const auto* chunkRows = partitionIndex.chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows, "Index partition has no chunk rows");
  return {chunkRows->data(), chunkRows->size()};
}

MetadataSection ClusterIndexBase::partitionSection(uint32_t partitionId) const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  const auto& indexRoot = *velox::checkedNotNull(indexRoot_);
  const auto* indexPartitions = indexRoot.index_partitions();
  NIMBLE_CHECK_NOT_NULL(indexPartitions);
  const auto* metadata = indexPartitions->Get(partitionId);
  const auto& section = *velox::checkedNotNull(metadata);
  const auto rawUncompressedSize = section.uncompressed_size();
  return MetadataSection{
      section.offset(),
      section.size(),
      static_cast<CompressionType>(section.compression_type()),
      rawUncompressedSize > 0 ? std::optional<uint32_t>(rawUncompressedSize)
                              : std::nullopt};
}

ClusterIndexBase::IndexPartition::IndexPartition(
    uint32_t _id,
    std::unique_ptr<MetadataBuffer> _metadata,
    bool pinIndex)
    : id{_id},
      metadata{std::move(_metadata)},
      index{flatbuffers::GetRoot<serialization::ClusterIndexPartition>(
          velox::checkedNotNull(metadata.get())->content().data())} {
  const auto& partitionIndex = *velox::checkedNotNull(index);
  const auto* chunkKeys = partitionIndex.chunk_keys();
  NIMBLE_CHECK_NOT_NULL(chunkKeys);
  const auto* chunkRows = partitionIndex.chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_EQ(chunkRows->size(), chunkKeys->size());
  const auto* chunkOffsets = partitionIndex.chunk_offsets();
  NIMBLE_CHECK_NOT_NULL(chunkOffsets);
  NIMBLE_CHECK_EQ(chunkOffsets->size(), chunkKeys->size());
  // Allocate the cache slot vector. When pinIndex is true, size to the
  // number of chunks so every chunk has a dedicated slot. When false,
  // allocate a single scratch slot that retains at most one decoded chunk.
  decodedChunks.resize(pinIndex ? numChunks() : 1);
}

uint32_t ClusterIndexBase::IndexPartition::chunkIndex(
    std::string_view encodedKey) const {
  const auto& partitionIndex = *velox::checkedNotNull(index);
  const auto* chunkKeys = partitionIndex.chunk_keys();
  auto it = std::lower_bound(
      chunkKeys->begin(),
      chunkKeys->end(),
      encodedKey,
      [](const flatbuffers::String* a, std::string_view b) {
        return a->string_view() < b;
      });
  NIMBLE_CHECK(
      it != chunkKeys->end(), "Key must be within partition's chunk range");
  return static_cast<uint32_t>(it - chunkKeys->begin());
}

velox::common::Region ClusterIndexBase::IndexPartition::chunkStreamRegion(
    const ChunkLocation& chunkLocation) const {
  const auto& partitionIndex = *velox::checkedNotNull(index);
  const uint64_t offset =
      partitionIndex.key_stream_offset() + chunkLocation.chunkOffset;
  return velox::common::Region{offset, chunkLocation.chunkSize};
}

void ClusterIndexBase::initializePartition(
    uint32_t partitionId,
    MetadataBuffer&& buffer) const {
  NIMBLE_CHECK_NULL(partitions_[partitionId]);
  partitions_[partitionId] = std::make_unique<IndexPartition>(
      partitionId,
      std::make_unique<MetadataBuffer>(std::move(buffer)),
      pinIndex_);
}

const ClusterIndexBase::IndexPartition* ClusterIndexBase::loadPartition(
    uint32_t partitionId) const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  {
    std::shared_lock l(mutex_);
    if (partitions_[partitionId] != nullptr) {
      return partitions_[partitionId].get();
    }
  }
  std::unique_lock l(mutex_);
  if (partitions_[partitionId] == nullptr) {
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::index::ClusterIndexBase::loadPartition",
        &partitionId);
    const auto section = partitionSection(partitionId);
    auto results = metadataInput_->load({&section, 1});
    NIMBLE_CHECK_EQ(results.size(), 1);
    initializePartition(partitionId, std::move(*results.at(0)));
  }
  return partitions_.at(partitionId).get();
}

ChunkLocation ClusterIndexBase::lookupChunk(
    const IndexPartition* partition,
    std::string_view encodedKey) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t idx = loadedPartition.chunkIndex(encodedKey);
  return ChunkLocation{
      idx,
      loadedPartition.chunkOffset(idx),
      loadedPartition.chunkSize(idx),
      loadedPartition.rowOffset(idx)};
}

ClusterIndexBase::Layout ClusterIndexBase::layout(bool detail) const {
  Layout result;
  result.indexColumns = indexColumns_;
  result.sortOrders = sortOrders_;
  result.numPartitions = numPartitions_;

  // Always populate partition metadata sections (cheap, from root FlatBuffer).
  result.partitions.resize(numPartitions_);
  for (uint32_t i = 0; i < numPartitions_; ++i) {
    result.partitions.at(i).metadataSection = partitionSection(i);
  }

  if (!detail) {
    return result;
  }

  // Populate per-partition detail (requires loading partition metadata via
  // I/O).
  for (uint32_t i = 0; i < numPartitions_; ++i) {
    const auto* partition = loadPartition(i);
    const auto& loadedPartition = *velox::checkedNotNull(partition);
    const auto& partitionIndex = *velox::checkedNotNull(loadedPartition.index);
    auto& partitionLayout = result.partitions.at(i);
    partitionLayout.keyStreamRegion = velox::common::Region{
        partitionIndex.key_stream_offset(), partitionIndex.key_stream_size()};
    partitionLayout.numChunks = loadedPartition.numChunks();
    partitionLayout.numRows = partitionRows_.at(i + 1) - partitionRows_.at(i);
    partitionLayout.metadataSizeBytes = static_cast<uint32_t>(
        velox::checkedNotNull(loadedPartition.metadata.get())
            ->content()
            .size());
  }
  return result;
}

// ---------------------------------------------------------------------------
// ClusterIndexBase::DecodedChunk
// ---------------------------------------------------------------------------

ClusterIndexBase::DecodedChunk ClusterIndexBase::getDecodedChunk(
    const IndexPartition* partition,
    const ChunkLocation& chunkLocation) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t slotIdx = pinIndex_ ? chunkLocation.chunkIndex : 0;
  // Fast path: shared lock for cache hit.
  {
    std::shared_lock l(loadedPartition.mutex);
    NIMBLE_CHECK_LT(slotIdx, loadedPartition.decodedChunks.size());
    const auto& slot = loadedPartition.decodedChunks.at(slotIdx);
    if (slot.data != nullptr &&
        (pinIndex_ || slot.chunkOffset == chunkLocation.chunkOffset)) {
      return slot;
    }
  }

  // Decode outside the lock to avoid blocking concurrent readers.
  DecodedChunk decodedChunk;
  const auto region = loadedPartition.chunkStreamRegion(chunkLocation);
  decodedChunk.chunkOffset = chunkLocation.chunkOffset;
  velox::BufferPtr scratch;
  decodedChunk.data = decodeKeyChunk(
      dataInput_->read(
          region.offset,
          region.length,
          velox::dwio::common::LogType::GROUP_INDEX),
      keyReaderFactory_,
      scratch,
      pool_);

  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      decodedChunk.data.get());

  // Install under exclusive lock. Another thread may have filled the slot
  // while we were decoding — that's fine, both produce the same result.
  std::unique_lock l(loadedPartition.mutex);
  auto& slot = loadedPartition.decodedChunks.at(slotIdx);
  if (slot.data == nullptr || slot.chunkOffset != chunkLocation.chunkOffset) {
    NIMBLE_DCHECK(
        !pinIndex_ || slot.data == nullptr,
        "Pinned slot should only be filled once");
    slot = std::move(decodedChunk);
  }
  return slot;
}

uint32_t ClusterIndexBase::seekInChunk(
    const IndexPartition* partition,
    const ChunkLocation& chunkLocation,
    std::string_view encodedKey,
    bool inclusive) const {
  const auto chunk = getDecodedChunk(partition, chunkLocation);
  const auto rowInChunk = chunk.data->reader->seek(encodedKey, inclusive);
  if (!rowInChunk.has_value()) {
    if (!inclusive) {
      return chunkLocation.rowOffset + chunk.data->reader->rowCount();
    }
    NIMBLE_CHECK(false, "Key must be found within a matched chunk");
  }
  return chunkLocation.rowOffset + rowInChunk.value();
}

ChunkLocation ClusterIndexBase::lookupChunk(
    const IndexPartition* partition,
    uint32_t partitionRow) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto& partitionIndex = *velox::checkedNotNull(loadedPartition.index);
  // chunk_rows is cumulative: chunk_rows[i] = total rows through chunk i.
  // Binary search for the first chunk whose cumulative rows > partitionRow.
  const auto* chunkRows = partitionIndex.chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  const auto it =
      std::upper_bound(chunkRows->begin(), chunkRows->end(), partitionRow);
  NIMBLE_CHECK(
      it != chunkRows->end(),
      "Row {} is beyond partition {} total rows",
      partitionRow,
      loadedPartition.id);
  const auto idx = static_cast<uint32_t>(it - chunkRows->begin());
  return ChunkLocation{
      idx,
      loadedPartition.chunkOffset(idx),
      loadedPartition.chunkSize(idx),
      loadedPartition.rowOffset(idx)};
}

std::string ClusterIndexBase::keyAtRow(uint32_t row) const {
  const auto* partition = lookupPartition(row);
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto& partitionIndex = *velox::checkedNotNull(loadedPartition.index);
  const auto partitionRow = this->partitionRow(loadedPartition.id, row);

  const auto* chunkRows = partitionIndex.chunk_rows();
  const auto* chunkKeys = partitionIndex.chunk_keys();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_NOT_NULL(chunkKeys);

  const auto chunkIt =
      std::upper_bound(chunkRows->begin(), chunkRows->end(), partitionRow);
  NIMBLE_CHECK(chunkIt != chunkRows->end());
  const auto chunkIdx = static_cast<uint32_t>(chunkIt - chunkRows->begin());

  // Fast path: the last row of each chunk has its key in chunk_keys metadata.
  const uint32_t chunkEndRow = chunkRows->Get(chunkIdx);
  if (partitionRow == chunkEndRow - 1) {
    return std::string(chunkKeys->Get(chunkIdx)->string_view());
  }

  const ChunkLocation chunkLocation{
      chunkIdx,
      loadedPartition.chunkOffset(chunkIdx),
      loadedPartition.chunkSize(chunkIdx),
      loadedPartition.rowOffset(chunkIdx)};

  const auto decodedChunk = getDecodedChunk(partition, chunkLocation);
  const uint32_t rowInChunk = partitionRow - chunkLocation.rowOffset;
  return decodedChunk.data->reader->get(rowInChunk);
}

// ---------------------------------------------------------------------------
// ClusterIndexBase::KeyIterator
// ---------------------------------------------------------------------------

class ClusterIndexBase::KeyIterator final : public KeyCursor {
 public:
  KeyIterator(const ClusterIndexBase& index, RowRange rows);

  bool hasNext() const override {
    return nextRow_ < endRow_;
  }

  std::string_view next() override;

 private:
  // Loads the chunk holding 'row' and positions the key cursor at it. Runs
  // once per chunk, so the binary searches it performs are amortized over
  // the chunk's rows.
  void openChunkAt(uint32_t row);

  // Index being scanned.
  const ClusterIndexBase& index_;

  // File-level row the iteration stops at, exclusive.
  const uint32_t endRow_;

  // File-level row that the next next() call reads.
  uint32_t nextRow_;

  // Chunk the key cursor reads. Held so that keys returned from a
  // pre-materialized encoding stay valid while the chunk is current.
  std::shared_ptr<DecodedKeyChunk> chunk_;

  // Reads keys from chunk_. Exhausted exactly at the chunk boundary, which
  // is what drives the transition to the next chunk.
  std::unique_ptr<KeyCursor> cursor_;
};

std::unique_ptr<KeyCursor> ClusterIndexBase::keyCursor(RowRange rows) const {
  return std::make_unique<KeyIterator>(*this, rows);
}

ClusterIndexBase::KeyIterator::KeyIterator(
    const ClusterIndexBase& index,
    RowRange rows)
    : index_{index}, endRow_{rows.endRow}, nextRow_{rows.startRow} {
  NIMBLE_CHECK_LE(
      rows.startRow,
      rows.endRow,
      "Cluster index iterator row range is inverted: {}",
      rows.toString());
  NIMBLE_CHECK_LE(
      rows.endRow,
      index_.numRows_,
      "Row range {} is beyond file total rows {}",
      rows.toString(),
      index_.numRows_);
}

std::string_view ClusterIndexBase::KeyIterator::next() {
  NIMBLE_CHECK(
      hasNext(), "Cluster index iterator is exhausted at row {}", nextRow_);
  // The key cursor runs out exactly at the chunk boundary, so its own
  // exhaustion is what says to move on.
  if (cursor_ == nullptr || !cursor_->hasNext()) {
    openChunkAt(nextRow_);
  }
  // Advance only once the key is in hand: a throwing cursor must not leave
  // nextRow_ claiming a row that was never returned.
  const auto key = cursor_->next();
  ++nextRow_;
  return key;
}

void ClusterIndexBase::KeyIterator::openChunkAt(uint32_t row) {
  const auto* partition = index_.lookupPartition(row);
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto& partitionIndex = *velox::checkedNotNull(loadedPartition.index);
  const uint32_t partitionRow = index_.partitionRow(loadedPartition.id, row);
  const auto chunkLocation = index_.lookupChunk(partition, partitionRow);
  auto chunk = index_.getDecodedChunk(partition, chunkLocation).data;

  // chunk_rows is a partition-relative prefix sum, so this chunk's entry is
  // the row it ends at and its distance from rowOffset is its row count.
  const uint32_t chunkEndRowInPartition =
      velox::checkedNotNull(partitionIndex.chunk_rows())
          ->Get(chunkLocation.chunkIndex);
  NIMBLE_CHECK_EQ(
      chunk->reader->rowCount(),
      chunkEndRowInPartition - chunkLocation.rowOffset,
      "Decoded key chunk holds a different row count than partition {} chunk {} metadata",
      loadedPartition.id,
      chunkLocation.chunkIndex);

  auto cursor = chunk->reader->cursor(partitionRow - chunkLocation.rowOffset);

  // Commit after every step that can throw, so a failure leaves the iterator
  // on its previous chunk rather than half-moved onto this one.
  chunk_ = std::move(chunk);
  cursor_ = std::move(cursor);
}

} // namespace facebook::nimble::index
