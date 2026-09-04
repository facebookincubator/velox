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

#include "velox/dwio/nimble/index/ClusterIndex.h"

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
  const auto* indexPartitions = indexRoot->index_partitions();
  NIMBLE_CHECK_NOT_NULL(indexPartitions);
  NIMBLE_CHECK_GT(indexPartitions->size(), 0, "ClusterIndex cannot be empty");
  return indexPartitions->size();
}

std::string_view getFirstKey(const serialization::ClusterIndex* indexRoot) {
  const auto* keys = indexRoot->partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(
      keys->size(),
      1,
      "partition_keys must have at least 2 entries (firstKey + one lastKey per partition)");
  return keys->Get(0)->string_view();
}

std::string_view getLastKey(const serialization::ClusterIndex* indexRoot) {
  const auto* keys = indexRoot->partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(keys->size(), 1);
  return keys->Get(keys->size() - 1)->string_view();
}

std::vector<std::string_view> getPartitionLastKeys(
    const serialization::ClusterIndex* indexRoot) {
  const auto* keys = indexRoot->partition_keys();
  NIMBLE_CHECK_NOT_NULL(keys);
  NIMBLE_CHECK_GT(keys->size(), 1);
  std::vector<std::string_view> result;
  result.reserve(keys->size() - 1);
  for (uint32_t i = 1; i < keys->size(); ++i) {
    result.emplace_back(keys->Get(i)->string_view());
  }
  return result;
}

std::vector<std::string> getIndexColumns(
    const serialization::ClusterIndex* indexRoot) {
  const auto* indexColumns = indexRoot->index_columns();
  NIMBLE_CHECK_NOT_NULL(indexColumns);
  std::vector<std::string> result;
  result.reserve(indexColumns->size());
  for (const auto* column : *indexColumns) {
    result.emplace_back(column->string_view());
  }
  return result;
}

std::vector<SortOrder> getSortOrders(
    const serialization::ClusterIndex* indexRoot) {
  const auto* sortOrders = indexRoot->sort_orders();
  NIMBLE_CHECK_NOT_NULL(sortOrders);
  std::vector<SortOrder> result;
  result.reserve(sortOrders->size());
  for (const auto* sortOrder : *sortOrders) {
    const auto sv = sortOrder->string_view();
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
  const auto* rowCounts = indexRoot->partition_row_counts();
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
// ClusterIndex
// ---------------------------------------------------------------------------

ClusterIndex::~ClusterIndex() = default;

std::unique_ptr<ClusterIndex> ClusterIndex::create(
    Section rootSection,
    velox::memory::MemoryPool* pool,
    const Options& options) {
  options.validate();
  NIMBLE_CHECK_EQ(
      static_cast<const void*>(&options.ioOptions->memoryPool()),
      static_cast<const void*>(pool),
      "ioOptions pool must match the provided pool");
  return std::unique_ptr<ClusterIndex>(new ClusterIndex(
      std::move(rootSection),
      createIndexMetadataInput(options),
      createIndexDataInput(options),
      options.pinIndex,
      options.preloadIndex,
      pool));
}

ClusterIndex::ClusterIndex(
    Section rootSection,
    std::shared_ptr<MetadataInput> metadataInput,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
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
      numRows_{partitionRows_.back()},
      metadataInput_{std::move(metadataInput)},
      dataInput_{std::move(dataInput)},
      pool_{pool},
      pinIndex_{pinIndex},
      partitions_(numPartitions_) {
  NIMBLE_CHECK(!indexColumns_.empty());
  NIMBLE_CHECK_EQ(indexColumns_.size(), sortOrders_.size());
  NIMBLE_CHECK_EQ(partitionRows_.size(), numPartitions_ + 1);
  NIMBLE_CHECK_EQ(partitionRows_.front(), 0);
  NIMBLE_CHECK_EQ(partitionKeys_.size(), numPartitions_);
  NIMBLE_CHECK_NOT_NULL(metadataInput_);
  NIMBLE_CHECK_NOT_NULL(dataInput_);
  NIMBLE_CHECK_NOT_NULL(pool_);
  if (!preloadIndex) {
    return;
  }
  NIMBLE_CHECK(
      pinIndex_,
      "preloadIndex requires pinIndex=true to retain preloaded chunks");
  this->preloadIndex();
}

void ClusterIndex::preloadIndex() {
  NIMBLE_CHECK_GT(numPartitions_, 0);
  std::vector<uint32_t> partitionIds(numPartitions_);
  std::iota(partitionIds.begin(), partitionIds.end(), 0);
  loadPartitions(partitionIds);

  for (uint32_t partitionId = 0; partitionId < numPartitions_; ++partitionId) {
    loadPartitionKeyStream(partitions_[partitionId].get());
  }
}

void ClusterIndex::loadPartitionKeyStream(IndexPartition* partition) const {
  const uint32_t numChunks = partition->numChunks();
  NIMBLE_CHECK_GT(numChunks, 0, "Partition {} has no chunks", partition->id);
  std::vector<std::unique_ptr<velox::dwio::common::SeekableInputStream>>
      streams;
  streams.reserve(numChunks);
  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    const ChunkLocation chunkLocation{
        chunkIdx,
        partition->chunkOffset(chunkIdx),
        partition->chunkSize(chunkIdx),
        partition->rowOffset(chunkIdx)};
    streams.push_back(
        dataInput_->enqueue(partition->chunkStreamRegion(chunkLocation)));
  }
  dataInput_->load(velox::dwio::common::LogType::GROUP_INDEX);

  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    auto& decodedChunk = partition->decodedChunks[chunkIdx];
    decodedChunk.chunkOffset = partition->chunkOffset(chunkIdx);
    // Per-iteration scratch so the previous chunk's buffer isn't aliased
    // and overwritten if the next chunk fits in its capacity.
    velox::BufferPtr scratch;
    decodedChunk.data =
        decodeKeyChunk(std::move(streams[chunkIdx]), *pool_, scratch);
  }
}

void ClusterIndex::loadPartitions(
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
  for (size_t i = 0; i < partitionIds.size(); ++i) {
    initializePartition(partitionIds[i], std::move(*buffers[i]));
  }
}

std::optional<uint32_t> ClusterIndex::lookupPartition(
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

const ClusterIndex::IndexPartition* ClusterIndex::lookupPartition(
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

uint32_t ClusterIndex::partitionRow(uint32_t partitionId, uint32_t row) const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  NIMBLE_CHECK_GE(row, partitionRows_[partitionId]);
  NIMBLE_CHECK_LT(row, partitionRows_[partitionId + 1]);
  return row - partitionRows_[partitionId];
}

void ClusterIndex::resolvePartitionBounds(
    std::vector<PartitionLookup>& partitionLookups) const {
  std::sort(
      partitionLookups.begin(),
      partitionLookups.end(),
      [](const auto& a, const auto& b) {
        return a.partitionId < b.partitionId;
      });

  for (size_t i = 0; i < partitionLookups.size();) {
    const auto* partition = loadPartition(partitionLookups[i].partitionId);

    while (i < partitionLookups.size() &&
           partitionLookups[i].partitionId == partition->id) {
      const auto& entry = partitionLookups[i];
      *entry.targetRow =
          resolvePartitionRow(partition, entry.encodedKey, entry.inclusive);
      ++i;
    }
  }
}

IndexLookup::LookupResult ClusterIndex::buildLookupResult(
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

void ClusterIndex::lookupPartitions(
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

void ClusterIndex::partitionPointLookups(
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
        &partitionRowRanges[i].startRow);
    // Upper bound: first row > key (exclusive).
    partitionLookups.emplace_back(
        partitionId.value(),
        encodedKey,
        /*inclusive=*/false,
        &partitionRowRanges[i].endRow);
  }
}

void ClusterIndex::partitionRangeLookups(
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
          &partitionRowRanges[i].startRow);
    }

    if (keyBound.upperKey.has_value()) {
      const auto partitionId = lookupPartition(keyBound.upperKey.value());
      if (partitionId.has_value()) {
        // Upper bound is exclusive: seekAtOrAfter gives the correct cutoff.
        partitionLookups.emplace_back(
            partitionId.value(),
            keyBound.upperKey.value(),
            /*inclusive=*/true,
            &partitionRowRanges[i].endRow);
      } else {
        partitionRowRanges[i].endRow = numRows_;
      }
    } else {
      partitionRowRanges[i].endRow = numRows_;
    }
  }
}

IndexLookup::LookupResult ClusterIndex::lookup(
    const LookupRequest& request) const {
  std::vector<PartitionLookup> partitionLookups;
  std::vector<RowRange> partitionRowRanges;
  lookupPartitions(request, partitionLookups, partitionRowRanges);
  NIMBLE_CHECK_EQ(partitionRowRanges.size(), request.size());
  NIMBLE_CHECK_LE(partitionLookups.size(), 2 * request.size());
  resolvePartitionBounds(partitionLookups);

  return buildLookupResult(request.options(), partitionRowRanges);
}

uint32_t ClusterIndex::resolvePartitionRow(
    const IndexPartition* partition,
    std::string_view encodedKey,
    bool inclusive) const {
  const auto chunkLocation = lookupChunk(partition, encodedKey);
  const auto partitionRow =
      seekInChunk(partition, chunkLocation, encodedKey, inclusive);
  return partitionRows_[partition->id] + partitionRow;
}

MetadataSection ClusterIndex::partitionSection(uint32_t partitionId) const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  const auto* indexPartitions = indexRoot_->index_partitions();
  NIMBLE_CHECK_NOT_NULL(indexPartitions);
  const auto* metadata = indexPartitions->Get(partitionId);
  const auto rawUncompressedSize = metadata->uncompressed_size();
  return MetadataSection{
      metadata->offset(),
      metadata->size(),
      static_cast<CompressionType>(metadata->compression_type()),
      rawUncompressedSize > 0 ? std::optional<uint32_t>(rawUncompressedSize)
                              : std::nullopt};
}

ClusterIndex::IndexPartition::IndexPartition(
    uint32_t _id,
    std::unique_ptr<MetadataBuffer> _metadata,
    bool pinIndex)
    : id{_id},
      metadata{std::move(_metadata)},
      index{flatbuffers::GetRoot<serialization::ClusterIndexPartition>(
          metadata->content().data())} {
  const auto* chunkKeys = index->chunk_keys();
  NIMBLE_CHECK_NOT_NULL(chunkKeys);
  const auto* chunkRows = index->chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_EQ(chunkRows->size(), chunkKeys->size());
  const auto* chunkOffsets = index->chunk_offsets();
  NIMBLE_CHECK_NOT_NULL(chunkOffsets);
  NIMBLE_CHECK_EQ(chunkOffsets->size(), chunkKeys->size());
  // Allocate the cache slot vector. When pinIndex is true, size to the
  // number of chunks so every chunk has a dedicated slot. When false,
  // allocate a single scratch slot that retains at most one decoded chunk.
  decodedChunks.resize(pinIndex ? numChunks() : 1);
}

uint32_t ClusterIndex::IndexPartition::chunkIndex(
    std::string_view encodedKey) const {
  const auto* chunkKeys = index->chunk_keys();
  auto it = std::lower_bound(
      chunkKeys->begin(),
      chunkKeys->end(),
      encodedKey,
      [](const flatbuffers::String* a, std::string_view b) {
        return a->string_view() < b;
      });
  NIMBLE_CHECK(
      it != chunkKeys->end(), "Key must be within partition's chunk range");
  return it - chunkKeys->begin();
}

velox::common::Region ClusterIndex::IndexPartition::chunkStreamRegion(
    const ChunkLocation& chunkLocation) const {
  const uint64_t offset =
      index->key_stream_offset() + chunkLocation.chunkOffset;
  return velox::common::Region{offset, chunkLocation.chunkSize};
}

void ClusterIndex::initializePartition(
    uint32_t partitionId,
    MetadataBuffer&& buffer) const {
  NIMBLE_CHECK_NULL(partitions_[partitionId]);
  partitions_[partitionId] = std::make_unique<IndexPartition>(
      partitionId,
      std::make_unique<MetadataBuffer>(std::move(buffer)),
      pinIndex_);
}

const ClusterIndex::IndexPartition* ClusterIndex::loadPartition(
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
        "facebook::nimble::index::ClusterIndex::loadPartition", &partitionId);
    const auto section = partitionSection(partitionId);
    auto results = metadataInput_->load({&section, 1});
    NIMBLE_CHECK_EQ(results.size(), 1);
    initializePartition(partitionId, std::move(*results.front()));
  }
  return partitions_[partitionId].get();
}

ChunkLocation ClusterIndex::lookupChunk(
    const IndexPartition* partition,
    std::string_view encodedKey) const {
  NIMBLE_DCHECK_NOT_NULL(partition);
  const uint32_t idx = partition->chunkIndex(encodedKey);
  return ChunkLocation{
      idx,
      partition->chunkOffset(idx),
      partition->chunkSize(idx),
      partition->rowOffset(idx)};
}

ClusterIndex::Layout ClusterIndex::layout(bool detail) const {
  Layout result;
  result.indexColumns = indexColumns_;
  result.sortOrders = sortOrders_;
  result.numPartitions = numPartitions_;

  // Always populate partition metadata sections (cheap, from root FlatBuffer).
  result.partitions.resize(numPartitions_);
  for (uint32_t i = 0; i < numPartitions_; ++i) {
    result.partitions[i].metadataSection = partitionSection(i);
  }

  if (!detail) {
    return result;
  }

  // Populate per-partition detail (requires loading partition metadata via
  // I/O).
  for (uint32_t i = 0; i < numPartitions_; ++i) {
    const auto* partition = loadPartition(i);
    auto& partitionLayout = result.partitions[i];
    partitionLayout.keyStreamRegion = velox::common::Region{
        partition->index->key_stream_offset(),
        partition->index->key_stream_size()};
    partitionLayout.numChunks = partition->numChunks();
    partitionLayout.numRows = partitionRows_[i + 1] - partitionRows_[i];
    partitionLayout.metadataSizeBytes =
        static_cast<uint32_t>(partition->metadata->content().size());
  }
  return result;
}

// ---------------------------------------------------------------------------
// ClusterIndex::DecodedChunk
// ---------------------------------------------------------------------------

ClusterIndex::DecodedChunk ClusterIndex::getDecodedChunk(
    const IndexPartition* partition,
    const ChunkLocation& chunkLocation) const {
  NIMBLE_DCHECK_NOT_NULL(partition);
  const uint32_t slotIdx = pinIndex_ ? chunkLocation.chunkIndex : 0;
  // Fast path: shared lock for cache hit.
  {
    std::shared_lock l(partition->mutex);
    NIMBLE_CHECK_LT(slotIdx, partition->decodedChunks.size());
    const auto& slot = partition->decodedChunks.at(slotIdx);
    if (slot.data != nullptr &&
        (pinIndex_ || slot.chunkOffset == chunkLocation.chunkOffset)) {
      return slot;
    }
  }

  // Decode outside the lock to avoid blocking concurrent readers.
  DecodedChunk decodedChunk;
  const auto region = partition->chunkStreamRegion(chunkLocation);
  decodedChunk.chunkOffset = chunkLocation.chunkOffset;
  velox::BufferPtr scratch;
  decodedChunk.data = decodeKeyChunk(
      dataInput_->read(
          region.offset,
          region.length,
          velox::dwio::common::LogType::GROUP_INDEX),
      *pool_,
      scratch);

  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::index::ClusterIndex::getDecodedChunk",
      decodedChunk.data.get());

  // Install under exclusive lock. Another thread may have filled the slot
  // while we were decoding — that's fine, both produce the same result.
  std::unique_lock l(partition->mutex);
  auto& slot = partition->decodedChunks.at(slotIdx);
  if (slot.data == nullptr || slot.chunkOffset != chunkLocation.chunkOffset) {
    NIMBLE_DCHECK(
        !pinIndex_ || slot.data == nullptr,
        "Pinned slot should only be filled once");
    slot = std::move(decodedChunk);
  }
  return slot;
}

uint32_t ClusterIndex::seekInChunk(
    const IndexPartition* partition,
    const ChunkLocation& chunkLocation,
    std::string_view encodedKey,
    bool inclusive) const {
  const auto chunk = getDecodedChunk(partition, chunkLocation);
  const auto rowInChunk = chunk.data->encoding->seek(encodedKey, inclusive);
  if (!rowInChunk.has_value()) {
    if (!inclusive) {
      return chunkLocation.rowOffset + chunk.data->encoding->rowCount();
    }
    NIMBLE_CHECK(false, "Key must be found within a matched chunk");
  }
  return chunkLocation.rowOffset + rowInChunk.value();
}

ChunkLocation ClusterIndex::lookupChunk(
    const IndexPartition* partition,
    uint32_t partitionRow) const {
  NIMBLE_DCHECK_NOT_NULL(partition);
  // chunk_rows is cumulative: chunk_rows[i] = total rows through chunk i.
  // Binary search for the first chunk whose cumulative rows > partitionRow.
  const auto* chunkRows = partition->index->chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  const auto it =
      std::upper_bound(chunkRows->begin(), chunkRows->end(), partitionRow);
  NIMBLE_CHECK(
      it != chunkRows->end(),
      "Row {} is beyond partition {} total rows",
      partitionRow,
      partition->id);
  const uint32_t idx = it - chunkRows->begin();
  return ChunkLocation{
      idx,
      partition->chunkOffset(idx),
      partition->chunkSize(idx),
      partition->rowOffset(idx)};
}

std::string ClusterIndex::keyAtRow(uint32_t row) const {
  const auto* partition = lookupPartition(row);
  const auto partitionRow = this->partitionRow(partition->id, row);

  const auto* chunkRows = partition->index->chunk_rows();
  const auto* chunkKeys = partition->index->chunk_keys();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_NOT_NULL(chunkKeys);

  const auto chunkIt =
      std::upper_bound(chunkRows->begin(), chunkRows->end(), partitionRow);
  NIMBLE_CHECK(chunkIt != chunkRows->end());
  const uint32_t chunkIdx = chunkIt - chunkRows->begin();

  // Fast path: the last row of each chunk has its key in chunk_keys metadata.
  const uint32_t chunkEndRow = chunkRows->Get(chunkIdx);
  if (partitionRow == chunkEndRow - 1) {
    return std::string(chunkKeys->Get(chunkIdx)->string_view());
  }

  const ChunkLocation chunkLocation{
      chunkIdx,
      partition->chunkOffset(chunkIdx),
      partition->chunkSize(chunkIdx),
      partition->rowOffset(chunkIdx)};

  const auto decodedChunk = getDecodedChunk(partition, chunkLocation);
  const uint32_t rowInChunk = partitionRow - chunkLocation.rowOffset;
  return decodedChunk.data->encoding->get(rowInChunk);
}

// ---------------------------------------------------------------------------
// ClusterIndex::KeyIterator
// ---------------------------------------------------------------------------

std::unique_ptr<IndexKeyCursor> ClusterIndex::keyCursor(RowRange rows) const {
  return std::make_unique<KeyIterator>(*this, rows);
}

ClusterIndex::KeyIterator::KeyIterator(const ClusterIndex& index, RowRange rows)
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

std::string_view ClusterIndex::KeyIterator::next() {
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

void ClusterIndex::KeyIterator::openChunkAt(uint32_t row) {
  const auto* partition = index_.lookupPartition(row);
  const uint32_t partitionRow = index_.partitionRow(partition->id, row);
  const auto chunkLocation = index_.lookupChunk(partition, partitionRow);
  auto chunk = index_.getDecodedChunk(partition, chunkLocation).data;

  // chunk_rows is a partition-relative prefix sum, so this chunk's entry is
  // the row it ends at and its distance from rowOffset is its row count.
  const uint32_t chunkEndRowInPartition =
      partition->index->chunk_rows()->Get(chunkLocation.chunkIndex);
  NIMBLE_CHECK_EQ(
      chunk->encoding->rowCount(),
      chunkEndRowInPartition - chunkLocation.rowOffset,
      "Decoded key chunk holds a different row count than partition {} chunk {} metadata",
      partition->id,
      chunkLocation.chunkIndex);

  auto cursor = chunk->encoding->cursor(partitionRow - chunkLocation.rowOffset);

  // Commit after every step that can throw, so a failure leaves the iterator
  // on its previous chunk rather than half-moved onto this one.
  chunk_ = std::move(chunk);
  cursor_ = std::move(cursor);
}

} // namespace facebook::nimble::index
