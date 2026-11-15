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
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/DataInput.h"
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

KeyStreamLayout getKeyStreamLayout(
    const serialization::ClusterIndex* indexRoot) {
  // An index root without key_stream_layout, such as every root written before
  // the Stripe layout, reads as the default, Partition.
  const uint32_t layout = velox::checkedNotNull(indexRoot)->key_stream_layout();
  NIMBLE_CHECK_FILE_LE(
      layout,
      static_cast<uint32_t>(serialization::KeyStreamLayout_MAX),
      "Unknown cluster index key stream layout");
  return static_cast<KeyStreamLayout>(layout);
}

// Returns the metadata sections of the partitions: the ClusterIndexPartition
// sections in the Partition layout, the ClusterIndexGroup sections of the
// stripe groups in the Stripe layout.
const flatbuffers::Vector<flatbuffers::Offset<serialization::MetadataSection>>*
getPartitionSections(const serialization::ClusterIndex& root) {
  if (getKeyStreamLayout(&root) == KeyStreamLayout::kPartition) {
    NIMBLE_CHECK_FILE_NOT_NULL(
        root.index_partitions(),
        "Partition layout index must set index_partitions");
    return root.index_partitions();
  }
  NIMBLE_CHECK_FILE_NULL(
      root.index_partitions(),
      "Stripe layout index must not set index_partitions");
  NIMBLE_CHECK_FILE_NOT_NULL(
      root.index_groups(), "Stripe layout index must set index_groups");
  return root.index_groups();
}

uint32_t getIndexPartitionCount(const serialization::ClusterIndex* indexRoot) {
  const auto& root = *velox::checkedNotNull(indexRoot);
  const auto* indexPartitions = getPartitionSections(root);
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

// Returns the file region of the key stream of each stripe of a Stripe layout
// partition, from the metadata of its stripes with the location of the key
// stream in each. Checks that each stripe has a key stream, which every key
// chunk read then trusts.
std::vector<velox::common::Region> getKeyStreamRegions(
    std::span<const StripeMetadata> stripes) {
  std::vector<velox::common::Region> regions;
  regions.reserve(stripes.size());
  for (const auto& stripe : stripes) {
    NIMBLE_CHECK_EQ(
        stripe.streams.size(),
        1,
        "Key stream locator must locate one stream per stripe");
    const auto& keyStream = stripe.streams[0];
    NIMBLE_CHECK_FILE_GT(keyStream.size, 0, "Stripe has no key stream");
    regions.emplace_back(stripe.offset + keyStream.offset, keyStream.size);
  }
  return regions;
}

// Returns the accumulated row counts of the stripes of a Stripe layout
// partition. Checks that 'group' has a key for each stripe and that each
// stripe has rows.
std::vector<uint32_t> getStripeRows(
    const serialization::ClusterIndexGroup& group,
    std::span<const StripeMetadata> stripes) {
  const auto* stripeKeys = group.stripe_keys();
  NIMBLE_CHECK_FILE_NOT_NULL(
      stripeKeys, "Cluster index group has no stripe_keys");
  NIMBLE_CHECK_FILE_GT(stripes.size(), 0, "Cluster index group has no stripes");
  NIMBLE_CHECK_FILE_EQ(
      stripeKeys->size(),
      stripes.size(),
      "Cluster index group must have one key per stripe");

  std::vector<uint32_t> stripeRows;
  stripeRows.reserve(stripes.size());
  uint32_t endRow{0};
  for (const auto& stripe : stripes) {
    NIMBLE_CHECK_FILE_GT(stripe.numRows, 0, "Stripe has no rows");
    endRow += stripe.numRows;
    stripeRows.push_back(endRow);
  }
  return stripeRows;
}

// Returns the stripe of the file of each stripe of a Stripe layout partition.
std::vector<uint32_t> getStripes(std::span<const StripeMetadata> stripes) {
  std::vector<uint32_t> result;
  result.reserve(stripes.size());
  for (const auto& stripe : stripes) {
    result.emplace_back(stripe.stripe);
  }
  return result;
}

// Returns the caller's bytes of the key chunk of 'stripe', the whole key stream
// of the stripe of 'chunkSize' bytes, from the entry of 'keyChunks' for the
// stripe.
std::string_view keyChunkData(
    uint32_t stripe,
    uint32_t chunkSize,
    std::span<const ClusterIndexBase::KeyChunk> keyChunks) {
  for (const auto& keyChunk : keyChunks) {
    if (keyChunk.stripe != stripe) {
      continue;
    }
    NIMBLE_CHECK_EQ(
        keyChunk.data.size(),
        chunkSize,
        "The given key chunk of stripe {} is not the key stream of the stripe",
        stripe);
    return keyChunk.data;
  }
  NIMBLE_FAIL("No key chunk is given for stripe {}", stripe);
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
    std::shared_ptr<DataInput> dataInput,
    KeyStreamLocator keyStreamLocator,
    KeyReaderFactory keyReaderFactory,
    const KeyComparator* keyComparator,
    bool pinIndex,
    bool preloadIndex,
    velox::memory::MemoryPool* pool)
    : IndexLookup{IndexType::Cluster},
      rootSection_{std::move(rootSection)},
      indexRoot_{getIndexRoot(rootSection_)},
      numPartitions_{getIndexPartitionCount(indexRoot_)},
      indexColumns_{getIndexColumns(indexRoot_)},
      sortOrders_{getSortOrders(indexRoot_)},
      keyStreamLayout_{getKeyStreamLayout(indexRoot_)},
      firstKey_{getFirstKey(indexRoot_)},
      lastKey_{getLastKey(indexRoot_)},
      partitionKeys_{getPartitionLastKeys(indexRoot_)},
      partitionRows_{getPartitionRows(indexRoot_)},
      numRows_{partitionRows_.at(partitionRows_.size() - 1)},
      metadataInput_{std::move(metadataInput)},
      dataInput_{std::move(dataInput)},
      keyStreamLocator_{std::move(keyStreamLocator)},
      keyStreamId_{indexRoot_->key_stream_id()},
      keyReaderFactory_{std::move(keyReaderFactory)},
      keyComparator_{keyComparator},
      pool_{pool},
      pinIndex_{pinIndex},
      partitions_(numPartitions_) {
  // Only the Stripe layout locates key streams, in the stripes of the file.
  if (keyStreamLayout_ == KeyStreamLayout::kStripe) {
    NIMBLE_CHECK_NOT_NULL(
        keyStreamLocator_,
        "A cluster index that stores its key chunks in stripes needs a key stream locator");
  }
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

void ClusterIndexBase::loadPartitionKeyStream(
    IndexPartitionBase* partition) const {
  auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t numChunks = loadedPartition.numChunks();
  NIMBLE_CHECK_GT(
      numChunks, 0, "Partition {} has no chunks", loadedPartition.id());

  // Loads the chunks in one batch, with each chunk as its own group, so a
  // cached input caches one entry per chunk.
  dataInput_->reserve(numChunks);
  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    const ChunkLocation chunkLocation{
        chunkIdx,
        loadedPartition.chunkOffset(chunkIdx),
        loadedPartition.chunkSize(chunkIdx),
        loadedPartition.rowOffset(chunkIdx)};
    const auto region = loadedPartition.chunkStreamRegion(chunkLocation);
    dataInput_->startGroup(region);
    dataInput_->enqueue(region);
  }
  const auto handle = dataInput_->load();

  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
    const auto& bufferRef = dataInput_->bufferRef(chunkIdx);
    auto& decodedChunk = loadedPartition.decodedChunks().at(chunkIdx);
    decodedChunk.chunkIndex = chunkIdx;
    decodedChunk.data = decodeKeyChunk(
        std::string_view{bufferRef.data, bufferRef.length},
        handle,
        keyReaderFactory_,
        pool_);
  }
  dataInput_->clear();
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
  return withKeyOrder(
      keyComparator_, [&](auto keyOrder) -> std::optional<uint32_t> {
        if (keyOrder(key, firstKey_)) {
          return 0;
        }
        const auto it = std::lower_bound(
            partitionKeys_.begin(), partitionKeys_.end(), key, keyOrder);
        if (it == partitionKeys_.end()) {
          return std::nullopt;
        }
        return static_cast<uint32_t>(it - partitionKeys_.begin());
      });
}

const ClusterIndexBase::IndexPartitionBase* ClusterIndexBase::lookupPartition(
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
    std::optional<std::span<const KeyChunk>> keyChunks,
    std::vector<PartitionLookup>& partitionLookups,
    LowerBoundChunk* lowerBoundChunk) const {
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
           partitionLookups.at(i).partitionId == loadedPartition.id()) {
      const auto& entry = partitionLookups.at(i);
      *velox::checkedNotNull(entry.targetRow) =
          resolvePartitionRow(partition, entry, keyChunks, lowerBoundChunk);
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
        /*lowerBound=*/true,
        &partitionRowRanges.at(i).startRow);
    // Upper bound: first row > key (exclusive).
    partitionLookups.emplace_back(
        partitionId.value(),
        encodedKey,
        /*inclusive=*/false,
        /*lowerBound=*/false,
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
        compareKeys(
            keyBound.upperKey.value(),
            keyBound.lowerKey.value(),
            keyComparator_) <= 0) {
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
          /*lowerBound=*/true,
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
            /*lowerBound=*/false,
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
  return lookup(
      request, /*keyChunks=*/std::nullopt, /*lowerBoundChunk=*/nullptr);
}

IndexLookup::LookupResult ClusterIndexBase::lookup(
    const LookupRequest& request,
    std::span<const KeyChunk> keyChunks) const {
  NIMBLE_CHECK_EQ(
      keyStreamLayout_,
      KeyStreamLayout::kStripe,
      "Only a Stripe layout index looks up keys in given key chunks");
  NIMBLE_CHECK(!keyChunks.empty(), "A lookup in given key chunks needs some");
  return lookup(request, keyChunks, /*lowerBoundChunk=*/nullptr);
}

IndexLookup::LookupResult ClusterIndexBase::lookup(
    const LookupRequest& request,
    std::optional<std::span<const KeyChunk>> keyChunks,
    LowerBoundChunk* lowerBoundChunk) const {
  std::vector<PartitionLookup> partitionLookups;
  std::vector<RowRange> partitionRowRanges;
  lookupPartitions(request, partitionLookups, partitionRowRanges);
  NIMBLE_CHECK_EQ(partitionRowRanges.size(), request.size());
  NIMBLE_CHECK_LE(partitionLookups.size(), 2 * request.size());
  resolvePartitionBounds(keyChunks, partitionLookups, lowerBoundChunk);

  return buildLookupResult(request.options(), partitionRowRanges);
}

std::vector<uint32_t> ClusterIndexBase::lookupStripes(
    const LookupRequest& request) const {
  NIMBLE_CHECK_EQ(
      keyStreamLayout_,
      KeyStreamLayout::kStripe,
      "Only a Stripe layout index looks up the stripes of a request");
  std::vector<PartitionLookup> partitionLookups;
  std::vector<RowRange> partitionRowRanges;
  lookupPartitions(request, partitionLookups, partitionRowRanges);
  std::vector<uint32_t> stripes;
  stripes.reserve(partitionLookups.size());
  for (const auto& bound : partitionLookups) {
    const auto& partition =
        *velox::checkedNotNull(loadPartition(bound.partitionId));
    // The chunk that lookup() decodes to resolve the bound.
    const auto chunkLocation = lookupChunk(&partition, bound.encodedKey);
    stripes.emplace_back(partition.chunkStripe(chunkLocation.chunkIndex));
  }
  std::sort(stripes.begin(), stripes.end());
  stripes.erase(std::unique(stripes.begin(), stripes.end()), stripes.end());
  return stripes;
}

uint32_t ClusterIndexBase::keyStreamId() const {
  NIMBLE_CHECK_EQ(
      keyStreamLayout_,
      KeyStreamLayout::kStripe,
      "Only a Stripe layout index keeps its key chunks in a stream of each stripe");
  return keyStreamId_;
}

uint32_t ClusterIndexBase::resolvePartitionRow(
    const IndexPartitionBase* partition,
    const PartitionLookup& bound,
    std::optional<std::span<const KeyChunk>> keyChunks,
    LowerBoundChunk* lowerBoundChunk) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto chunkLocation = lookupChunk(partition, bound.encodedKey);
  const auto partitionRow =
      seekInChunk(partition, chunkLocation, bound, keyChunks, lowerBoundChunk);
  return partitionRows_.at(loadedPartition.id()) + partitionRow;
}

folly::Range<const uint32_t*> ClusterIndexBase::partitionChunkRows(
    uint32_t partitionId) const {
  const auto* partition = loadPartition(partitionId);
  return velox::checkedNotNull(partition)->chunkRows();
}

MetadataSection ClusterIndexBase::partitionSection(uint32_t partitionId) const {
  NIMBLE_CHECK_LT(partitionId, numPartitions_);
  const auto& indexRoot = *velox::checkedNotNull(indexRoot_);
  const auto* indexPartitions = getPartitionSections(indexRoot);
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

ClusterIndexBase::IndexPartitionBase::IndexPartitionBase(
    uint32_t id,
    std::unique_ptr<MetadataBuffer> metadata)
    : id_{id}, metadata_{std::move(metadata)} {
  NIMBLE_CHECK_NOT_NULL(metadata_);
}

void ClusterIndexBase::IndexPartitionBase::allocateDecodedChunks(
    bool pinIndex) {
  decodedChunks_.resize(pinIndex ? numChunks() : 1);
}

uint32_t ClusterIndexBase::IndexPartitionBase::chunkIndex(
    std::string_view encodedKey,
    const KeyComparator* keyComparator) const {
  const auto& keys = chunkKeys();
  const auto it = withKeyOrder(keyComparator, [&](auto keyOrder) {
    return std::lower_bound(
        keys.begin(),
        keys.end(),
        encodedKey,
        [keyOrder](const flatbuffers::String* chunkKey, std::string_view key) {
          return keyOrder(chunkKey->string_view(), key);
        });
  });
  NIMBLE_CHECK(it != keys.end(), "Key must be within partition's chunk range");
  return static_cast<uint32_t>(it - keys.begin());
}

velox::common::Region ClusterIndexBase::IndexPartitionBase::chunkStreamRegion(
    const ChunkLocation& chunkLocation) const {
  const auto keyStream =
      keyStreamRegion(keyStreamIndex(chunkLocation.chunkIndex));
  return velox::common::Region{
      keyStream.offset + chunkLocation.chunkOffset, chunkLocation.chunkSize};
}

namespace detail {

// A Partition layout partition, read from its ClusterIndexPartition.
class IndexPartition final : public ClusterIndexBase::IndexPartitionBase {
 public:
  IndexPartition(
      uint32_t id,
      std::unique_ptr<MetadataBuffer> metadata,
      bool pinIndex)
      : IndexPartitionBase{id, std::move(metadata)},
        partition_{flatbuffers::GetRoot<serialization::ClusterIndexPartition>(
            metadata_->content().data())} {
    const auto* chunkKeys = partition_->chunk_keys();
    NIMBLE_CHECK_NOT_NULL(chunkKeys);
    const auto* chunkRows = partition_->chunk_rows();
    NIMBLE_CHECK_NOT_NULL(chunkRows);
    NIMBLE_CHECK_EQ(chunkRows->size(), chunkKeys->size());
    const auto* chunkOffsets = partition_->chunk_offsets();
    NIMBLE_CHECK_NOT_NULL(chunkOffsets);
    NIMBLE_CHECK_EQ(chunkOffsets->size(), chunkKeys->size());
    allocateDecodedChunks(pinIndex);
  }

  KeyStreamLayout layout() const override {
    return KeyStreamLayout::kPartition;
  }

  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>&
  chunkKeys() const override {
    return *partition_->chunk_keys();
  }

  folly::Range<const uint32_t*> chunkRows() const override {
    const auto* rows = partition_->chunk_rows();
    return {rows->data(), rows->size()};
  }

  uint32_t chunkOffset(uint32_t chunkIdx) const override {
    return partition_->chunk_offsets()->Get(chunkIdx);
  }

  uint32_t chunkSize(uint32_t chunkIdx) const override {
    const uint32_t offset = chunkOffset(chunkIdx);
    const uint32_t nextOffset = chunkIdx + 1 < numChunks()
        ? chunkOffset(chunkIdx + 1)
        : partition_->key_stream_size();
    return nextOffset - offset;
  }

  // Every chunk of the partition is in its one key stream.
  uint32_t keyStreamIndex(uint32_t chunkIdx) const override {
    NIMBLE_CHECK_LT(chunkIdx, numChunks());
    return 0;
  }

  velox::common::Region keyStreamRegion(uint32_t keyStreamIdx) const override {
    NIMBLE_CHECK_EQ(keyStreamIdx, 0);
    return velox::common::Region{
        partition_->key_stream_offset(), partition_->key_stream_size()};
  }

  uint32_t chunkStripe(uint32_t /*chunkIdx*/) const override {
    NIMBLE_FAIL("A Partition layout partition keeps no key chunk in a stripe");
  }

 private:
  const serialization::ClusterIndexPartition* const partition_;
};

// A Stripe layout partition, read from its ClusterIndexGroup and the metadata
// of its stripe group.
class IndexGroup final : public ClusterIndexBase::IndexPartitionBase {
 public:
  // 'keyStreamMetadata' holds the metadata of each stripe of the group, with
  // the location of its key stream.
  IndexGroup(
      uint32_t id,
      std::unique_ptr<MetadataBuffer> metadata,
      std::span<const StripeMetadata> keyStreamMetadata,
      bool pinIndex)
      : IndexPartitionBase{id, std::move(metadata)},
        group_{flatbuffers::GetRoot<serialization::ClusterIndexGroup>(
            metadata_->content().data())},
        keyStreamRegions_{getKeyStreamRegions(keyStreamMetadata)},
        stripes_{getStripes(keyStreamMetadata)},
        stripeRows_{getStripeRows(*group_, keyStreamMetadata)} {
    allocateDecodedChunks(pinIndex);
  }

  KeyStreamLayout layout() const override {
    return KeyStreamLayout::kStripe;
  }

  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>&
  chunkKeys() const override {
    return *group_->stripe_keys();
  }

  folly::Range<const uint32_t*> chunkRows() const override {
    return {stripeRows_.data(), stripeRows_.size()};
  }

  // The key chunk of a stripe is its whole key stream.
  uint32_t chunkOffset(uint32_t /*chunkIdx*/) const override {
    return 0;
  }

  uint32_t chunkSize(uint32_t chunkIdx) const override {
    return static_cast<uint32_t>(
        keyStreamRegion(keyStreamIndex(chunkIdx)).length);
  }

  // Each stripe holds its chunk in a key stream of its own.
  uint32_t keyStreamIndex(uint32_t chunkIdx) const override {
    return chunkIdx;
  }

  velox::common::Region keyStreamRegion(uint32_t keyStreamIdx) const override {
    NIMBLE_CHECK_LT(keyStreamIdx, keyStreamRegions_.size());
    return keyStreamRegions_[keyStreamIdx];
  }

  uint32_t chunkStripe(uint32_t chunkIdx) const override {
    NIMBLE_CHECK_LT(chunkIdx, stripes_.size());
    return stripes_[chunkIdx];
  }

 private:
  const serialization::ClusterIndexGroup* const group_;
  // File region of the key stream of each stripe, without its stream trailer.
  const std::vector<velox::common::Region> keyStreamRegions_;
  // Stripe of the file of each stripe of the group.
  const std::vector<uint32_t> stripes_;
  // Accumulated row counts per stripe (prefix sum, group-wide): the chunk rows
  // of the group.
  const std::vector<uint32_t> stripeRows_;
};

} // namespace detail

void ClusterIndexBase::initializePartition(
    uint32_t partitionId,
    MetadataBuffer&& buffer) const {
  NIMBLE_CHECK_NULL(partitions_[partitionId]);
  auto metadata = std::make_unique<MetadataBuffer>(std::move(buffer));
  if (keyStreamLayout_ == KeyStreamLayout::kPartition) {
    partitions_[partitionId] = std::make_unique<detail::IndexPartition>(
        partitionId, std::move(metadata), pinIndex_);
    return;
  }

  // A partition holds the stripes of the stripe group of the same index.
  const auto keyStreamMetadata = keyStreamLocator_(
      partitionId, std::span<const uint32_t>{&keyStreamId_, 1});
  auto group = std::make_unique<detail::IndexGroup>(
      partitionId, std::move(metadata), keyStreamMetadata, pinIndex_);
  NIMBLE_CHECK_FILE_EQ(
      group->chunkRows().back(),
      partitionRows_.at(partitionId + 1) - partitionRows_.at(partitionId),
      "The stripes of a cluster index group must hold its rows");
  partitions_[partitionId] = std::move(group);
}

const ClusterIndexBase::IndexPartitionBase* ClusterIndexBase::loadPartition(
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
    const IndexPartitionBase* partition,
    std::string_view encodedKey) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t idx = loadedPartition.chunkIndex(encodedKey, keyComparator_);
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
  result.keyStreamLayout = keyStreamLayout_;
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
    auto& partitionLayout = result.partitions.at(i);
    // Every chunk of a Partition layout partition is in its one key stream,
    // and each stripe of a Stripe layout partition holds one key chunk.
    const uint32_t numKeyStreams =
        loadedPartition.layout() == KeyStreamLayout::kPartition
        ? 1
        : loadedPartition.numChunks();
    partitionLayout.keyStreamRegions.reserve(numKeyStreams);
    for (uint32_t keyStream = 0; keyStream < numKeyStreams; ++keyStream) {
      partitionLayout.keyStreamRegions.emplace_back(
          loadedPartition.keyStreamRegion(keyStream));
    }
    partitionLayout.numChunks = loadedPartition.numChunks();
    partitionLayout.numRows = partitionRows_.at(i + 1) - partitionRows_.at(i);
    partitionLayout.metadataSizeBytes =
        static_cast<uint32_t>(loadedPartition.metadata().content().size());
  }
  return result;
}

// ---------------------------------------------------------------------------
// ClusterIndexBase::DecodedChunk
// ---------------------------------------------------------------------------

ClusterIndexBase::DecodedChunk ClusterIndexBase::getDecodedChunk(
    const IndexPartitionBase* partition,
    const ChunkLocation& chunkLocation,
    std::optional<std::span<const KeyChunk>> keyChunks) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t slotIdx = pinIndex_ ? chunkLocation.chunkIndex : 0;
  // Fast path: shared lock for cache hit.
  {
    std::shared_lock l(loadedPartition.mutex());
    NIMBLE_CHECK_LT(slotIdx, loadedPartition.decodedChunks().size());
    const auto& slot = loadedPartition.decodedChunks().at(slotIdx);
    if (slot.data != nullptr &&
        (pinIndex_ || slot.chunkIndex == chunkLocation.chunkIndex)) {
      return slot;
    }
  }

  // Decode outside the lock to avoid blocking concurrent readers.
  DecodedChunk decodedChunk;
  const auto region = loadedPartition.chunkStreamRegion(chunkLocation);
  decodedChunk.chunkIndex = chunkLocation.chunkIndex;
  auto [chunk, handle] = keyChunks.has_value()
      ? std::make_pair(
            keyChunkData(
                loadedPartition.chunkStripe(chunkLocation.chunkIndex),
                chunkLocation.chunkSize,
                keyChunks.value()),
            DataInput::Handle{})
      : dataInput_->read(region);
  decodedChunk.data =
      decodeKeyChunk(chunk, std::move(handle), keyReaderFactory_, pool_);

  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::index::ClusterIndexBase::getDecodedChunk",
      decodedChunk.data.get());

  // A chunk decoded from the caller's bytes can read its keys from them, so
  // it must not outlive the call.
  if (keyChunks.has_value()) {
    return decodedChunk;
  }

  // Install under exclusive lock. Another thread may have filled the slot
  // while we were decoding — that's fine, both produce the same result.
  std::unique_lock l(loadedPartition.mutex());
  auto& slot = loadedPartition.decodedChunks().at(slotIdx);
  if (slot.data == nullptr || slot.chunkIndex != chunkLocation.chunkIndex) {
    NIMBLE_DCHECK(
        !pinIndex_ || slot.data == nullptr,
        "Pinned slot should only be filled once");
    slot = std::move(decodedChunk);
  }
  return slot;
}

uint32_t ClusterIndexBase::seekInChunk(
    const IndexPartitionBase* partition,
    const ChunkLocation& chunkLocation,
    const PartitionLookup& bound,
    std::optional<std::span<const KeyChunk>> keyChunks,
    LowerBoundChunk* lowerBoundChunk) const {
  // The upper bound of a key is usually in the chunk of its lower bound.
  const bool inLowerBoundChunk = lowerBoundChunk != nullptr &&
      lowerBoundChunk->data != nullptr &&
      lowerBoundChunk->partitionId == velox::checkedNotNull(partition)->id() &&
      lowerBoundChunk->chunkIndex == chunkLocation.chunkIndex;
  const auto chunk = inLowerBoundChunk
      ? lowerBoundChunk->data
      : getDecodedChunk(partition, chunkLocation, keyChunks).data;
  if (bound.lowerBound && lowerBoundChunk != nullptr) {
    NIMBLE_CHECK_NULL(lowerBoundChunk->data, "A seek has one lower bound");
    *lowerBoundChunk = {
        .partitionId = velox::checkedNotNull(partition)->id(),
        .chunkIndex = chunkLocation.chunkIndex,
        .data = chunk};
  }
  const auto rowInChunk =
      chunk->reader->seek(bound.encodedKey, bound.inclusive);
  if (!rowInChunk.has_value()) {
    if (!bound.inclusive) {
      return chunkLocation.rowOffset + chunk->reader->rowCount();
    }
    NIMBLE_CHECK(false, "Key must be found within a matched chunk");
  }
  return chunkLocation.rowOffset + rowInChunk.value();
}

ChunkLocation ClusterIndexBase::lookupChunk(
    const IndexPartitionBase* partition,
    uint32_t partitionRow) const {
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  // chunk_rows is cumulative: chunk_rows[i] = total rows through chunk i.
  // Binary search for the first chunk whose cumulative rows > partitionRow.
  const auto chunkRows = loadedPartition.chunkRows();
  const auto it =
      std::upper_bound(chunkRows.begin(), chunkRows.end(), partitionRow);
  NIMBLE_CHECK(
      it != chunkRows.end(),
      "Row {} is beyond partition {} total rows",
      partitionRow,
      loadedPartition.id());
  const auto idx = static_cast<uint32_t>(it - chunkRows.begin());
  return ChunkLocation{
      idx,
      loadedPartition.chunkOffset(idx),
      loadedPartition.chunkSize(idx),
      loadedPartition.rowOffset(idx)};
}

std::string ClusterIndexBase::keyAtRow(uint32_t row) const {
  const auto* partition = lookupPartition(row);
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const auto partitionRow = this->partitionRow(loadedPartition.id(), row);

  const auto chunkRows = loadedPartition.chunkRows();
  const auto& chunkKeys = loadedPartition.chunkKeys();

  const auto chunkIt =
      std::upper_bound(chunkRows.begin(), chunkRows.end(), partitionRow);
  NIMBLE_CHECK(chunkIt != chunkRows.end());
  const auto chunkIdx = static_cast<uint32_t>(chunkIt - chunkRows.begin());

  // Fast path: the last row of each chunk has its key in chunk_keys metadata.
  const uint32_t chunkEndRow = chunkRows[chunkIdx];
  if (partitionRow == chunkEndRow - 1) {
    return std::string(chunkKeys.Get(chunkIdx)->string_view());
  }

  const ChunkLocation chunkLocation{
      chunkIdx,
      loadedPartition.chunkOffset(chunkIdx),
      loadedPartition.chunkSize(chunkIdx),
      loadedPartition.rowOffset(chunkIdx)};

  const auto decodedChunk =
      getDecodedChunk(partition, chunkLocation, /*keyChunks=*/std::nullopt);
  const uint32_t rowInChunk = partitionRow - chunkLocation.rowOffset;
  return decodedChunk.data->reader->get(rowInChunk);
}

// ---------------------------------------------------------------------------
// ClusterIndexBase::KeyIterator
// ---------------------------------------------------------------------------

class ClusterIndexBase::KeyIterator final : public KeyCursor {
 public:
  KeyIterator(
      const ClusterIndexBase& index,
      RowRange rows,
      std::shared_ptr<DecodedKeyChunk> chunk);

  bool hasNext() const override {
    return nextRow_ < endRow_;
  }

  std::string_view next() override;

  // Reads the keys of each chunk that the batch crosses from the key cursor of
  // the chunk, which writes them to the caller's memory, so a key stays valid
  // past its chunk.
  void next(
      size_t maxKeys,
      std::vector<std::string_view>& keys,
      const BufferAllocator& bufferAllocator) override;

  // Moves past the rows without reading their keys. Opens a key cursor of the
  // held chunk at the next row when the chunk holds it, which the chunk's
  // reader does without decoding the chunk again, and otherwise releases the
  // chunk, so that the next read opens the chunk of its row.
  void skip(size_t numKeys) override;

 private:
  // Loads the chunk holding 'row' and positions the key cursor at it. Runs
  // once per chunk, so the binary searches it performs are amortized over
  // the chunk's rows. Opens 'chunk' when it is not null, which must be the
  // chunk of 'row', and otherwise the chunk that the index caches.
  void openChunkAt(uint32_t row, std::shared_ptr<DecodedKeyChunk> chunk);

  // Index being scanned.
  const ClusterIndexBase& index_;

  // File-level row the iteration stops at, exclusive.
  const uint32_t endRow_;

  // File-level row that the next next() call reads.
  uint32_t nextRow_;

  // Chunk the key cursor reads. Held so that keys returned from a
  // pre-materialized encoding stay valid while the chunk is current.
  std::shared_ptr<DecodedKeyChunk> chunk_;

  // File-level rows of chunk_, the first and the end, exclusive.
  uint32_t chunkStartRow_{0};
  uint32_t chunkEndRow_{0};

  // Reads keys from chunk_. Exhausted exactly at the chunk boundary, which
  // is what drives the transition to the next chunk. Null until the first
  // read, so that a cursor that is never read loads no chunk, except for the
  // cursor of seek(), which opens the chunk of its first row that the lookup
  // decoded.
  std::unique_ptr<KeyCursor> cursor_;
};

std::unique_ptr<KeyCursor> ClusterIndexBase::keyCursor(RowRange rows) const {
  return std::make_unique<KeyIterator>(*this, rows, /*chunk=*/nullptr);
}

ClusterIndexBase::SeekResult ClusterIndexBase::seek(
    const velox::serializer::EncodedKeyBounds& bounds) const {
  LowerBoundChunk lowerBoundChunk;
  const auto result = lookup(
      LookupRequest::rangeScan({bounds}),
      /*keyChunks=*/std::nullopt,
      &lowerBoundChunk);
  NIMBLE_CHECK_EQ(result.size(), 1);
  const auto& ranges = result[0];
  NIMBLE_CHECK_LE(
      ranges.size(), 1, "Cluster index returned multiple ranges for one seek");
  if (ranges.empty() || ranges.front().empty()) {
    return {};
  }
  const auto rows = ranges.front();
  // The lower bound resolves to a row in its chunk, and the lookup does not
  // narrow the range, so the chunk of the lower bound holds the first row. A
  // range without a lower bound has no chunk.
  return {
      .rows = rows,
      .keyCursor = std::make_unique<KeyIterator>(
          *this, rows, std::move(lowerBoundChunk.data))};
}

ClusterIndexBase::KeyIterator::KeyIterator(
    const ClusterIndexBase& index,
    RowRange rows,
    std::shared_ptr<DecodedKeyChunk> chunk)
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
  if (chunk != nullptr && hasNext()) {
    openChunkAt(nextRow_, std::move(chunk));
  }
}

std::string_view ClusterIndexBase::KeyIterator::next() {
  NIMBLE_CHECK(
      hasNext(), "Cluster index iterator is exhausted at row {}", nextRow_);
  // The key cursor runs out exactly at the chunk boundary, so its own
  // exhaustion is what says to move on.
  if (cursor_ == nullptr || !cursor_->hasNext()) {
    openChunkAt(nextRow_, /*chunk=*/nullptr);
  }
  // Advance only once the key is in hand: a throwing cursor must not leave
  // nextRow_ claiming a row that was never returned.
  const auto key = cursor_->next();
  ++nextRow_;
  return key;
}

void ClusterIndexBase::KeyIterator::next(
    size_t maxKeys,
    std::vector<std::string_view>& keys,
    const BufferAllocator& bufferAllocator) {
  auto remainingKeys = maxKeys;
  while (remainingKeys > 0 && hasNext()) {
    // cursor_ is null on the first read and exhausted at the end of its chunk.
    // openChunkAt() sets it or throws, so it is not null past this point.
    if (cursor_ == nullptr || !cursor_->hasNext()) {
      openChunkAt(nextRow_, /*chunk=*/nullptr);
    }

    // The key cursor stops at the end of its chunk, and the iteration may stop
    // before it.
    const auto numKeys = keys.size();
    cursor_->next(
        std::min<size_t>(remainingKeys, endRow_ - nextRow_),
        keys,
        bufferAllocator);
    const auto numRead = keys.size() - numKeys;
    NIMBLE_CHECK_GT(numRead, 0, "Cluster index key cursor returned no key");
    nextRow_ += static_cast<uint32_t>(numRead);
    remainingKeys -= numRead;
  }
}

void ClusterIndexBase::KeyIterator::skip(size_t numKeys) {
  if (numKeys == 0) {
    return;
  }
  const auto row = static_cast<uint32_t>(
      std::min<uint64_t>(endRow_, static_cast<uint64_t>(nextRow_) + numKeys));
  if (chunk_ != nullptr && row >= chunkStartRow_ && row < chunkEndRow_) {
    // Commit after the cursor, which can throw, as openChunkAt() does.
    auto cursor = chunk_->reader->cursor(row - chunkStartRow_);
    cursor_ = std::move(cursor);
    nextRow_ = row;
    return;
  }
  cursor_.reset();
  chunk_.reset();
  nextRow_ = row;
}

void ClusterIndexBase::KeyIterator::openChunkAt(
    uint32_t row,
    std::shared_ptr<DecodedKeyChunk> chunk) {
  const auto* partition = index_.lookupPartition(row);
  const auto& loadedPartition = *velox::checkedNotNull(partition);
  const uint32_t partitionRow = index_.partitionRow(loadedPartition.id(), row);
  const auto chunkLocation = index_.lookupChunk(partition, partitionRow);
  if (chunk == nullptr) {
    chunk = index_
                .getDecodedChunk(
                    partition, chunkLocation, /*keyChunks=*/std::nullopt)
                .data;
  }

  // chunk_rows is a partition-relative prefix sum, so this chunk's entry is
  // the row it ends at and its distance from rowOffset is its row count.
  const uint32_t chunkEndRowInPartition =
      loadedPartition.chunkRows()[chunkLocation.chunkIndex];
  NIMBLE_CHECK_EQ(
      chunk->reader->rowCount(),
      chunkEndRowInPartition - chunkLocation.rowOffset,
      "Decoded key chunk holds a different row count than partition {} chunk {} metadata",
      loadedPartition.id(),
      chunkLocation.chunkIndex);

  auto cursor = chunk->reader->cursor(partitionRow - chunkLocation.rowOffset);
  const uint32_t partitionStartRow =
      index_.partitionRows_.at(loadedPartition.id());

  // Commit after every step that can throw, so a failure leaves the iterator
  // on its previous chunk rather than half-moved onto this one.
  chunk_ = std::move(chunk);
  cursor_ = std::move(cursor);
  chunkStartRow_ = partitionStartRow + chunkLocation.rowOffset;
  chunkEndRow_ = partitionStartRow + chunkEndRowInPartition;
}

} // namespace facebook::nimble::index
