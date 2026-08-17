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
#include "velox/dwio/nimble/index/HashIndexWriter.h"

#include "velox/dwio/nimble/index/HashIndexConfig.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/ranges.h>

#include "flatbuffers/flatbuffers.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/BloomFilter.h"
#include "velox/dwio/nimble/index/HashIndexUtils.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/tablet/HashIndexGenerated.h"

#include "velox/common/base/BitUtil.h"

namespace facebook::nimble::index {

// static
// Estimates the serialized size of one bucket's keys.
// Accounts for encoded key strings + row numbers + bucket directory overhead.
size_t HashIndexWriter::estimateBucketSize(
    const std::vector<uint32_t>& keyIndices,
    const IndexAccumulator& accumulator) {
  // Per-bucket overhead: bucket_offsets (uint32).
  size_t size = sizeof(uint32_t);
  for (const auto idx : keyIndices) {
    // Per-key: row_numbers (uint32) + encoded key bytes + FlatBuffer string
    // overhead (length prefix + padding). Conservative estimate.
    size += sizeof(uint32_t) + accumulator.entries[idx].key.size() + 8;
  }
  return size;
}

// static
// Builds a HashIndexPartition FlatBuffer for a range of buckets.
void HashIndexWriter::buildIndexPartitionFlatBuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<std::vector<uint32_t>>& keyIndices,
    const IndexAccumulator& accumulator,
    uint32_t startBucket,
    uint32_t bucketCount) {
  // bucketCount + 1 offsets: offsets[i] is start, offsets[i+1] is end
  // (exclusive) for bucket i. The last element is the total entry count.
  std::vector<uint32_t> bucketOffsets(bucketCount + 1);
  std::vector<flatbuffers::Offset<flatbuffers::String>> keys;
  std::vector<uint32_t> rows;

  uint32_t rowOffset = 0;
  for (uint32_t i = 0; i < bucketCount; ++i) {
    bucketOffsets[i] = rowOffset;
    const auto& indices = keyIndices[startBucket + i];
    for (const auto idx : indices) {
      const auto& entry = accumulator.entries[idx];
      keys.emplace_back(
          builder.CreateString(entry.key.data(), entry.key.size()));
      rows.emplace_back(entry.row);
    }
    rowOffset += indices.size();
  }
  bucketOffsets[bucketCount] = rowOffset;

  builder.Finish(
      serialization::CreateHashIndexPartition(
          builder,
          builder.CreateVector(bucketOffsets),
          builder.CreateVector(keys),
          builder.CreateVector(rows)));
}

flatbuffers::Offset<serialization::BloomFilter>
HashIndexWriter::buildBloomFilter(
    flatbuffers::FlatBufferBuilder& builder,
    const IndexAccumulator& accumulator) const {
  if (!accumulator.options.bloomFilterBitsPerKey.has_value()) {
    return 0;
  }
  const auto bitsPerKey = accumulator.options.bloomFilterBitsPerKey.value();
  BloomFilter bloomFilter(accumulator.entries.size(), bitsPerKey, pool_);
  for (const auto& entry : accumulator.entries) {
    bloomFilter.insert(entry.key);
  }
  auto dataVec =
      builder.CreateVector(bloomFilter.data(), bloomFilter.dataSize());
  return serialization::CreateBloomFilter(
      builder, bloomFilter.numBlocks(), bitsPerKey, dataVec);
}

void HashIndexWriter::buildIndexFlatBuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const IndexAccumulator& accumulator,
    const CreateMetadataSectionFn& createMetadataFn) const {
  NIMBLE_CHECK(!accumulator.entries.empty());
  const auto minIt = std::min_element(
      accumulator.entries.begin(),
      accumulator.entries.end(),
      [](const auto& a, const auto& b) { return a.key < b.key; });
  const auto maxIt = std::max_element(
      accumulator.entries.begin(),
      accumulator.entries.end(),
      [](const auto& a, const auto& b) { return a.key < b.key; });

  const uint64_t numKeys = accumulator.entries.size();
  const uint32_t numBuckets = static_cast<uint32_t>(velox::bits::nextPowerOfTwo(
      std::max(
          1u,
          static_cast<uint32_t>(std::ceil(
              static_cast<float>(numKeys) / accumulator.options.loadFactor)))));
  const uint32_t bucketMask = numBuckets - 1;

  // Assign keys to buckets.
  std::vector<std::vector<uint32_t>> bucketKeyIndices(numBuckets);
  for (uint64_t i = 0; i < numKeys; ++i) {
    const uint32_t bucket = bucketIndex(accumulator.entries[i].key, bucketMask);
    bucketKeyIndices[bucket].emplace_back(i);
  }

  const auto bloomFilterOffset = buildBloomFilter(builder, accumulator);

  const float actualLoadFactor =
      static_cast<float>(numKeys) / static_cast<float>(numBuckets);

  // Determine partition boundaries. Always create at least one partition.
  // When maxPartitionSizeBytes is set and data exceeds it, split into
  // multiple partitions for on-demand loading.
  uint32_t bucketsPerPartition = numBuckets;
  if (accumulator.options.maxPartitionSizeBytes > 0 && numBuckets > 1) {
    size_t estimatedTotalBucketSize = 0;
    for (uint32_t i = 0; i < numBuckets; ++i) {
      estimatedTotalBucketSize +=
          estimateBucketSize(bucketKeyIndices[i], accumulator);
    }
    if (estimatedTotalBucketSize > accumulator.options.maxPartitionSizeBytes) {
      const uint32_t numPartitions =
          static_cast<uint32_t>(velox::bits::divRoundUp(
              estimatedTotalBucketSize,
              accumulator.options.maxPartitionSizeBytes));
      bucketsPerPartition = velox::bits::divRoundUp(numBuckets, numPartitions);
    }
  }

  std::vector<uint32_t> partitionStartBuckets;
  std::vector<flatbuffers::Offset<serialization::MetadataSection>>
      partitionSections;

  for (uint32_t startBucket = 0; startBucket < numBuckets;
       startBucket += bucketsPerPartition) {
    const uint32_t endBucket =
        std::min(startBucket + bucketsPerPartition, numBuckets);
    const uint32_t bucketCount = endBucket - startBucket;

    flatbuffers::FlatBufferBuilder partitionBuilder;
    buildIndexPartitionFlatBuffer(
        partitionBuilder,
        bucketKeyIndices,
        accumulator,
        startBucket,
        bucketCount);
    const auto partitionSection =
        createMetadataFn(asStringView(partitionBuilder));

    partitionStartBuckets.emplace_back(startBucket);
    partitionSections.emplace_back(
        serialization::CreateMetadataSection(
            builder,
            partitionSection.offset(),
            partitionSection.size(),
            static_cast<serialization::CompressionType>(
                partitionSection.compressionType())));
  }

  auto partitionStartBucketsVec = builder.CreateVector(partitionStartBuckets);
  auto partitionSectionsVec = builder.CreateVector(partitionSections);
  auto minKeyOffset =
      builder.CreateString(minIt->key.data(), minIt->key.size());
  auto maxKeyOffset =
      builder.CreateString(maxIt->key.data(), maxIt->key.size());

  auto hashIndex = serialization::CreateHashIndex(
      builder,
      numKeys,
      numBuckets,
      numKeys,
      actualLoadFactor,
      bloomFilterOffset,
      minKeyOffset,
      maxKeyOffset,
      partitionStartBucketsVec,
      partitionSectionsVec);
  builder.Finish(hashIndex);
}

void HashIndexWriter::buildIndexDirectoryFlatBuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<MetadataSection>& indexSections) const {
  NIMBLE_CHECK(!accumulators_.empty());
  NIMBLE_CHECK_EQ(indexSections.size(), accumulators_.size());

  std::vector<flatbuffers::Offset<serialization::HashIndexSection>>
      sectionOffsets;
  sectionOffsets.reserve(accumulators_.size());

  for (size_t i = 0; i < accumulators_.size(); ++i) {
    // Build column names for the directory entry.
    std::vector<flatbuffers::Offset<flatbuffers::String>> columnOffsets;
    columnOffsets.reserve(accumulators_[i].options.columns.size());
    for (const auto& col : accumulators_[i].options.columns) {
      columnOffsets.emplace_back(builder.CreateString(col));
    }
    auto columnsVec = builder.CreateVector(columnOffsets);

    // Build MetadataSection reference.
    const auto& section = indexSections[i];
    auto metadataSection = serialization::CreateMetadataSection(
        builder,
        section.offset(),
        section.size(),
        static_cast<serialization::CompressionType>(section.compressionType()));

    sectionOffsets.emplace_back(
        serialization::CreateHashIndexSection(
            builder, columnsVec, metadataSection));
  }

  auto indicesVec = builder.CreateVector(sectionOffsets);
  auto directory = serialization::CreateHashIndexDirectory(builder, indicesVec);
  builder.Finish(directory);
}

HashIndexWriter::Options HashIndexWriter::makeOptions(
    const IndexConfig& config) {
  const auto& hashIndexConfig = checkedIndexConfig<HashIndexConfig>(config);
  auto options = Options{
      .columns = hashIndexConfig.columns,
      .loadFactor = hashIndexConfig.loadFactor,
      .bloomFilterBitsPerKey = hashIndexConfig.bloomFilter.has_value()
          ? std::optional<float>{hashIndexConfig.bloomFilter->bitsPerKey}
          : std::nullopt,
      .maxPartitionSizeBytes = hashIndexConfig.maxPartitionSizeBytes,
  };
  NIMBLE_USER_CHECK(
      std::isfinite(options.loadFactor) && options.loadFactor > 0 &&
          options.loadFactor <= 1,
      "Hash index load factor must be finite and in (0, 1], but got: {}",
      options.loadFactor);
  NIMBLE_USER_CHECK(
      !options.columns.empty(), "Hash index must have at least one column");
  NIMBLE_USER_CHECK(
      !options.bloomFilterBitsPerKey.has_value() ||
          (std::isfinite(options.bloomFilterBitsPerKey.value()) &&
           options.bloomFilterBitsPerKey.value() > 0),
      "Bloom filter bits per key must be finite and positive, but got: {}",
      options.bloomFilterBitsPerKey.value_or(0));
  return options;
}

std::unique_ptr<HashIndexWriter> HashIndexWriter::create(
    std::span<const IndexConfig*> configs,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) {
  if (configs.empty()) {
    return nullptr;
  }
  NIMBLE_CHECK_NOT_NULL(pool, "memory pool must not be null");
  NIMBLE_CHECK_NOT_NULL(configs.front());
  const auto indexName = configs.front()->name;
  NIMBLE_USER_CHECK_EQ(
      indexName,
      kDenseHashIndexName,
      "Hash index writer must use the built-in hash index name");
  std::vector<Options> options;
  options.reserve(configs.size());
  for (const auto* config : configs) {
    NIMBLE_CHECK_NOT_NULL(config);
    NIMBLE_USER_CHECK_EQ(config->family, IndexFamily::Dense);
    NIMBLE_USER_CHECK_EQ(config->name, indexName);
    options.emplace_back(makeOptions(*config));
  }
  return std::unique_ptr<HashIndexWriter>(new HashIndexWriter(
      indexName, velox::asRowType(inputType), std::move(options), pool));
}

std::vector<std::vector<std::string>> HashIndexWriter::extractColumnSets(
    const std::vector<Options>& options) {
  std::vector<std::vector<std::string>> columnSets;
  columnSets.reserve(options.size());
  for (const auto& option : options) {
    columnSets.emplace_back(option.columns);
  }
  return columnSets;
}

HashIndexWriter::HashIndexWriter(
    std::string indexName,
    const velox::RowTypePtr& inputType,
    std::vector<Options> options,
    velox::memory::MemoryPool* pool)
    : indexName_{std::move(indexName)},
      pool_{pool},
      keyColumnIndices_{
          getKeyColumnIndices(extractColumnSets(options), inputType)} {
  NIMBLE_CHECK(!options.empty(), "Hash index configs must not be empty");
  accumulators_.reserve(options.size());
  for (auto& option : options) {
    NIMBLE_USER_CHECK(
        !option.columns.empty(), "Hash index must have at least one column");
    // Check for duplicate index columns.
    for (const auto& accumulator : accumulators_) {
      NIMBLE_USER_CHECK(
          accumulator.options.columns != option.columns,
          "Duplicate hash index columns: [{}]",
          fmt::join(option.columns, ", "));
    }
    accumulators_.emplace_back(
        IndexAccumulator{
            .options = std::move(option),
        });
    auto& accumulator = accumulators_.back();
    accumulator.encoder = createNimbleIndexKeyEncoder(
        accumulator.options.columns,
        inputType,
        std::vector<SortOrder>(
            accumulator.options.columns.size(), SortOrder{.ascending = true}),
        pool);
  }
}

void HashIndexWriter::write(const velox::VectorPtr& input) {
  checkNotClosed();
  if (input->size() == 0) {
    return;
  }

  NIMBLE_USER_CHECK_LE(
      numRows_ + static_cast<uint64_t>(input->size()),
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
      "Hash index row count exceeds uint32 limit");

  validateNoNullKeys(input, keyColumnIndices_);
  ensureEncodingBuffer();

  for (auto& accumulator : accumulators_) {
    // Encode keys from this batch.
    std::vector<std::string_view> keys;
    accumulator.encoder->encode(input, keys, [this](size_t size) {
      return encodingBuffer_->reserve(size);
    });
    NIMBLE_CHECK_EQ(keys.size(), input->size());

    // Record index entries. Encoded keys are string_views into
    // encodingBuffer_ which guarantees pointer stability.
    const auto newSize = accumulator.entries.size() + input->size();
    if (accumulator.entries.capacity() < newSize) {
      accumulator.entries.reserve(
          std::max(accumulator.entries.size() * 2, newSize));
    }
    for (velox::vector_size_t i = 0; i < input->size(); ++i) {
      accumulator.entries.emplace_back(IndexEntry{keys[i], numRows_ + i});
    }
  }

  numRows_ += input->size();
}

std::optional<IndexDescriptor> HashIndexWriter::close(
    const WriteDataFn& /*writeDataFn*/,
    const CreateMetadataSectionFn& createMetadataFn) {
  setClosed();

  if (numRows_ == 0) {
    return std::nullopt;
  }

  // Write each hash index as a separate section.
  std::vector<MetadataSection> indexSections;
  indexSections.reserve(accumulators_.size());
  for (auto& accumulator : accumulators_) {
    flatbuffers::FlatBufferBuilder indexBuilder;
    buildIndexFlatBuffer(indexBuilder, accumulator, createMetadataFn);
    indexSections.emplace_back(createMetadataFn(asStringView(indexBuilder)));
  }

  // Free encoded keys now that they've been serialized into FlatBuffers.
  encodingBuffer_.reset();

  flatbuffers::FlatBufferBuilder directoryBuilder;
  buildIndexDirectoryFlatBuffer(directoryBuilder, indexSections);

  for (auto& accumulator : accumulators_) {
    accumulator.clear();
  }

  return IndexDescriptor{
      .family = IndexFamily::Dense,
      .name = indexName_,
      .root = createMetadataFn(asStringView(directoryBuilder))};
}

} // namespace facebook::nimble::index
