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
#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::nimble::serialization {
struct SortedIndex;
} // namespace facebook::nimble::serialization

namespace facebook::nimble::index {

class IndexConfig;

namespace test {
class SortedIndexWriterTestHelper;
} // namespace test

/// Builds one or more sorted indices during file write.
///
/// Each sorted index stores a sorted key stream where entries are
/// encoded_key || fixed_width_row_id. At close time, entries are sorted by
/// key, chunked, and encoded using the same pipeline as ClusterIndex
/// (EncodingFactory with Prefix/Trivial → ChunkedStreamWriter).
///
/// Supports both point lookups (binary search) and range scans on unsorted
/// data. Unlike HashIndex (point-only, O(1)), this provides O(log n) lookups
/// with range scan support.
class SortedIndexWriter : public IndexWriter {
 public:
  /// Factory method. Returns nullptr if configs is empty.
  static std::unique_ptr<SortedIndexWriter> create(
      std::span<const IndexConfig*> configs,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool);

  ~SortedIndexWriter() override = default;

  SortedIndexWriter(const SortedIndexWriter&) = delete;
  SortedIndexWriter& operator=(const SortedIndexWriter&) = delete;
  SortedIndexWriter(SortedIndexWriter&&) = delete;
  SortedIndexWriter& operator=(SortedIndexWriter&&) = delete;

  void write(const velox::VectorPtr& input) override;

  std::optional<IndexDescriptor> close(
      const WriteDataFn& writeDataFn,
      const CreateMetadataSectionFn& createMetadataFn) override;

 private:
  struct Options {
    std::vector<std::string> columns;
    EncodingLayout encodingLayout;
    uint64_t maxRowsPerKeyChunk;
  };

  static Options makeOptions(const IndexConfig& config);
  static std::vector<std::vector<std::string>> extractColumnSets(
      const std::vector<Options>& options);

  SortedIndexWriter(
      std::string indexName,
      const velox::RowTypePtr& inputType,
      std::vector<Options> options,
      velox::memory::MemoryPool* pool);

  struct IndexEntry {
    std::string_view key;
    uint32_t row;
  };

  struct IndexAccumulator {
    Options options;
    std::unique_ptr<IndexKeyEncoder> encoder{};
    std::vector<IndexEntry> entries{};
    // Backs encoded key string_views in entries. Each accumulator owns its
    // buffer so it can be freed independently after composite entries are
    // built.
    std::unique_ptr<Buffer> encodingBuffer{};

    void sort();
    void clear();
  };

  // Sorted composite entries with backing buffer.
  struct CompositeEntries {
    // Contiguous buffer backing all entry string_views.
    velox::BufferPtr buffer;
    // Views into buffer, one per entry: encoded_key || big_endian_row_id.
    std::vector<std::string_view> entries;
  };

  // Determines the row ID width in bytes for the given row count.
  static uint8_t rowIdWidth(uint32_t numRows);

  // Extracts the key prefix (without row ID suffix) from a composite entry.
  static std::string_view extractKey(
      std::string_view compositeEntry,
      uint8_t idWidth) {
    return compositeEntry.substr(0, compositeEntry.size() - idWidth);
  }

  // Builds sorted composite entries from the accumulator's index entries.
  // Each composite entry is encoded_key || big_endian_row_id packed into a
  // single contiguous buffer.
  CompositeEntries buildCompositeEntries(
      const std::vector<IndexEntry>& entries,
      uint8_t idWidth) const;

  // Builds a single sorted index and writes its key stream data to the file.
  void buildIndex(
      IndexAccumulator& accumulator,
      const WriteDataFn& writeDataFn,
      flatbuffers::FlatBufferBuilder& builder);

  // Builds the directory FlatBuffer with all index descriptors.
  void buildDirectoryFlatBuffer(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<MetadataSection>& indexSections) const;

  static std::string_view asStringView(
      const flatbuffers::FlatBufferBuilder& builder) {
    return {
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize()};
  }

  const std::string indexName_;
  velox::memory::MemoryPool* const pool_;
  const std::vector<velox::column_index_t> keyColumnIndices_;
  std::vector<IndexAccumulator> accumulators_;
  uint32_t numRows_{0};

  friend class test::SortedIndexWriterTestHelper;
};

} // namespace facebook::nimble::index
