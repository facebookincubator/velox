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

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/index/ClusterIndexFactory.h"

namespace facebook::nimble::index {

/// Defines the keys of a hierarchical cluster index: the key that the values of
/// the levels of a row make, where a seek to a key starts, and the order of the
/// keys. The writer of the index bounds its partitions and chunks with these
/// keys, and its readers and key encoder take and return them. The built-in
/// index uses Nimble's key encoding: for each level, a zero null indicator and
/// the value big-endian, which sorts bytewise. An index over the keys of
/// another store, such as RocksDB internal keys, gives its factory a codec of
/// that format.
///
/// For the levels [7, 1] of widths [4, 1], Nimble's key is 00 00000007 00 01.
class HierarchicalKeyCodec {
 public:
  virtual ~HierarchicalKeyCodec() = default;

  /// Returns the size of a key whose levels are 'levelWidths' bytes wide.
  virtual size_t keySize(std::span<const uint8_t> levelWidths) const = 0;

  /// Writes the key of 'levels', the values of the levels of a row, whose
  /// widths are 'levelWidths', to the keySize(levelWidths) bytes at 'key'.
  virtual void encodeKey(
      std::span<const uint64_t> levels,
      std::span<const uint8_t> levelWidths,
      char* key) const = 0;

  /// Returns the key of 'levels', whose widths are 'levelWidths'.
  std::string encodeKey(
      std::span<const uint64_t> levels,
      std::span<const uint8_t> levelWidths) const;

  /// Writes the keys of a batch of rows. 'levels' holds the values of the
  /// levels of each row after those of the row before, and each key takes
  /// keySize(levelWidths) bytes of 'keys' after the key before. A codec
  /// overrides it to write a batch without a call per row.
  virtual void encodeKeys(
      std::span<const uint64_t> levels,
      std::span<const uint8_t> levelWidths,
      char* keys) const;

  /// Stores in 'levels' where a seek to 'target' starts among keys whose
  /// levels are 'levelWidths' bytes wide, and returns whether the seek lands
  /// past the keys equal to 'levels' rather than on the first of them. A target
  /// that is no key of the index, such as a prefix of one, starts at the first
  /// key that sorts at or after it.
  virtual bool parseKey(
      std::string_view target,
      std::span<const uint8_t> levelWidths,
      std::span<uint64_t> levels) const = 0;

  /// Returns the order of the keys, or null when they sort bytewise. The
  /// comparator lives as long as the codec.
  virtual const KeyComparator* keyComparator() const {
    return nullptr;
  }
};

/// Creates the writers, readers and key encoders of a hierarchical cluster
/// index, whose keys its codec defines. An index over keys of another format
/// registers a factory of its codec under its own name.
class NimbleHierarchicalClusterIndexFactory : public ClusterIndexFactory {
 public:
  /// Creates the factory of the built-in index, kHierarchicalClusterIndexName,
  /// whose codec is Nimble's key encoding. It takes only complete keys.
  NimbleHierarchicalClusterIndexFactory();

  /// Creates the factory of the index named 'name', whose keys 'keyCodec'
  /// defines.
  explicit NimbleHierarchicalClusterIndexFactory(
      std::string name,
      std::shared_ptr<const HierarchicalKeyCodec> keyCodec);

  std::string_view name() const override;

  std::unique_ptr<IndexWriter> createWriter(
      const IndexConfig& config,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override;

  /// Returns a key encoder that turns the values of the key columns of a row
  /// into the values of the levels, as the writer of the index does, and writes
  /// their key with the codec. Like the Velox key encoder, it encodes lookup
  /// bounds as an inclusive lower key and an exclusive upper key: an exclusive
  /// lower bound and an inclusive upper bound move to the next values of the
  /// levels, in the order of the index. An inclusive upper bound with no next
  /// values has no upper key.
  ///
  /// For a BIGINT key column, ascending, the values 5 and 6 are the levels
  /// 5 ^ 2^63 and 6 ^ 2^63, and an exclusive lower bound of 5 is the key of 6.
  std::unique_ptr<IndexKeyEncoder> createKeyEncoder(
      const std::vector<std::string>& columns,
      const velox::RowTypePtr& inputType,
      const std::vector<SortOrder>& sortOrders,
      velox::memory::MemoryPool* pool) const override;

  std::unique_ptr<ClusterIndexBase> createReader(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const IndexLookup::Options& options) const override;

  /// Returns the order of the keys of the codec.
  const KeyComparator* keyComparator() const override;

  /// Returns the codec of the index. Test-only.
  const std::shared_ptr<const HierarchicalKeyCodec>& testingKeyCodec() const;

 private:
  // Names the index in the registry and in the index descriptors of files.
  const std::string name_;
  // Defines the keys that the writer, the readers and the key encoder of the
  // index take and return.
  const std::shared_ptr<const HierarchicalKeyCodec> keyCodec_;
};

} // namespace facebook::nimble::index
