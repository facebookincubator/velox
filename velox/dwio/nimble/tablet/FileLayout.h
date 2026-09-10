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

#include <string_view>
#include <unordered_map>
#include <vector>
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/dwio/nimble/tablet/Postscript.h"

namespace facebook::nimble {

/// Nimble Physical File Layout
/// ============================
///
/// A Nimble file has the following physical structure:
///
/// +===================================================================+
/// |                         STRIPE DATA                               |
/// +===================================================================+
/// |  Stripe 0 Streams (belongs to Stripe Group 0)                     |
/// |  +-------------------------------------------------------------+  |
/// |  | Stream 0 | Stream 1 | Stream 2 | ... | Stream N             |  |
/// |  +-------------------------------------------------------------+  |
/// +-------------------------------------------------------------------+
/// |  Stripe 1 Streams (belongs to Stripe Group 0)                     |
/// +-------------------------------------------------------------------+
/// |  ...                                                              |
/// +-------------------------------------------------------------------+
/// |  Stripe M Streams (last stripe in Stripe Group 0)                 |
/// +===================================================================+
/// |  Stripe Group 0 Metadata (stream_offsets, stream_sizes)           |
/// +-------------------------------------------------------------------+
/// |  Index Group 0 Metadata (if indexing enabled, written immediately |
/// |                          after its corresponding stripe group)    |
/// +===================================================================+
/// |  Stripe M+1 Streams (first stripe in Stripe Group 1)              |
/// +-------------------------------------------------------------------+
/// |  ...                                                              |
/// +===================================================================+
/// |  Stripe Group 1 Metadata                                          |
/// +-------------------------------------------------------------------+
/// |  Index Group 1 Metadata (if indexing enabled)                     |
/// +===================================================================+
/// |  ... (more stripe groups with their index groups)                 |
/// +===================================================================+
/// |                      GLOBAL METADATA                              |
/// +===================================================================+
/// |  Stripes Metadata (row_counts, offsets, sizes, group_indices)     |
/// +-------------------------------------------------------------------+
/// |  Optional: "columnar.indexes" (root index manifest with named     |
/// |            cluster and dense index payloads)                      |
/// +-------------------------------------------------------------------+
/// |  Optional: "columnar.chunk.stats" (root ChunkStats flatbuffer     |
/// |            with stripe_indexes)                                   |
/// +-------------------------------------------------------------------+
/// |  Optional: "schema", "stats", etc.                                |
/// +===================================================================+
/// |                          FOOTER                                   |
/// +===================================================================+
/// |  Footer (row_count, stripes ref, stripe_groups refs,              |
/// |          optional_sections refs)                                  |
/// +===================================================================+
/// |                     POSTSCRIPT (20 bytes)                         |
/// +===================================================================+
/// |  Footer Size (4B) | Compression (1B) | Checksum Type (1B)         |
/// |  Checksum (8B) | Major (2B) | Minor (2B) | Magic "NI" (2B)        |
/// +===================================================================+
///
/// Key Layout Properties:
/// ----------------------
/// Stripe Groups: Each stripe group contains metadata for a contiguous
/// range of stripes. Written after all stripes in the group.
///
/// Index Groups: When cluster indexing is enabled, each index group is
/// written IMMEDIATELY AFTER its corresponding stripe group. This
/// co-location enables efficient single IO to load both stripe and
/// index metadata.
///
/// Index Manifest ("columnar.indexes"):
///   Root IndexRoot flatbuffer (see Index.fbs) containing named index payloads.
///
/// Cluster index payload (see ClusterIndex.fbs) contains:
///   - stripe_keys: key boundaries per stripe (size = stripe_count + 1)
///   - index_columns: names of indexed columns
///   - sort_orders: sort order per column ("ASC NULLS FIRST", etc.)
///   - stripe_indexes: refs to per-group StripeClusterIndex metadata
///
///   ClusterIndexPartition (per stripe group / partition):
///   +---------------------------------------------------------------+
///   | key_data_offset: file offset of key data blob                  |
///   | key_data_size: total size of key data blob                     |
///   | SubPartitionIndex (null if single sub-partition):              |
///   |   - sub_partition_rows: accumulated rows per sub-partition     |
///   |   - sub_partition_offsets: byte offset per sub-partition       |
///   |   - sub_partition_keys: last key value per sub-partition       |
///   | stripe_row_counts: row count per stripe                        |
///   +---------------------------------------------------------------+
///
/// Chunk Stats ("columnar.chunk.stats"):
///   Root ChunkStats flatbuffer (see ChunkStats.fbs) containing:
///   - stripe_indexes: refs to per-group StripeChunkStats metadata
///
///   StripeChunkStats (per stripe group):
///   +---------------------------------------------------------------+
///   | stream_count: total number of indexed streams                   |
///   | stream_chunk_counts: accumulated chunks per (stripe, stream)   |
///   | stream_chunk_rows: accumulated rows per chunk                  |
///   | stream_chunk_offsets: byte offset per chunk                    |
///   +---------------------------------------------------------------+

/// Describes the physical layout of a Nimble file, including offsets and sizes
/// of the footer and metadata sections. This information can be used by readers
/// to optimize IO by skipping the discovery phase when layout is known.
struct FileLayout {
  /// Per-stripe layout information.
  struct StripeInfo {
    /// Offset from start of file to first byte of stripe data.
    uint64_t offset{0};
    /// Total size of stripe data in bytes (sum of all stream sizes).
    uint64_t size{0};
    /// Index of the stripe group this stripe belongs to.
    uint32_t stripeGroupIndex{0};
  };

  /// Total file size in bytes.
  uint64_t fileSize{0};

  /// Postscript information (parsed from last 20 bytes of file).
  nimble::Postscript postscript;

  /// Footer section location and compression.
  MetadataSection footer;

  /// Stripes metadata section location and compression.
  /// Contains per-stripe info: row counts, file offsets, sizes, group indices.
  /// Only valid when stripeGroups is not empty.
  MetadataSection stripes;

  /// Stripe group metadata sections (per-stream offsets/sizes within stripes).
  std::vector<MetadataSection> stripeGroups;

  /// Index partition metadata sections (empty if indexing not enabled).
  std::vector<MetadataSection> indexPartitions;

  /// Per-stripe layout information.
  std::vector<StripeInfo> stripesInfo;

  /// Optional sections (e.g., index, schema, column stats).
  std::unordered_map<std::string, MetadataSection> optionalSections;

  /// Reads file layout from a Nimble file.
  /// @param file The file to read layout from.
  /// @param pool Memory pool for allocations (must not be null).
  /// @return The file layout.
  static FileLayout create(
      std::shared_ptr<velox::ReadFile> file,
      velox::memory::MemoryPool* pool);

  /// Reads file layout from a Nimble file at the given path.
  /// @param path Path to the Nimble file.
  /// @param pool Memory pool for allocations (must not be null).
  /// @return The file layout.
  static FileLayout create(
      const std::string& path,
      velox::memory::MemoryPool* pool);
};

} // namespace facebook::nimble
