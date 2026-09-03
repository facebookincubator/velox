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

#include <fmt/format.h>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>

#include "folly/Executor.h"
#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"

#include <set>
#include "velox/type/Type.h"

namespace facebook::nimble {

/// Serialization format version.
///
/// - kLegacy: Simple zstd compression with inline u32 sizes.
///   Wire:
///   [version:1B][rowCount:u32][size_0:u32][stream_data_0]...[size_N:u32][stream_data_N]
///
/// - kLegacyCompact: READ-ONLY post the two-array sparse trailer change.
///   Existing production blobs at this version are decoded via the frozen
///   legacy reader (`nimble::serde::legacy::`), whose wire format is:
///   Wire: [version:1B][rowCount:varint][stream_data_0]...[stream_data_N]
///         [encodingByte:1B][denseSizes:u32[]|sparsePair]
///         [trailer_size:u32]
///   Serializer upgrades new kLegacyCompact writes to kSerialization.
///
/// - kLegacySerialization: READ-ONLY old encoded Serializer format. Uses the
///   two-array sparse trailer and [version:1B][rowCount:varint], without a
///   header flags byte. Serializer upgrades new writes to kSerialization.
///
/// - kSerialization / kProjection: Writer versions using the two-array sparse
///   trailer. Wire body identical to kLegacySerialization; trailer is:
///   [version:1B][rowCount:varint][flags:1B][stream_data...]
///   [indicesEncType:1B][indicesPayload]
///   [sizesEncType:1B][sizesPayload]
///   [trailer_size:u32]
///   trailer_size = 2 + len(indicesPayload) + len(sizesPayload).
///   Indices and sizes axes independently choose from EncodingTypes
///   {Trivial, Varint, Delta, FixedBitWidth}. See getTrailerEncodingType()
///   for validation. kSerialization is the Serializer writer version and
///   kProjection is the Projector writer version. Both carry a compact-header
///   flags byte after the row count.
///   The flags byte also records whether stream encoding row counts are
///   varint (default) or fixed u32. Fixed u32 is used when kProjection carries
///   stream bodies sliced from kTablet payloads.
///
/// - kTablet: Tablet stream passthrough format. Like kSerialization but:
///   (1) Chunk slice headers are:
///       [version:1B][rowCount:varint][flags:1B][rowRange/resumeKey...]
///   (2) Each stream's data includes tablet chunk headers:
///       [chunkSize:u32][compressionType:1B][encoded_data...]
///       which the Deserializer strips before decoding.
///   (3) Encoding headers within streams use fixed u32 row counts
///       (useVarintRowCount=false), matching the tablet's default format.
///   Wire: same two-array sparse trailer layout as kSerialization.
enum class SerializationVersion : uint8_t {
  kLegacy = 0,
  // READ-ONLY post the two-array trailer change: production blobs with this
  // version byte are still decoded via `nimble::serde::legacy::`. New
  // Serializer writes are upgraded to kSerialization; Projector writers emit
  // kProjection.
  kLegacyCompact = 2,
  // Source-level migration spelling for the old non-nullable encoded
  // serializer wire format. Serializer upgrades new writes to kSerialization.
  kLegacySerialization = 3,
  // Default writer version for the Serializer: null-capable two-array sparse
  // trailer with a compact-header flags byte.
  kSerialization = 4,
  // Default writer version for the Projector: two-array sparse trailer.
  // Same wire format as kSerialization today; the distinct version byte lets
  // the two writers' formats evolve independently in the future.
  kProjection = 5,
  kTablet = 6,
};

std::string toString(SerializationVersion version);

/// Returns true if the version is any non-legacy format. All non-legacy
/// formats have a version header, encoded streams, and a stream sizes trailer.
inline bool nonLegacyFormat(SerializationVersion version) {
  return version != SerializationVersion::kLegacy;
}

/// Returns true if the version is the tablet passthrough format.
inline bool isTabletVersion(SerializationVersion version) {
  return version == SerializationVersion::kTablet;
}

/// Returns true if the version was written using the legacy kLegacyCompact
/// trailer format (decoded via `nimble::serde::legacy::`). All other
/// non-legacy formats (kLegacySerialization, kTablet, kSerialization,
/// kProjection) use the new two-array sparse trailer.
///
/// kLegacyCompact is the only frozen version here because it is the only one
/// with existing production blobs that must keep using the legacy trailer
/// reader.
inline bool usesLegacyTrailer(SerializationVersion version) {
  return version == SerializationVersion::kLegacyCompact;
}

/// Returns true if the version carries serialization header flags.
inline bool hasSerializationHeaderFlags(SerializationVersion version) {
  return version == SerializationVersion::kSerialization ||
      version == SerializationVersion::kProjection ||
      version == SerializationVersion::kTablet;
}

/// Returns true if the optional version carries serialization header flags.
inline bool hasSerializationHeaderFlags(
    std::optional<SerializationVersion> version) {
  return version.has_value() && hasSerializationHeaderFlags(version.value());
}

/// Returns true if the version's compact header carries a flags byte after the
/// row count. kTablet carries its flag in the tablet chunk header instead;
/// kLegacy, the frozen kLegacyCompact, and kLegacySerialization have no flags
/// byte.
inline bool usesCompactHeaderFlags(SerializationVersion version) {
  return version == SerializationVersion::kSerialization ||
      version == SerializationVersion::kProjection;
}

/// Returns true if the optional version's compact header carries a flags byte.
inline bool usesCompactHeaderFlags(
    std::optional<SerializationVersion> version) {
  return version.has_value() && usesCompactHeaderFlags(version.value());
}

/// Returns true if the version is the tablet passthrough format.
inline bool isTabletVersion(std::optional<SerializationVersion> version) {
  return version.has_value() && isTabletVersion(version.value());
}

/// Returns true if the version uses varint for the header row count.
/// All non-legacy versioned formats (kLegacyCompact, kTablet) use varint row
/// counts in the header. The raw stream bodies inside kTablet may use fixed
/// u32 row counts in their encoding headers.
inline bool usesVarintRowCount(SerializationVersion version) {
  return version != SerializationVersion::kLegacy;
}

/// Returns true if the optional version uses varint for the header row count.
inline bool usesVarintRowCount(std::optional<SerializationVersion> version) {
  return version.has_value() && usesVarintRowCount(version.value());
}

/// Validates and returns the EncodingType for a stream-sizes trailer section
/// (used independently for both the indices and the sizes arrays).
/// Supported: Trivial, Varint, Delta, FixedBitWidth.
EncodingType getTrailerEncodingType(EncodingType encodingType);

inline std::ostream& operator<<(
    std::ostream& os,
    SerializationVersion version) {
  return os << toString(version);
}

/// Returns the default encoding selection policy creator. The underlying
/// ManualEncodingSelectionPolicyFactory (and its default read-factor vector) is
/// initialized once and shared across all returned creators.
EncodingSelectionPolicyCreator defaultEncodingSelectionPolicyCreator();

struct SerializerOptions {
  /// Legacy compression settings. These only apply when version is kLegacy or
  /// nullopt. Encoded formats handle compression through the encoding selection
  /// policy.
  CompressionType compressionType{CompressionType::Uncompressed};
  uint32_t compressionThreshold{0};
  int32_t compressionLevel{0};

  /// Serialization format version.
  /// - nullopt: Legacy format with no version header.
  /// - kSerialization: Encoded Serializer format.
  /// - kLegacy / kLegacyCompact / kLegacySerialization: Legacy spellings for
  ///   migration. Existing kLegacyCompact production blobs are decoded via
  ///   `nimble::serde::legacy::`; new Serializer writes are upgraded to
  ///   kSerialization while callers migrate.
  /// - kProjection / kTablet are not valid Serializer writer versions
  ///   (Projector + tablet pipelines write them, respectively).
  std::optional<SerializationVersion> version{};

  /// Columns that should be encoded as flat maps. Maps column name to a set
  /// of predefined key strings. When the set is empty, the column is
  /// treated as a flat map with dynamic key discovery. When non-empty, keys
  /// are predefined in sorted order to ensure all serializers produce
  /// identical schemas regardless of data arrival order. Unknown keys not in
  /// the set will cause an error during serialization.
  folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns{};

  /// Factory for creating encoding selection policies.
  /// Used by encoded serializer writes.
  /// When encodingLayoutTree is specified, used as fallback for streams or
  /// nested encodings not captured in the layout tree. When encodingLayoutTree
  /// is not specified, used directly for all streams.
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator =
      defaultEncodingSelectionPolicyCreator();

  /// Optional captured encoding layout tree.
  /// When specified, encodings are replayed from this tree instead of using
  /// encodingSelectionPolicyCreator. This speeds up writes by skipping runtime
  /// encoding selection and can provide better encodings based on historical
  /// data. Falls back to encodingSelectionPolicyCreator for streams not in the
  /// tree.
  std::optional<EncodingLayoutTree> encodingLayoutTree{};

  /// Compression options used with encodingLayoutTree.
  /// When encodingLayoutTree is specified, these options are passed to
  /// ReplayedEncodingSelectionPolicy to control compression during encoding
  /// replay. When nullopt (default), compression is disabled.
  std::optional<CompressionOptions> compressionOptions{};

  /// Encoding type for the indices array of the sparse stream-sizes trailer.
  /// Indices are the offsets of non-zero stream slots in scan order
  /// (sorted ascending). Supported types: Trivial, Varint, Delta,
  /// FixedBitWidth.
  EncodingType streamIndicesEncodingType{EncodingType::FixedBitWidth};

  /// Returns the default per-encoding options used by Serializer writes.
  static Encoding::Options defaultEncodingOptions() {
    Encoding::Options options;
    options.useVarintRowCount = true;
    return options;
  }

  /// Per-encoding options passed to EncodingFactory::encode(). Controls
  /// format-level settings (varint row count) and per-encoding config
  /// (e.g., BlockBitPacking block size).
  Encoding::Options encodingOptions{defaultEncodingOptions()};

  /// Maximum number of scratch buffers retained by the serializer-owned
  /// nested encoding buffer pool. 0 disables nested encoding buffer caching.
  /// Disabled by default; callers can set a non-zero value to opt in.
  uint32_t maxCachedNestedEncodingBuffers{0};

  /// Encoding type for the sizes array of the sparse stream-sizes trailer.
  /// Sizes are the byte sizes of non-zero streams, parallel to the indices
  /// array. Supported types: Trivial, Varint, Delta, FixedBitWidth.
  EncodingType streamSizesEncodingType{EncodingType::FixedBitWidth};

  /// Returns true if the serialized data has a version header byte.
  bool hasVersionHeader() const;

  /// Returns the effective serialization version.
  SerializationVersion serializationVersion() const;

  /// Returns true if nimble encoding is enabled.
  bool enableEncoding() const;
};

struct DeserializerOptions {
  /// Whether the serialized data has a header byte.
  /// - false (default): Legacy format (version 0) with no header.
  /// - true: Version is auto-detected from the first byte of serialized data.
  bool hasHeader{false};

  /// Output type for deserializing flatmap columns as struct (ROW).
  /// When provided, each top-level flatmap column whose corresponding field in
  /// outputType is ROW will be deserialized as a struct instead of a map. The
  /// ROW field names specify which flatmap keys to select and their order.
  /// Fields not present in the flatmap schema will be filled with nulls.
  /// When nullopt (default), all flatmap columns are deserialized as maps.
  velox::RowTypePtr outputType{};

  /// Maximum number of scratch buffers each per-stream BufferPool retains
  /// for reuse across encoding lifetimes. Higher values reduce MemoryPool
  /// allocation churn at the cost of resident memory. 0 disables the pool
  /// entirely (every encoding allocates and frees through MemoryPool).
  size_t bufferPoolCapacity{velox::BufferPool::kDefaultCapacity};

  /// Executor for parallel decoding of child fields.
  /// When set, RowFieldReader and StructFlatMapFieldReader dispatch child reads
  /// as coroutines to this executor, using co_await to yield threads back to
  /// the pool. This prevents deadlock from nested parallelism.
  /// When nullptr (default), child reads are performed sequentially.
  folly::Executor* decodeExecutor{nullptr};

  /// Maximum number of parallel coroutine tasks scheduled by each field
  /// reader. Children are grouped into this many batches, each decoded
  /// sequentially within a single coroutine task. This is a per-reader limit;
  /// the executor bounds the number of tasks that run concurrently across the
  /// reader tree. 0 disables parallel decoding.
  uint32_t maxDecodeParallelism{0};

  /// Minimum number of child streams per parallel decode task. Ensures each
  /// coroutine task has enough work to amortize threading overhead.
  uint32_t minStreamsPerDecodeTask{1};
};

} // namespace facebook::nimble

template <>
struct fmt::formatter<facebook::nimble::SerializationVersion>
    : fmt::formatter<std::string> {
  auto format(
      facebook::nimble::SerializationVersion version,
      format_context& ctx) const {
    return fmt::formatter<std::string>::format(
        facebook::nimble::toString(version), ctx);
  }
};
