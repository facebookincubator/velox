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

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "folly/container/F14Map.h"
#include "folly/io/IOBuf.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble::serde {

/// Produces compact serialized row-range slices from serialized Nimble
/// payloads.
///
/// The slicer rewrites stream metadata and copies only the encoded byte ranges
/// needed for the requested top-level rows. It preserves the schema while
/// preparing compact server-side payloads for wire transfer without
/// materializing values into Velox vectors.
class StreamSlicer {
 public:
  /// Controls raw-stream input and sliced output formats.
  struct Options {
    /// Encoding used for output stream indices.
    EncodingType streamIndicesEncodingType{EncodingType::FixedBitWidth};
    /// Encoding used for output stream sizes.
    EncodingType streamSizesEncodingType{EncodingType::FixedBitWidth};
    /// Serialization format used by each raw input stream.
    SerializationVersion streamVersion{SerializationVersion::kProjection};
    /// Whether each raw input stream retains tablet storage chunk framing.
    bool streamHasChunkHeader{false};
    /// Whether encoding prefixes store row counts as varints.
    bool streamsUseVarintRowCount{true};
  };

  /// Creates a slicer for payloads that use the supplied Nimble schema.
  StreamSlicer(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool,
      Options options);

  /// Returns a compact serialization containing rows [offset, offset + length).
  folly::IOBuf slice(std::string_view input, uint32_t offset, uint32_t length)
      const;

  /// Sliced stream-set result.
  struct SlicedStreams {
    /// Owns the encoded bytes referenced by streams.
    folly::IOBuf data;

    /// Views ordered by stream offset, backed by data.
    std::vector<std::string_view> streams;

    /// Indicates whether the stream set needs a row null-barrier on read.
    bool requiresNullBarrier{false};
  };

  /// Returns compact raw streams containing rows [offset, offset + length).
  /// The raw-stream input format is configured in Options. The returned
  /// streams never contain chunk headers.
  SlicedStreams slice(
      const std::vector<std::string_view>& inputStreams,
      uint32_t offset,
      uint32_t length) const;

 private:
  // Top-level row range in the current stream's row domain.
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  // Recursively slices all streams reachable from a schema node.
  void sliceType(
      const Type& type,
      Range range,
      const std::vector<std::string_view>& inputStreams,
      SlicedStreams& outputStreams,
      Buffer& outputBuffer,
      const Encoding::Options& encodingOptions) const;

  // Slices one stream descriptor and propagates null-barrier requirements.
  void sliceDescriptor(
      const StreamDescriptor& descriptor,
      Range range,
      const std::vector<std::string_view>& inputStreams,
      bool isRowOrFlatMapNullStream,
      SlicedStreams& outputStreams,
      Buffer& outputBuffer,
      const Encoding::Options& encodingOptions) const;

  // Returns sliced stream views backed by an owned output buffer.
  SlicedStreams sliceStreams(
      const std::vector<std::string_view>& inputStreams,
      Range range,
      Buffer& outputBuffer,
      const Encoding::Options& encodingOptions) const;

  // Returns true when the descriptor points to a present, non-empty stream.
  bool hasStream(
      const std::vector<std::string_view>& inputStreams,
      const StreamDescriptor& descriptor) const;

  // Returns true when a FlatMap child has any encoded value stream.
  bool hasFlatMapValues(
      const Type& type,
      const std::vector<std::string_view>& inputStreams) const;

  // Hashes stream views by backing storage identity, not by contents.
  struct StreamViewIdentityHash {
    size_t operator()(std::string_view stream) const {
      const auto dataHash = std::hash<const void*>{}(stream.data());
      const auto sizeHash = std::hash<size_t>{}(stream.size());
      return dataHash ^
          (sizeHash + 0x9e3779b9 + (dataHash << 6) + (dataHash >> 2));
    }
  };

  // Compares stream views by backing pointer and length.
  struct StreamViewIdentityEqual {
    bool operator()(std::string_view lhs, std::string_view rhs) const {
      return lhs.data() == rhs.data() && lhs.size() == rhs.size();
    }
  };

  using StrippedStreamCache = folly::F14FastMap<
      std::string_view,
      std::string_view,
      StreamViewIdentityHash,
      StreamViewIdentityEqual>;

  // Removes tablet chunk headers from the input stream set into
  // strippedStreams. Reuses stripped views for aliased physical stream bytes
  // within one slice.
  static void stripChunkHeaders(
      const std::vector<std::string_view>& inputStreams,
      Buffer& strippedStreamBuffer,
      std::vector<std::string_view>& strippedStreams,
      StrippedStreamCache& strippedStreamCache);

  // Maps a nullable stream range to its non-null child-value range.
  Range nonNullRange(
      std::string_view encoded,
      Range range,
      const Encoding::Options& encodingOptions) const;

  // Maps a boolean stream range to the range covering true positions.
  Range trueRange(
      std::string_view encoded,
      Range range,
      const Encoding::Options& encodingOptions) const;

  // Maps an offsets stream range to the child-element range it references.
  Range offsetsRange(
      std::string_view encoded,
      Range range,
      const Encoding::Options& encodingOptions) const;

  // Counts non-null rows in the range.
  uint32_t countNonNull(
      std::string_view encoded,
      Range range,
      const Encoding::Options& encodingOptions) const;

  // Counts true values in the range.
  uint32_t countTrue(
      std::string_view encoded,
      Range range,
      const Encoding::Options& encodingOptions) const;

  const std::shared_ptr<const Type> schema_;
  velox::memory::MemoryPool* const pool_;
  const Options options_;
  const uint32_t streamCount_;
  // Scratch Velox buffers reused by temporary Vector materialization.
  mutable velox::BufferPool bufferPool_;
  // Scratch arenas reused by nested EncodingFactory::slice() calls.
  mutable EncodingBufferPool encodingBufferPool_;
  // Scratch arena used while removing tablet chunk headers before slicing.
  mutable EncodingBufferPool strippedStreamBufferPool_;
  // Per-call cache for duplicate projected streams backed by the same bytes.
  mutable StrippedStreamCache strippedStreamCache_;
  mutable std::vector<std::string_view> inputStreams_;
  mutable std::vector<uint32_t> streamSizes_;
  mutable Vector<char> headerBuffer_;
  mutable Vector<char> trailerBuffer_;
};

} // namespace facebook::nimble::serde
