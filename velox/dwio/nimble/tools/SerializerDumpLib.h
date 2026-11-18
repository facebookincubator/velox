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
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/StreamLabels.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/serializer/Options.h"

namespace facebook::nimble::tools {

/// Analyzes nimble serialized buffers and aggregates per-stream encoding
/// statistics across multiple buffers. Uses a pre-loaded nimble schema for
/// stream labeling (e.g., flat map key names).
/// Callers provide serialized buffers one at a time via addSerialization(),
/// or in batch via addSerializations().
class SerializationDump {
 public:
  /// Per-stream encoding statistics extracted from a nimble serialized buffer.
  struct StreamStats {
    /// Zero-based index of this stream in the serialized buffer.
    uint32_t streamIndex{0};
    /// Encoding type used to encode this stream (e.g., RLE, Dictionary).
    EncodingType encodingType{};
    /// Data type of the elements in this stream.
    DataType dataType{};
    /// Compression type applied to this stream's encoded data.
    CompressionType compressionType{CompressionType::Uncompressed};
    /// Number of rows (elements) in this stream.
    uint32_t rowCount{0};
    /// Uncompressed size in bytes (rowCount * element size).
    uint64_t rawSize{0};
    /// Encoded (on-disk) size in bytes.
    uint64_t encodedSize{0};
    /// Human-readable schema label (e.g., "Scalar (trait_name)").
    std::string schemaLabel;
    /// True if the stream has zero encoded bytes (no data written).
    bool empty{false};
  };

  /// Aggregated analysis of a nimble serialized buffer.
  struct SerializationStats {
    /// Serialization format version.
    SerializationVersion version{};
    /// Total number of rows across all analyzed buffers.
    uint32_t rowCount{0};
    /// Per-stream statistics.
    std::vector<StreamStats> streams;
    /// Total uncompressed size in bytes across all streams.
    uint64_t rawSize{0};
    /// Total encoded (on-disk) size in bytes across all streams.
    uint64_t encodedSize{0};

    /// Returns a human-readable string representation of the stats.
    std::string toString() const;
  };

  /// Creates a SerializationDump that uses the given nimble schema for stream
  /// labeling.
  ///
  /// @param schema Nimble schema with flat map keys for stream labeling.
  /// @param pool Memory pool for stats parsing.
  SerializationDump(
      std::shared_ptr<const Type> schema,
      velox::memory::MemoryPool* pool);

  /// Analyze a single serialized buffer and accumulate aggregate stats.
  ///
  /// @param serialized The raw serialized buffer to parse for encoding stats.
  void addSerialization(std::string_view serialized);

  /// Analyze multiple serialized buffers and accumulate aggregate stats.
  ///
  /// @param serializations The raw serialized buffers to parse.
  void addSerializations(const std::vector<std::string_view>& serializations);

  /// Returns aggregated per-stream stats.
  const SerializationStats& stats() const {
    return stats_;
  }

  /// Reset all accumulated stats.
  void reset();

  /// Format aggregated stats as a human-readable string.
  std::string toString() const;

 private:
  /// Parses a nimble serialized buffer and returns per-stream encoding info.
  velox::memory::MemoryPool* const pool_;

  SerializationStats serializationStats(std::string_view serialized);

  StreamLabels streamLabels_;
  SerializationStats stats_;
};

} // namespace facebook::nimble::tools
