/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <cuda_runtime.h>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include "velox/common/base/Exceptions.h"

/// Definitions needed for the Cudf exchange protocol.

namespace facebook::velox::cudf_exchange {

// Data and metadata tags are a uint64_t split into 3 fields, most-significant
// first:
// - Bits 63..32 (4 bytes): FNV-1a hash of the producing taskId, which is
//   unique within a cluster.
// - Bits 31..24 (1 byte): Operation type (metadata, data, or handshake
//   response).
// - Bits 23..0  (3 bytes): Sequence number of the chunk exchanged between 2
//   tasks.

// Definition of the operations.
constexpr uint64_t METADATA_TAG = 0x02000000;
constexpr uint64_t DATA_TAG = 0x03000000;
constexpr uint64_t HANDSHAKE_RESPONSE_TAG = 0x04000000;

// Implementation of the fowler-noll-vo hash function for 32 bits.
uint32_t fnv1a_32(const std::string& s);

// Gets the tag used for metadata communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getMetadataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | METADATA_TAG | sequenceNumber;
}

// Gets the tag used for data communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getDataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | DATA_TAG | sequenceNumber;
}

// Gets the tag used for handshake response communication.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getHandshakeResponseTag(uint64_t taskHash) {
  return (taskHash << 32) | HANDSHAKE_RESPONSE_TAG;
}

/// @brief Request that is sent from the client (CudfExchangeSource) to the
/// server (CudfExchangeServer) after connection.
///
/// The handshake establishes the partition key for data exchange.
/// The workerId identifies the source's Communicator instance (process).
/// If the server's workerId matches, both are in the same process, enabling
/// intra-node transfer via IntraNodeTransferRegistry instead of UCXX.
struct HandshakeMsg {
  char taskId[256];
  uint32_t destination;
  /// Unique identifier for the source's Communicator instance.
  /// Generated randomly at Communicator startup. The server compares this
  /// against its own workerId to detect same-process (intra-node) transfers.
  uint64_t workerId{0};
};

/// @brief Response sent from server to source after handshake.
/// Informs the source whether intra-node transfer optimization is available,
/// allowing the source to bypass UCXX for all subsequent data transfers.
struct HandshakeResponse {
  /// True if server and source are on the same node (same Communicator).
  /// When true, source should use IntraNodeTransferRegistry instead of UCXX.
  bool isIntraNodeTransfer{false};
  /// Padding for alignment
  uint8_t padding[7]{};
};

constexpr uint32_t kMagicNumber = 0x12345678;
/// Maximum metadata buffer size for receiving. This should be large enough
/// to handle tables with many columns. 1MB allows for ~10,000+ columns.
/// The sender allocates exact size needed; receiver pre-allocates this max.
constexpr uint32_t kMaxMetaBufSize = 1024 * 1024; // 1MB

/// Minimum header size needed to read the totalSize field.
/// Format: [magic (4 bytes)][totalSize (4 bytes)]
constexpr uint32_t kMetaHeaderSize = sizeof(kMagicNumber) + sizeof(uint32_t);

/// Wire-format types for MetadataMsg serialization. Using shared type aliases
/// ensures serialize() and deserializeMetadataMsg() agree on field widths.
using WireLengthType = uint64_t;
using WireDataSizeType = int64_t;
using WireRemainingElementType = int64_t;

struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  WireDataSizeType dataSizeBytes;
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd;

  uint32_t getSerializedSize() const {
    // The header: the magic number and the metadata length.
    uint32_t totalSize = sizeof(kMagicNumber) + sizeof(totalSize);
    // cudfMetadata: length info and then the data.
    WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
    totalSize += sizeof(cudfSize);
    totalSize += cudfSize;
    // dataSizeBytes
    totalSize += sizeof(dataSizeBytes);
    // remainingBytes: length and then the data.
    totalSize += sizeof(WireLengthType); // for numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);
    // atEnd, encoded in a byte.
    totalSize += sizeof(uint8_t);

    return totalSize;
  }

  // serializes the metadata record.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize() {
    uint32_t totalSize = getSerializedSize();

    // Validate that the serialized size fits in the maximum allowed buffer.
    // The receiver allocates kMaxMetaBufSize; sender must not exceed this.
    VELOX_CHECK_LE(
        totalSize,
        kMaxMetaBufSize,
        "Metadata serialized size ({}) exceeds maximum buffer size ({}). "
        "This can happen with extremely wide tables. "
        "Consider reducing table width or increasing kMaxMetaBufSize.",
        totalSize,
        kMaxMetaBufSize);

    // Allocate exact size needed - no wasted memory for small metadata.
    auto deleter = [](uint8_t* p) { delete[] p; };
    std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);

    uint8_t* ptr = buffer.get();

    // Write the magic number.
    std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
    ptr += sizeof(kMagicNumber);

    // Write the size of the metadata.
    std::memcpy(ptr, &totalSize, sizeof(totalSize));
    ptr += sizeof(totalSize);

    // Serialize cudfMetadata size.
    WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
    std::memcpy(ptr, &cudfSize, sizeof(cudfSize));
    ptr += sizeof(cudfSize);

    // If data exists, serialize each byte.
    if (cudfSize > 0) {
      std::memcpy(ptr, cudfMetadata->data(), cudfSize);
      ptr += cudfSize;
    }

    // Serialize dataSizeBytes.
    std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
    ptr += sizeof(dataSizeBytes);

    // Serialize number of remainingBytes elements.
    WireLengthType numRemaining = remainingBytes.size();
    std::memcpy(ptr, &numRemaining, sizeof(numRemaining));
    ptr += sizeof(numRemaining);

    // Serialize remainingBytes elements.
    if (numRemaining > 0) {
      auto bytesSize = numRemaining * sizeof(remainingBytes[0]);
      std::memcpy(ptr, remainingBytes.data(), bytesSize);
      ptr += bytesSize;
    }

    // Serialize atEnd bool as 0/1.
    uint8_t atEndByte = atEnd ? 1 : 0;
    *ptr = atEndByte;

    return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
        std::move(buffer), totalSize);
  }

  // Deserialization function that constructs a MetadataMsg from a
  // buffer that points to a serialized metadata.
  static MetadataMsg deserializeMetadataMsg(const uint8_t* buffer) {
    // We'll use a pointer to scan through the memory.
    const uint8_t* ptr = buffer;

    MetadataMsg record;

    // Extract magic number.
    uint32_t magicNumber = 0;
    std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
    VELOX_CHECK_EQ(magicNumber, kMagicNumber);
    ptr += sizeof(magicNumber);

    // Extract the total size.
    uint32_t totalSize = 0;
    std::memcpy(&totalSize, ptr, sizeof(totalSize));
    ptr += sizeof(totalSize);

    const uint8_t* endPtr = buffer + totalSize;

    // Deserialize cudfMetadata:
    // First read the size of the metadata.
    WireLengthType metaSize = 0;
    if (ptr + sizeof(metaSize) > endPtr)
      throw std::runtime_error("Insufficient data for cudfMetadata size");
    std::memcpy(&metaSize, ptr, sizeof(metaSize));
    ptr += sizeof(metaSize);

    // Allocate a vector of the correct size.
    record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metaSize);
    if (metaSize > 0) {
      if (ptr + metaSize > endPtr)
        throw std::runtime_error("Insufficient data for cudfMetadata bytes");
      std::memcpy(record.cudfMetadata->data(), ptr, metaSize);
      ptr += metaSize;
    }

    // Deserialize dataSizeBytes.
    if (ptr + sizeof(record.dataSizeBytes) > endPtr)
      throw std::runtime_error("Insufficient data for dataSizeBytes");
    std::memcpy(&record.dataSizeBytes, ptr, sizeof(record.dataSizeBytes));
    ptr += sizeof(record.dataSizeBytes);

    // Deserialize remainingBytes vector:
    // Start with the count of elements.
    WireLengthType numRemaining = 0;
    if (ptr + sizeof(numRemaining) > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes count");
    std::memcpy(&numRemaining, ptr, sizeof(numRemaining));
    ptr += sizeof(numRemaining);

    // Reserve space in the vector and read each element.
    record.remainingBytes.resize(numRemaining);
    if (numRemaining > 0) {
      auto bytesSize = numRemaining * sizeof(record.remainingBytes[0]);
      if (ptr + bytesSize > endPtr)
        throw std::runtime_error("Insufficient data for remainingBytes values");
      std::memcpy(record.remainingBytes.data(), ptr, bytesSize);
      ptr += bytesSize;
    }

    // Deserialize bool `atEnd` (stored as a single byte: 1 for true, 0 for
    // false)
    record.atEnd = (*ptr != 0);

    return record;
  }
};

} // namespace facebook::velox::cudf_exchange
