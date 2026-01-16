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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "velox/common/base/Exceptions.h"

/// Definitions needed for the Cudf exchange protocol.

namespace facebook::velox::cudf_exchange {

void cudaCheck(CUresult result);

// data and metadata tags are split into 3 parts:
// - 4 bytes: A hash of the producing taskId. The taskId is unique within a
// cluster.
// - 1 byte: The operation, which is either metadata, or data.
// - 3 bytes: The sequence number of the chunk that is exchanged between 2
// tasks.

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

/// Maximum length for listener IP address (supports IPv6).
constexpr size_t kMaxListenerIpLen = 64;

/// @brief Request that is sent from the client (CudfExchangeSource) to the
/// server (CudfExchangeServer) after connection.
///
/// The handshake establishes the partition key for data exchange and includes
/// the source's Communicator listener address. The server uses this to detect
/// if the source is on the same node (same Communicator instance) by comparing
/// with its own listener address. Same-node detection enables local exchange
/// optimizations that bypass UCXX transfers.
struct HandshakeMsg {
  char taskId[256];
  uint32_t destination;
  /// Source's Communicator listener IP address for same-node detection.
  char sourceListenerIp[kMaxListenerIpLen];
  /// Source's Communicator listener port for same-node detection.
  uint16_t sourceListenerPort;
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
constexpr uint32_t kMetaBufSize = 4096;

struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  int64_t dataSizeBytes;
  std::vector<int64_t> remainingBytes;
  bool atEnd;

  uint32_t getSerializedSize() {
    // The header: the magic number and the metadata length (an uint32_t).
    uint32_t totalSize = 2 * sizeof(uint32_t);
    // cudfMetadata: lenght info and then the data.
    totalSize += sizeof(size_t);
    if (cudfMetadata && cudfMetadata->size() > 0) {
      totalSize += cudfMetadata->size();
    }
    totalSize += sizeof(size_t); // dataSizeBytes

    // remainingBytes: length and then the data.
    totalSize += sizeof(size_t);
    totalSize += remainingBytes.size() * sizeof(uint64_t);

    totalSize += sizeof(uint8_t); // atEnd, encoded in a byte.

    return totalSize;
  }

  // serializes the metadata record.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize() {
    uint32_t totalSize = getSerializedSize();

    // Allocate a contiguous block of memory
    // Use shared_ptr with a custom deleter for arrays.
    auto deleter = [](uint8_t* p) { delete[] p; };
    // allocate a fixed size buffer, make it easier for the receiving side.
    // TODO: Extend the exchange protocol and send the actual size.
    std::shared_ptr<uint8_t> buffer(new uint8_t[kMetaBufSize], deleter);

    uint8_t* ptr = buffer.get();

    // write the magic number
    std::memcpy(ptr, &kMagicNumber, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    // write the size of the metadata
    std::memcpy(ptr, &totalSize, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // Serialize cudfMetadata size.

    size_t cudfSize = cudfMetadata ? cudfMetadata->size() : 0;

    std::memcpy(ptr, &cudfSize, sizeof(size_t));
    ptr += sizeof(size_t);

    // If data exists, serialize each byte.
    if (cudfSize > 0) {
      std::memcpy(ptr, cudfMetadata->data(), cudfSize);
      ptr += cudfSize;
    }

    // Serialize dataSizeBytes.
    std::memcpy(ptr, &dataSizeBytes, sizeof(size_t));
    ptr += sizeof(size_t);

    // Serialize number of remainingBytes elements.
    size_t numRemaining = remainingBytes.size();
    std::memcpy(ptr, &numRemaining, sizeof(size_t));
    ptr += sizeof(size_t);

    // Serialize remainingBytes elements.
    if (numRemaining > 0) {
      std::memcpy(ptr, remainingBytes.data(), numRemaining * sizeof(uint64_t));
      ptr += numRemaining * sizeof(uint64_t);
    }

    // Serialize atEnd bool as 0/1.
    *ptr = atEnd ? 1 : 0;

    return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
        std::move(buffer), totalSize);
  }

  // Deserialization function that constructs a MetadataMsg from a
  // buffer that points to a serialized metadata.
  static MetadataMsg deserializeMetadataMsg(const uint8_t* buffer) {
    // We'll use a pointer to scan through the memory.
    const uint8_t* ptr = buffer;

    MetadataMsg record;

    uint32_t magicNumber = 0;
    // extract magic number.
    std::memcpy(&magicNumber, ptr, sizeof(uint32_t));
    VELOX_CHECK_EQ(magicNumber, kMagicNumber);

    ptr += sizeof(uint32_t);
    // extract the total size.
    uint32_t totalSize = 0;
    std::memcpy(&totalSize, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    const uint8_t* endPtr = buffer + totalSize;

    // Deserialize cudfMetadata:
    // First read the size of the metadata (stored as size_t)
    if (ptr + sizeof(size_t) > endPtr)
      throw std::runtime_error("Insufficient data for cudfMetadata size");
    uint64_t metaSize = 0;
    std::memcpy(&metaSize, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    // Allocate a vector of the correct size.
    record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metaSize);
    if (metaSize > 0) {
      if (ptr + metaSize * sizeof(uint8_t) > endPtr)
        throw std::runtime_error("Insufficient data for cudfMetadata bytes");
      std::memcpy(record.cudfMetadata->data(), ptr, metaSize);
      ptr += metaSize;
    }

    // Deserialize dataSizeBytes, stored as a size_t.
    if (ptr + sizeof(size_t) > endPtr)
      throw std::runtime_error("Insufficient data for dataSizeBytes");
    std::memcpy(&record.dataSizeBytes, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    // Deserialize remainingBytes vector:
    // Start with the count of elements (stored as size_t).
    if (ptr + sizeof(size_t) > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes count");
    size_t numRemaining = 0;
    std::memcpy(&numRemaining, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    // Reserve space in the vector and read each element of type uint64_t.
    if (ptr + numRemaining * sizeof(uint64_t) > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes values");
    record.remainingBytes.resize(numRemaining);
    if (numRemaining > 0) {
      std::memcpy(
          record.remainingBytes.data(), ptr, numRemaining * sizeof(uint64_t));
      ptr += numRemaining * sizeof(uint64_t);
    }

    // Deserialize bool `atEnd` (stored as a single byte: 1 for true, 0 for
    // false)
    record.atEnd = (*ptr != 0);

    return record;
  }
};

} // namespace facebook::velox::cudf_exchange
