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

#include <array>
#include <cinttypes>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"

/// Definitions needed for the Ucx exchange protocol.
///
/// Byte order: all multi-byte fields are serialized with std::memcpy, which
/// preserves host byte order. The protocol assumes matching endianness between
/// peers (little-endian on x86 and ARM). Cross-endian transfers are not
/// supported.

namespace facebook::velox::ucx_exchange {

// Data and metadata tags are a uint64_t split into 3 fields, most-significant
// first:
// - Bits 63..32 (4 bytes): FNV-1a hash of the producing taskId, which is
//   unique within a cluster.
// - Bits 31..24 (1 byte): Operation type (metadata, data, handshake response,
//   or close).
// - Bits 23..0  (3 bytes): Sequence number of the chunk exchanged between 2
//   tasks.

// Definition of the operations.
constexpr uint64_t METADATA_TAG = 0x02000000;
constexpr uint64_t DATA_TAG = 0x03000000;
constexpr uint64_t HANDSHAKE_RESPONSE_TAG = 0x04000000;
constexpr uint64_t CLOSE_TAG = 0x05000000;

/// The bits of a tag that hold the operation.
inline constexpr uint64_t kOperationTagMask{0x00000000FF000000};

/// The version of the wire protocol in this file. Peers of different versions
/// do not exchange data.
inline constexpr uint32_t kProtocolVersion{1};

/// The size of the handshake before protocol version 1 added its version. An
/// older peer sends this many bytes.
inline constexpr size_t kLegacyHandshakeSize{272};

// Implementation of the fowler-noll-vo hash function for 32 bits.
uint32_t fnv1a_32(std::string_view s);

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

// Gets the tag used for the close message, which a source sends to its server.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getCloseTag(uint64_t taskHash) {
  return (taskHash << 32) | CLOSE_TAG;
}

/// @brief Request that is sent from the client (UcxExchangeSource) to the
/// server (UcxExchangeServer) after connection.
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
  /// The protocol version of the sender. A server that runs another version
  /// refuses the handshake. The fields before it keep their offsets from
  /// version 0.
  uint32_t protocolVersion{kProtocolVersion};
  uint32_t reserved{0};
};

static_assert(offsetof(HandshakeMsg, destination) == 256);
static_assert(offsetof(HandshakeMsg, workerId) == 264);
static_assert(offsetof(HandshakeMsg, protocolVersion) == kLegacyHandshakeSize);
static_assert(sizeof(HandshakeMsg) == 280);

/// Whether the server accepted the handshake.
enum class HandshakeStatus : uint8_t {
  kAccepted = 0,
  /// The server runs another protocol version and will not serve the source.
  kVersionMismatch = 1,
};

/// @brief Response sent from server to source after handshake.
/// Informs the source whether intra-node transfer optimization is available,
/// allowing the source to bypass UCXX for all subsequent data transfers. The
/// response has 16 bytes in every protocol version.
struct HandshakeResponse {
  /// True if server and source are on the same node (same Communicator).
  /// When true, source should use IntraNodeTransferRegistry instead of UCXX.
  bool isIntraNodeTransfer{false};
  HandshakeStatus status{HandshakeStatus::kAccepted};
  /// Padding for alignment
  uint8_t padding[6]{};
  /// The protocol version of the server. A server before version 1 sends 8
  /// bytes only, so its version reads as 0.
  uint32_t protocolVersion{kProtocolVersion};
  uint32_t reserved{0};
};

static_assert(offsetof(HandshakeResponse, protocolVersion) == 8);
static_assert(sizeof(HandshakeResponse) == 16);

/// Reads a received response. 'bytes' is a zero-filled 16-byte receive buffer,
/// so an older server's 8 bytes read as version 0.
HandshakeResponse decodeHandshakeResponse(
    const std::array<uint8_t, sizeof(HandshakeResponse)>& bytes);

/// Tells a server that its consumer closed. A source sends it with
/// getCloseTag(), as a tagged message.
struct CloseMsg {
  char taskId[256];
  uint32_t destination;
};

static_assert(sizeof(CloseMsg) == 260);

/// Builds the close message of 'key'. A task ID of 256 bytes or more is cut,
/// as in the handshake.
CloseMsg makeCloseMsg(const PartitionKey& key);

/// Returns the partition that 'message' closes.
PartitionKey decodeCloseMsg(const CloseMsg& message);

constexpr uint32_t kMagicNumber = 0x12345678;
/// Maximum metadata buffer size for receiving. This should be large enough
/// to handle tables with many columns. 1MB allows for ~10,000+ columns.
/// The sender allocates exact size needed; receiver pre-allocates this max.
constexpr uint32_t kMaxMetaBufSize = 1024 * 1024; // 1MB

/// The header of every message that a producer sends on the metadata tag.
/// Format: [magic (4 bytes)][totalSize (4 bytes)][kind (1 byte)]
constexpr uint32_t kMetaHeaderSize =
    sizeof(kMagicNumber) + sizeof(uint32_t) + sizeof(uint8_t);

/// The kind of a message that a producer sends on the metadata tag.
enum class ProducerMessageKind : uint8_t {
  /// A chunk's metadata. The chunk's data follows on the data tag.
  kChunk = 1,
  /// The end of the stream, clean or with the producer's error.
  kEnd = 2,
};

/// Wire-format types for MetadataMsg serialization. Using shared type aliases
/// ensures serialize() and decodeProducerMessage() agree on field widths.
using WireLengthType = uint64_t;
using WireDataSizeType = int64_t;
using WireRemainingElementType = int64_t;
/// A cuDF table's row count is a cudf::size_type, so it cannot exceed
/// 2^31 - 1 and needs no more than 32 bits on the wire.
using WireRowCountType = int32_t;

/// The metadata of one chunk: a message of kind ProducerMessageKind::kChunk.
/// Body: [cudf metadata size u64][cudf metadata][dataSizeBytes i64]
/// [numRows i32][remaining count u64][remainingBytes i64...]
struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  WireDataSizeType dataSizeBytes;

  /// Logical rows in the payload. Sent explicitly because cuDF derives a
  /// table's row count from its columns, so a payload with no columns — an
  /// exchange fragment whose output layout is empty — cannot report its own
  /// row count once it has been packed. Bounded by cudf::size_type, which is
  /// also what bounds the CudfVector the consumer rebuilds.
  WireRowCountType numRows;

  std::vector<WireRemainingElementType> remainingBytes;

  uint32_t getSerializedSize() const {
    // The header: the magic number, the message size and the kind.
    uint32_t totalSize = kMetaHeaderSize;
    // cudfMetadata: length info and then the data.
    WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
    totalSize += sizeof(cudfSize);
    totalSize += cudfSize;
    // dataSizeBytes
    totalSize += sizeof(dataSizeBytes);
    // numRows
    totalSize += sizeof(numRows);
    // remainingBytes: length and then the data.
    totalSize += sizeof(WireLengthType); // for numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);

    return totalSize;
  }

  /// Serializes this metadata record into a newly allocated buffer.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize();
};

/// Serializes the end of a stream. Body: [outcome u8: 0 clean, 1 producer
/// error][error size u32][error bytes]. 'outcome' must be clean or a producer
/// error, and a clean outcome has no error. The error is at most
/// kMaxProducerErrorSize bytes.
std::pair<std::shared_ptr<uint8_t>, size_t> serializeEndMessage(
    const ChannelOutcome& outcome);

/// What a producer sends on the metadata tag: a chunk's metadata, or the end
/// of the stream.
using ProducerMessage = std::variant<MetadataMsg, ChannelOutcome>;

/// Decodes a message that a producer sent. 'bufferSize' is the size of the
/// receive buffer, which may be larger than the message. Throws
/// VeloxRuntimeError with a message naming the defect.
ProducerMessage decodeProducerMessage(const uint8_t* buffer, size_t bufferSize);

} // namespace facebook::velox::ucx_exchange
