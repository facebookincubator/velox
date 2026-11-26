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

#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

#include <cstring>
#include <limits>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {

uint32_t fnv1a_32(std::string_view s) {
  uint32_t hash = 0x811C9DC5u; // FNV offset basis
  for (unsigned char c : s) {
    hash ^= c;
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}

namespace {

std::shared_ptr<uint8_t> allocateMessage(uint32_t totalSize) {
  return std::shared_ptr<uint8_t>(
      new uint8_t[totalSize], [](uint8_t* p) { delete[] p; });
}

// Writes the header of a message and returns the position after it.
uint8_t*
writeHeader(uint8_t* ptr, uint32_t totalSize, ProducerMessageKind kind) {
  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

  *ptr = static_cast<uint8_t>(kind);
  return ptr + sizeof(uint8_t);
}

// Reads the fields of a message from a byte range, checking each length
// against the bytes that remain.
class MessageReader {
 public:
  MessageReader(const uint8_t* begin, const uint8_t* end)
      : ptr_(begin), end_(end) {}

  size_t remaining() const {
    return end_ - ptr_;
  }

  template <typename T>
  bool read(T& value) {
    if (remaining() < sizeof(T)) {
      return false;
    }
    std::memcpy(&value, ptr_, sizeof(T));
    ptr_ += sizeof(T);
    return true;
  }

  // Copies 'size' bytes to 'out', if that many remain.
  bool readBytes(void* out, size_t size) {
    if (remaining() < size) {
      return false;
    }
    if (size > 0) {
      std::memcpy(out, ptr_, size);
    }
    ptr_ += size;
    return true;
  }

 private:
  const uint8_t* ptr_;
  const uint8_t* const end_;
};

MetadataMsg decodeChunk(MessageReader& reader) {
  constexpr auto kTruncated = "Truncated chunk metadata";

  MetadataMsg chunk;

  WireLengthType cudfSize = 0;
  VELOX_CHECK(reader.read(cudfSize), "{}: cudf metadata size", kTruncated);
  VELOX_CHECK_LE(
      cudfSize,
      reader.remaining(),
      "{}: cudf metadata of {} bytes, {} bytes remain",
      kTruncated,
      cudfSize,
      reader.remaining());
  chunk.cudfMetadata = std::make_unique<std::vector<uint8_t>>(cudfSize);
  VELOX_CHECK(
      reader.readBytes(chunk.cudfMetadata->data(), cudfSize),
      "{}: cudf metadata",
      kTruncated);

  VELOX_CHECK(reader.read(chunk.dataSizeBytes), "{}: data size", kTruncated);
  VELOX_CHECK(reader.read(chunk.numRows), "{}: row count", kTruncated);

  WireLengthType numRemaining = 0;
  VELOX_CHECK(reader.read(numRemaining), "{}: remaining count", kTruncated);
  VELOX_CHECK_LE(
      numRemaining,
      reader.remaining() / sizeof(WireRemainingElementType),
      "{}: {} remaining sizes, {} bytes remain",
      kTruncated,
      numRemaining,
      reader.remaining());
  chunk.remainingBytes.resize(numRemaining);
  VELOX_CHECK(
      reader.readBytes(
          chunk.remainingBytes.data(),
          numRemaining * sizeof(WireRemainingElementType)),
      "{}: remaining sizes",
      kTruncated);

  return chunk;
}

ChannelOutcome decodeEnd(MessageReader& reader) {
  constexpr auto kTruncated = "Truncated end message";

  uint8_t outcome = 0;
  VELOX_CHECK(reader.read(outcome), "{}: outcome", kTruncated);

  uint32_t errorSize = 0;
  VELOX_CHECK(reader.read(errorSize), "{}: error size", kTruncated);
  VELOX_CHECK_LE(
      errorSize,
      reader.remaining(),
      "{}: error of {} bytes, {} bytes remain",
      kTruncated,
      errorSize,
      reader.remaining());

  VELOX_CHECK_LE(
      outcome, 1, "Invalid end message outcome: {}", static_cast<int>(outcome));
  VELOX_CHECK_LE(
      errorSize,
      kMaxProducerErrorSize,
      "Producer error exceeds its bound, {} bytes",
      errorSize);

  std::string error(errorSize, '\0');
  VELOX_CHECK(
      reader.readBytes(error.data(), errorSize), "{}: error", kTruncated);

  if (outcome == 0) {
    VELOX_CHECK(
        error.empty(),
        "Invalid end message outcome: clean with an error of {} bytes",
        errorSize);
    return ChannelOutcome::clean();
  }
  return ChannelOutcome::producerError(std::move(error));
}

} // namespace

HandshakeResponse decodeHandshakeResponse(
    const std::array<uint8_t, sizeof(HandshakeResponse)>& bytes) {
  HandshakeResponse response;
  // Read the flag as a byte: any other value in a bool is undefined.
  response.isIntraNodeTransfer =
      bytes[offsetof(HandshakeResponse, isIntraNodeTransfer)] != 0;
  response.status =
      static_cast<HandshakeStatus>(bytes[offsetof(HandshakeResponse, status)]);
  std::memcpy(
      &response.protocolVersion,
      bytes.data() + offsetof(HandshakeResponse, protocolVersion),
      sizeof(response.protocolVersion));
  std::memcpy(
      &response.reserved,
      bytes.data() + offsetof(HandshakeResponse, reserved),
      sizeof(response.reserved));
  return response;
}

CloseMsg makeCloseMsg(const PartitionKey& key) {
  CloseMsg message{};
  // Copy at most sizeof - 1 bytes: the buffer stays null-terminated even if
  // the task ID is longer.
  std::strncpy(message.taskId, key.taskId.c_str(), sizeof(message.taskId) - 1);
  message.destination = key.destination;
  return message;
}

PartitionKey decodeCloseMsg(const CloseMsg& message) {
  // The sender null-terminates, but a corrupt message may not.
  return PartitionKey{
      std::string(
          message.taskId, strnlen(message.taskId, sizeof(message.taskId))),
      message.destination};
}

std::pair<std::shared_ptr<uint8_t>, size_t> MetadataMsg::serialize() {
  uint32_t totalSize = getSerializedSize();

  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "Metadata serialized size ({}) exceeds maximum buffer size ({}). "
      "This can happen with extremely wide tables. "
      "Consider reducing table width or increasing kMaxMetaBufSize.",
      totalSize,
      kMaxMetaBufSize);

  auto buffer = allocateMessage(totalSize);
  uint8_t* ptr =
      writeHeader(buffer.get(), totalSize, ProducerMessageKind::kChunk);

  WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
  std::memcpy(ptr, &cudfSize, sizeof(cudfSize));
  ptr += sizeof(cudfSize);

  if (cudfSize > 0) {
    std::memcpy(ptr, cudfMetadata->data(), cudfSize);
    ptr += cudfSize;
  }

  std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
  ptr += sizeof(dataSizeBytes);

  std::memcpy(ptr, &numRows, sizeof(numRows));
  ptr += sizeof(numRows);

  WireLengthType numRemaining = remainingBytes.size();
  std::memcpy(ptr, &numRemaining, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(remainingBytes[0]);
    std::memcpy(ptr, remainingBytes.data(), bytesSize);
    ptr += bytesSize;
  }

  return {std::move(buffer), totalSize};
}

std::pair<std::shared_ptr<uint8_t>, size_t> serializeEndMessage(
    const ChannelOutcome& outcome) {
  VELOX_CHECK(
      outcome.kind == OutcomeKind::kClean ||
          outcome.kind == OutcomeKind::kProducerError,
      "Only a clean end or a producer error travels in an end message, got {}",
      outcome.kind);
  VELOX_CHECK(
      outcome.kind != OutcomeKind::kClean || outcome.error.empty(),
      "A clean end carries no error, got {} bytes",
      outcome.error.size());
  VELOX_CHECK_LE(
      outcome.error.size(),
      kMaxProducerErrorSize,
      "Producer error exceeds its bound, {} bytes",
      outcome.error.size());

  const uint32_t errorSize = outcome.error.size();
  const uint32_t totalSize =
      kMetaHeaderSize + sizeof(uint8_t) + sizeof(errorSize) + errorSize;

  auto buffer = allocateMessage(totalSize);
  uint8_t* ptr =
      writeHeader(buffer.get(), totalSize, ProducerMessageKind::kEnd);

  *ptr = outcome.kind == OutcomeKind::kClean ? 0 : 1;
  ptr += sizeof(uint8_t);

  std::memcpy(ptr, &errorSize, sizeof(errorSize));
  ptr += sizeof(errorSize);

  if (errorSize > 0) {
    std::memcpy(ptr, outcome.error.data(), errorSize);
  }

  return {std::move(buffer), totalSize};
}

ProducerMessage decodeProducerMessage(
    const uint8_t* buffer,
    size_t bufferSize) {
  VELOX_CHECK_GE(
      bufferSize,
      kMetaHeaderSize,
      "Message shorter than its header, {} bytes",
      bufferSize);

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, buffer, sizeof(magicNumber));
  VELOX_CHECK_EQ(
      magicNumber, kMagicNumber, "Unexpected magic number: {}", magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, buffer + sizeof(magicNumber), sizeof(totalSize));
  VELOX_CHECK_LE(
      totalSize,
      bufferSize,
      "Message size exceeds the receive buffer: {} > {}",
      totalSize,
      bufferSize);
  VELOX_CHECK_GE(
      totalSize,
      kMetaHeaderSize,
      "Message shorter than its header, size {}",
      totalSize);

  const uint8_t kind = buffer[kMetaHeaderSize - sizeof(uint8_t)];
  MessageReader reader(buffer + kMetaHeaderSize, buffer + totalSize);
  switch (static_cast<ProducerMessageKind>(kind)) {
    case ProducerMessageKind::kChunk:
      return decodeChunk(reader);
    case ProducerMessageKind::kEnd:
      return decodeEnd(reader);
  }
  VELOX_FAIL("Unknown message kind: {}", static_cast<int>(kind));
}

} // namespace facebook::velox::ucx_exchange
