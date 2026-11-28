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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/experimental/ucx-exchange/PartitionKey.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"

namespace facebook::velox::ucx_exchange {
namespace {

// Byte offsets of the header fields: [magic u32][totalSize u32][kind u8].
constexpr size_t kTotalSizeOffset = 4;
constexpr size_t kKindOffset = 8;
constexpr size_t kHeaderSize = 9;
// The end body follows the header: [outcome u8][errorSize u32][error bytes].
constexpr size_t kOutcomeOffset = kHeaderSize;
constexpr size_t kErrorSizeOffset = kHeaderSize + 1;

MetadataMsg makeChunk() {
  MetadataMsg chunk;
  chunk.cudfMetadata =
      std::make_unique<std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3});
  chunk.dataSizeBytes = 1'024;
  chunk.numRows = 7;
  chunk.remainingBytes = {5, 6};
  return chunk;
}

std::vector<uint8_t> toBytes(
    const std::pair<std::shared_ptr<uint8_t>, size_t>& serialized) {
  return {serialized.first.get(), serialized.first.get() + serialized.second};
}

template <typename T>
void put(std::vector<uint8_t>& bytes, size_t offset, T value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

TEST(UcxExchangeProtocolTest, handshakeKeepsLegacyPrefix) {
  EXPECT_EQ(offsetof(HandshakeMsg, taskId), 0);
  EXPECT_EQ(offsetof(HandshakeMsg, destination), 256);
  EXPECT_EQ(offsetof(HandshakeMsg, workerId), 264);
  EXPECT_EQ(offsetof(HandshakeMsg, protocolVersion), 272);
  EXPECT_EQ(sizeof(HandshakeMsg), 280);
  EXPECT_EQ(kLegacyHandshakeSize, offsetof(HandshakeMsg, protocolVersion));
  EXPECT_EQ(HandshakeMsg{}.protocolVersion, kProtocolVersion);
}

TEST(UcxExchangeProtocolTest, handshakeResponseIsSixteenBytes) {
  EXPECT_EQ(sizeof(HandshakeResponse), 16);
  EXPECT_EQ(offsetof(HandshakeResponse, protocolVersion), 8);
}

TEST(UcxExchangeProtocolTest, olderResponseReadsVersionZero) {
  // An older server sends 8 bytes into the 16-byte buffer, which stays zero
  // beyond them.
  std::array<uint8_t, sizeof(HandshakeResponse)> bytes{};
  bytes[0] = 1;

  const auto response = decodeHandshakeResponse(bytes);
  EXPECT_TRUE(response.isIntraNodeTransfer);
  EXPECT_EQ(response.protocolVersion, 0);
  EXPECT_EQ(response.status, HandshakeStatus::kAccepted);
}

TEST(UcxExchangeProtocolTest, currentResponseRoundTrips) {
  HandshakeResponse sent;
  sent.status = HandshakeStatus::kVersionMismatch;
  std::array<uint8_t, sizeof(HandshakeResponse)> bytes{};
  std::memcpy(bytes.data(), &sent, sizeof(sent));

  const auto response = decodeHandshakeResponse(bytes);
  EXPECT_FALSE(response.isIntraNodeTransfer);
  EXPECT_EQ(response.status, HandshakeStatus::kVersionMismatch);
  EXPECT_EQ(response.protocolVersion, kProtocolVersion);
}

TEST(UcxExchangeProtocolTest, closeTagUsesTheCloseOperation) {
  const uint64_t taskHash = 0xABCD1234;
  const auto tag = getCloseTag(taskHash);
  EXPECT_EQ(tag >> 32, taskHash);
  EXPECT_EQ(tag & kOperationTagMask, CLOSE_TAG);
  EXPECT_NE(tag, getHandshakeResponseTag(taskHash));
  EXPECT_NE(tag, getMetadataTag(taskHash, 0));
  EXPECT_NE(tag, getDataTag(taskHash, 0));
}

TEST(UcxExchangeProtocolTest, chunkRoundTrips) {
  auto chunk = makeChunk();
  const auto serialized = chunk.serialize();
  EXPECT_EQ(serialized.second, chunk.getSerializedSize());

  const auto message = decodeProducerMessage(
      serialized.first.get(), static_cast<size_t>(serialized.second));
  ASSERT_TRUE(std::holds_alternative<MetadataMsg>(message));
  const auto& decoded = std::get<MetadataMsg>(message);
  ASSERT_NE(decoded.cudfMetadata, nullptr);
  EXPECT_EQ(*decoded.cudfMetadata, (std::vector<uint8_t>{1, 2, 3}));
  EXPECT_EQ(decoded.dataSizeBytes, 1'024);
  EXPECT_EQ(decoded.numRows, 7);
  EXPECT_EQ(decoded.remainingBytes, (std::vector<int64_t>{5, 6}));
}

TEST(UcxExchangeProtocolTest, chunkReadsFromALargerBuffer) {
  // The source receives into a buffer of kMaxMetaBufSize bytes, most of which
  // the message does not fill.
  auto chunk = makeChunk();
  const auto serialized = chunk.serialize();
  std::vector<uint8_t> buffer(kMaxMetaBufSize);
  std::memcpy(buffer.data(), serialized.first.get(), serialized.second);

  const auto message = decodeProducerMessage(buffer.data(), buffer.size());
  ASSERT_TRUE(std::holds_alternative<MetadataMsg>(message));
  EXPECT_EQ(std::get<MetadataMsg>(message).numRows, 7);
}

TEST(UcxExchangeProtocolTest, endRoundTripsClean) {
  const auto serialized = serializeEndMessage(ChannelOutcome::clean());
  EXPECT_LE(serialized.second, 16);

  const auto message =
      decodeProducerMessage(serialized.first.get(), serialized.second);
  ASSERT_TRUE(std::holds_alternative<ChannelOutcome>(message));
  EXPECT_EQ(std::get<ChannelOutcome>(message), ChannelOutcome::clean());
}

TEST(UcxExchangeProtocolTest, endRoundTripsProducerError) {
  const auto outcome =
      ChannelOutcome::producerError("Injected producer failure");
  const auto serialized = serializeEndMessage(outcome);

  const auto message =
      decodeProducerMessage(serialized.first.get(), serialized.second);
  ASSERT_TRUE(std::holds_alternative<ChannelOutcome>(message));
  EXPECT_EQ(std::get<ChannelOutcome>(message), outcome);
}

TEST(UcxExchangeProtocolTest, endRoundTripsLongestProducerError) {
  const auto outcome =
      ChannelOutcome::producerError(std::string(kMaxProducerErrorSize, 'x'));
  const auto serialized = serializeEndMessage(outcome);

  const auto message =
      decodeProducerMessage(serialized.first.get(), serialized.second);
  EXPECT_EQ(std::get<ChannelOutcome>(message), outcome);
}

TEST(UcxExchangeProtocolTest, endRejectsOversizedError) {
  const auto oversized = ChannelOutcome::producerError(std::string(4'097, 'x'));
  VELOX_ASSERT_THROW(
      serializeEndMessage(oversized), "Producer error exceeds its bound");
}

TEST(UcxExchangeProtocolTest, endRejectsOutcomesThatDoNotTravel) {
  VELOX_ASSERT_THROW(
      serializeEndMessage(ChannelOutcome::transportError("Cannot send")),
      "Only a clean end or a producer error travels in an end message");
  VELOX_ASSERT_THROW(
      serializeEndMessage(ChannelOutcome::versionMismatch("Other version")),
      "Only a clean end or a producer error travels in an end message");
}

TEST(UcxExchangeProtocolTest, endRejectsCleanWithError) {
  ChannelOutcome outcome;
  outcome.kind = OutcomeKind::kClean;
  outcome.error = "Unexpected";
  VELOX_ASSERT_THROW(
      serializeEndMessage(outcome), "A clean end carries no error");
}

TEST(UcxExchangeProtocolTest, decodeRejectsShortBuffer) {
  const auto serialized = serializeEndMessage(ChannelOutcome::clean());
  VELOX_ASSERT_THROW(
      decodeProducerMessage(serialized.first.get(), kHeaderSize - 1),
      "Message shorter than its header");
}

TEST(UcxExchangeProtocolTest, decodeRejectsBadMagic) {
  auto bytes = toBytes(serializeEndMessage(ChannelOutcome::clean()));
  put<uint32_t>(bytes, 0, kMagicNumber + 1);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Unexpected magic number");
}

TEST(UcxExchangeProtocolTest, decodeRejectsSizeBeyondBuffer) {
  auto bytes = toBytes(serializeEndMessage(ChannelOutcome::clean()));
  put<uint32_t>(
      bytes, kTotalSizeOffset, static_cast<uint32_t>(bytes.size() + 1));
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Message size exceeds the receive buffer");
}

TEST(UcxExchangeProtocolTest, decodeRejectsSizeBelowHeader) {
  auto bytes = toBytes(serializeEndMessage(ChannelOutcome::clean()));
  put<uint32_t>(
      bytes, kTotalSizeOffset, static_cast<uint32_t>(kHeaderSize - 1));
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Message shorter than its header");
}

TEST(UcxExchangeProtocolTest, decodeRejectsUnknownKind) {
  auto bytes = toBytes(serializeEndMessage(ChannelOutcome::clean()));
  put<uint8_t>(bytes, kKindOffset, 3);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Unknown message kind");
  put<uint8_t>(bytes, kKindOffset, 0);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Unknown message kind");
}

TEST(UcxExchangeProtocolTest, decodeRejectsTruncatedEnd) {
  auto bytes = toBytes(
      serializeEndMessage(ChannelOutcome::producerError("Injected failure")));
  // The error is longer than the bytes that remain in the message.
  auto longer = bytes;
  put<uint32_t>(longer, kErrorSizeOffset, 1'000);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(longer.data(), longer.size()),
      "Truncated end message");

  // The message ends before the error size.
  auto cut = bytes;
  put<uint32_t>(
      cut, kTotalSizeOffset, static_cast<uint32_t>(kOutcomeOffset + 2));
  VELOX_ASSERT_THROW(
      decodeProducerMessage(cut.data(), cut.size()), "Truncated end message");
}

TEST(UcxExchangeProtocolTest, decodeRejectsInvalidEnd) {
  auto bytes = toBytes(serializeEndMessage(ChannelOutcome::clean()));
  put<uint8_t>(bytes, kOutcomeOffset, 2);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Invalid end message outcome");

  // A clean end carries no error.
  auto withError = toBytes(
      serializeEndMessage(ChannelOutcome::producerError("Injected failure")));
  put<uint8_t>(withError, kOutcomeOffset, 0);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(withError.data(), withError.size()),
      "Invalid end message outcome");
}

TEST(UcxExchangeProtocolTest, decodeRejectsOversizedError) {
  // Build the oversized message by hand: serializeEndMessage refuses it.
  const std::string error(kMaxProducerErrorSize + 1, 'x');
  std::vector<uint8_t> bytes(
      kErrorSizeOffset + sizeof(uint32_t) + error.size());
  put<uint32_t>(bytes, 0, kMagicNumber);
  put<uint32_t>(bytes, kTotalSizeOffset, static_cast<uint32_t>(bytes.size()));
  put<uint8_t>(
      bytes, kKindOffset, static_cast<uint8_t>(ProducerMessageKind::kEnd));
  put<uint8_t>(bytes, kOutcomeOffset, 1);
  put<uint32_t>(bytes, kErrorSizeOffset, static_cast<uint32_t>(error.size()));
  std::memcpy(
      bytes.data() + kErrorSizeOffset + sizeof(uint32_t),
      error.data(),
      error.size());
  VELOX_ASSERT_THROW(
      decodeProducerMessage(bytes.data(), bytes.size()),
      "Producer error exceeds its bound");
}

TEST(UcxExchangeProtocolTest, decodeRejectsTruncatedChunk) {
  auto chunk = makeChunk();
  const auto bytes = toBytes(chunk.serialize());

  // Each cut keeps a valid header and magic but ends inside a field.
  for (size_t cut = kHeaderSize; cut < bytes.size(); ++cut) {
    SCOPED_TRACE(cut);
    auto truncated = bytes;
    put<uint32_t>(truncated, kTotalSizeOffset, static_cast<uint32_t>(cut));
    VELOX_ASSERT_THROW(
        decodeProducerMessage(truncated.data(), truncated.size()),
        "Truncated chunk metadata");
  }
}

TEST(UcxExchangeProtocolTest, decodeRejectsChunkLengthsBeyondMessage) {
  auto chunk = makeChunk();
  const auto bytes = toBytes(chunk.serialize());

  // The cuDF metadata length is the first field of the body.
  auto metadata = bytes;
  put<uint64_t>(metadata, kHeaderSize, UINT64_MAX);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(metadata.data(), metadata.size()),
      "Truncated chunk metadata");

  // The count of remaining bytes is the last 8-byte field before its values.
  auto remaining = bytes;
  put<uint64_t>(
      remaining,
      bytes.size() - 2 * sizeof(int64_t) - sizeof(uint64_t),
      UINT64_MAX);
  VELOX_ASSERT_THROW(
      decodeProducerMessage(remaining.data(), remaining.size()),
      "Truncated chunk metadata");
}

TEST(UcxExchangeProtocolTest, closeMessageRoundTrips) {
  EXPECT_EQ(sizeof(CloseMsg), 260);
  EXPECT_EQ(
      decodeCloseMsg(makeCloseMsg({"task.1.0.0.0", 3})),
      (PartitionKey{"task.1.0.0.0", 3}));
}

TEST(UcxExchangeProtocolTest, closeMessageTruncatesLongTaskId) {
  const std::string taskId(300, 'a');
  const auto message = makeCloseMsg({taskId, 9});
  EXPECT_EQ(message.destination, 9);
  EXPECT_EQ(message.taskId[sizeof(message.taskId) - 1], '\0');
  EXPECT_EQ(
      decodeCloseMsg(message),
      (PartitionKey{std::string(sizeof(message.taskId) - 1, 'a'), 9}));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
