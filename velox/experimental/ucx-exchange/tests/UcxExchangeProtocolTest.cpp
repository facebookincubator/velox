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

#include <gtest/gtest.h>

#include <cstring>
#include <limits>

namespace facebook::velox::ucx_exchange::test {
namespace {

MetadataMsg makeMetadata() {
  MetadataMsg metadata;
  metadata.cudfMetadata = std::make_unique<std::vector<uint8_t>>(
      std::initializer_list<uint8_t>{1, 3, 5, 7});
  metadata.dataSizeBytes = 12345;
  metadata.numRows = 67;
  metadata.remainingBytes = {890, 12};
  metadata.atEnd = false;
  return metadata;
}

template <typename T>
void overwrite(std::shared_ptr<uint8_t>& buffer, size_t offset, T value) {
  std::memcpy(buffer.get() + offset, &value, sizeof(value));
}

TEST(UcxExchangeProtocolTest, metadataRoundTrip) {
  auto metadata = makeMetadata();
  auto [buffer, size] = metadata.serialize();

  auto decoded = MetadataMsg::deserializeMetadataMsg(buffer.get(), size);

  ASSERT_NE(decoded.cudfMetadata, nullptr);
  EXPECT_EQ(*decoded.cudfMetadata, *metadata.cudfMetadata);
  EXPECT_EQ(decoded.dataSizeBytes, metadata.dataSizeBytes);
  EXPECT_EQ(decoded.numRows, metadata.numRows);
  EXPECT_EQ(decoded.remainingBytes, metadata.remainingBytes);
  EXPECT_EQ(decoded.atEnd, metadata.atEnd);
}

TEST(UcxExchangeProtocolTest, rejectsTruncatedMetadata) {
  auto metadata = makeMetadata();
  auto [buffer, size] = metadata.serialize();

  for (size_t receivedSize = 0; receivedSize < size; ++receivedSize) {
    SCOPED_TRACE(receivedSize);
    EXPECT_ANY_THROW(
        MetadataMsg::deserializeMetadataMsg(buffer.get(), receivedSize));
  }
  EXPECT_ANY_THROW(MetadataMsg::deserializeMetadataMsg(nullptr, size));
}

TEST(UcxExchangeProtocolTest, rejectsInvalidRecordBoundsBeforeAllocation) {
  auto metadata = makeMetadata();
  auto [buffer, size] = metadata.serialize();

  EXPECT_ANY_THROW(
      MetadataMsg::deserializeMetadataMsg(buffer.get(), kMetaHeaderSize - 1));

  overwrite<uint32_t>(buffer, sizeof(kMagicNumber), kMaxMetaBufSize + 1);
  EXPECT_ANY_THROW(MetadataMsg::deserializeMetadataMsg(buffer.get(), size));

  auto [hugeMetadataBuffer, hugeMetadataSize] = metadata.serialize();
  overwrite<WireLengthType>(
      hugeMetadataBuffer,
      kMetaHeaderSize,
      std::numeric_limits<WireLengthType>::max());
  EXPECT_ANY_THROW(
      MetadataMsg::deserializeMetadataMsg(
          hugeMetadataBuffer.get(), hugeMetadataSize));

  MetadataMsg noCudfMetadata;
  noCudfMetadata.dataSizeBytes = 1;
  noCudfMetadata.numRows = 1;
  noCudfMetadata.atEnd = false;
  auto [hugeRemainingBuffer, hugeRemainingSize] = noCudfMetadata.serialize();
  constexpr size_t kRemainingCountOffset = kMetaHeaderSize +
      sizeof(WireLengthType) + sizeof(WireDataSizeType) +
      sizeof(WireRowCountType);
  overwrite<WireLengthType>(
      hugeRemainingBuffer,
      kRemainingCountOffset,
      std::numeric_limits<WireLengthType>::max());
  EXPECT_ANY_THROW(
      MetadataMsg::deserializeMetadataMsg(
          hugeRemainingBuffer.get(), hugeRemainingSize));
}

TEST(UcxExchangeProtocolTest, rejectsMalformedEndFlagAndTrailingBytes) {
  MetadataMsg metadata;
  metadata.dataSizeBytes = 0;
  metadata.numRows = 0;
  metadata.atEnd = true;
  auto [buffer, size] = metadata.serialize();

  overwrite<uint8_t>(buffer, size - 1, 2);
  EXPECT_ANY_THROW(MetadataMsg::deserializeMetadataMsg(buffer.get(), size));

  auto [trailingBuffer, trailingSize] = metadata.serialize();
  std::vector<uint8_t> recordWithTrailingByte(
      trailingBuffer.get(), trailingBuffer.get() + trailingSize);
  recordWithTrailingByte.push_back(0);
  const auto oversizedRecord = static_cast<uint32_t>(trailingSize + 1);
  std::memcpy(
      recordWithTrailingByte.data() + sizeof(kMagicNumber),
      &oversizedRecord,
      sizeof(oversizedRecord));
  EXPECT_ANY_THROW(
      MetadataMsg::deserializeMetadataMsg(
          recordWithTrailingByte.data(), recordWithTrailingByte.size()));
}

TEST(UcxExchangeProtocolTest, rejectsOversizedSerialization) {
  MetadataMsg metadata;
  metadata.cudfMetadata =
      std::make_unique<std::vector<uint8_t>>(kMaxMetaBufSize);
  metadata.dataSizeBytes = 1;
  metadata.numRows = 1;
  metadata.atEnd = false;

  EXPECT_ANY_THROW(metadata.serialize());
}

} // namespace
} // namespace facebook::velox::ucx_exchange::test
