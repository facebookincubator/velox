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

uint32_t MetadataMsg::getSerializedSize() const {
  uint64_t totalSize = 0;
  auto addField = [&](uint64_t bytes, std::string_view field) {
    VELOX_CHECK_LE(
        bytes,
        static_cast<uint64_t>(kMaxMetaBufSize) - totalSize,
        "UCX metadata {} exceeds the maximum serialized size of {} bytes",
        field,
        kMaxMetaBufSize);
    totalSize += bytes;
  };

  addField(sizeof(kMagicNumber), "magic number");
  addField(sizeof(uint32_t), "total-size field");
  addField(sizeof(WireLengthType), "cuDF metadata length");
  addField(cudfMetadata ? cudfMetadata->size() : 0, "cuDF metadata");
  addField(sizeof(dataSizeBytes), "data size");
  addField(sizeof(numRows), "row count");
  addField(sizeof(WireLengthType), "remaining-byte count");
  VELOX_CHECK_LE(
      remainingBytes.size(),
      kMaxMetaBufSize / sizeof(remainingBytes[0]),
      "UCX remaining-byte count exceeds the maximum metadata buffer size");
  addField(
      remainingBytes.size() * sizeof(remainingBytes[0]),
      "remaining-byte values");
  addField(sizeof(uint8_t), "end flag");

  return static_cast<uint32_t>(totalSize);
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

  auto deleter = [](uint8_t* p) { delete[] p; };
  std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);

  uint8_t* ptr = buffer.get();

  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

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

  uint8_t atEndByte = atEnd ? 1 : 0;
  *ptr = atEndByte;

  return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
      std::move(buffer), totalSize);
}

MetadataMsg MetadataMsg::deserializeMetadataMsg(
    const uint8_t* buffer,
    size_t bufferCapacity) {
  VELOX_CHECK_NOT_NULL(buffer, "UCX metadata buffer is null");
  VELOX_CHECK_GE(
      bufferCapacity,
      kMetaHeaderSize,
      "UCX metadata buffer is too small for its header");

  const uint8_t* ptr = buffer;
  size_t remaining = bufferCapacity;

  auto readField =
      [&](void* destination, size_t bytes, std::string_view field) {
        VELOX_CHECK_LE(
            bytes,
            remaining,
            "Insufficient UCX metadata for {}: need {} bytes, have {}",
            field,
            bytes,
            remaining);
        std::memcpy(destination, ptr, bytes);
        ptr += bytes;
        remaining -= bytes;
      };

  MetadataMsg record;

  uint32_t magicNumber = 0;
  readField(&magicNumber, sizeof(magicNumber), "magic number");
  VELOX_CHECK_EQ(
      magicNumber, kMagicNumber, "Invalid UCX metadata magic number");

  uint32_t totalSize = 0;
  readField(&totalSize, sizeof(totalSize), "total size");
  constexpr size_t kMinimumSerializedSize = kMetaHeaderSize +
      sizeof(WireLengthType) + sizeof(WireDataSizeType) +
      sizeof(WireRowCountType) + sizeof(WireLengthType) + sizeof(uint8_t);
  VELOX_CHECK_GE(
      totalSize,
      kMinimumSerializedSize,
      "UCX metadata total size is too small");
  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "UCX metadata total size exceeds the protocol maximum");
  VELOX_CHECK_LE(
      totalSize,
      bufferCapacity,
      "UCX metadata total size exceeds the receive buffer");
  // The receive buffer is normally the fixed one-megabyte maximum. Restrict
  // every subsequent read to the peer-declared record within that capacity.
  remaining = totalSize - kMetaHeaderSize;

  WireLengthType metaSize = 0;
  readField(&metaSize, sizeof(metaSize), "cuDF metadata size");
  VELOX_CHECK_LE(
      metaSize,
      remaining,
      "UCX cuDF metadata size exceeds the serialized record");
  VELOX_CHECK_LE(
      metaSize,
      kMaxMetaBufSize,
      "UCX cuDF metadata size exceeds the protocol maximum");

  record.cudfMetadata =
      std::make_unique<std::vector<uint8_t>>(static_cast<size_t>(metaSize));
  if (metaSize > 0) {
    readField(
        record.cudfMetadata->data(),
        static_cast<size_t>(metaSize),
        "cuDF metadata bytes");
  }

  readField(&record.dataSizeBytes, sizeof(record.dataSizeBytes), "data size");

  readField(&record.numRows, sizeof(record.numRows), "row count");

  WireLengthType numRemaining = 0;
  readField(&numRemaining, sizeof(numRemaining), "remaining-byte count");
  VELOX_CHECK_GE(
      remaining, sizeof(uint8_t), "UCX metadata is missing its end flag");
  VELOX_CHECK_LE(
      numRemaining,
      (remaining - sizeof(uint8_t)) / sizeof(WireRemainingElementType),
      "UCX remaining-byte count exceeds the serialized record");

  record.remainingBytes.resize(static_cast<size_t>(numRemaining));
  if (numRemaining > 0) {
    const auto bytesSize =
        static_cast<size_t>(numRemaining) * sizeof(record.remainingBytes[0]);
    readField(record.remainingBytes.data(), bytesSize, "remaining-byte values");
  }

  uint8_t atEnd = 0;
  readField(&atEnd, sizeof(atEnd), "end flag");
  VELOX_CHECK_LE(atEnd, 1, "Invalid UCX metadata end flag");
  record.atEnd = atEnd != 0;
  VELOX_CHECK_EQ(remaining, 0, "UCX metadata record contains trailing bytes");

  return record;
}

} // namespace facebook::velox::ucx_exchange
