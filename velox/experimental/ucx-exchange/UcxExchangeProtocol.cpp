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
#include <stdexcept>
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

MetadataMsg MetadataMsg::deserializeMetadataMsg(const uint8_t* buffer) {
  const uint8_t* ptr = buffer;

  MetadataMsg record;

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
  VELOX_CHECK_EQ(magicNumber, kMagicNumber);
  ptr += sizeof(magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, ptr, sizeof(totalSize));
  ptr += sizeof(totalSize);

  const uint8_t* endPtr = buffer + totalSize;

  WireLengthType metaSize = 0;
  if (ptr + sizeof(metaSize) > endPtr)
    throw std::runtime_error("Insufficient data for cudfMetadata size");
  std::memcpy(&metaSize, ptr, sizeof(metaSize));
  ptr += sizeof(metaSize);

  record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metaSize);
  if (metaSize > 0) {
    if (ptr + metaSize > endPtr)
      throw std::runtime_error("Insufficient data for cudfMetadata bytes");
    std::memcpy(record.cudfMetadata->data(), ptr, metaSize);
    ptr += metaSize;
  }

  if (ptr + sizeof(record.dataSizeBytes) > endPtr)
    throw std::runtime_error("Insufficient data for dataSizeBytes");
  std::memcpy(&record.dataSizeBytes, ptr, sizeof(record.dataSizeBytes));
  ptr += sizeof(record.dataSizeBytes);

  WireLengthType numRemaining = 0;
  if (ptr + sizeof(numRemaining) > endPtr)
    throw std::runtime_error("Insufficient data for remainingBytes count");
  std::memcpy(&numRemaining, ptr, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  record.remainingBytes.resize(numRemaining);
  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(record.remainingBytes[0]);
    if (ptr + bytesSize > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes values");
    std::memcpy(record.remainingBytes.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  if (ptr + 1 > endPtr) {
    throw std::runtime_error("Insufficient data for atEnd flag");
  }
  record.atEnd = (*ptr != 0);

  return record;
}

} // namespace facebook::velox::ucx_exchange
