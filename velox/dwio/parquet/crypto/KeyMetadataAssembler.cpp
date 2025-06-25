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
#include "velox/dwio/parquet/crypto/KeyMetadataAssembler.h"
#include <memory>
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"

namespace facebook::velox::parquet {

const char ASSEMBLER_VERSION = 2;
const char IV = 1;
const char KEYVERSION = 2;
const char EEK = 3;
const char MKNAME = 4;

const size_t LENGTH_BYTES = 4;

int getDataLength(const std::string& metadata, size_t offset) {
  size_t remainingSpace = metadata.size() - offset;

  VELOX_CHECK(
      remainingSpace >= LENGTH_BYTES,
      "[CLAC] Not enough space in metadata array after offset to copy length bytes.");

  uint8_t lengthBytes[LENGTH_BYTES];

  // Copy bytes from metadata starting at the offset to lengthBytes
  std::memcpy(lengthBytes, metadata.data() + offset, LENGTH_BYTES);

  // Convert the byte array to an integer
  int dataLength = (lengthBytes[0] << 24) | (lengthBytes[1] << 16) |
      (lengthBytes[2] << 8) | (lengthBytes[3]);

  return dataLength;
}

static std::string getDataByteArray(
    const std::string& metadata,
    size_t offset) {
  int length = getDataLength(metadata, offset);
  size_t dataStart = offset + LENGTH_BYTES;

  // Check if there is enough space in metadata to extract the data
  VELOX_CHECK(
      dataStart + length <= metadata.size(),
      "[CLAC] Not enough data after the offset to copy.");

  return metadata.substr(dataStart, length);
}

static std::string getIv(const std::string& metadata, size_t offset) {
  // Check metadata[0] for assembler version. IV is base64 encoded for versions
  // > 1.
  uint8_t metadataVersion = metadata[0];

  switch (metadataVersion) {
    case 1:
      return getDataByteArray(metadata, offset);
    default: {
      // Get data byte array and decode it using Base64
      std::string dataBytes = getDataByteArray(metadata, offset);
      std::string decodedData = velox::encoding::Base64::decode(dataBytes);
      return decodedData;
    }
  }
}

int getVersion(const std::string& metadata, size_t offset) {
  std::string versionBytes = getDataByteArray(metadata, offset);

  // Convert the version bytes to an integer
  return (versionBytes[0] << 24) | (versionBytes[1] << 16) |
      (versionBytes[2] << 8) | versionBytes[3];
}

std::string getEEK(const std::string& metadata, size_t offset) {
  // Check metadata[0] for assembler version. EEK is base64 encoded for versions
  // > 1.
  uint8_t metadataVersion = metadata[0];

  switch (metadataVersion) {
    case 1:
      return getDataByteArray(metadata, offset);
    default:
      std::string dataBytes = getDataByteArray(metadata, offset);
      return velox::encoding::Base64::decode(dataBytes);
  }
}

std::string getName(const std::string& metadata, size_t offset) {
  return getDataByteArray(metadata, offset);
}

KeyMetadata KeyMetadataAssembler::unAssembly(const std::string& keyMetadata) {
  // KeyMetadataAssembler.java#74
  size_t offset = 0;
  VELOX_CHECK(
      keyMetadata[offset] > 0 && keyMetadata[offset] <= ASSEMBLER_VERSION,
      "[CLAC] Illegal keyMetadata assembler version {}",
      keyMetadata[offset]);

  offset = 1;
  std::string iv;
  std::string eek;
  int version = 0;
  std::string mkName;
  while (offset < keyMetadata.size()) {
    if (keyMetadata[offset] == IV) {
      iv = getIv(keyMetadata, offset + 1);
    } else if (keyMetadata[offset] == KEYVERSION) {
      version = getVersion(keyMetadata, offset + 1);
    } else if (keyMetadata[offset] == EEK) {
      eek = getEEK(keyMetadata, offset + 1);
    } else if (keyMetadata[offset] == MKNAME) {
      mkName = getName(keyMetadata, offset + 1);
    } else {
      VELOX_FAIL("[CLAC] MetadataType does not exist.");
    }
    int dataLength = getDataLength(keyMetadata, offset + 1);
    offset += LENGTH_BYTES + dataLength + 1;
  }
  return {mkName, iv, version, eek};
}

} // namespace facebook::velox::parquet
