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

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox {

constexpr std::string_view kVeloxTimestampTypeMetadataKey{
    "velox.logical_timestamp"};

// Metadata value to indicate a timezone-agnostic timestamp type.
constexpr std::string_view kVeloxTimestampUtcMetadataValue{"utc"};

inline void appendInt32(std::string& metadata, int32_t value) {
  constexpr auto kInt32Size = sizeof(int32_t);
  const auto start = metadata.size();
  metadata.resize(start + kInt32Size);
  std::memcpy(metadata.data() + start, &value, kInt32Size);
}

inline int32_t readInt32(const char*& metadata) {
  int32_t value;
  std::memcpy(&value, metadata, sizeof(int32_t));
  metadata += sizeof(int32_t);
  return value;
}

/// Encode a single key-value pair into a metadata string. The format is:
/// [numPairs][keySize][key][valueSize][value].
inline std::string encodeSingleKeyValue(
    std::string_view key,
    std::string_view value) {
  std::string metadata;
  appendInt32(metadata, 1);
  appendInt32(metadata, key.size());
  metadata.append(key);
  appendInt32(metadata, value.size());
  metadata.append(value);
  return metadata;
}

inline std::optional<std::string_view> findValue(
    const char* metadata,
    std::string_view key) {
  if (metadata == nullptr) {
    return std::nullopt;
  }

  int32_t numPairs = readInt32(metadata);
  if (numPairs < 0) {
    return std::nullopt;
  }

  for (int32_t i = 0; i < numPairs; ++i) {
    int32_t keySize = readInt32(metadata);
    if (keySize < 0) {
      return std::nullopt;
    }
    std::string_view currentKey{metadata, static_cast<size_t>(keySize)};
    metadata += keySize;

    int32_t valueSize = readInt32(metadata);
    if (valueSize < 0) {
      return std::nullopt;
    }
    std::string_view currentValue{metadata, static_cast<size_t>(valueSize)};
    metadata += valueSize;

    if (currentKey == key) {
      return currentValue;
    }
  }

  return std::nullopt;
}

} // namespace facebook::velox
