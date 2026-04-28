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

#include "velox/common/file/File.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Loads a fixed width value from a string view without assuming aligned
/// memory.
template <typename T>
T inline unalignedLoad(std::string_view payload, std::size_t offset = 0)
  requires(std::is_integral_v<T>)
{
  T value;
  std::memcpy(&value, payload.data() + offset, sizeof(T));
  return value;
}

/// Checks whether the 32-bit roaring block is normalized for cuco.
bool is32bitBitmapNormalized(std::string_view payload);

/// Checks if all 32 bit roaring bitmaps in the payload are normalized.
bool is64bitBitmapNormalized(std::string_view payload, uint64_t numKeys);

/// Normalizes a single 32 bit roaring bitmap for cuco.
std::string normalizeRoaring32(std::string_view payload);

/// Walks the 64 bit roaring bitmap payload and normalizes each 32 bit
/// roaring bitmap bucket for cuco.
std::string normalizeRoaring64(std::string_view payload, uint64_t numKeys);

/// Representation of deletion vector v1 (DV-v1) blob source.
struct BlobSource {
  std::shared_ptr<velox::ReadFile> file;
  std::size_t payloadFileOffset{0};
  std::size_t payloadSize{0};
  bool isRawRoaring32{false};
};

/// Loads a DV v1 blob from the file.
BlobSource loadBlobSource(
    const std::string_view filePath,
    uint64_t fileSizeInBytes,
    const std::unordered_map<int32_t, std::string>& lowerBounds,
    const std::unordered_map<int32_t, std::string>& upperBounds);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
