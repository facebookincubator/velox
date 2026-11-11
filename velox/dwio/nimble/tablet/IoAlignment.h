/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <cstdint>
#include <span>

#include "velox/common/base/BitUtil.h"
#include "velox/common/file/File.h"
#include "velox/common/file/Region.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

/// Returns the alignment that reads from 'file' must use for offsets, lengths,
/// and buffers: the direct-I/O alignment, or 1 when the file does not use
/// direct I/O.
inline uint64_t readAlignment(const velox::ReadFile& file) {
  uint64_t alignment{0};
  return file.directIo(/*alignment=*/alignment) ? alignment : 1;
}

/// Returns the read of 'region' from a file of 'fileSize' bytes: 'region'
/// widened to multiples of 'alignment', which must be a power of two. The read
/// may run past the end of the file, where a direct-I/O read returns no bytes.
/// Rejects a region that starts or ends past the end of the file, since its
/// bytes past the end do not exist.
inline velox::common::Region alignedRegion(
    velox::common::Region region,
    uint64_t alignment,
    uint64_t fileSize) {
  NIMBLE_DCHECK_GT(alignment, 0);
  NIMBLE_DCHECK(velox::bits::isPowerOfTwo(alignment));

  // Compared without adding offset and length, which a corrupt offset can
  // overflow.
  NIMBLE_CHECK_FILE_LE(
      region.offset, fileSize, "Read region starts past the end of the file");
  NIMBLE_CHECK_FILE_LE(
      region.length,
      fileSize - region.offset,
      "Read region extends past the end of the file: offset {}, file size {}",
      region.offset,
      fileSize);
  const auto alignedOffset = region.offset & ~(alignment - 1);
  const auto alignedEnd =
      (region.offset + region.length + alignment - 1) & ~(alignment - 1);
  return {alignedOffset, alignedEnd - alignedOffset};
}

/// Returns the bytes of 'readRegions' before the end of a file of 'fileSize'
/// bytes, which is all that a direct-I/O read past the end returns. Each read
/// comes from alignedRegion(), so none starts past the end of the file.
inline uint64_t fileReadBytes(
    std::span<const velox::common::Region> readRegions,
    uint64_t fileSize) {
  uint64_t bytes{0};
  for (const auto& region : readRegions) {
    bytes += std::min(region.length, fileSize - region.offset);
  }
  return bytes;
}

} // namespace facebook::nimble
