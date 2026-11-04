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

#include <cstdint>

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

/// Expands 'region' outward to multiples of 'alignment', which must be a power
/// of two.
inline velox::common::Region alignedRegion(
    velox::common::Region region,
    uint64_t alignment) {
  NIMBLE_DCHECK_GT(alignment, 0);
  NIMBLE_DCHECK(velox::bits::isPowerOfTwo(alignment));
  const auto alignedOffset = region.offset & ~(alignment - 1);
  const auto alignedEnd =
      (region.offset + region.length + alignment - 1) & ~(alignment - 1);
  return {alignedOffset, alignedEnd - alignedOffset};
}

} // namespace facebook::nimble
