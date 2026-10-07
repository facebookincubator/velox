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
#include <string_view>

#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

// Sorted-position slot helper. Extracted into its own header so both
// EncodingUtils.h and PFOREncoding.h can include it without forming an
// include cycle (EncodingUtils.h includes PFOREncoding.h for its encoding
// dispatch, so PFOR templates cannot include EncodingUtils.h directly).

namespace facebook::nimble::detail {

// Slot index range into the encoded position array returned by
// findSortedPositionSlots. Half-open: retained slots are [slotStart, slotEnd).
struct SortedPositionSlots {
  uint32_t slotStart;
  uint32_t slotEnd;
};

// Locates the slot index range covering values in
// [valueStartOffset, valueEndOffset) in a sorted-position encoded stream.
// Shared by SparseBoolEncoding (positions terminator) and PFOREncoding (no
// terminator).
//
// Uses random-access binary search + view->readAt when the position encoding
// has an EncodingView (Trivial, RLE, ...). Otherwise (Varint, DeltaEncoding,
// ...) materializes every position once and binary searches the decoded
// values; the decoded buffer is used only for the search and is released
// when the function returns.
//
// Callers add wire-specific checks on the returned bounds (SparseBool
// verifies the terminator slot lives past slotEnd).
SortedPositionSlots findSortedPositionSlots(
    std::string_view encodedPositions,
    uint32_t positionCount,
    uint32_t valueStartOffset,
    uint32_t valueEndOffset,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options);

} // namespace facebook::nimble::detail
