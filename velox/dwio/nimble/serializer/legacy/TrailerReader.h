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
#include <utility>
#include <vector>

#include <folly/io/IOBuf.h>

// Read-only legacy reader for the original kLegacyCompact trailer wire format
// (single encoding-type byte + dense payload, or MainlyConstant-special-case
// sparse pair). Used by callers that need to decode existing production hybrid
// Nimble blobs after the main serializer trailer wire format evolves.
//
// The public API normalizes master's dense-vs-sparse output divergence into a
// uniform sparse (streamIndices, streamSizes) pair — the same shape the new
// `detail::readTrailerStreamMetadata` returns — so dispatchers can pick
// between the two implementations with a single-line version branch.

namespace facebook::nimble::serde::legacy {

/// Reads the legacy kLegacyCompact trailer from the end of a contiguous buffer.
/// Fills `streamIndices` (offsets of non-zero stream slots, sorted ascending)
/// and `streamSizes` (their byte sizes), parallel arrays of identical length.
/// Both vectors are reusable buffers owned by the caller (e.g. members on
/// `StreamDataParser`) to keep the per-blob hot path alloc-free across
/// invocations.
void readLegacyTrailerStreamMetadata(
    const char* end,
    std::vector<uint32_t>& streamIndices,
    std::vector<uint32_t>& streamSizes);

/// Value-returning convenience overload for cold-path consumers (tests, dump
/// tools). Returns parallel (streamIndices, streamSizes) arrays.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readLegacyTrailerStreamMetadata(const char* end);

/// IOBuf overload: reads the trailer from a (possibly chained) IOBuf. Tries
/// the fast path first (entire trailer in the tail segment), falls back to
/// cursor + pull() when the trailer spans a chain boundary.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
readLegacyTrailerStreamMetadata(const folly::IOBuf& input);

/// Convenience overloads that scatter the sparse pair returned by
/// `readLegacyTrailerStreamMetadata` into a dense vector indexed by stream
/// offset. Empty stream slots remain 0; the returned vector's length is
/// `lastNonEmptyOffset + 1` (or empty if no non-zero entries exist).
/// For callers that need the dense layout (projector helpers, dump tools).
std::vector<uint32_t> readLegacyTrailerStreamSizesDense(const char* end);

std::vector<uint32_t> readLegacyTrailerStreamSizesDense(
    const folly::IOBuf& input);

} // namespace facebook::nimble::serde::legacy
