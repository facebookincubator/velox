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
#include <vector>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

/// Whole-blob GPU compression for the UCX exchange payload.
/// Wraps the DietGPU byte-rANS codec. The blob is split into
/// fixed-size segments compressed as one batch (intra-blob parallelism);
/// the compressed buffer is the plain concatenation of the segments and the
/// per-segment sizes travel in the wire metadata (MetadataMsg
/// remainingBytes), so decode needs no device-side header parsing.
namespace facebook::velox::ucx_exchange {

/// Codec identifiers carried on the wire.
enum class ExchangeCodec : int64_t {
  kNone = 0,
  kByteRans = 1, // DietGPU order-0 byte-rANS, precision 10
};

/// Segment size for batched whole-blob compression.
constexpr std::size_t kCompressSegmentBytes = 32u << 20; // 32 MiB

struct CompressResult {
  /// False when compression would not pay (incompressible or tiny input);
  /// the caller must send the original buffer uncompressed.
  bool used{false};
  rmm::device_buffer data; // concatenated compressed segments
  std::vector<uint32_t> segSizes; // compressed bytes per segment

  /// Cheap telemetry retained even when the candidate is rejected. This lets
  /// callers distinguish "not attempted" from "attempted but did not pay"
  /// without adding CUDA events or synchronization.
  struct Stats {
    bool attempted{false};
    std::size_t inputBytes{0};
    std::size_t candidateBytes{0}; // 16-byte-padded bytes if transmitted
  } stats;
};

/// Compresses `size` bytes at `src` (device memory) on `stream`.
/// Synchronizes `stream` once (to compact segments); the result is ready to
/// hand to UCX after the caller's usual pre-send synchronize.
CompressResult compressBlob(
    const void* src,
    std::size_t size,
    rmm::cuda_stream_view stream,
    double minGain = 0.02,
    std::size_t minBytes = 1u << 16);

/// Decompresses a kByteRans blob (concatenated segments of `segSizes`
/// compressed bytes each, `uncompressedBytes` total output) into a new
/// device buffer on `stream`. Byte-exact; throws on decode failure.

rmm::device_buffer decompressBlob(
    const void* src,
    const std::vector<uint32_t>& segSizes,
    std::size_t uncompressedBytes,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::ucx_exchange
