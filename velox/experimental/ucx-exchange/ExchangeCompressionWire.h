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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Tells the receiver how to decode the GPU data in an exchange payload.
enum class ExchangePayloadCodec : uint8_t {
  kNone = 0,
  kFusedFor = 1,
  kCascaded = 2,
};

/// Metadata needed to reconstruct a received exchange payload. `cudfMetadata`
/// contains the original cuDF metadata after the compression header is removed.
struct ExchangePayloadMetadata {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  ExchangePayloadCodec codec{ExchangePayloadCodec::kNone};
  /// Size of the packed GPU data before compression.
  std::size_t logicalDataSize{0};
  /// Number of fused FOR segments. Zero for other codecs.
  std::size_t auxiliaryCount{0};
};

/// Prepends the codec, uncompressed byte count, and number of fused FOR
/// segments to serialized cuDF metadata. The segment count is zero for
/// Cascaded. This is used only for compressed payloads.
std::unique_ptr<std::vector<uint8_t>> wrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
    ExchangePayloadCodec codec,
    std::size_t logicalDataSize,
    std::size_t auxiliaryCount);

/// Removes and parses the compression header. If `metadata` has no compression
/// header, returns it unchanged with codec kNone.
ExchangePayloadMetadata unwrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> metadata);

} // namespace facebook::velox::ucx_exchange
