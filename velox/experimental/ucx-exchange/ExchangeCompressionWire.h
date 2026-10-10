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

/// Identifies the encoding of an exchange payload.
enum class ExchangePayloadCodec : uint8_t {
  kNone = 0,
  kFusedFor = 1,
  kCascaded = 2,
};

/// Owns the cuDF metadata and compression metadata for an exchange payload.
struct ExchangePayloadMetadata {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  ExchangePayloadCodec codec{ExchangePayloadCodec::kNone};
  std::size_t logicalDataSize{0};
  std::size_t auxiliaryCount{0}; // FOR segment count, zero for Cascaded.
};

/// Adds compression metadata to serialized cuDF metadata.
std::unique_ptr<std::vector<uint8_t>> wrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
    ExchangePayloadCodec codec,
    std::size_t logicalDataSize,
    std::size_t auxiliaryCount);

/// Parses compression metadata, or returns unwrapped cuDF metadata for a raw
/// payload.
ExchangePayloadMetadata unwrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> metadata);

} // namespace facebook::velox::ucx_exchange
