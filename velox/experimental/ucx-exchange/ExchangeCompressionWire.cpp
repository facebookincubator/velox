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

#include "velox/experimental/ucx-exchange/ExchangeCompressionWire.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr std::array<uint8_t, 8> kMagic{'V', 'L', 'X', 'P', 'A', 'C', 'K', 0};
constexpr uint16_t kVersion = 1;
constexpr std::size_t kHeaderSize = kMagic.size() + sizeof(kVersion) +
    sizeof(uint16_t) + sizeof(uint8_t) + 3 * sizeof(uint64_t);

static_assert(kHeaderSize == 37);

template <typename T>
void appendScalar(std::vector<uint8_t>& output, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto position = output.size();
  output.resize(position + sizeof(T));
  std::memcpy(output.data() + position, &value, sizeof(T));
}

template <typename T>
T readScalar(const uint8_t*& current, const uint8_t* end) {
  static_assert(std::is_trivially_copyable_v<T>);
  VELOX_CHECK_LE(
      sizeof(T),
      static_cast<std::size_t>(end - current),
      "Truncated exchange compression metadata envelope");
  T value;
  std::memcpy(&value, current, sizeof(T));
  current += sizeof(T);
  return value;
}

} // namespace

std::unique_ptr<std::vector<uint8_t>> wrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
    ExchangePayloadCodec codec,
    std::size_t logicalDataSize,
    std::size_t auxiliaryCount) {
  VELOX_CHECK_NOT_NULL(cudfMetadata);
  VELOX_CHECK(
      codec == ExchangePayloadCodec::kFusedFor ||
          codec == ExchangePayloadCodec::kCascaded,
      "Unsupported exchange payload codec");
  VELOX_CHECK(
      codec != ExchangePayloadCodec::kCascaded || auxiliaryCount == 0,
      "Cascaded cannot carry an auxiliary segment count");
  VELOX_CHECK_LE(
      cudfMetadata->size(),
      std::numeric_limits<std::size_t>::max() - kHeaderSize,
      "Exchange compression metadata envelope size overflow");

  auto output = std::make_unique<std::vector<uint8_t>>();
  output->reserve(kHeaderSize + cudfMetadata->size());
  output->insert(output->end(), kMagic.begin(), kMagic.end());
  appendScalar(*output, kVersion);
  appendScalar(*output, static_cast<uint16_t>(kHeaderSize));
  appendScalar(*output, static_cast<uint8_t>(codec));
  appendScalar(*output, static_cast<uint64_t>(logicalDataSize));
  appendScalar(*output, static_cast<uint64_t>(auxiliaryCount));
  appendScalar(*output, static_cast<uint64_t>(cudfMetadata->size()));
  output->insert(output->end(), cudfMetadata->begin(), cudfMetadata->end());
  return output;
}

ExchangePayloadMetadata unwrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> metadata) {
  VELOX_CHECK_NOT_NULL(metadata);
  if (metadata->size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), metadata->begin())) {
    return ExchangePayloadMetadata{std::move(metadata)};
  }

  VELOX_CHECK_GE(
      metadata->size(),
      kHeaderSize,
      "Truncated exchange compression metadata envelope");
  const auto* current = metadata->data() + kMagic.size();
  const auto* end = metadata->data() + metadata->size();
  const auto version = readScalar<uint16_t>(current, end);
  const auto headerSize = readScalar<uint16_t>(current, end);
  const auto codecValue = readScalar<uint8_t>(current, end);
  const auto logicalDataSize = readScalar<uint64_t>(current, end);
  const auto auxiliaryCount = readScalar<uint64_t>(current, end);
  const auto cudfMetadataSize = readScalar<uint64_t>(current, end);

  VELOX_CHECK_EQ(
      version, kVersion, "Unsupported exchange compression metadata version");
  VELOX_CHECK_EQ(
      headerSize, kHeaderSize, "Invalid exchange compression header size");
  VELOX_CHECK(
      codecValue == static_cast<uint8_t>(ExchangePayloadCodec::kFusedFor) ||
          codecValue == static_cast<uint8_t>(ExchangePayloadCodec::kCascaded),
      "Unsupported exchange payload codec");
  VELOX_CHECK_LE(
      logicalDataSize,
      std::numeric_limits<std::size_t>::max(),
      "Exchange logical payload size exceeds size_t");
  VELOX_CHECK_LE(
      auxiliaryCount,
      std::numeric_limits<std::size_t>::max(),
      "Exchange auxiliary count exceeds size_t");
  VELOX_CHECK(
      codecValue != static_cast<uint8_t>(ExchangePayloadCodec::kCascaded) ||
          auxiliaryCount == 0,
      "Cascaded cannot carry an auxiliary segment count");
  VELOX_CHECK_EQ(
      cudfMetadataSize,
      static_cast<uint64_t>(end - current),
      "Invalid exchange cuDF metadata size");

  const auto codec = static_cast<ExchangePayloadCodec>(codecValue);
  auto cudfMetadata = std::make_unique<std::vector<uint8_t>>(
      current, current + static_cast<std::size_t>(cudfMetadataSize));
  return ExchangePayloadMetadata{
      std::move(cudfMetadata),
      codec,
      static_cast<std::size_t>(logicalDataSize),
      static_cast<std::size_t>(auxiliaryCount)};
}

} // namespace facebook::velox::ucx_exchange
