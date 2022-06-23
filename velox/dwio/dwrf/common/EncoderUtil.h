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

#include "velox/dwio/dwrf/common/IntEncoder.h"
#include "velox/dwio/dwrf/common/RLEv1.h"

namespace facebook::velox::dwrf {

/**
 * Create an RLE encoder.
 * @param version version of RLE encoding to do
 * @param output the output stream to write to
 * @param useVInts whether varint encoding will be used
 * @param numBytes number of bytes the encoder will write for an integer
 */
template <bool isSigned>
std::unique_ptr<IntEncoder<isSigned>> createRleEncoder(
    RleVersion version,
    std::unique_ptr<BufferedOutputStream> output,
    bool useVInts,
    uint32_t numBytes) {
  switch (version) {
    case RleVersion::RleVersion_1:
      return std::make_unique<RleEncoderV1<isSigned>>(
          std::move(output), useVInts, numBytes);
    case RleVersion::RleVersion_2:
    default:
      DWIO_RAISE("RleVersion not supported");
      return {};
  }
}

template <bool isSigned>
std::unique_ptr<IntEncoder<isSigned>> createDirectEncoder(
    std::unique_ptr<BufferedOutputStream> output,
    bool useVInts,
    uint32_t numBytes) {
  return std::make_unique<IntEncoder<isSigned>>(
      std::move(output), useVInts, numBytes);
}

} // namespace facebook::velox::dwrf
