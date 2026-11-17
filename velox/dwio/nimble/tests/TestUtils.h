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

#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

namespace facebook::nimble::test {
// Calculate the raw Stream Size.
inline std::uint64_t getRawStreamSize(
    velox::memory::MemoryPool& pool,
    nimble::TabletReader& tablet) {
  // Calculate expected size by summing stream sizes.
  uint64_t expected = 0;
  for (auto i = 0; i < tablet.stripeCount(); ++i) {
    auto stripeIdentifier = tablet.stripeIdentifier(i);

    auto numStreams = tablet.streamCount(stripeIdentifier);
    std::vector<uint32_t> identifiers(numStreams);
    std::iota(identifiers.begin(), identifiers.end(), 0);
    auto streams = tablet.load(stripeIdentifier, identifiers);

    // Skip nullStreams indicated by nullptr.
    for (auto& stream : streams) {
      if (stream == nullptr) {
        continue;
      }
      nimble::InMemoryChunkedStream chunkedStream{pool, std::move(stream)};
      while (chunkedStream.hasNext()) {
        auto chunk = chunkedStream.nextChunk();
        expected += TestUtils::getRawDataSize(pool, chunk);
      }
    }
  }
  return expected;
}

} // namespace facebook::nimble::test
