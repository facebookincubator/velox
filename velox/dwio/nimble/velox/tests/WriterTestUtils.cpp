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

#include "velox/dwio/nimble/velox/tests/WriterTestUtils.h"

#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook {

std::vector<velox::RowVectorPtr> generateBatches(
    const std::shared_ptr<const velox::RowType>& type,
    size_t batchCount,
    size_t size,
    uint32_t seed,
    velox::memory::MemoryPool& pool) {
  velox::VectorFuzzer fuzzer(
      {.vectorSize = size, .nullRatio = 0.1}, &pool, seed);
  std::vector<velox::RowVectorPtr> batches;
  batches.reserve(batchCount);
  for (size_t i = 0; i < batchCount; ++i) {
    batches.push_back(fuzzer.fuzzInputFlatRow(type));
  }
  return batches;
}

ChunkSizeResults validateChunkSize(
    nimble::BatchReader& reader,
    const uint64_t minStreamChunkRawSize,
    const uint64_t maxStreamChunkRawSize) {
  constexpr int kRowCountOffset = 2;
  constexpr double kMaxErrorRate = 0.2;
  const auto& tablet = reader.tabletReader();
  auto& pool = reader.memoryPool();

  std::unordered_set<uint32_t> stringStreamOffsets;
  nimble::SchemaReader::traverseSchema(
      reader.schema(),
      [&](uint32_t /*level*/,
          const nimble::Type& type,
          const nimble::SchemaReader::NodeInfo& /*info*/) {
        if (type.isScalar()) {
          auto scalarKind = type.asScalar().scalarDescriptor().scalarKind();
          if (scalarKind == nimble::ScalarKind::String ||
              scalarKind == nimble::ScalarKind::Binary) {
            stringStreamOffsets.insert(
                type.asScalar().scalarDescriptor().offset());
          }
        }
      });

  const uint32_t stripeCount = tablet.stripeCount();
  uint32_t maxChunkCount = 0;
  uint32_t minChunkCount = std::numeric_limits<uint32_t>::max();

  for (uint32_t stripeIndex = 0; stripeIndex < stripeCount; ++stripeIndex) {
    const auto stripeIdentifier = tablet.stripeIdentifier(stripeIndex);
    const auto streamCount = tablet.streamCount(stripeIdentifier);

    std::vector<uint32_t> streamIds(streamCount);
    std::iota(streamIds.begin(), streamIds.end(), 0);
    auto streamLoaders = tablet.load(stripeIdentifier, streamIds);

    for (uint32_t streamId = 0; streamId < streamLoaders.size(); ++streamId) {
      if (!streamLoaders[streamId]) {
        continue;
      }
      nimble::InMemoryChunkedStream chunkedStream{
          pool, std::move(streamLoaders[streamId])};
      uint32_t currentStreamChunkCount = 0;
      const bool isStringStream = stringStreamOffsets.contains(streamId);
      while (chunkedStream.hasNext()) {
        ++currentStreamChunkCount;
        const auto chunk = chunkedStream.nextChunk();
        const uint64_t chunkRawDataSize =
            nimble::test::TestUtils::getRawDataSize(pool, chunk);
        const uint32_t rowCount =
            *reinterpret_cast<const uint32_t*>(chunk.data() + kRowCountOffset);
        const double stringError =
            isStringStream ? rowCount * sizeof(std::string_view) : 0;

        // For string streams, a single row may exceed max chunk size if the
        // string itself is larger than the limit. This is acceptable since we
        // cannot split a single string value.
        if (!(isStringStream && rowCount == 1)) {
          const double maxError =
              kMaxErrorRate * maxStreamChunkRawSize + stringError;
          EXPECT_LE(chunkRawDataSize, maxStreamChunkRawSize + maxError)
              << "Stream " << streamId << " has a chunk with size "
              << chunkRawDataSize << " which is above max chunk size of "
              << maxStreamChunkRawSize;
        }

        // Validate min chunk size when not last chunk
        if (chunkedStream.hasNext() &&
            chunkRawDataSize < minStreamChunkRawSize) {
          const double minError =
              kMaxErrorRate * minStreamChunkRawSize + stringError;
          EXPECT_GE(chunkRawDataSize, minStreamChunkRawSize - minError)
              << "Stream " << streamId << " has a non-last chunk with size "
              << chunkRawDataSize << " which is below min chunk size of "
              << minStreamChunkRawSize;
        }
      }
      DWIO_ENSURE_GT(
          currentStreamChunkCount,
          0,
          "Non null streams should have at least one chunk");
      maxChunkCount = std::max(maxChunkCount, currentStreamChunkCount);
      minChunkCount = std::min(minChunkCount, currentStreamChunkCount);
    }
  }

  return ChunkSizeResults{
      .stripeCount = stripeCount,
      .minChunkCount = minChunkCount,
      .maxChunkCount = maxChunkCount,
  };
}

} // namespace facebook
