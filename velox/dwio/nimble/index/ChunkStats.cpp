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

#include "velox/dwio/nimble/index/ChunkStats.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/ChunkStatsGenerated.h"

namespace facebook::nimble::index {

std::unique_ptr<ChunkStats> ChunkStats::create(Section indexSection) {
  const auto* root = flatbuffers::GetRoot<serialization::ChunkStats>(
      indexSection.content().data());
  NIMBLE_CHECK_NOT_NULL(root);

  const auto* indexes = root->stripe_indexes();
  if (indexes == nullptr) {
    return nullptr;
  }

  std::vector<MetadataSection> groupSections;
  groupSections.reserve(indexes->size());
  for (uint32_t i = 0; i < indexes->size(); ++i) {
    const auto* ms = indexes->Get(i);
    const auto rawUncompressedSize = ms->uncompressed_size();
    groupSections.emplace_back(
        ms->offset(),
        ms->size(),
        static_cast<CompressionType>(ms->compression_type()),
        rawUncompressedSize > 0 ? std::optional<uint32_t>(rawUncompressedSize)
                                : std::nullopt);
  }

  return std::unique_ptr<ChunkStats>(
      new ChunkStats(std::move(indexSection), std::move(groupSections)));
}

ChunkStats::ChunkStats(
    Section indexSection,
    std::vector<MetadataSection> groupSections)
    : indexSection_{std::move(indexSection)},
      groupSections_{std::move(groupSections)} {}

const MetadataSection& ChunkStats::groupMetadata(uint32_t groupIndex) const {
  NIMBLE_CHECK_LT(
      groupIndex, groupSections_.size(), "Chunk stats group out of range");
  return groupSections_[groupIndex];
}

} // namespace facebook::nimble::index
