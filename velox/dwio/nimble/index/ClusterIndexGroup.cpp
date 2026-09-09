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
#include "velox/dwio/nimble/index/ClusterIndexGroup.h"

#include "velox/dwio/nimble/tablet/ClusterIndexGenerated.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <algorithm>

namespace facebook::nimble::index {

namespace {
template <typename T>
const T* asFlatBuffersRoot(std::string_view content) {
  return flatbuffers::GetRoot<T>(content.data());
}

uint32_t getStripeCount(
    const MetadataBuffer& rootIndex,
    uint32_t stripeGroupIndex) {
  const auto* indexRoot =
      asFlatBuffersRoot<serialization::ClusterIndex>(rootIndex.content());
  auto* stripeCounts = indexRoot->stripe_counts();
  NIMBLE_CHECK_NOT_NULL(stripeCounts);
  NIMBLE_CHECK_LT(stripeGroupIndex, stripeCounts->size());
  const uint32_t endStripe = stripeCounts->Get(stripeGroupIndex);
  const uint32_t startStripe =
      stripeGroupIndex == 0 ? 0 : stripeCounts->Get(stripeGroupIndex - 1);
  return endStripe - startStripe;
}

uint32_t getFirstStripe(
    const MetadataBuffer& rootIndex,
    uint32_t stripeGroupIndex) {
  const auto* indexRoot =
      asFlatBuffersRoot<serialization::ClusterIndex>(rootIndex.content());
  auto* stripeCounts = indexRoot->stripe_counts();
  NIMBLE_CHECK_NOT_NULL(stripeCounts);
  NIMBLE_CHECK_LT(stripeGroupIndex, stripeCounts->size());
  return stripeGroupIndex == 0 ? 0 : stripeCounts->Get(stripeGroupIndex - 1);
}

} // namespace

std::shared_ptr<ClusterIndexGroup> ClusterIndexGroup::create(
    uint32_t stripeGroupIndex,
    const MetadataBuffer& rootIndex,
    std::unique_ptr<MetadataBuffer> groupIndex) {
  const uint32_t firstStripe = getFirstStripe(rootIndex, stripeGroupIndex);
  const uint32_t stripeCount = getStripeCount(rootIndex, stripeGroupIndex);

  return std::shared_ptr<ClusterIndexGroup>(new ClusterIndexGroup(
      stripeGroupIndex, firstStripe, stripeCount, std::move(groupIndex)));
}

ClusterIndexGroup::ClusterIndexGroup(
    uint32_t groupIndex,
    uint32_t firstStripe,
    uint32_t stripeCount,
    std::unique_ptr<MetadataBuffer> metadata)
    : metadata_{std::move(metadata)},
      groupIndex_{groupIndex},
      firstStripe_{firstStripe},
      stripeCount_{stripeCount} {}

std::optional<ChunkLocation> ClusterIndexGroup::lookupChunk(
    uint32_t stripe,
    std::string_view encodedKey) const {
  const uint32_t stripeOffset = this->stripeOffset(stripe);
  const auto* root = asFlatBuffersRoot<serialization::StripeClusterIndex>(
      metadata_->content());

  // No chunk index sub-table means single chunk per stripe — return default
  // location at offset 0.
  const auto* chunkIndex = root->chunk_index();
  if (chunkIndex == nullptr) {
    return ChunkLocation{0, 0, 0, 0};
  }

  // Get chunk counts to determine the search range for this stripe.
  const auto* chunkCounts = chunkIndex->chunk_counts();
  NIMBLE_CHECK_NOT_NULL(chunkCounts);
  NIMBLE_CHECK_EQ(stripeCount_, chunkCounts->size());

  // Determine the chunk range for this stripe
  const uint32_t startChunkIndex =
      stripeOffset == 0 ? 0 : chunkCounts->Get(stripeOffset - 1);
  const uint32_t endChunkIndex = chunkCounts->Get(stripeOffset);

  const auto* chunkKeys = chunkIndex->chunk_keys();
  NIMBLE_CHECK_NOT_NULL(chunkKeys);
  NIMBLE_CHECK_LE(endChunkIndex, chunkKeys->size());

  // Binary search for the chunk key within this stripe's range.
  // Chunk keys are the last key in each chunk. lower_bound finds the first
  // chunk whose last key >= encodedKey, meaning the chunk that may contain
  // the target key.
  auto it = std::lower_bound(
      chunkKeys->begin() + startChunkIndex,
      chunkKeys->begin() + endChunkIndex,
      encodedKey,
      [](const flatbuffers::String* a, std::string_view b) {
        return a->string_view() < b;
      });
  if (it == chunkKeys->begin() + endChunkIndex) {
    return std::nullopt;
  }

  const uint32_t chunkOffset = it - chunkKeys->begin();

  const auto* chunkRows = chunkIndex->chunk_rows();
  NIMBLE_CHECK_NOT_NULL(chunkRows);
  NIMBLE_CHECK_EQ(chunkRows->size(), chunkKeys->size());

  const auto* chunkOffsets = chunkIndex->chunk_offsets();
  NIMBLE_CHECK_NOT_NULL(chunkOffsets);
  NIMBLE_CHECK_EQ(chunkOffsets->size(), chunkKeys->size());

  // Return the found chunk location.
  const uint32_t rowOffset =
      chunkOffset == startChunkIndex ? 0 : chunkRows->Get(chunkOffset - 1);
  return ChunkLocation{
      chunkOffset - startChunkIndex,
      chunkOffsets->Get(chunkOffset),
      0,
      rowOffset};
}

velox::common::Region ClusterIndexGroup::keyStreamRegion(
    uint32_t stripeIndex,
    uint64_t stripeBaseOffset) const {
  const uint32_t stripeOffset = this->stripeOffset(stripeIndex);
  const auto* root = asFlatBuffersRoot<serialization::StripeClusterIndex>(
      metadata_->content());

  const auto* keyStream = root->key_stream();
  NIMBLE_CHECK_NOT_NULL(keyStream);

  const auto* offsets = keyStream->offsets();
  NIMBLE_CHECK_NOT_NULL(offsets);
  const auto* sizes = keyStream->sizes();
  NIMBLE_CHECK_NOT_NULL(sizes);
  NIMBLE_CHECK_EQ(
      offsets->size(),
      sizes->size(),
      "offsets and sizes must have the same size");
  NIMBLE_CHECK_LT(stripeOffset, offsets->size());

  const uint32_t keyStreamOffset = offsets->Get(stripeOffset);
  const uint32_t keyStreamSize = sizes->Get(stripeOffset);
  NIMBLE_CHECK_GT(keyStreamSize, 0);

  return velox::common::Region{
      stripeBaseOffset + keyStreamOffset, keyStreamSize};
}

} // namespace facebook::nimble::index
