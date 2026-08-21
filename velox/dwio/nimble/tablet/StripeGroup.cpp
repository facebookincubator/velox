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
#include "velox/dwio/nimble/tablet/StripeGroup.h"

#include <algorithm>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/tablet/FooterGenerated.h"

#include "flatbuffers/flatbuffers.h"

namespace facebook::nimble {

namespace {

template <typename T>
const T* asFlatBuffersRoot(std::string_view content) {
  return flatbuffers::GetRoot<T>(content.data());
}

const serialization::Stripes* stripesRoot(const MetadataBuffer& stripes) {
  return asFlatBuffersRoot<serialization::Stripes>(stripes.content());
}

// Detects the StripeGroup encoding layout from the blob's FlatBuffers file
// identifier. Each non-kRaw layout carries its own identifier; kRaw blobs are
// bare, so an absent/unknown identifier (including pre-existing files) is kRaw.
StripeGroup::EncodingLayout stripeGroupEncodingLayout(
    std::string_view content) {
  if (flatbuffers::BufferHasIdentifier(
          content.data(), StripeGroup::kStreamMajorLayoutIdentifier.data())) {
    return StripeGroup::EncodingLayout::kStreamMajor;
  }
  return StripeGroup::EncodingLayout::kRaw;
}

// Creates a random-access view over one encoded uint32 array and validates its
// length against `expectedRowCount` (the number of values the array must hold —
// stripeCount for a per-stream array).
std::unique_ptr<EncodingView> makeStripeGroupView(
    const flatbuffers::Vector<uint8_t>* fbVec,
    uint32_t expectedRowCount,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(fbVec, "Missing encoded payload in stripe group.");
  const std::string_view encoded(
      reinterpret_cast<const char*>(fbVec->data()), fbVec->size());
  auto view = createEncodingView(encoded, pool);
  NIMBLE_CHECK_EQ(
      view->rowCount(),
      expectedRowCount,
      "Encoded stripe-group row count mismatch.");
  return view;
}

} // namespace

StripeGroup::StripeGroup(
    uint32_t stripeGroupIndex,
    const MetadataBuffer& stripes,
    std::unique_ptr<MetadataBuffer> stripeGroup,
    velox::memory::MemoryPool* pool)
    : metadata_{std::move(stripeGroup)},
      index_{stripeGroupIndex},
      // The blob self-identifies its layout via a FlatBuffers file identifier;
      // a kRaw blob carries none (see stripeGroupEncodingLayout).
      encodingLayout_{stripeGroupEncodingLayout(metadata_->content())} {
  NIMBLE_CHECK_NOT_NULL(pool);

  // The group's first stripe is the first stripe assigned to this group in the
  // tablet's per-stripe group_indices.
  const auto* groupIndices = stripesRoot(stripes)->group_indices();
  NIMBLE_CHECK_NOT_NULL(groupIndices, "Missing group_indices in stripes.");
  firstStripe_ = 0;
  bool found = false;
  for (uint32_t stripe = 0; stripe < groupIndices->size(); ++stripe) {
    if (groupIndices->Get(stripe) == stripeGroupIndex) {
      firstStripe_ = stripe;
      found = true;
      break;
    }
  }
  NIMBLE_CHECK(found, "No stripes found for stripe group.");

  switch (encodingLayout_) {
    case EncodingLayout::kRaw: {
      const auto* root =
          asFlatBuffersRoot<serialization::StripeGroup>(metadata_->content());
      stripeCount_ = root->stripe_count();
      NIMBLE_CHECK_GT(stripeCount_, 0, "Unexpected stripe count");
      // Flattened stripe-major arrays. streamCount is derived from the array
      // length; an all-null group (e.g. every selected column null) has
      // zero-length arrays and streamCount 0.
      const auto* offsets = root->stream_offsets();
      const auto* sizes = root->stream_sizes();
      const auto offsetsLen = offsets != nullptr ? offsets->size() : 0;
      const auto sizesLen = sizes != nullptr ? sizes->size() : 0;
      NIMBLE_CHECK_EQ(
          offsetsLen, sizesLen, "stream_offsets/stream_sizes length mismatch.");
      if (offsetsLen == 0) {
        streamCount_ = 0;
        return;
      }
      NIMBLE_CHECK_EQ(
          offsetsLen % stripeCount_,
          0,
          "stream_offsets length must be a multiple of stripe_count.");
      streamCount_ = offsetsLen / stripeCount_;
      raw_.offsets = offsets->data();
      raw_.sizes = sizes->data();
      return;
    }

    case EncodingLayout::kStreamMajor: {
      const auto* root =
          asFlatBuffersRoot<serialization::StreamMajorStripeGroup>(
              metadata_->content());
      stripeCount_ = root->stripe_count();
      NIMBLE_CHECK_GT(stripeCount_, 0, "Unexpected stripe count");
      streamCount_ = root->stream_count();
      // streamCount_ may legitimately be 0 (every column all-null); leave the
      // encoding lists empty.
      if (streamCount_ == 0) {
        return;
      }
      const auto* offsetsVec = root->stream_offsets();
      const auto* sizesVec = root->stream_sizes();
      NIMBLE_CHECK_NOT_NULL(
          offsetsVec, "Missing stream_offsets in StreamMajorStripeGroup.");
      NIMBLE_CHECK_NOT_NULL(
          sizesVec, "Missing stream_sizes in StreamMajorStripeGroup.");
      NIMBLE_CHECK_EQ(
          offsetsVec->size(),
          streamCount_,
          "stream_offsets length must equal stream_count.");
      NIMBLE_CHECK_EQ(
          sizesVec->size(),
          streamCount_,
          "stream_sizes length must equal stream_count.");
      streamMajor_.offsets.reserve(streamCount_);
      streamMajor_.sizes.reserve(streamCount_);
      for (uint32_t i = 0; i < streamCount_; ++i) {
        const auto* offsetsEntry = offsetsVec->Get(i);
        const auto* sizesEntry = sizesVec->Get(i);
        NIMBLE_CHECK_NOT_NULL(
            offsetsEntry,
            "Null stream_offsets entry in StreamMajorStripeGroup.");
        NIMBLE_CHECK_NOT_NULL(
            sizesEntry, "Null stream_sizes entry in StreamMajorStripeGroup.");
        streamMajor_.offsets.emplace_back(
            makeStripeGroupView(offsetsEntry->data(), stripeCount_, pool));
        streamMajor_.sizes.emplace_back(
            makeStripeGroupView(sizesEntry->data(), stripeCount_, pool));
      }
      return;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

StripeGroup::~StripeGroup() = default;

uint32_t StripeGroup::stripeOffset(uint32_t stripeIndex) const {
  NIMBLE_DCHECK_GE(
      stripeIndex, firstStripe_, "stripeIndex precedes this stripe group.");
  const uint32_t localIndex = stripeIndex - firstStripe_;
  NIMBLE_DCHECK_LT(
      localIndex, stripeCount_, "stripeIndex is beyond this stripe group.");
  return localIndex;
}

uint32_t StripeGroup::streamOffset(uint32_t stripeIndex, uint32_t streamId)
    const {
  NIMBLE_CHECK_LT(streamId, streamCount_, "streamId is out of range.");
  const auto stripeOffset = this->stripeOffset(stripeIndex);
  switch (encodingLayout_) {
    case EncodingLayout::kRaw:
      return raw_
          .offsets[static_cast<size_t>(stripeOffset) * streamCount_ + streamId];
    case EncodingLayout::kStreamMajor: {
      uint32_t offset;
      streamMajor_.offsets[streamId]->readAt(stripeOffset, &offset);
      return offset;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

uint32_t StripeGroup::streamSize(uint32_t stripeIndex, uint32_t streamId)
    const {
  NIMBLE_CHECK_LT(streamId, streamCount_, "streamId is out of range.");
  const auto stripeOffset = this->stripeOffset(stripeIndex);
  switch (encodingLayout_) {
    case EncodingLayout::kRaw:
      return raw_
          .sizes[static_cast<size_t>(stripeOffset) * streamCount_ + streamId];
    case EncodingLayout::kStreamMajor: {
      uint32_t size;
      streamMajor_.sizes[streamId]->readAt(stripeOffset, &size);
      return size;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

void StripeGroup::streamOffsets(uint32_t stripeIndex, std::span<uint32_t> out)
    const {
  NIMBLE_CHECK_EQ(
      out.size(), streamCount_, "output size must equal streamCount.");
  const uint32_t stripeOffset = this->stripeOffset(stripeIndex);
  switch (encodingLayout_) {
    case EncodingLayout::kRaw: {
      const auto* base =
          raw_.offsets + static_cast<size_t>(stripeOffset) * streamCount_;
      std::copy(base, base + streamCount_, out.begin());
      return;
    }
    case EncodingLayout::kStreamMajor:
      // TODO: add a range-read API to EncodingView and decode the stripe's full
      // offsets array in one call instead of per-stream readAt.
      for (uint32_t i = 0; i < streamCount_; ++i) {
        streamMajor_.offsets[i]->readAt(stripeOffset, &out[i]);
      }
      return;
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

void StripeGroup::streamSizes(uint32_t stripeIndex, std::span<uint32_t> out)
    const {
  NIMBLE_CHECK_EQ(
      out.size(), streamCount_, "output size must equal streamCount.");
  const uint32_t stripeOffset = this->stripeOffset(stripeIndex);
  switch (encodingLayout_) {
    case EncodingLayout::kRaw: {
      const auto* base =
          raw_.sizes + static_cast<size_t>(stripeOffset) * streamCount_;
      std::copy(base, base + streamCount_, out.begin());
      return;
    }
    case EncodingLayout::kStreamMajor:
      // TODO: add a range-read API to EncodingView and decode the stripe's full
      // sizes array in one call instead of per-stream readAt.
      for (uint32_t i = 0; i < streamCount_; ++i) {
        streamMajor_.sizes[i]->readAt(stripeOffset, &out[i]);
      }
      return;
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

std::string_view toString(StripeGroup::EncodingLayout layout) {
  switch (layout) {
    case StripeGroup::EncodingLayout::kRaw:
      return "kRaw";
    case StripeGroup::EncodingLayout::kStreamMajor:
      return "kStreamMajor";
  }
  return "Unknown";
}

std::ostream& operator<<(std::ostream& os, StripeGroup::EncodingLayout layout) {
  return os << toString(layout);
}

} // namespace facebook::nimble
