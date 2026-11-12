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
#include <optional>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/views/EliasFanoEncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewUtils.h"
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
// stripeCount for a per-stream array). A stream's stripe list passes
// std::nullopt, because its length is the stream's own stripe count.
std::unique_ptr<EncodingView> makeStripeGroupView(
    const flatbuffers::Vector<uint8_t>* fbVec,
    std::optional<uint32_t> expectedRowCount,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(fbVec, "Missing encoded payload in stripe group.");
  const std::string_view encoded(
      reinterpret_cast<const char*>(fbVec->data()), fbVec->size());
  auto view = createEncodingView(
      encoded, pool, Encoding::Options{.cacheEncodingView = true});
  NIMBLE_CHECK_FILE_EQ(
      view->dataType(),
      DataType::Uint32,
      "Encoded stripe-group array must hold uint32 values.");
  if (expectedRowCount.has_value()) {
    NIMBLE_CHECK_FILE_EQ(
        view->rowCount(),
        *expectedRowCount,
        "Encoded stripe-group row count mismatch.");
  }
  return view;
}

// Builds the views of a stream-major stripe group into `views`. Arrays that
// reference one EncodedStream payload (TabletWriter stripe-group metadata
// deduplication) share one view, which is safe because view reads are
// stateless.
class StripeGroupViewBuilder {
 public:
  StripeGroupViewBuilder(
      std::vector<std::unique_ptr<EncodingView>>& views,
      velox::memory::MemoryPool* pool)
      : pool_{pool}, views_{views} {}

  // Returns the view over `payload`, checking that it holds `expectedRowCount`
  // values when given.
  EncodingView* view(
      const flatbuffers::Vector<uint8_t>* payload,
      std::optional<uint32_t> expectedRowCount) {
    const auto [it, inserted] = viewMap_.try_emplace(payload, nullptr);
    if (inserted) {
      it->second = views_
                       .emplace_back(makeStripeGroupView(
                           payload, expectedRowCount, pool_))
                       .get();
    } else if (expectedRowCount.has_value()) {
      NIMBLE_CHECK_FILE_EQ(
          it->second->rowCount(),
          *expectedRowCount,
          "Encoded stripe-group row count mismatch.");
    }
    return it->second;
  }

 private:
  // Allocates the views.
  velox::memory::MemoryPool* const pool_;

  // Owns the created views.
  std::vector<std::unique_ptr<EncodingView>>& views_;

  // Maps each payload to its view.
  folly::F14FastMap<const flatbuffers::Vector<uint8_t>*, EncodingView*>
      viewMap_;
};

// Returns whether a stream-major entry carries encoded data. In a sparse group,
// an entry without data marks an omitted array.
bool hasEncodedData(
    const serialization::EncodedStream* entry,
    std::string_view fieldName) {
  NIMBLE_CHECK_FILE_NOT_NULL(
      entry, "Null entry in StreamMajorStripeGroup: {}", fieldName);
  const auto* data = entry->data();
  if (data == nullptr) {
    return false;
  }
  NIMBLE_CHECK_FILE_GT(
      data->size(), 0, "Empty entry in StreamMajorStripeGroup: {}", fieldName);
  return true;
}

// Checks that `stripeList` is not empty, not longer than the group of
// `stripeCount` stripes, and ends below `stripeCount`. These checks read one
// value, so they run on every load.
void checkStripeListBounds(EncodingView& stripeList, uint32_t stripeCount) {
  const uint32_t numNonEmptyStripes = stripeList.rowCount();
  NIMBLE_CHECK_FILE_GT(
      numNonEmptyStripes, 0, "Empty stripe list in stripe group.");
  NIMBLE_CHECK_FILE_LE(
      numNonEmptyStripes,
      stripeCount,
      "Stripe list is longer than the stripe group.");
  const uint32_t lastStripe =
      readEncodingViewAt<uint32_t>(&stripeList, numNonEmptyStripes - 1);
  NIMBLE_CHECK_FILE_LT(
      lastStripe,
      stripeCount,
      "Stripe list references a stripe beyond the stripe group.");
}

#ifndef NDEBUG
// Checks that `stripeList` is strictly ascending, as findStripe() requires.
void checkStripeListOrder(
    const EncodingView& stripeList,
    velox::memory::MemoryPool* pool) {
  Vector<uint32_t> values(pool, stripeList.rowCount());
  stripeList.read(0, stripeList.rowCount(), values.data());
  for (size_t row = 1; row < values.size(); ++row) {
    NIMBLE_CHECK_FILE_LT(
        values[row - 1], values[row], "Stripe list is not strictly ascending.");
  }
}
#endif

// Returns the position of group-local stripe `stripeIndex` in an ascending
// stripe list, or std::nullopt when the list does not contain it.
std::optional<uint32_t> findStripe(
    EncodingView& stripeList,
    uint32_t stripeIndex) {
  // EliasFano searches its index. The other encodings probe rows through
  // readEncodingViewAt, which skips the vtable dispatch that the generic
  // TypedEncodingView::lowerBound pays on every probe.
  if (stripeList.encodingType() == EncodingType::EliasFano) {
    uint32_t stripe{0};
    const uint32_t row =
        static_cast<EliasFanoEncodingView<uint32_t>&>(stripeList)
            .lowerBound(0, stripeList.rowCount(), stripeIndex, stripe);
    if (row == stripeList.rowCount() || stripe != stripeIndex) {
      return std::nullopt;
    }
    return row;
  }
  uint32_t begin{0};
  uint32_t end{stripeList.rowCount()};
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const uint32_t stripe = readEncodingViewAt<uint32_t>(&stripeList, middle);
    if (stripe == stripeIndex) {
      return middle;
    }
    if (stripe < stripeIndex) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return std::nullopt;
}

} // namespace

StripeGroup::StripeGroup(
    uint32_t stripeGroupIndex,
    const MetadataBuffer& stripes,
    std::unique_ptr<MetadataBuffer> stripeGroup,
    bool encodingViewDeduplicationEnabled,
    velox::memory::MemoryPool* pool)
    : metadata_{std::move(stripeGroup)},
      index_{stripeGroupIndex},
      // The blob self-identifies its layout via a FlatBuffers file identifier;
      // a kRaw blob carries none (see stripeGroupEncodingLayout).
      encodingLayout_{stripeGroupEncodingLayout(metadata_->content())},
      streamMajor_{makeStreamMajorMetadata(
          encodingLayout_,
          metadata_->content(),
          encodingViewDeduplicationEnabled,
          pool)} {
  NIMBLE_CHECK_NOT_NULL(pool);

  // The group's first stripe is the first stripe assigned to this group in the
  // tablet's per-stripe group_indices, which is non-decreasing because stripes
  // are assigned to groups in order.
  const auto* groupIndices = stripesRoot(stripes)->group_indices();
  NIMBLE_CHECK_NOT_NULL(groupIndices, "Missing group_indices in stripes.");
  const auto first = std::lower_bound(
      groupIndices->begin(), groupIndices->end(), stripeGroupIndex);
  NIMBLE_CHECK(
      first != groupIndices->end(),
      "No stripes found for stripe group {}.",
      stripeGroupIndex);
  NIMBLE_CHECK_EQ(
      *first, stripeGroupIndex, "No stripes found for stripe group.");
  firstStripe_ = static_cast<uint32_t>(first - groupIndices->begin());

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
      // streamCount_ may legitimately be 0 (every column all-null), which
      // leaves streamMajor_ empty.
      streamCount_ = root->stream_count();
      return;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

StripeGroup::~StripeGroup() = default;

StripeGroup::StreamMajorMetadata StripeGroup::makeStreamMajorMetadata(
    EncodingLayout encodingLayout,
    std::string_view content,
    bool encodingViewDeduplicationEnabled,
    velox::memory::MemoryPool* pool) {
  if (encodingLayout != EncodingLayout::kStreamMajor) {
    return {};
  }
  const auto* root =
      asFlatBuffersRoot<serialization::StreamMajorStripeGroup>(content);
  // A group without streams has nothing to build, and the constructor rejects
  // a group without stripes.
  if (root->stream_count() == 0 || root->stripe_count() == 0) {
    return {};
  }
  return {*root, root->stripe_count(), encodingViewDeduplicationEnabled, pool};
}

StripeGroup::StreamMajorMetadata::StreamMajorMetadata(
    const serialization::StreamMajorStripeGroup& root,
    uint32_t stripeCount,
    bool encodingViewDeduplicationEnabled,
    velox::memory::MemoryPool* pool)
    : sparseLayout_{root.stream_stripes() != nullptr} {
  const uint32_t streamCount = root.stream_count();
  const auto checkStreamVector = [&](const auto* vector,
                                     std::string_view fieldName) {
    NIMBLE_CHECK_FILE_NOT_NULL(
        vector, "Missing vector in StreamMajorStripeGroup: {}", fieldName);
    NIMBLE_CHECK_FILE_EQ(
        vector->size(),
        streamCount,
        "Vector length must equal stream_count: {}",
        fieldName);
  };
  const auto* stripesVec = root.stream_stripes();
  const auto* offsetsVec = root.stream_offsets();
  const auto* sizesVec = root.stream_sizes();
  if (sparseLayout_) {
    checkStreamVector(stripesVec, "stream_stripes");
  }
  checkStreamVector(offsetsVec, "stream_offsets");
  checkStreamVector(sizesVec, "stream_sizes");

  // Deduplicated arrays reference one EncodedStream payload, so they share one
  // view instead of building an identical one per stream. The writer always
  // deduplicates a sparse group, so streams with the same stripe list share
  // one view of it, and a batched lookup searches it once. The number of
  // distinct payloads is unknown up front, so views_ grows as needed. Without
  // sharing, every array gets its own view.
  const bool shareViews = sparseLayout_ || encodingViewDeduplicationEnabled;
  if (!shareViews) {
    views_.reserve(2 * static_cast<size_t>(streamCount));
  }
  StripeGroupViewBuilder viewBuilder{views_, pool};
  const auto viewFor = [&](const flatbuffers::Vector<uint8_t>* payload,
                           std::optional<uint32_t> expectedRowCount) {
    if (shareViews) {
      return viewBuilder.view(payload, expectedRowCount);
    }
    return views_
        .emplace_back(makeStripeGroupView(payload, expectedRowCount, pool))
        .get();
  };

  streams_.reserve(streamCount);
  for (uint32_t i = 0; i < streamCount; ++i) {
    const auto* offsetsEntry = offsetsVec->Get(i);
    const auto* sizesEntry = sizesVec->Get(i);
    const bool hasOffsets = hasEncodedData(offsetsEntry, "stream_offsets");
    const bool hasSizes = hasEncodedData(sizesEntry, "stream_sizes");
    NIMBLE_CHECK_FILE_EQ(
        hasOffsets,
        hasSizes,
        "A stream must carry both offsets and sizes, or neither.");
    const bool hasStripeList =
        sparseLayout_ && hasEncodedData(stripesVec->Get(i), "stream_stripes");
    auto& stream = streams_.emplace_back();
    if (!hasSizes) {
      NIMBLE_CHECK_FILE(
          sparseLayout_, "A dense stripe group must carry every array.");
      NIMBLE_CHECK_FILE(
          !hasStripeList, "A stream without data must omit its stripe list.");
      continue;
    }
    uint32_t numRows{stripeCount};
    if (hasStripeList) {
      stream.stripeList = viewFor(
          stripesVec->Get(i)->data(), /*expectedRowCount=*/std::nullopt);
      checkStripeListBounds(*stream.stripeList, stripeCount);
      numRows = stream.stripeList->rowCount();
    }
    stream.offsets = viewFor(offsetsEntry->data(), numRows);
    stream.sizes = viewFor(sizesEntry->data(), numRows);
  }

#ifndef NDEBUG
  // Checking the order decodes every stripe list, so only debug builds do it.
  // A list out of order only makes findStripe() miss, since its values are
  // compared, never used as indexes. Streams share deduplicated lists, so each
  // distinct list is checked once.
  folly::F14FastSet<const EncodingView*> validatedStripeLists;
  for (const auto& stream : streams_) {
    if (stream.stripeList != nullptr &&
        validatedStripeLists.insert(stream.stripeList).second) {
      checkStripeListOrder(*stream.stripeList, pool);
    }
  }
#endif
}

std::optional<uint32_t> StripeGroup::StreamMajorMetadata::row(
    uint32_t streamId,
    uint32_t stripeIndex,
    StripeSearch& lastSearch) const {
  const auto& stream = streams_[streamId];
  if (stream.sizes == nullptr) {
    return std::nullopt;
  }
  if (stream.stripeList == nullptr) {
    return stripeIndex;
  }
  if (lastSearch.stripeList != stream.stripeList) {
    lastSearch = {
        stream.stripeList, findStripe(*stream.stripeList, stripeIndex)};
  }
  return lastSearch.row;
}

uint32_t StripeGroup::StreamMajorMetadata::offset(
    uint32_t streamId,
    uint32_t row) const {
  return readEncodingViewAt<uint32_t>(streams_[streamId].offsets, row);
}

uint32_t StripeGroup::StreamMajorMetadata::size(uint32_t streamId, uint32_t row)
    const {
  const auto size = readEncodingViewAt<uint32_t>(streams_[streamId].sizes, row);
  if (sparseLayout_) {
    NIMBLE_CHECK_FILE_GT(
        size, 0, "A sparse stripe group must store only stripes with data.");
  }
  return size;
}

StripeGroup::StreamMetadata StripeGroup::StreamMajorMetadata::location(
    uint32_t streamId,
    uint32_t stripeIndex,
    StripeSearch& lastSearch) const {
  const auto row = this->row(streamId, stripeIndex, lastSearch);
  if (!row.has_value()) {
    return {};
  }
  const auto size = this->size(streamId, *row);
  if (size == 0) {
    return {};
  }
  return {offset(streamId, *row), size};
}

size_t StripeGroup::testingEncodingViewCount() const {
  return streamMajor_.numViews();
}

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
      StreamMajorMetadata::StripeSearch search;
      const auto row = streamMajor_.row(streamId, stripeOffset, search);
      return row.has_value() ? streamMajor_.offset(streamId, *row) : 0;
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
      StreamMajorMetadata::StripeSearch search;
      const auto row = streamMajor_.row(streamId, stripeOffset, search);
      return row.has_value() ? streamMajor_.size(streamId, *row) : 0;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

void StripeGroup::streamLocations(
    uint32_t stripeIndex,
    std::span<StreamMetadata> locations) const {
  NIMBLE_CHECK_EQ(
      locations.size(), streamCount_, "locations size must equal streamCount.");
  const auto localStripe = stripeOffset(stripeIndex);

  switch (encodingLayout_) {
    case EncodingLayout::kRaw: {
      const auto base = static_cast<size_t>(localStripe) * streamCount_;
      for (uint32_t streamId{0}; streamId < streamCount_; ++streamId) {
        const auto size = raw_.sizes[base + streamId];
        locations[streamId] = size == 0
            ? StreamMetadata{}
            : StreamMetadata{raw_.offsets[base + streamId], size};
      }
      return;
    }
    case EncodingLayout::kStreamMajor: {
      StreamMajorMetadata::StripeSearch lastSearch;
      for (uint32_t streamId{0}; streamId < streamCount_; ++streamId) {
        locations[streamId] =
            streamMajor_.location(streamId, localStripe, lastSearch);
      }
      return;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

void StripeGroup::streamLocations(
    uint32_t stripeIndex,
    std::span<const uint32_t> streamIds,
    std::span<StreamMetadata> locations) const {
  NIMBLE_CHECK_EQ(
      streamIds.size(),
      locations.size(),
      "streamIds and locations sizes must match.");
  const auto localStripe = stripeOffset(stripeIndex);

  switch (encodingLayout_) {
    case EncodingLayout::kRaw: {
      const auto base = static_cast<size_t>(localStripe) * streamCount_;
      for (size_t i = 0; i < streamIds.size(); ++i) {
        const auto streamId = streamIds[i];
        if (streamId >= streamCount_) {
          locations[i] = StreamMetadata{};
          continue;
        }
        const auto size = raw_.sizes[base + streamId];
        locations[i] = size == 0
            ? StreamMetadata{}
            : StreamMetadata{raw_.offsets[base + streamId], size};
      }
      return;
    }
    case EncodingLayout::kStreamMajor: {
      StreamMajorMetadata::StripeSearch lastSearch;
      for (size_t i = 0; i < streamIds.size(); ++i) {
        const auto streamId = streamIds[i];
        locations[i] = streamId < streamCount_
            ? streamMajor_.location(streamId, localStripe, lastSearch)
            : StreamMetadata{};
      }
      return;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

StripeGroup::StreamMetadata StripeGroup::streamLocation(
    uint32_t stripeIndex,
    uint32_t streamId) const {
  StreamMetadata location;
  streamLocations(stripeIndex, {&streamId, 1}, {&location, 1});
  return location;
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
