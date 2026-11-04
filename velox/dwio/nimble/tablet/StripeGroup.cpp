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
// bare, so an absent identifier (including pre-existing files) is kRaw.
StripeGroup::EncodingLayout stripeGroupEncodingLayout(
    std::string_view content) {
  constexpr size_t kIdentifierLength =
      flatbuffers::FlatBufferBuilder::kFileIdentifierLength;
  if (content.size() < sizeof(flatbuffers::uoffset_t) + kIdentifierLength) {
    return StripeGroup::EncodingLayout::kRaw;
  }
  const std::string_view identifier{
      flatbuffers::GetBufferIdentifier(content.data()), kIdentifierLength};
  if (identifier == StripeGroup::kStreamMajorLayoutIdentifier.data()) {
    return StripeGroup::EncodingLayout::kStreamMajor;
  }
  if (identifier == StripeGroup::kStreamMajorSparseLayoutIdentifier.data()) {
    return StripeGroup::EncodingLayout::kStreamMajorSparse;
  }
  // A kRaw blob cannot start with a layout identifier, so one here names a
  // layout from a newer writer.
  if (identifier.starts_with(StripeGroup::kLayoutIdentifierPrefix)) {
    NIMBLE_UNSUPPORTED("Unsupported stripe group layout: {}", identifier);
  }
  return StripeGroup::EncodingLayout::kRaw;
}

// Creates a random-access view over one encoded uint32 array and validates its
// length against `expectedRowCount` (the number of values the array must hold —
// stripeCount for a per-stream array). A kStreamMajorSparse stripe list passes
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

// Builds the views of a stream-major stripe group into `views`. With
// deduplication, arrays that reference one EncodedStream payload (TabletWriter
// stripe-group metadata deduplication) share one view, which is safe because
// view reads are stateless.
class StripeGroupViewBuilder {
 public:
  StripeGroupViewBuilder(
      std::vector<std::unique_ptr<EncodingView>>& views,
      bool deduplicate,
      velox::memory::MemoryPool* pool)
      : views_{views}, deduplicate_{deduplicate}, pool_{pool} {}

  // Returns the view over `payload`, checking that it holds `expectedRowCount`
  // values when given.
  EncodingView* view(
      const flatbuffers::Vector<uint8_t>* payload,
      std::optional<uint32_t> expectedRowCount) {
    if (!deduplicate_) {
      return create(payload, expectedRowCount);
    }
    const auto [it, inserted] = viewsByPayload_.try_emplace(payload, nullptr);
    if (inserted) {
      it->second = create(payload, expectedRowCount);
    } else if (expectedRowCount.has_value()) {
      NIMBLE_CHECK_FILE_EQ(
          it->second->rowCount(),
          *expectedRowCount,
          "Encoded stripe-group row count mismatch.");
    }
    return it->second;
  }

 private:
  // Creates a view over `payload` and keeps it in `views_`.
  EncodingView* create(
      const flatbuffers::Vector<uint8_t>* payload,
      std::optional<uint32_t> expectedRowCount) {
    return views_
        .emplace_back(makeStripeGroupView(payload, expectedRowCount, pool_))
        .get();
  }

  // Owns the created views.
  std::vector<std::unique_ptr<EncodingView>>& views_;

  // Shares one view among the arrays of one payload when true.
  const bool deduplicate_;

  // Allocates the views.
  velox::memory::MemoryPool* const pool_;

  // Maps each payload to its view when deduplicating.
  folly::F14FastMap<const flatbuffers::Vector<uint8_t>*, EncodingView*>
      viewsByPayload_;
};

// Returns whether a kStreamMajorSparse entry carries an array. An entry without
// data marks an omitted array.
bool hasArray(
    const serialization::EncodedStream* entry,
    std::string_view fieldName) {
  NIMBLE_CHECK_FILE_NOT_NULL(
      entry, "Null entry in StreamMajorSparseStripeGroup: {}", fieldName);
  const auto* data = entry->data();
  if (data == nullptr) {
    return false;
  }
  NIMBLE_CHECK_FILE_GT(
      data->size(),
      0,
      "Empty entry in StreamMajorSparseStripeGroup: {}",
      fieldName);
  return true;
}

// Checks that `stripeList` is strictly ascending and below `stripeCount`, as
// findStripe() requires.
void validateStripeList(
    const EncodingView& stripeList,
    uint32_t stripeCount,
    velox::memory::MemoryPool* pool) {
  const uint32_t numStripes = stripeList.rowCount();
  NIMBLE_CHECK_FILE_GT(numStripes, 0, "Empty stripe list in stripe group.");
  NIMBLE_CHECK_FILE_LE(
      numStripes, stripeCount, "Stripe list is longer than the stripe group.");
  Vector<uint32_t> values(pool, numStripes);
  stripeList.read(0, numStripes, values.data());
  for (size_t row = 1; row < values.size(); ++row) {
    NIMBLE_CHECK_FILE_LT(
        values[row - 1], values[row], "Stripe list is not strictly ascending.");
  }
  NIMBLE_CHECK_FILE_LT(
      values.back(),
      stripeCount,
      "Stripe list references a stripe beyond the stripe group.");
}

// Returns the position of `localStripe` in an ascending stripe list, or
// std::nullopt when the list does not contain it.
std::optional<uint32_t> findStripe(
    EncodingView& stripeList,
    uint32_t localStripe) {
  // EliasFano searches its index. The other encodings probe rows through
  // readEncodingViewAt, which skips the vtable dispatch that the generic
  // TypedEncodingView::lowerBound pays on every probe.
  if (stripeList.encodingType() == EncodingType::EliasFano) {
    uint32_t stripe{0};
    const uint32_t row =
        static_cast<EliasFanoEncodingView<uint32_t>&>(stripeList)
            .lowerBound(0, stripeList.rowCount(), localStripe, stripe);
    if (row == stripeList.rowCount() || stripe != localStripe) {
      return std::nullopt;
    }
    return row;
  }
  uint32_t begin{0};
  uint32_t end{stripeList.rowCount()};
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const uint32_t stripe = readEncodingViewAt<uint32_t>(&stripeList, middle);
    if (stripe == localStripe) {
      return middle;
    }
    if (stripe < localStripe) {
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
      streamMajor_.views.reserve(2 * static_cast<size_t>(streamCount_));
      StripeGroupViewBuilder viewBuilder{
          streamMajor_.views,
          encodingViewDeduplicationEnabled,
          pool,
      };
      for (uint32_t i = 0; i < streamCount_; ++i) {
        const auto* offsetsEntry = offsetsVec->Get(i);
        const auto* sizesEntry = sizesVec->Get(i);
        NIMBLE_CHECK_NOT_NULL(
            offsetsEntry,
            "Null stream_offsets entry in StreamMajorStripeGroup.");
        NIMBLE_CHECK_NOT_NULL(
            sizesEntry, "Null stream_sizes entry in StreamMajorStripeGroup.");
        streamMajor_.offsets.emplace_back(
            viewBuilder.view(offsetsEntry->data(), stripeCount_));
        streamMajor_.sizes.emplace_back(
            viewBuilder.view(sizesEntry->data(), stripeCount_));
      }
      return;
    }

    case EncodingLayout::kStreamMajorSparse: {
      const auto* root =
          asFlatBuffersRoot<serialization::StreamMajorSparseStripeGroup>(
              metadata_->content());
      stripeCount_ = root->stripe_count();
      NIMBLE_CHECK_FILE_GT(stripeCount_, 0, "Unexpected stripe count");
      streamCount_ = root->stream_count();

      // As with kStreamMajor, a group whose columns are all null has no
      // streams.
      if (streamCount_ == 0) {
        return;
      }
      const auto checkStreamVector = [&](const auto* vector,
                                         std::string_view fieldName) {
        NIMBLE_CHECK_FILE_NOT_NULL(
            vector,
            "Missing vector in StreamMajorSparseStripeGroup: {}",
            fieldName);
        NIMBLE_CHECK_FILE_EQ(
            vector->size(),
            streamCount_,
            "Vector length must equal stream_count: {}",
            fieldName);
      };
      const auto* stripesVec = root->stream_stripes();
      const auto* offsetsVec = root->stream_offsets();
      const auto* sizesVec = root->stream_sizes();
      checkStreamVector(stripesVec, "stream_stripes");
      checkStreamVector(offsetsVec, "stream_offsets");
      checkStreamVector(sizesVec, "stream_sizes");

      // The writer deduplicates kStreamMajorSparse arrays, so streams with the
      // same stripe list share its view, its validation and its searches.
      StripeGroupViewBuilder viewBuilder{
          streamMajorSparse_.views, /*deduplicate=*/true, pool};

      // Stripe lists validated so far. A payload can back both a stripe list
      // and another stream's offsets or sizes, so a list is validated when it
      // is first used as one, not when its view is created.
      folly::F14FastSet<const EncodingView*> validatedStripeLists;
      streamMajorSparse_.streams.reserve(streamCount_);
      for (uint32_t i = 0; i < streamCount_; ++i) {
        const auto* stripesEntry = stripesVec->Get(i);
        const auto* offsetsEntry = offsetsVec->Get(i);
        const auto* sizesEntry = sizesVec->Get(i);
        const bool hasOffsets = hasArray(offsetsEntry, "stream_offsets");
        const bool hasSizes = hasArray(sizesEntry, "stream_sizes");
        NIMBLE_CHECK_FILE_EQ(
            hasOffsets,
            hasSizes,
            "A stream must carry both offsets and sizes, or neither.");
        auto& stream = streamMajorSparse_.streams.emplace_back();
        uint32_t numStripes{stripeCount_};
        if (hasArray(stripesEntry, "stream_stripes")) {
          NIMBLE_CHECK_FILE(
              hasSizes, "A stream without data must omit its stripe list.");
          stream.stripeList = viewBuilder.view(
              stripesEntry->data(), /*expectedRowCount=*/std::nullopt);

          // Validates each distinct stripe list once.
          if (validatedStripeLists.insert(stream.stripeList).second) {
            validateStripeList(*stream.stripeList, stripeCount_, pool);
          }
          numStripes = stream.stripeList->rowCount();
        }
        if (hasSizes) {
          stream.offsets = viewBuilder.view(offsetsEntry->data(), numStripes);
          stream.sizes = viewBuilder.view(sizesEntry->data(), numStripes);
        }
      }
      return;
    }
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unknown StripeGroup encoding layout: {}", encodingLayout_));
}

StripeGroup::~StripeGroup() = default;

size_t StripeGroup::testingEncodingViewCount() const {
  return streamMajor_.views.size() + streamMajorSparse_.views.size();
}

StripeGroup::StreamMetadata StripeGroup::sparseLocation(
    uint32_t streamId,
    uint32_t localStripe,
    StripeSearch& lastSearch) const {
  const auto& stream = streamMajorSparse_.streams[streamId];
  if (stream.sizes == nullptr) {
    return {};
  }
  uint32_t row{localStripe};
  if (stream.stripeList != nullptr) {
    if (lastSearch.stripeList != stream.stripeList) {
      lastSearch = {
          stream.stripeList, findStripe(*stream.stripeList, localStripe)};
    }
    if (!lastSearch.row.has_value()) {
      return {};
    }
    row = *lastSearch.row;
  }
  const auto size = readEncodingViewAt<uint32_t>(stream.sizes, row);
  NIMBLE_CHECK_FILE_GT(
      size, 0, "Sparse stripe group lists a stripe without stream data.");
  return {readEncodingViewAt<uint32_t>(stream.offsets, row), size};
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
    case EncodingLayout::kStreamMajor:
      return readEncodingViewAt<uint32_t>(
          streamMajor_.offsets[streamId], stripeOffset);
    case EncodingLayout::kStreamMajorSparse: {
      StripeSearch search;
      return sparseLocation(streamId, stripeOffset, search).offset;
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
    case EncodingLayout::kStreamMajor:
      return readEncodingViewAt<uint32_t>(
          streamMajor_.sizes[streamId], stripeOffset);
    case EncodingLayout::kStreamMajorSparse: {
      StripeSearch search;
      return sparseLocation(streamId, stripeOffset, search).size;
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
      for (uint32_t streamId{0}; streamId < streamCount_; ++streamId) {
        const auto size = readEncodingViewAt<uint32_t>(
            streamMajor_.sizes[streamId], localStripe);
        if (size == 0) {
          locations[streamId] = StreamMetadata{};
          continue;
        }
        const auto offset = readEncodingViewAt<uint32_t>(
            streamMajor_.offsets[streamId], localStripe);
        locations[streamId] = StreamMetadata{offset, size};
      }
      return;
    }
    case EncodingLayout::kStreamMajorSparse: {
      StripeSearch lastSearch;
      for (uint32_t streamId{0}; streamId < streamCount_; ++streamId) {
        locations[streamId] = sparseLocation(streamId, localStripe, lastSearch);
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
      for (size_t i = 0; i < streamIds.size(); ++i) {
        const auto streamId = streamIds[i];
        if (streamId >= streamCount_) {
          locations[i] = StreamMetadata{};
          continue;
        }
        const auto size = readEncodingViewAt<uint32_t>(
            streamMajor_.sizes[streamId], localStripe);
        if (size == 0) {
          locations[i] = StreamMetadata{};
          continue;
        }
        const auto offset = readEncodingViewAt<uint32_t>(
            streamMajor_.offsets[streamId], localStripe);
        locations[i] = StreamMetadata{offset, size};
      }
      return;
    }
    case EncodingLayout::kStreamMajorSparse: {
      StripeSearch lastSearch;
      for (size_t i = 0; i < streamIds.size(); ++i) {
        const auto streamId = streamIds[i];
        locations[i] = streamId < streamCount_
            ? sparseLocation(streamId, localStripe, lastSearch)
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
    case StripeGroup::EncodingLayout::kStreamMajorSparse:
      return "kStreamMajorSparse";
  }
  return "Unknown";
}

std::ostream& operator<<(std::ostream& os, StripeGroup::EncodingLayout layout) {
  return os << toString(layout);
}

} // namespace facebook::nimble
