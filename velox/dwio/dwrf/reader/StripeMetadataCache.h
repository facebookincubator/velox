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

#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"

namespace facebook::velox::dwrf {

constexpr uint64_t INVALID_INDEX = std::numeric_limits<uint64_t>::max();

class StripeMetadataCache {
 public:
  StripeMetadataCache(
      const proto::PostScript& ps,
      const proto::Footer& footer,
      std::shared_ptr<dwio::common::DataBuffer<char>> buffer)
      : StripeMetadataCache{
            ps.cachemode(),
            std::move(buffer),
            getOffsets(footer)} {}

  StripeMetadataCache(
      proto::StripeCacheMode mode,
      std::shared_ptr<dwio::common::DataBuffer<char>> buffer,
      std::vector<uint32_t>&& offsets)
      : mode_{mode}, buffer_{std::move(buffer)}, offsets_{std::move(offsets)} {}

  StripeMetadataCache(
      const proto::PostScript& ps,
      const proto::Footer& footer,
      std::unique_ptr<dwio::common::SeekableInputStream> input)
      : mode_(ps.cachemode()),
        input_(std::move(input)),
        offsets_(getOffsets(footer)) {
    VELOX_CHECK(dynamic_cast<dwio::common::CacheInputStream*>(input_.get()));
  }

  ~StripeMetadataCache() = default;

  StripeMetadataCache(const StripeMetadataCache&) = delete;
  StripeMetadataCache(StripeMetadataCache&&) = delete;
  StripeMetadataCache& operator=(const StripeMetadataCache&) = delete;
  StripeMetadataCache& operator=(StripeMetadataCache&&) = delete;

  bool has(proto::StripeCacheMode mode, uint64_t stripeIndex) const {
    return getIndex(mode, stripeIndex) != INVALID_INDEX;
  }

  std::unique_ptr<dwio::common::SeekableInputStream> get(
      proto::StripeCacheMode mode,
      uint64_t stripeIndex) const {
    auto index = getIndex(mode, stripeIndex);
    if (index != INVALID_INDEX) {
      auto offset = offsets_[index];
      if (buffer_) {
        return std::make_unique<dwio::common::SeekableArrayInputStream>(
            buffer_->data() + offset, offsets_[index + 1] - offset);
      } else {
        auto clone =
	  reinterpret_cast<dwio::common::CacheInputStream*>(input_.get())->clone();
        clone->Skip(offset);
	clone->setRemainingBytes(offsets_[index + 1] - offset);
        return clone;
      }
    }
    return {};
  }

 private:
  proto::StripeCacheMode mode_;
  std::shared_ptr<dwio::common::DataBuffer<char>> buffer_;
  std::unique_ptr<dwio::common::SeekableInputStream> input_;
  std::vector<uint32_t> offsets_;

  uint64_t getIndex(proto::StripeCacheMode mode, uint64_t stripeIndex) const {
    if (mode_ & mode) {
      uint64_t index =
          (mode_ == mode
               ? stripeIndex
               : stripeIndex * 2 + mode - proto::StripeCacheMode::INDEX);
      // offsets has N + 1 items, so length[N] = offset[N+1]- offset[N]
      if (index < offsets_.size() - 1) {
        return index;
      }
    }
    return INVALID_INDEX;
  }

  std::vector<uint32_t> getOffsets(const proto::Footer& footer) {
    std::vector<uint32_t> offsets;
    offsets.reserve(footer.stripecacheoffsets_size());
    const auto& from = footer.stripecacheoffsets();
    offsets.assign(from.begin(), from.end());
    return offsets;
  }
};

} // namespace facebook::velox::dwrf
