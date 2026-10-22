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
#include "velox/dwio/nimble/index/KeyChunkBuilder.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "folly/String.h"
#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/velox/BufferGrowthPolicy.h"
#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble::index {
namespace {

const StreamDescriptorBuilder& keyStreamDescriptor() {
  static const StreamDescriptorBuilder descriptor{
      kKeyStreamId, ScalarKind::Binary};
  return descriptor;
}

const InputBufferGrowthPolicy& keyStreamGrowthPolicy() {
  static const auto policy =
      DefaultInputBufferGrowthPolicy::withDefaultRanges();
  return *velox::checkedNotNull(policy.get());
}

bool isKeyOrdered(int32_t comparison, bool noDuplicateKey) {
  return noDuplicateKey ? comparison > 0 : comparison >= 0;
}

[[noreturn]] void failKeyOrder(
    bool noDuplicateKey,
    size_t currIndex,
    std::string_view currKey,
    size_t prevIndex,
    std::string_view prevKey) {
  NIMBLE_USER_FAIL(
      noDuplicateKey
          ? "Encoded keys must be in strictly ascending order (duplicates are not allowed). "
            "Key at index {} (hex: {}) is not greater than key at index {} (hex: {})"
          : "Encoded keys must be in ascending order. "
            "Key at index {} (hex: {}) is less than key at index {} (hex: {})",
      currIndex,
      folly::hexlify(currKey),
      prevIndex,
      folly::hexlify(prevKey));
}

class FlatKeyChunkBuilder final : public KeyChunkBuilder {
 public:
  FlatKeyChunkBuilder(
      std::unique_ptr<IndexKeyEncoder> keyEncoder,
      EncodingLayout encodingLayout,
      bool enforceKeyOrder,
      bool noDuplicateKey,
      velox::memory::MemoryPool* pool)
      : keyEncoder_{std::move(keyEncoder)},
        encodingLayout_{std::move(encodingLayout)},
        enforceKeyOrder_{enforceKeyOrder},
        noDuplicateKey_{noDuplicateKey},
        pool_{pool},
        keyStream_{std::make_unique<ContentStreamData<std::string_view>>(
            *pool_,
            keyStreamDescriptor(),
            keyStreamGrowthPolicy())},
        keyBuffer_{std::make_unique<Buffer>(*pool_)} {
    NIMBLE_CHECK_NOT_NULL(keyEncoder_);
  }

  void append(const velox::VectorPtr& input) override {
    const auto newKeyStart = keyStream_->mutableData().size();
    keyStream_->ensureMutableDataCapacity(newKeyStart + input->size());

    encodedKeys_.clear();
    encodedKeys_.reserve(input->size());
    keyEncoder_->encode(input, encodedKeys_, [this](size_t size) {
      return keyBuffer_->reserve(size);
    });
    for (const auto key : encodedKeys_) {
      keyStream_->mutableData().emplace_back(key);
    }
    validateOrder(newKeyStart);
  }

  size_t size() const override {
    return keyStream_->mutableData().size();
  }

  std::string keyAt(size_t row) const override {
    // The interface returns ownership because the hierarchical builder
    // synthesizes boundary keys. Keep both implementations interchangeable.
    return std::string{keyAtView(row)};
  }

  std::string_view encode(size_t offset, uint32_t count, Buffer& buffer)
      const override {
    return encodeFlatKeys(
        encodingLayout_,
        std::span<const std::string_view>{keyStream_->mutableData()}.subspan(
            offset, count),
        buffer);
  }

  void clear() override {
    const auto& keys = keyStream_->mutableData();
    if (keys.empty()) {
      return;
    }
    lastKey_ = std::string(keys.back());
    keyStream_->reset();
    keyBuffer_ = std::make_unique<Buffer>(*velox::checkedNotNull(pool_));
  }

 private:
  std::string_view keyAtView(size_t row) const {
    NIMBLE_CHECK_LT(row, size());
    return keyStream_->mutableData()[row];
  }

  void validateOrder(size_t newKeyStart) const {
    if (!enforceKeyOrder_) {
      return;
    }

    const auto& keys = keyStream_->mutableData();
    if (newKeyStart == 0 && lastKey_.has_value() &&
        !isKeyOrdered(
            keyAtView(0).compare(lastKey_.value()), noDuplicateKey_)) {
      failKeyOrder(noDuplicateKey_, 0, keyAtView(0), 0, lastKey_.value());
    }

    for (auto row = std::max<size_t>(newKeyStart, 1); row < keys.size();
         ++row) {
      if (FOLLY_UNLIKELY(!isKeyOrdered(
              keyAtView(row).compare(keyAtView(row - 1)), noDuplicateKey_))) {
        failKeyOrder(
            noDuplicateKey_, row, keyAtView(row), row - 1, keyAtView(row - 1));
      }
    }
  }

  const std::unique_ptr<IndexKeyEncoder> keyEncoder_;
  const EncodingLayout encodingLayout_;
  const bool enforceKeyOrder_;
  const bool noDuplicateKey_;
  velox::memory::MemoryPool* const pool_;
  const std::unique_ptr<ContentStreamData<std::string_view>> keyStream_;
  std::optional<std::string> lastKey_;
  std::unique_ptr<Buffer> keyBuffer_;
  std::vector<std::string_view> encodedKeys_;
};

} // namespace

std::string_view encodeFlatKeys(
    const EncodingLayout& layout,
    std::span<const std::string_view> keys,
    Buffer& buffer) {
  auto policy =
      std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
          layout,
          CompressionOptions{},
          [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
            return nullptr;
          });
  return EncodingFactory::encode<std::string_view>(
      std::move(policy), keys, buffer);
}

std::unique_ptr<KeyChunkBuilder> createFlatKeyChunkBuilder(
    std::unique_ptr<IndexKeyEncoder> keyEncoder,
    EncodingLayout encodingLayout,
    bool enforceKeyOrder,
    bool noDuplicateKey,
    velox::memory::MemoryPool* pool) {
  return std::make_unique<FlatKeyChunkBuilder>(
      std::move(keyEncoder),
      std::move(encodingLayout),
      enforceKeyOrder,
      noDuplicateKey,
      pool);
}

} // namespace facebook::nimble::index
