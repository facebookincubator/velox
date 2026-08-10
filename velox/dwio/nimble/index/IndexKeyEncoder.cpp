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
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {
namespace {

class NimbleIndexKeyEncoder final : public IndexKeyEncoder {
 public:
  explicit NimbleIndexKeyEncoder(
      std::unique_ptr<velox::serializer::KeyEncoder> encoder)
      : encoder_{std::move(encoder)} {}

  void encode(
      const velox::VectorPtr& input,
      std::vector<std::string_view>& encodedKeys,
      const BufferAllocator& bufferAllocator) override {
    encoder_->encode(input, encodedKeys, bufferAllocator);
  }

  std::vector<velox::serializer::EncodedKeyBounds> encodeIndexBounds(
      const velox::serializer::IndexBounds& indexBounds) override {
    return encoder_->encodeIndexBounds(indexBounds);
  }

 private:
  std::unique_ptr<velox::serializer::KeyEncoder> encoder_;
};

} // namespace

std::unique_ptr<IndexKeyEncoder> createNimbleIndexKeyEncoder(
    const std::vector<std::string>& columns,
    const velox::RowTypePtr& inputType,
    const std::vector<SortOrder>& sortOrders,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_EQ(
      sortOrders.size(),
      columns.size(),
      "sortOrders and columns must have the same size");
  std::vector<velox::core::SortOrder> veloxSortOrders;
  veloxSortOrders.reserve(sortOrders.size());
  for (const auto& sortOrder : sortOrders) {
    veloxSortOrders.emplace_back(sortOrder.toVeloxSortOrder());
  }
  return std::make_unique<NimbleIndexKeyEncoder>(
      velox::serializer::KeyEncoder::create(
          columns, inputType, std::move(veloxSortOrders), pool));
}

} // namespace facebook::nimble::index
