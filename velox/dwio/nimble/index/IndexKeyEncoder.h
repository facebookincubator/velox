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

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::nimble::index {

/// Encodes index keys for writing and lookup using a consistent byte format.
class IndexKeyEncoder {
 public:
  /// Allocates stable storage for one encoded batch.
  using BufferAllocator = std::function<void*(size_t)>;

  virtual ~IndexKeyEncoder() = default;

  /// Encodes one key per input row into memory provided by bufferAllocator.
  virtual void encode(
      const velox::VectorPtr& input,
      std::vector<std::string_view>& encodedKeys,
      const BufferAllocator& bufferAllocator) = 0;

  /// Encodes lookup bounds using the same format as write-path keys.
  virtual std::vector<velox::serializer::EncodedKeyBounds> encodeIndexBounds(
      const velox::serializer::IndexBounds& indexBounds) = 0;
};

/// Creates the built-in Velox-backed index key encoder.
std::unique_ptr<IndexKeyEncoder> createNimbleIndexKeyEncoder(
    const std::vector<std::string>& columns,
    const velox::RowTypePtr& inputType,
    const std::vector<SortOrder>& sortOrders,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index
