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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/velox/Decoder.h"

namespace facebook::nimble {

/// Adapts a single encoded stream chunk to FieldReader's selected-row API.
class EncodingViewDecoder final : public Decoder {
 public:
  /// Creates an EncodingView for the unwrapped chunk payload.
  using EncodingViewFactory =
      std::function<std::shared_ptr<const EncodingView>(std::string_view)>;

  /// Creates a decoder over one raw encoded chunk. `pool` and the view returned
  /// by `encodingViewFactory` must be non-null.
  EncodingViewDecoder(
      std::string_view encoded,
      velox::memory::MemoryPool* pool,
      const EncodingViewFactory& encodingViewFactory);

  uint32_t next(
      uint32_t count,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) override;

  /// Decodes the selected source rows densely into `output`.
  uint32_t read(
      std::span<const uint32_t> rows,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  /// Decodes ordered, disjoint source ranges densely into `output`.
  uint32_t read(
      std::span<const RowRange> ranges,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  void skip(uint32_t count) override;

  void reset() override;

  const Encoding* FOLLY_NULLABLE encoding() const override;

 private:
  // Memory pool used to allocate retained string buffers.
  velox::memory::MemoryPool* const pool_;

  // Random-access view over the encoded values.
  std::shared_ptr<const EncodingView> encodingView_;
};

} // namespace facebook::nimble
