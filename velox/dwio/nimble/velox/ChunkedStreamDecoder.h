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

#include "velox/dwio/nimble/common/MetricsLogger.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/Decoder.h"

namespace facebook::nimble {

class ChunkedStreamDecoder : public Decoder {
 public:
  /// Creates a decoder for one encoded chunk.
  using EncodingFactoryCallback = std::function<std::unique_ptr<Encoding>(
      velox::memory::MemoryPool&,
      std::string_view,
      std::function<void*(uint32_t)>)>;

  ChunkedStreamDecoder(
      velox::memory::MemoryPool& pool,
      std::unique_ptr<ChunkedStream> stream,
      EncodingFactoryCallback encodingFactory,
      bool stringDecoderZeroCopy,
      const MetricsLogger& logger)
      : pool_{&pool},
        stream_{std::move(stream)},
        encodingFactory_{std::move(encodingFactory)},
        stringDecoderZeroCopy_{stringDecoderZeroCopy},
        logger_{logger} {}

  uint32_t next(
      uint32_t count,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) override;

  uint32_t read(
      std::span<const uint32_t> rows,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  uint32_t read(
      std::span<const RowRange> ranges,
      DataType dataType,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override;

  void skip(uint32_t count) override;

  void reset() override;

  void ensureLoaded();

  const Encoding* encoding() const override {
    return encoding_.get();
  }

 private:
  velox::memory::MemoryPool* const pool_;
  std::unique_ptr<ChunkedStream> stream_;
  std::unique_ptr<Encoding> encoding_;
  uint32_t remaining_{0};
  const EncodingFactoryCallback encodingFactory_;
  const bool stringDecoderZeroCopy_;
  const MetricsLogger& logger_;
  std::vector<velox::BufferPtr> currentStringBuffers_;
  // For compatibility with old encoding behavior.
  std::vector<Vector<char>> stringBuffers_;
};

} // namespace facebook::nimble
