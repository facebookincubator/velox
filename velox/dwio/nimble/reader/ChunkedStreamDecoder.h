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

#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/common/MetricsLogger.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/reader/Decoder.h"

namespace facebook::nimble {

class ChunkedStreamDecoder : public Decoder {
 public:
  ChunkedStreamDecoder(
      velox::memory::MemoryPool& pool,
      std::unique_ptr<ChunkedStream> stream,
      std::function<std::unique_ptr<Encoding>(
          velox::memory::MemoryPool&,
          std::string_view,
          std::function<void*(uint32_t)>)> encodingFactory,
      bool optimizeStringBufferHandling,
      const MetricsLogger& logger)
      : pool_{pool},
        stream_{std::move(stream)},
        encodingFactory_{std::move(encodingFactory)},
        optimizeStringBufferHandling_{optimizeStringBufferHandling},
        logger_{logger} {}

  uint32_t next(
      uint32_t count,
      void* output,
      std::vector<velox::BufferPtr>& stringBuffers,
      std::function<void*()> getOutputNulls = nullptr,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr) override;

  void skip(uint32_t count) override;

  void reset() override;

  void ensureLoaded();

  const Encoding* encoding() const override {
    return encoding_.get();
  }

 private:
  velox::memory::MemoryPool& pool_;
  std::unique_ptr<ChunkedStream> stream_;
  std::unique_ptr<Encoding> encoding_;
  uint32_t remaining_{0};
  std::function<std::unique_ptr<Encoding>(
      velox::memory::MemoryPool&,
      std::string_view,
      std::function<void*(uint32_t)>)>
      encodingFactory_;
  bool optimizeStringBufferHandling_;
  const MetricsLogger& logger_;
  std::vector<velox::BufferPtr> currentStringBuffers_;
  // For compatibility with old encoding behavior.
  std::vector<Vector<char>> stringBuffers_;
};

} // namespace facebook::nimble
