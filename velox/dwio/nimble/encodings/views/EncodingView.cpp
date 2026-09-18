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

#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

EncodingView::~EncodingView() {
  if (auto* bufferPool = options_.bufferPool) {
    bufferPool->release(std::move(decompressedPayload_));
  }
}

uint32_t EncodingView::read(
    std::span<const uint32_t> indices,
    const std::function<void(uint32_t)>& /*setNull*/,
    void* output) const {
  NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
  readAt(indices, output);
  return static_cast<uint32_t>(indices.size());
}

std::string_view EncodingView::decompressPayload(
    CompressionType compressionType,
    DataType dataType,
    std::string_view payload) {
  if (compressionType == CompressionType::Uncompressed) {
    return payload;
  }
  NIMBLE_CHECK_NULL(decompressedPayload_);
  decompressedPayload_ = Compression::uncompress(
      *pool_,
      compressionType,
      dataType,
      payload,
      options_.decompressCounter(),
      options_.bufferPool);
  return {decompressedPayload_->as<char>(), decompressedPayload_->size()};
}

} // namespace facebook::nimble
