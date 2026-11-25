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

#include <algorithm>

namespace facebook::nimble {

EncodingView::~EncodingView() {
  if (auto* bufferPool = options_.bufferPool) {
    bufferPool->release(std::move(decompressedPayload_));
  }
}

uint32_t EncodingView::read(
    std::span<const uint32_t> indices,
    const std::function<void(uint32_t)>& /*setNull*/,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    void* output) const {
  NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
  readAt(indices, output);
  if (dataType_ == DataType::String && addStringBuffer != nullptr) {
    addStrings(
        addStringBuffer,
        {static_cast<std::string_view*>(output), indices.size()});
  }
  return static_cast<uint32_t>(indices.size());
}

void EncodingView::addInputStrings(
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::span<std::string_view> values) const {
  if (addStringBuffer == nullptr || values.empty()) {
    return;
  }
  if (options_.inputStringBuffer == nullptr) {
    copyStrings(addStringBuffer, values);
    return;
  }

  // A velox::Buffer counts its own references, so a BufferPtr made from the
  // pointer shares the ownership of the buffer that the caller keeps alive.
  addStringBuffer(velox::BufferPtr{options_.inputStringBuffer});
}

namespace {

// Returns whether 'lhs' and 'rhs' are the same bytes, not just equal ones.
bool sameBytes(std::string_view lhs, std::string_view rhs) {
  return lhs.data() == rhs.data() && lhs.size() == rhs.size();
}

} // namespace

void EncodingView::copyStrings(
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::span<std::string_view> values) const {
  if (addStringBuffer == nullptr) {
    return;
  }
  uint64_t numBytes{0};
  std::string_view prevValue;
  for (const auto value : values) {
    if (value.empty() || sameBytes(value, prevValue)) {
      continue;
    }
    numBytes += value.size();
    prevValue = value;
  }
  if (numBytes == 0) {
    return;
  }
  auto buffer = velox::AlignedBuffer::allocate<char>(numBytes, pool_);
  auto* rawBuffer = buffer->asMutable<char>();
  std::string_view prevSource;
  std::string_view prevCopy;
  for (auto& value : values) {
    if (value.empty()) {
      continue;
    }
    if (!sameBytes(value, prevSource)) {
      std::copy(value.cbegin(), value.cend(), rawBuffer);
      prevSource = value;
      prevCopy = {rawBuffer, value.size()};
      rawBuffer += value.size();
    }
    value = prevCopy;
  }
  NIMBLE_CHECK_EQ(
      static_cast<uint64_t>(rawBuffer - buffer->as<char>()),
      numBytes,
      "Copied bytes must equal the bytes the first pass counted");
  addStringBuffer(std::move(buffer));
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
