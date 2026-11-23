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

#include "velox/vector/ConstantVector.h"

#include "velox/vector/FlatVector.h"

namespace facebook::velox {
namespace {

// Returns the buffer among a flat vector's 'stringBuffers' that holds all bytes
// of 'value' and that a constant vector in 'pool' may share instead of copying
// 'value', or nullptr. A flat vector reuses its string buffers and packs many
// values into one, so:
// - Only values longer than FlatVector's kMaxStringSizeForReuse are shared.
//   Their buffers are too large for the flat vector to reuse, so sharing never
//   stops it from reusing a buffer, and copying a shorter value is cheap.
// - The value must fill at least half of the buffer, so the constant keeps
//   alive at most twice the memory a copy would.
// - The buffer must be allocated from 'pool', so the constant never holds
//   memory charged to another pool. Views have no pool and never qualify.
BufferPtr findSharableStringBuffer(
    StringView value,
    const std::vector<BufferPtr>& stringBuffers,
    memory::MemoryPool* pool) {
  if (value.size() <= FlatVector<StringView>::kMaxStringSizeForReuse) {
    return nullptr;
  }
  const auto valueBegin = reinterpret_cast<uintptr_t>(value.data());
  const auto valueEnd = valueBegin + value.size();
  for (const auto& buffer : stringBuffers) {
    if (buffer == nullptr || buffer->pool() != pool) {
      continue;
    }
    const auto bufferBegin = reinterpret_cast<uintptr_t>(buffer->as<char>());
    if (valueBegin >= bufferBegin && valueEnd <= bufferBegin + buffer->size()) {
      return buffer->capacity() <= 2 * value.size() ? buffer : nullptr;
    }
  }
  return nullptr;
}

} // namespace

template <>
void ConstantVector<StringView>::setStringValue(
    StringView value,
    BufferPtr buffer) {
  if (value.isInline()) {
    value_ = value;
    return;
  }
  if (buffer == nullptr) {
    buffer = AlignedBuffer::allocate<char>(value.size(), pool());
    memcpy(buffer->asMutable<char>(), value.data(), value.size());
    value = StringView(buffer->as<char>(), static_cast<int32_t>(value.size()));
  }
  value_ = value;
  stringBuffer_ = std::move(buffer);
}

template <>
void ConstantVector<StringView>::setStringValue(
    StringView value,
    const BaseVector& source) {
  // A lazy vector can load into a dictionary or a constant, so look through
  // them for the vector that holds the bytes.
  const auto* base = source.wrappedVector();
  if (const auto* flat = base->asFlatVector<StringView>()) {
    setStringValue(
        value, findSharableStringBuffer(value, flat->stringBuffers(), pool()));
    return;
  }
  if (base->isConstantEncoding()) {
    // A constant holds just its value and never reuses its buffer, so share
    // the buffer at any size.
    auto buffer = base->as<ConstantVector<StringView>>()->getStringBuffer();
    setStringValue(
        value,
        buffer != nullptr && buffer->pool() == pool() ? std::move(buffer)
                                                      : nullptr);
    return;
  }
  setStringValue(value, BufferPtr(nullptr));
}

} // namespace facebook::velox
