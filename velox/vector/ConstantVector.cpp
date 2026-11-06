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

#include <gflags/gflags.h>

#include "velox/vector/FlatVector.h"

DECLARE_int64(velox_constant_vector_share_string_min_bytes);

namespace facebook::velox {
namespace {

// Returns the buffer among 'candidates' that holds all bytes of 'value' and
// that a constant vector in 'pool' may share instead of copying 'value', or
// nullptr. A buffer qualifies only if it is allocated from 'pool', so the
// constant never holds memory charged to another pool. FlatVector::copy follows
// the same rule, but checks the source vector's pool rather than each buffer's.
// Views have no pool and never qualify.
BufferPtr findSharableStringBuffer(
    StringView value,
    folly::Range<const BufferPtr*> candidates,
    memory::MemoryPool* pool) {
  const int64_t minBytes = FLAGS_velox_constant_vector_share_string_min_bytes;
  if (minBytes < 0 || value.isInline() ||
      static_cast<int64_t>(value.size()) < minBytes) {
    return nullptr;
  }
  const auto valueBegin = reinterpret_cast<uintptr_t>(value.data());
  const auto valueEnd = valueBegin + value.size();
  for (const auto& buffer : candidates) {
    if (buffer == nullptr || buffer->pool() != pool) {
      continue;
    }
    const auto bufferBegin = reinterpret_cast<uintptr_t>(buffer->as<char>());
    if (valueBegin >= bufferBegin && valueEnd <= bufferBegin + buffer->size()) {
      return buffer;
    }
  }
  return nullptr;
}

} // namespace

template <>
void ConstantVector<StringView>::setStringValue(
    StringView value,
    const BufferPtr& candidate) {
  if (value.isInline()) {
    value_ = value;
    return;
  }
  if (auto buffer = findSharableStringBuffer(
          value, folly::Range<const BufferPtr*>(&candidate, 1), pool())) {
    stringBuffer_ = std::move(buffer);
    value_ = value;
    return;
  }
  auto buffer = AlignedBuffer::allocate<char>(value.size(), pool());
  memcpy(buffer->asMutable<char>(), value.data(), value.size());
  value_ = StringView(buffer->as<char>(), static_cast<int32_t>(value.size()));
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
    const auto& stringBuffers = flat->stringBuffers();
    setStringValue(
        value,
        findSharableStringBuffer(
            value,
            folly::Range<const BufferPtr*>(
                stringBuffers.data(), stringBuffers.size()),
            pool()));
    return;
  }
  if (base->isConstantEncoding()) {
    setStringValue(
        value, base->as<ConstantVector<StringView>>()->getStringBuffer());
    return;
  }
  setStringValue(value, BufferPtr(nullptr));
}

} // namespace facebook::velox
