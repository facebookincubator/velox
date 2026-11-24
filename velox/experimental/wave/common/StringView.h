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

#pragma once

#include <assert.h>
#include <stdint.h>
#include "velox/experimental/wave/common/CompilerDefines.h"

namespace facebook::velox::wave {

WAVE_DEVICE_HOST inline int
stringview_memcmp(const void* lhs, const void* rhs, size_t n) {
  auto* a = reinterpret_cast<const unsigned char*>(lhs);
  auto* b = reinterpret_cast<const unsigned char*>(rhs);
  for (size_t i = 0; i < n; ++i) {
    if (int c = (int)a[i] - (int)b[i]) {
      return c;
    }
  }
  return 0;
}

/// A string reference with the layout of velox::StringView: the size, the
/// first 4 bytes, and then the next 8 bytes for strings of up to 12 bytes or
/// a pointer to the whole string. The bytes after the end of an inline string
/// are 0. Vectors of strings can be copied between device and host as is,
/// with only the pointers of non-inline strings to translate.
class StringView {
 public:
  static constexpr int32_t kInlineSize = 12;
  static constexpr int32_t kPrefixSize = 4;

  WAVE_DEVICE_HOST void init(const char* data, uint32_t len) {
    size_ = len;
    prefixWord() = 0;
    value_.data = nullptr;
    if (len <= kInlineSize) {
      memcpy(prefix_, data, len);
    } else {
      memcpy(prefix_, data, kPrefixSize);
      value_.data = data;
    }
  }

  WAVE_DEVICE_HOST uint32_t size() const {
    return size_;
  }

  WAVE_DEVICE_HOST bool isInline() const {
    return size_ <= kInlineSize;
  }

  WAVE_DEVICE_HOST const char* data() const {
    return isInline() ? prefix_ : value_.data;
  }

  /// Sets the pointer of a non-inline string, e.g. to translate a device
  /// address to host.
  WAVE_DEVICE_HOST void setData(const char* data) {
    value_.data = data;
  }

  WAVE_DEVICE_HOST bool operator==(StringView other) const {
    // The size and prefix are compared as one word.
    if (sizeAndPrefix() != other.sizeAndPrefix()) {
      return false;
    }
    if (isInline()) {
      return value_.word == other.value_.word;
    }
    return stringview_memcmp(
               value_.data + kPrefixSize,
               other.value_.data + kPrefixSize,
               size_ - kPrefixSize) == 0;
  }

  WAVE_DEVICE_HOST bool operator!=(StringView other) const {
    return !(*this == other);
  }

#if !defined(__NVCC__) && !defined(__CUDACC_RTC__)
  operator std::string_view() const {
    return {data(), size()};
  }
#endif

 private:
  WAVE_DEVICE_HOST uint32_t& prefixWord() {
    return *reinterpret_cast<uint32_t*>(prefix_);
  }

  WAVE_DEVICE_HOST uint64_t sizeAndPrefix() const {
    return *reinterpret_cast<const uint64_t*>(this);
  }

  uint32_t size_;
  char prefix_[kPrefixSize];
  union {
    char inlined[8];
    const char* data;
    uint64_t word;
  } value_;
};

// Non-trivial class does not play well in device code.
#ifndef __CUDACC_RTC__
static_assert(std::is_trivial_v<StringView>);
#endif
static_assert(sizeof(StringView) == 16);

} // namespace facebook::velox::wave
