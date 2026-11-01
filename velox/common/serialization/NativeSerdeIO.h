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

#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {

/// Appends fixed-width values and raw bytes to a std::string in host byte
/// order. The produced bytes are only portable between processes running the
/// same binary on the same architecture.
class NativeStringWriter {
 public:
  explicit NativeStringWriter(std::string& data) : data_(data) {}

  void write(const void* data, size_t size) {
    data_.append(static_cast<const char*>(data), size);
  }

  template <typename T>
  void writeValue(T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    write(&value, sizeof(value));
  }

 private:
  std::string& data_;
};

/// Reads back the bytes produced by NativeStringWriter.
class NativeStringReader {
 public:
  explicit NativeStringReader(std::string_view data) : data_(data) {}

  void read(void* out, size_t size) {
    memcpy(out, view(size).data(), size);
  }

  template <typename T>
  T readValue() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    read(&value, sizeof(value));
    return value;
  }

  /// Returns the next 'size' bytes without copying them and advances past
  /// them. The returned view is valid for as long as the underlying data is.
  std::string_view view(size_t size) {
    VELOX_CHECK_LE(offset_ + size, data_.size(), "Corrupted binary data");
    const auto result = data_.substr(offset_, size);
    offset_ += size;
    return result;
  }

  /// Advances past the next 'size' bytes without reading them.
  void skip(size_t size) {
    VELOX_CHECK_LE(offset_ + size, data_.size(), "Corrupted binary data");
    offset_ += size;
  }

  bool atEnd() const {
    return offset_ == data_.size();
  }

 private:
  std::string_view data_;
  size_t offset_{0};
};

} // namespace facebook::velox::common
