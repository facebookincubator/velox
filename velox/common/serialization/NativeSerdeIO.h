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

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common {

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

class NativeStringReader {
 public:
  explicit NativeStringReader(std::string_view data) : data_(data) {}

  void read(void* out, size_t size) {
    VELOX_CHECK_LE(offset_ + size, data_.size(), "Corrupted binary data");
    memcpy(out, data_.data() + offset_, size);
    offset_ += size;
  }

  template <typename T>
  T readValue() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    read(&value, sizeof(value));
    return value;
  }

  bool atEnd() const {
    return offset_ == data_.size();
  }

 private:
  std::string_view data_;
  size_t offset_{0};
};

class NativeBufferedWriter {
 public:
  NativeBufferedWriter(std::ostream& out, size_t bufferSize)
      : out_(out), buffer_(bufferSize), pos_(0) {}

  ~NativeBufferedWriter() {
    flush();
  }

  void write(const void* data, size_t size) {
    const char* src = static_cast<const char*>(data);
    while (size > 0) {
      size_t available = buffer_.size() - pos_;
      if (available == 0) {
        flush();
        available = buffer_.size();
      }

      const size_t toWrite = std::min(size, available);
      std::memcpy(buffer_.data() + pos_, src, toWrite);
      pos_ += toWrite;
      src += toWrite;
      size -= toWrite;
    }
  }

  template <typename T>
  void writeValue(T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    write(&value, sizeof(value));
  }

  void flush() {
    if (pos_ > 0) {
      out_.write(buffer_.data(), pos_);
      pos_ = 0;
    }
  }

 private:
  std::ostream& out_;
  std::vector<char> buffer_;
  size_t pos_;
};

class NativeBufferedReader {
 public:
  NativeBufferedReader(std::istream& in, size_t bufferSize)
      : in_(in), buffer_(bufferSize), pos_(0), size_(0) {}

  void read(void* data, size_t size) {
    char* dest = static_cast<char*>(data);
    while (size > 0) {
      if (pos_ >= size_) {
        refill();
      }

      const size_t available = size_ - pos_;
      const size_t toRead = std::min(size, available);
      std::memcpy(dest, buffer_.data() + pos_, toRead);
      pos_ += toRead;
      dest += toRead;
      size -= toRead;
    }
  }

  template <typename T>
  T readValue() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    read(&value, sizeof(value));
    return value;
  }

 private:
  void refill() {
    in_.read(buffer_.data(), buffer_.size());
    size_ = in_.gcount();
    pos_ = 0;
    VELOX_CHECK_GT(size_, 0, "Unexpected end of stream");
  }

  std::istream& in_;
  std::vector<char> buffer_;
  size_t pos_;
  size_t size_;
};

} // namespace facebook::velox::common
