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

#include "velox/common/memory/BufferInputStream.h"

namespace facebook::velox {

std::string BufferInputStream::toString() const {
  std::stringstream oss;
  oss << ranges_.size() << " ranges (position/size) [";
  for (const auto& range : ranges_) {
    oss << "(" << range.position << "/" << range.size
        << (&range == current_ ? " current" : "") << ")";
    if (&range != &ranges_.back()) {
      oss << ",";
    }
  }
  oss << "]";
  return oss.str();
}

bool BufferInputStream::atEnd() const {
  if (current_ == nullptr) {
    return false;
  }
  if (current_->position < current_->size) {
    return false;
  }

  VELOX_CHECK(current_ >= ranges_.data() && current_ <= &ranges_.back());
  return current_ == &ranges_.back();
}

size_t BufferInputStream::size() const {
  size_t total = 0;
  for (const auto& range : ranges_) {
    total += range.size;
  }
  return total;
}

size_t BufferInputStream::remainingSize() const {
  if (ranges_.empty()) {
    return 0;
  }
  const auto* lastRange = &ranges_.back();
  auto* cur = current_;
  size_t remainingBytes = cur->availableBytes();
  while (++cur <= lastRange) {
    remainingBytes += cur->size;
  }
  return remainingBytes;
}

std::streampos BufferInputStream::tellp() const {
  if (ranges_.empty()) {
    return 0;
  }
  assert(current_);
  int64_t size = 0;
  for (auto& range : ranges_) {
    if (&range == current_) {
      return current_->position + size;
    }
    size += range.size;
  }
  VELOX_FAIL("BufferInputStream 'current_' is not in 'ranges_'.");
}

void BufferInputStream::seekp(std::streampos position) {
  if (ranges_.empty() && position == 0) {
    return;
  }
  int64_t toSkip = position;
  for (auto& range : ranges_) {
    if (toSkip <= range.size) {
      current_ = &range;
      current_->position = toSkip;
      return;
    }
    toSkip -= range.size;
  }
  static_assert(sizeof(std::streamsize) <= sizeof(long long));
  VELOX_FAIL(
      "Seeking past end of BufferInputStream: {}",
      static_cast<long long>(position));
}

void BufferInputStream::nextRange() {
  VELOX_CHECK(current_ >= &ranges_[0]);
  const size_t rangeIndex = current_ - &ranges_[0];
  VELOX_CHECK_LT(
      rangeIndex + 1, ranges_.size(), "Reading past end of BufferInputStream");
  ++current_;
  current_->position = 0;
}

uint8_t BufferInputStream::readByte() {
  if (current_->position < current_->size) {
    return current_->buffer[current_->position++];
  }
  nextRange();
  return readByte();
}

void BufferInputStream::readBytes(uint8_t* bytes, int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to read negative number of bytes");
  int32_t offset = 0;
  for (;;) {
    const int32_t availableBytes = current_->size - current_->position;
    const int32_t readBytes = std::min(availableBytes, size);
    simd::memcpy(
        bytes + offset, current_->buffer + current_->position, readBytes);
    offset += readBytes;
    size -= readBytes;
    current_->position += readBytes;
    if (size == 0) {
      return;
    }
    nextRange();
  }
}

std::string_view BufferInputStream::nextView(int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to view negative number of bytes");
  if (current_->position == current_->size) {
    if (current_ == &ranges_.back()) {
      return std::string_view(nullptr, 0);
    }
    nextRange();
  }
  VELOX_CHECK_GT(current_->size, 0);
  const auto position = current_->position;
  const auto viewSize = std::min(current_->size - current_->position, size);
  current_->position += viewSize;
  return std::string_view(
      reinterpret_cast<char*>(current_->buffer) + position, viewSize);
}

void BufferInputStream::skip(int32_t size) {
  VELOX_CHECK_GE(size, 0, "Attempting to skip negative number of bytes");
  for (;;) {
    const int32_t numSkipped =
        std::min<int32_t>(current_->availableBytes(), size);
    size -= numSkipped;
    current_->position += numSkipped;
    if (size == 0) {
      return;
    }
    nextRange();
  }
}

} // namespace facebook::velox
