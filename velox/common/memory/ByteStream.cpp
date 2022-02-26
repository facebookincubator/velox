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

#include "velox/common/memory/ByteStream.h"

namespace facebook::velox {

std::streampos ByteStream::tellp() const {
  if (ranges_.empty()) {
    return 0;
  }
  assert(current_);
  int64_t size = 0;
  for (auto& range : ranges_) {
    if (&range == current_) {
      return current_->position + size;
    }
    size += range.numValues();
  }
  VELOX_FAIL("ByteStream 'current_' is not in 'ranges_'.");
}

void ByteStream::seekp(std::streampos position) {
  int64_t toSkip = position;
  if (current_ && current_->position > current_->fill) {
    current_->fill = current_->position;
  }
  if (ranges_.empty() && position == 0) {
    return;
  }
  for (auto& range : ranges_) {
    if (toSkip <= range.numValues()) {
      current_ = &range;
      current_->position = toSkip;
      return;
    }
    toSkip -= range.numValues();
  }
  VELOX_FAIL("Seeking past end of ByteStream: {}", position);
}

void ByteStream::flush(OutputStream* out) {
  for (int32_t i = 0; i < ranges_.size(); ++i) {
    int32_t count = ranges_[i].numValues();
    int32_t bytes = isBits_ ? bits::nbytes(count) : count;
    if (isBits_ && isReverseBitOrder_ && !isReversed_) {
      bits::reverseBits(ranges_[i].buffer, bytes);
    }
    out->write(reinterpret_cast<char*>(ranges_[i].buffer), bytes);
  }
  if (isBits_ && isReverseBitOrder_) {
    isReversed_ = true;
  }
}

void ByteStream::extend(int32_t bytes) {
  if (current_) {
    current_->fill = current_->position;
  }
  if (current_ && current_ != &ranges_.back()) {
    ++current_;
    current_->fill = current_->position;
    if (current_->fill < current_->position) {
    }
    current_->position = 0;
    return;
  }
  ranges_.emplace_back();
  current_ = &ranges_.back();
  arena_->newRange(bytes, current_);
}
namespace {
void freeFunc(void* /*data*/, void* userData) {
  auto ptr = reinterpret_cast<std::shared_ptr<StreamArena>*>(userData);
  delete ptr;
}
} // namespace

std::unique_ptr<folly::IOBuf> IOBufOutputStream::getIOBuf() {
  // Make an IOBuf for each range. The The IOBufs keep shared ownership of
  // 'arena_'.
  std::unique_ptr<folly::IOBuf> iobuf;
  for (auto& range : out_->ranges()) {
    auto userData = new std::shared_ptr<StreamArena>(arena_);
    auto newBuf = folly::IOBuf::takeOwnership(
        reinterpret_cast<char*>(range.buffer),
        range.numValues(),
        freeFunc,
        userData);
    if (iobuf) {
      iobuf->prev()->appendChain(std::move(newBuf));
    } else {
      iobuf = std::move(newBuf);
    }
  }
  return iobuf;
}

std::streampos IOBufOutputStream::tellp() const {
  return out_->tellp();
}

void IOBufOutputStream::seekp(std::streampos pos) {
  out_->seekp(pos);
}

} // namespace facebook::velox
