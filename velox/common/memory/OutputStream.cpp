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

#include "velox/common/memory/OutputStream.h"

namespace facebook::velox {

namespace {
// The user data structure passed to folly iobuf for buffer ownership handling.
struct FreeData {
  std::shared_ptr<StreamArena> arena;
  std::function<void()> releaseFn;
};

FreeData* newFreeData(
    const std::shared_ptr<StreamArena>& arena,
    const std::function<void()>& releaseFn) {
  auto freeData = new FreeData();
  freeData->arena = arena;
  freeData->releaseFn = releaseFn;
  return freeData;
}

void freeFunc(void* /*data*/, void* userData) {
  auto* freeData = reinterpret_cast<FreeData*>(userData);
  freeData->arena.reset();
  if (freeData->releaseFn != nullptr) {
    freeData->releaseFn();
  }
  delete freeData;
}
} // namespace

std::unique_ptr<folly::IOBuf> IOBufOutputStream::getIOBuf(
    const std::function<void()>& releaseFn) {
  // Make an IOBuf for each range. The IOBufs keep shared ownership of
  // 'arena_'.
  std::unique_ptr<folly::IOBuf> iobuf;
  auto& ranges = out_->ranges();
  for (auto& range : ranges) {
    auto numValues =
        &range == &ranges.back() ? out_->lastRangeEnd() : range.size;
    auto userData = newFreeData(arena_, releaseFn);
    auto newBuf = folly::IOBuf::takeOwnership(
        reinterpret_cast<char*>(range.buffer), numValues, freeFunc, userData);
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
