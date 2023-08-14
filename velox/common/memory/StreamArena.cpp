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
#include "velox/common/memory/StreamArena.h"
#include "velox/common/memory/ByteStream.h"

namespace facebook::velox {

StreamArena::StreamArena(memory::MemoryPool* pool) : pool_(pool) {}

void StreamArena::newRange(int32_t bytes, ByteRange* range) {
  VELOX_CHECK_GT(bytes, 0);
  const memory::MachinePageCount numPages =
      memory::AllocationTraits::numPages(bytes);
  const int32_t numRuns = allocation_.numRuns();
  if (currentRun_ >= numRuns) {
    if (numRuns > 0) {
      allocations_.push_back(
          std::make_unique<memory::Allocation>(std::move(allocation_)));
    }
    pool_->allocateNonContiguous(
        std::max(allocationQuantum_, numPages), allocation_);
    currentRun_ = 0;
    currentPage_ = 0;
    size_ += allocation_.byteSize();
  }
  auto run = allocation_.runAt(currentRun_);
  const int32_t available = run.numPages() - currentPage_;
  range->buffer =
      run.data() + memory::AllocationTraits::pageBytes(currentPage_);
  const int32_t allocatedPages = std::min<int32_t>(numPages, available);
  range->size = memory::AllocationTraits::pageBytes(allocatedPages);
  range->position = 0;
  currentPage_ += allocatedPages;
  if (currentPage_ == run.numPages()) {
    ++currentRun_;
    currentPage_ = 0;
  }
}

void StreamArena::newTinyRange(int32_t bytes, ByteRange* range) {
  tinyRanges_.emplace_back();
  tinyRanges_.back().resize(bytes);
  range->position = 0;
  range->buffer = reinterpret_cast<uint8_t*>(tinyRanges_.back().data());
  range->size = bytes;
}

} // namespace facebook::velox
