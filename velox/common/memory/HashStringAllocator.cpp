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
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/base/Portability.h"
#include "velox/common/base/SimdUtil.h"

namespace facebook::velox {

namespace {
/// Returns the size of the previous free block. The size is stored in the
/// last 4 bytes of the free block, e.g. 4 bytes just before the current
/// header.
uint32_t* previousFreeSize(HashStringAllocator::Header* header) {
  return reinterpret_cast<uint32_t*>(header) - 1;
}

/// Returns the header of the previous free block or nullptr if previous block
/// is not free.
HashStringAllocator::Header* FOLLY_NULLABLE
getPreviousFree(HashStringAllocator::Header* FOLLY_NONNULL header) {
  if (!header->isPreviousFree()) {
    return nullptr;
  }
  auto numBytes = *previousFreeSize(header);
  auto previous = reinterpret_cast<HashStringAllocator::Header*>(
      header->begin() - numBytes - 2 * sizeof(HashStringAllocator::Header));
  VELOX_CHECK_EQ(previous->size(), numBytes);
  VELOX_CHECK(previous->isFree());
  VELOX_CHECK(!previous->isPreviousFree());
  return previous;
}

/// Sets kFree flag in the 'header' and writes the size of the block to the
/// last 4 bytes of the block. Sets kPreviousFree flag in the next block's
/// 'header'.
void markAsFree(HashStringAllocator::Header* FOLLY_NONNULL header) {
  header->setFree();
  auto nextHeader = header->next();
  if (nextHeader) {
    nextHeader->setPreviousFree();
    *previousFreeSize(nextHeader) = header->size();
  }
}

int64_t tryAccommodate(int64_t destSize, int64_t srcSize, bool isSrcContinued) {
  static constexpr auto kMinAlloc = HashStringAllocator::kMinAlloc;
  static constexpr auto kMinBlockSize =
      HashStringAllocator::AllocationCompactor::kMinBlockSize;
  static constexpr auto kContinuedPtrSize =
      HashStringAllocator::Header::kContinuedPtrSize;

  VELOX_CHECK_GE(destSize, kMinAlloc);

  if (srcSize + kMinBlockSize <= destSize || srcSize == destSize) {
    return srcSize;
  }

  if (!isSrcContinued) {
    if (srcSize < destSize) {
      // After accommodating src block, the rest of space of the 'destBlock' is
      // not enough for a valid block, use it up.
      return srcSize;
    }
    // src block is larger than dest block, split src block so the first part
    // can fill the dest block.
    return destSize - kContinuedPtrSize;
  }

  // src block is continued.
  VELOX_CHECK_GE(srcSize, kMinAlloc);
  if (srcSize - destSize + kContinuedPtrSize >= kMinAlloc) {
    // src block is larger than dest block and when splitting src block to fill
    // the dest block, the second part of the splitted block is valid.
    return destSize - kContinuedPtrSize;
  }

  // src block's size is slightly different from dest block's size: If src block
  // is smaller than dest block, the diff is not enough for a valid free block;
  // otherwise, the diff is not enough to put in a valid continued block. Try
  // splitting src block into two blocks A and B, put A into dest block so
  // that remaining of dest block after accommodating A forms a minimum valid
  // free block(Will not be used for later accommodation, just for keeping the
  // invariant of HSA). We can put B into another page later. The constraint is
  // that A and B should both be valid(>= kMinAlloc), if unsatisfied, we can
  // only skip this dest block and put src block into later page.
  if (srcSize < destSize) {
    VELOX_CHECK_GT(srcSize, destSize - kMinBlockSize);
  } else {
    VELOX_CHECK_LT(srcSize - destSize + kContinuedPtrSize, kMinAlloc);
  }
  const auto sizeA{std::min<int32_t>(
      destSize - kMinBlockSize - HashStringAllocator::Header::kContinuedPtrSize,
      srcSize - HashStringAllocator::kMinAlloc)};
  const auto sizeB{srcSize - sizeA};
  if (sizeA + HashStringAllocator::Header::kContinuedPtrSize >= kMinAlloc &&
      sizeB >= kMinAlloc) {
    return sizeA;
  }
  return 0;
}

} // namespace

using Header = HashStringAllocator::Header;

std::string HashStringAllocator::Header::toString() {
  std::ostringstream out;
  if (isFree()) {
    out << "|free| ";
  }
  if (isContinued()) {
    out << "|multipart| ";
  }
  out << "size: " << size();
  if (isContinued()) {
    auto next = nextContinued();
    out << " [" << next->size();
    while (next->isContinued()) {
      next = next->nextContinued();
      out << ", " << next->size();
    }
    out << "]";
  }
  if (isPreviousFree()) {
    out << ", previous is free (" << *previousFreeSize(this) << " bytes)";
  }
  if (next() == nullptr) {
    out << ", at end";
  }
  return out.str();
}

HashStringAllocator::~HashStringAllocator() {
  clear();
}

void HashStringAllocator::clear() {
  numFree_ = 0;
  freeBytes_ = 0;
  std::fill(std::begin(freeNonEmpty_), std::end(freeNonEmpty_), 0);
  for (auto& pair : allocationsFromPool_) {
    pool()->free(pair.first, pair.second);
  }
  allocationsFromPool_.clear();
  for (auto i = 0; i < kNumFreeLists; ++i) {
    new (&free_[i]) CompactDoubleList();
  }
  pool_.clear();
}

void* HashStringAllocator::allocateFromPool(size_t size) {
  auto ptr = pool()->allocate(size);
  cumulativeBytes_ += size;
  allocationsFromPool_[ptr] = size;
  sizeFromPool_ += size;
  return ptr;
}

void HashStringAllocator::freeToPool(void* ptr, size_t size) {
  auto it = allocationsFromPool_.find(ptr);
  VELOX_CHECK(
      it != allocationsFromPool_.end(),
      "freeToPool for block not allocated from pool of HashStringAllocator");
  VELOX_CHECK_EQ(
      size, it->second, "Bad size in HashStringAllocator::freeToPool()");
  allocationsFromPool_.erase(it);
  sizeFromPool_ -= size;
  cumulativeBytes_ -= size;
  pool()->free(ptr, size);
}

// static
ByteInputStream HashStringAllocator::prepareRead(
    const Header* begin,
    size_t maxBytes) {
  std::vector<ByteRange> ranges;
  auto header = const_cast<Header*>(begin);

  size_t totalBytes = 0;

  for (;;) {
    ranges.push_back(ByteRange{
        reinterpret_cast<uint8_t*>(header->begin()), header->usableSize(), 0});
    totalBytes += ranges.back().size;
    if (!header->isContinued()) {
      break;
    }

    if (totalBytes >= maxBytes) {
      break;
    }

    header = header->nextContinued();
  }
  return ByteInputStream(std::move(ranges));
}

HashStringAllocator::Position HashStringAllocator::newWrite(
    ByteOutputStream& stream,
    int32_t preferredSize) {
  VELOX_CHECK(
      !currentHeader_,
      "Do not call newWrite before finishing the previous write to "
      "HashStringAllocator");
  currentHeader_ = allocate(preferredSize, false);

  stream.setRange(
      ByteRange{
          reinterpret_cast<uint8_t*>(currentHeader_->begin()),
          currentHeader_->size(),
          0},
      0);

  startPosition_ = Position::atOffset(currentHeader_, 0);

  return startPosition_;
}

void HashStringAllocator::extendWrite(
    Position position,
    ByteOutputStream& stream) {
  auto header = position.header;
  const auto offset = position.offset();
  VELOX_CHECK_GE(
      offset, 0, "Starting extendWrite outside of the current range");
  VELOX_CHECK_LE(
      offset,
      header->usableSize(),
      "Starting extendWrite outside of the current range");

  if (header->isContinued()) {
    free(header->nextContinued());
    header->clearContinued();
  }

  stream.setRange(
      ByteRange{
          reinterpret_cast<uint8_t*>(position.header->begin()),
          position.header->size(),
          static_cast<int32_t>(position.position - position.header->begin())},
      0);
  currentHeader_ = header;
  startPosition_ = position;
}

std::pair<HashStringAllocator::Position, HashStringAllocator::Position>
HashStringAllocator::finishWrite(
    ByteOutputStream& stream,
    int32_t numReserveBytes) {
  VELOX_CHECK(
      currentHeader_, "Must call newWrite or extendWrite before finishWrite");
  auto writePosition = stream.writePosition();
  const auto offset = writePosition - currentHeader_->begin();

  VELOX_CHECK_GE(
      offset, 0, "finishWrite called with writePosition out of range");
  VELOX_CHECK_LE(
      offset,
      currentHeader_->usableSize(),
      "finishWrite called with writePosition out of range");

  Position currentPosition = Position::atOffset(currentHeader_, offset);
  if (currentHeader_->isContinued()) {
    free(currentHeader_->nextContinued());
    currentHeader_->clearContinued();
  }
  // Free remainder of block if there is a lot left over.
  freeRestOfBlock(
      currentHeader_,
      writePosition - currentHeader_->begin() + numReserveBytes);
  currentHeader_ = nullptr;

  // The starting position may have shifted if it was at the end of the block
  // and the block was extended. Calculate the new position.
  if (startPosition_.header->isContinued()) {
    auto header = startPosition_.header;
    const auto offset = startPosition_.offset();
    const auto extra = offset - header->usableSize();
    if (extra > 0) {
      auto newHeader = header->nextContinued();
      auto newPosition = newHeader->begin() + extra;
      startPosition_ = {newHeader, newPosition};
    }
  }
  return {startPosition_, currentPosition};
}

void HashStringAllocator::newSlab() {
  constexpr int32_t kSimdPadding = simd::kPadding - sizeof(Header);
  const int64_t needed = pool_.allocatedBytes() >= pool_.hugePageThreshold()
      ? memory::AllocationTraits::kHugePageSize
      : kUnitSize;
  auto run = pool_.allocateFixed(needed);
  // We check we got exactly the requested amount. checkConsistency()
  // depends on slabs made here coinciding with ranges from
  // AllocationPool::rangeAt(). Sometimes the last range can be
  // several huge pages for severl huge page sized arenas but
  // checkConsistency() can interpret that.
  VELOX_CHECK_EQ(0, pool_.freeBytes());
  auto available = needed - sizeof(Header) - kSimdPadding;

  VELOX_CHECK_NOT_NULL(run);
  VELOX_CHECK_GT(available, 0);
  // Write end  marker.
  *reinterpret_cast<uint32_t*>(run + available) = Header::kArenaEnd;
  cumulativeBytes_ += available;

  // Add the new memory to the free list: Placement construct a header
  // that covers the space from start to the end marker and add this
  // to free list.
  free(new (run) Header(available - sizeof(Header)));
}

void HashStringAllocator::newRange(
    int32_t bytes,
    ByteRange* lastRange,
    ByteRange* range,
    bool contiguous) {
  // Allocates at least kMinContiguous or to the end of the current
  // run. At the end of the write the unused space will be made
  // free.
  VELOX_CHECK(
      currentHeader_,
      "Must have called newWrite or extendWrite before newRange");
  auto newHeader = allocate(bytes, contiguous);

  auto lastWordPtr = reinterpret_cast<void**>(
      currentHeader_->end() - Header::kContinuedPtrSize);
  *reinterpret_cast<void**>(newHeader->begin()) = *lastWordPtr;
  *lastWordPtr = newHeader;
  currentHeader_->setContinued();
  currentHeader_ = newHeader;
  if (lastRange) {
    // The last bytes of the last range are no longer payload. So do not
    // count them in size and do not overwrite them if overwriting the
    // multirange entry. Set position at the new end.
    lastRange->size -= sizeof(void*);
    lastRange->position = std::min(lastRange->size, lastRange->position);
  }
  *range = ByteRange{
      reinterpret_cast<uint8_t*>(currentHeader_->begin()),
      currentHeader_->size(),
      Header::kContinuedPtrSize};
}

void HashStringAllocator::newRange(
    int32_t bytes,
    ByteRange* lastRange,
    ByteRange* range) {
  newRange(bytes, lastRange, range, false);
}

void HashStringAllocator::newContiguousRange(int32_t bytes, ByteRange* range) {
  newRange(bytes, nullptr, range, true);
}

// static
StringView HashStringAllocator::contiguousString(
    StringView view,
    std::string& storage) {
  if (view.isInline()) {
    return view;
  }
  auto header = headerOf(view.data());
  if (view.size() <= header->size()) {
    return view;
  }

  auto stream = prepareRead(headerOf(view.data()));
  storage.resize(view.size());
  stream.readBytes(storage.data(), view.size());
  return StringView(storage);
}

void HashStringAllocator::freeRestOfBlock(Header* header, int32_t keepBytes) {
  keepBytes = std::max(keepBytes, kMinAlloc);
  int32_t freeSize = header->size() - keepBytes - sizeof(Header);
  if (freeSize <= kMinAlloc) {
    return;
  }

  header->setSize(keepBytes);
  auto newHeader = new (header->end()) Header(freeSize);
  free(newHeader);
}

int32_t HashStringAllocator::freeListIndex(int size) {
  return std::min(size - kMinAlloc, kNumFreeLists - 1);
}

void HashStringAllocator::removeFromFreeList(Header* header, bool clearFree) {
  VELOX_CHECK(header->isFree());
  if (clearFree) {
    header->clearFree();
  }
  auto index = freeListIndex(header->size());
  reinterpret_cast<CompactDoubleList*>(header->begin())->remove();
  if (free_[index].empty()) {
    bits::clearBit(freeNonEmpty_, index);
  }
}

HashStringAllocator::Header* FOLLY_NULLABLE
HashStringAllocator::allocate(int32_t size, bool exactSize) {
  requestedContiguous_ |= exactSize;
  if (size > kMaxAlloc && exactSize) {
    VELOX_CHECK_LE(size, Header::kSizeMask);
    auto header =
        reinterpret_cast<Header*>(allocateFromPool(size + sizeof(Header)));
    new (header) Header(size);
    return header;
  }
  auto header = allocateFromFreeLists(size, exactSize, exactSize);
  if (!header) {
    newSlab();
    header = allocateFromFreeLists(size, exactSize, exactSize);
    VELOX_CHECK(header != nullptr);
    VELOX_CHECK_GT(header->size(), 0);
  }

  return header;
}

HashStringAllocator::Header* FOLLY_NULLABLE
HashStringAllocator::allocateFromFreeLists(
    int32_t preferredSize,
    bool mustHaveSize,
    bool isFinalSize) {
  if (!numFree_) {
    return nullptr;
  }
  preferredSize = std::max(kMinAlloc, preferredSize);
  const auto index = freeListIndex(preferredSize);
  auto available = bits::findFirstBit(freeNonEmpty_, index, kNumFreeLists);
  if (!mustHaveSize && available == -1) {
    available = bits::findLastBit(freeNonEmpty_, 0, index);
  }
  if (available == -1) {
    return nullptr;
  }
  auto* header =
      allocateFromFreeList(preferredSize, mustHaveSize, isFinalSize, available);
  VELOX_CHECK_NOT_NULL(header);
  return header;
}

HashStringAllocator::Header* FOLLY_NULLABLE
HashStringAllocator::allocateFromFreeList(
    int32_t preferredSize,
    bool mustHaveSize,
    bool isFinalSize,
    int32_t freeListIndex) {
  auto* item = free_[freeListIndex].next();
  if (item == &free_[freeListIndex]) {
    return nullptr;
  }
  auto found = headerOf(item);
  VELOX_CHECK(
      found->isFree() && (!mustHaveSize || found->size() >= preferredSize));
  --numFree_;
  freeBytes_ -= found->size() + sizeof(Header);
  removeFromFreeList(found);
  auto next = found->next();
  if (next) {
    next->clearPreviousFree();
  }
  cumulativeBytes_ += found->size();
  if (isFinalSize) {
    freeRestOfBlock(found, preferredSize);
  }
  return found;
}

void HashStringAllocator::free(Header* _header) {
  Header* header = _header;

  do {
    Header* continued = nullptr;
    if (header->isContinued()) {
      continued = header->nextContinued();
      header->clearContinued();
    }
    if (header->size() > kMaxAlloc && !pool_.isInCurrentRange(header) &&
        allocationsFromPool_.find(header) != allocationsFromPool_.end()) {
      freeToPool(header, header->size() + sizeof(Header));
    } else {
      VELOX_CHECK(!header->isFree());
      freeBytes_ += header->size() + sizeof(Header);
      cumulativeBytes_ -= header->size();
      Header* next = header->next();
      if (next) {
        VELOX_CHECK(!next->isPreviousFree());
        if (next->isFree()) {
          --numFree_;
          removeFromFreeList(next);
          header->setSize(header->size() + next->size() + sizeof(Header));
          next = reinterpret_cast<Header*>(header->end());
          VELOX_CHECK(next->isArenaEnd() || !next->isFree());
        }
      }
      if (header->isPreviousFree()) {
        auto previousFree = getPreviousFree(header);
        removeFromFreeList(previousFree);
        previousFree->setSize(
            previousFree->size() + header->size() + sizeof(Header));

        header = previousFree;
      } else {
        ++numFree_;
      }
      auto freedSize = header->size();
      auto freeIndex = freeListIndex(freedSize);
      bits::setBit(freeNonEmpty_, freeIndex);
      free_[freeIndex].insert(
          reinterpret_cast<CompactDoubleList*>(header->begin()));
      markAsFree(header);
    }
    header = continued;
  } while (header);
}

// static
int64_t HashStringAllocator::offset(
    Header* FOLLY_NONNULL header,
    Position position) {
  static const int64_t kOutOfRange = -1;
  if (!position.isSet()) {
    return kOutOfRange;
  }

  int64_t size = 0;
  for (;;) {
    assert(header);
    const auto length = header->usableSize();
    const auto offset = position.position - header->begin();
    if (offset >= 0 && offset <= length) {
      return size + offset;
    }
    if (!header->isContinued()) {
      return kOutOfRange;
    }
    size += length;
    header = header->nextContinued();
  }
}

// static
HashStringAllocator::Position HashStringAllocator::seek(
    Header* FOLLY_NONNULL header,
    int64_t offset) {
  int64_t size = 0;
  for (;;) {
    assert(header);
    auto length = header->usableSize();
    if (offset <= size + length) {
      return Position::atOffset(header, offset - size);
    }
    if (!header->isContinued()) {
      return Position::null();
    }
    size += length;
    header = header->nextContinued();
  }
}

// static
int64_t HashStringAllocator::available(const Position& position) {
  auto header = position.header;
  const auto startOffset = position.offset();
  // startOffset bytes from the first block are already used.
  int64_t size = -startOffset;
  for (;;) {
    assert(header);
    size += header->usableSize();
    if (!header->isContinued()) {
      return size;
    }
    header = header->nextContinued();
  }
}

void HashStringAllocator::ensureAvailable(int32_t bytes, Position& position) {
  if (available(position) >= bytes) {
    return;
  }

  ByteOutputStream stream(this);
  extendWrite(position, stream);
  static char data[128];
  while (bytes) {
    auto written = std::min<size_t>(bytes, sizeof(data));
    stream.append(folly::StringPiece(data, written));
    bytes -= written;
  }
  position = finishWrite(stream, 0).first;
}

inline bool HashStringAllocator::storeStringFast(
    const char* bytes,
    int32_t numBytes,
    char* destination) {
  auto roundedBytes = std::max(numBytes, kMinAlloc);
  Header* header = nullptr;
  if (free_[kNumFreeLists - 1].empty()) {
    if (roundedBytes >= kMaxAlloc) {
      return false;
    }
    auto index = freeListIndex(roundedBytes);
    auto available = bits::findFirstBit(freeNonEmpty_, index, kNumFreeLists);
    if (available < 0) {
      return false;
    }
    header = allocateFromFreeList(roundedBytes, true, true, available);
    VELOX_CHECK_NOT_NULL(header);
  } else {
    auto& freeList = free_[kNumFreeLists - 1];
    header = headerOf(freeList.next());
    const auto spaceTaken = roundedBytes + sizeof(Header);
    if (spaceTaken > header->size()) {
      return false;
    }
    if (header->size() - spaceTaken > kMaxAlloc) {
      // The entry after allocation stays in the largest free list.
      // The size at the end of the block is changed in place.
      reinterpret_cast<int32_t*>(header->end())[-1] -= spaceTaken;
      auto freeHeader = new (header->begin() + roundedBytes)
          Header(header->size() - spaceTaken);
      freeHeader->setFree();
      header->clearFree();
      memcpy(freeHeader->begin(), header->begin(), sizeof(CompactDoubleList));
      freeList.nextMoved(
          reinterpret_cast<CompactDoubleList*>(freeHeader->begin()));
      header->setSize(roundedBytes);
      freeBytes_ -= spaceTaken;
      cumulativeBytes_ += roundedBytes;
    } else {
      header =
          allocateFromFreeList(roundedBytes, true, true, kNumFreeLists - 1);
      if (!header) {
        return false;
      }
    }
  }
  simd::memcpy(header->begin(), bytes, numBytes);
  *reinterpret_cast<StringView*>(destination) =
      StringView(reinterpret_cast<char*>(header->begin()), numBytes);
  return true;
}

void HashStringAllocator::copyMultipartNoInline(
    char* FOLLY_NONNULL group,
    int32_t offset) {
  auto string = reinterpret_cast<StringView*>(group + offset);
  const auto numBytes = string->size();
  if (storeStringFast(string->data(), numBytes, group + offset)) {
    return;
  }
  // Write the string as non-contiguous chunks.
  ByteOutputStream stream(this, false, false);
  auto position = newWrite(stream, numBytes);
  stream.appendStringView(*string);
  finishWrite(stream, 0);

  // The stringView has a pointer to the first byte and the total
  // size. Read with contiguousString().
  *string = StringView(reinterpret_cast<char*>(position.position), numBytes);
}

std::string HashStringAllocator::toString() const {
  std::ostringstream out;

  out << "allocated: " << cumulativeBytes_ << " bytes" << std::endl;
  out << "free: " << freeBytes_ << " bytes in " << numFree_ << " blocks"
      << std::endl;
  out << "standalone allocations: " << sizeFromPool_ << " bytes in "
      << allocationsFromPool_.size() << " allocations" << std::endl;
  out << "ranges: " << pool_.numRanges() << std::endl;

  static const auto kHugePageSize = memory::AllocationTraits::kHugePageSize;

  for (auto i = 0; i < pool_.numRanges(); ++i) {
    auto topRange = pool_.rangeAt(i);
    auto topRangeSize = topRange.size();

    out << "range " << i << ": " << topRangeSize << " bytes" << std::endl;

    // Some ranges are short and contain one arena. Some are multiples of huge
    // page size and contain one arena per huge page.
    for (int64_t subRangeStart = 0; subRangeStart < topRangeSize;
         subRangeStart += kHugePageSize) {
      auto range = folly::Range<char*>(
          topRange.data() + subRangeStart,
          std::min<int64_t>(topRangeSize, kHugePageSize));
      auto size = range.size() - simd::kPadding;

      auto end = reinterpret_cast<Header*>(range.data() + size);
      auto header = reinterpret_cast<Header*>(range.data());
      while (header != nullptr && header != end) {
        out << "\t" << header->toString() << std::endl;
        header = header->next();
      }
    }
  }

  return out.str();
}

int64_t HashStringAllocator::checkConsistency() const {
  static const auto kHugePageSize = memory::AllocationTraits::kHugePageSize;

  uint64_t numFree = 0;
  uint64_t freeBytes = 0;
  int64_t allocatedBytes = 0;
  for (auto i = 0; i < pool_.numRanges(); ++i) {
    auto topRange = pool_.rangeAt(i);
    auto topRangeSize = topRange.size();
    if (topRangeSize >= kHugePageSize) {
      VELOX_CHECK_EQ(0, topRangeSize % kHugePageSize);
    }
    // Some ranges are short and contain one arena. Some are multiples of huge
    // page size and contain one arena per huge page.
    for (int64_t subRangeStart = 0; subRangeStart < topRangeSize;
         subRangeStart += kHugePageSize) {
      auto range = folly::Range<char*>(
          topRange.data() + subRangeStart,
          std::min<int64_t>(topRangeSize, kHugePageSize));
      auto size = range.size() - simd::kPadding;
      bool previousFree = false;
      auto end = reinterpret_cast<Header*>(range.data() + size);
      auto header = reinterpret_cast<Header*>(range.data());
      while (header != end) {
        VELOX_CHECK_GE(reinterpret_cast<char*>(header), range.data());
        VELOX_CHECK_LT(
            reinterpret_cast<char*>(header), reinterpret_cast<char*>(end));
        VELOX_CHECK_LE(
            reinterpret_cast<char*>(header->end()),
            reinterpret_cast<char*>(end));
        VELOX_CHECK_EQ(header->isPreviousFree(), previousFree);

        if (header->isFree()) {
          VELOX_CHECK(!previousFree);
          VELOX_CHECK(!header->isContinued());
          if (header->next()) {
            VELOX_CHECK_EQ(
                header->size(),
                *(reinterpret_cast<int32_t*>(header->end()) - 1));
          }
          ++numFree;
          freeBytes += sizeof(Header) + header->size();
        } else if (header->isContinued()) {
          // If the content of the header is continued, check the
          // continue header is readable and not free.
          auto continued = header->nextContinued();
          VELOX_CHECK(!continued->isFree());
          allocatedBytes += header->size() - sizeof(void*);
        } else {
          allocatedBytes += header->size();
        }
        previousFree = header->isFree();
        header = reinterpret_cast<Header*>(header->end());
      }
    }
  }

  VELOX_CHECK_EQ(numFree, numFree_);
  VELOX_CHECK_EQ(freeBytes, freeBytes_);
  uint64_t numInFreeList = 0;
  uint64_t bytesInFreeList = 0;
  for (auto i = 0; i < kNumFreeLists; ++i) {
    bool hasData = bits::isBitSet(freeNonEmpty_, i);
    bool listNonEmpty = !free_[i].empty();
    VELOX_CHECK_EQ(hasData, listNonEmpty);
    for (auto free = free_[i].next(); free != &free_[i]; free = free->next()) {
      ++numInFreeList;
      VELOX_CHECK(
          free->next()->previous() == free,
          "free list previous link inconsistent");
      auto size = headerOf(free)->size();
      VELOX_CHECK_GE(size, kMinAlloc);
      if (size - kMinAlloc < kNumFreeLists - 1) {
        VELOX_CHECK_EQ(size - kMinAlloc, i);
      } else {
        VELOX_CHECK_GE(size - kMinAlloc, kNumFreeLists - 1);
      }
      bytesInFreeList += size + sizeof(Header);
    }
  }

  VELOX_CHECK_EQ(numInFreeList, numFree_);
  VELOX_CHECK_EQ(bytesInFreeList, freeBytes_);
  return allocatedBytes;
}

bool HashStringAllocator::isEmpty() const {
  return sizeFromPool_ == 0 && checkConsistency() == 0;
}

void HashStringAllocator::checkEmpty() const {
  VELOX_CHECK_EQ(0, sizeFromPool_);
  VELOX_CHECK_EQ(0, checkConsistency());
}

int64_t HashStringAllocator::estimateReclaimableSize() {
  if (requestedContiguous_ && !allowSplittingContiguous_) {
    return 0;
  }

  // The last allocation is excluded from the compaction since it's currently in
  // use for serving allocations.
  const auto allocations{pool_.numRanges() - 1};
  if (allocations == 0) {
    return 0;
  }
  compactors_.clear();
  compactors_.reserve(allocations);

  struct SortableEntry {
    int64_t size;
    int64_t nonFreeBlockSize;
    int32_t allocationIndex;
  };
  std::vector<SortableEntry> entries;
  entries.reserve(allocations);

  int64_t remainingUsableSize{0};
  int64_t totalNonFreeBlockSize{0};
  for (auto i = 0; i < allocations; ++i) {
    compactors_.emplace_back(pool_.rangeAt(i));
    const auto& compactor{compactors_.back()};
    remainingUsableSize += compactor.usableSize();
    totalNonFreeBlockSize += compactor.nonFreeBlockSize();
    entries.push_back(
        SortableEntry{compactor.size(), compactor.nonFreeBlockSize(), i});
  }

  // Priority is given to allocations with larger sizes. When allocations have
  // identical sizes, those with less stored data take precedence.
  std::sort(
      entries.begin(),
      entries.end(),
      [](const SortableEntry& lhs, const SortableEntry& rhs) {
        if (lhs.size == rhs.size) {
          return lhs.nonFreeBlockSize < rhs.nonFreeBlockSize;
        }
        return lhs.size > rhs.size;
      });

  int64_t reclaimableSize{0};
  for (const auto& entry : entries) {
    auto& compactor{compactors_[entry.allocationIndex]};
    if (remainingUsableSize - compactor.usableSize() >= totalNonFreeBlockSize) {
      remainingUsableSize -= compactor.usableSize();
      compactor.setReclaimable();
      reclaimableSize += compactor.size();
    }
  }
  return reclaimableSize;
}

// TODO: Update cumulativeBytes_.
std::pair<int64_t, folly::F14FastMap<Header*, Header*>>
HashStringAllocator::compact() {
  // TODO: move ac into hsa.
  using AC = AllocationCompactor;

  if (estimateReclaimableSize() == 0) {
    return {0, {}};
  }

  AC::HeaderMap movedBlocks;
  AC::HeaderMap multipartMap;
  // Accumulate multipart map from all alloctions.
  for (const auto& compactor : compactors_) {
    compactor.accumulateMultipartMap(multipartMap);
  }

  std::queue<Header*> destBlocks;
  for (auto& compactor : compactors_) {
    // Remove all the free blocks in candidate allocations from free list since:
    // 1. Reclaimable allocations will be freed to AllocationPool;
    // 2. Unreclaimable allocations's free blocks will be squeezed, at the
    // end of the compaction the remaining free blocks will be added back to
    // free list.
    compactor.foreachBlock([&](Header* header) {
      if (header->isFree()) {
        this->removeFromFreeList(header, false);
        --this->numFree_;
        this->freeBytes_ -= sizeof(Header) + header->size();
      }
    });

    // Squeeze unreclaimable allocations, whose remaining free blocks will be
    // the destination blocks for the compaction.
    if (!compactor.isReclaimable()) {
      auto freeBlocks = compactor.squeeze(multipartMap, movedBlocks);
      for (const auto freeBlock : freeBlocks) {
        destBlocks.push(freeBlock);
      }
    }
  }

  int64_t compactedSize{0};
  Header* destBlock{nullptr};
  for (auto& compactor : compactors_) {
    if (!compactor.isReclaimable()) {
      continue;
    }

    auto srcBlock = compactor.nextBlock(false);
    int64_t srcOffset{0};
    Header** prevContPtr{nullptr};
    while (srcBlock != nullptr) {
      if (destBlock == nullptr) {
        VELOX_CHECK(!destBlocks.empty());
        destBlock = destBlocks.front();
        destBlocks.pop();
      }

      VELOX_CHECK_EQ((srcOffset == 0), (prevContPtr == nullptr));
      auto moveResult = AC::moveBlock(
          srcBlock,
          srcOffset,
          prevContPtr,
          destBlock,
          multipartMap,
          movedBlocks);
      srcOffset += moveResult.srcMovedSize;
      if (moveResult.prevContPtr != nullptr) {
        VELOX_CHECK_GT(moveResult.srcMovedSize, 0);
        VELOX_CHECK_LT(srcOffset, srcBlock->size());
        prevContPtr = moveResult.prevContPtr;
      }
      if (srcOffset == srcBlock->size()) {
        srcBlock = compactor.nextBlock(false, srcBlock);
        srcOffset = 0;
        prevContPtr = nullptr;
      }
      destBlock = moveResult.remainingDestBlock;
    }
    compactedSize += compactor.size();
  }

  testCheckFreeInCurrentRange();
  addFreeBlocksToFreeList();

  // Free empty allocations.
  for (int32_t i = compactors_.size() - 1; i >= 0; --i) {
    if (compactors_[i].isReclaimable()) {
      pool_.freeRangeAt(i);
    }
  }
  compactors_.clear();

  checkConsistency();
  return {compactedSize, movedBlocks};
};

void HashStringAllocator::addFreeBlocksToFreeList() {
  for (auto& compactor : compactors_) {
    if (compactor.isReclaimable()) {
      continue;
    }
    compactor.foreachBlock([&](Header* header) {
      if (!header->isFree()) {
        return;
      }
      header->clearFree();
      free(header);
    });
  }
}

HashStringAllocator::AllocationCompactor::AllocationCompactor(
    AllocationRange allocationRange)
    : range_{allocationRange} {
  if (size() >= kHugePageSize) {
    VELOX_CHECK_EQ(0, size() % kHugePageSize);
  }

  // Collects allocation info.
  int64_t nonFreeBlockSize{0};
  foreachBlock([&nonFreeBlockSize](Header* header) {
    if (!header->isFree()) {
      nonFreeBlockSize += header->size() + sizeof(Header);
    }
  });
  nonFreeBlockSize_ = nonFreeBlockSize;
}

void HashStringAllocator::AllocationCompactor::accumulateMultipartMap(
    HeaderMap& multipartMap) const {
  foreachBlock([&multipartMap](Header* header) {
    if (!header->isContinued()) {
      return;
    }
    const auto nextContinued{header->nextContinued()};
    VELOX_CHECK(!multipartMap.contains(nextContinued));
    multipartMap[nextContinued] = header;
  });
}

void HashStringAllocator::AllocationCompactor::foreachBlock(
    folly::Range<char*> range,
    const std::function<void(Header*)>& func) {
  for (int64_t subRangeOffset = 0; subRangeOffset < range.size();
       subRangeOffset += kHugePageSize) {
    auto header = reinterpret_cast<Header*>(range.data() + subRangeOffset);
    while (header) {
      func(header);
      header = header->next();
    }
  }
}

Header* HashStringAllocator::AllocationCompactor::squeezeArena(
    AllocationRange arena,
    HeaderMap& multipartMap,
    HeaderMap& movedBlocks) {
  VELOX_CHECK(
      reinterpret_cast<Header*>(arena.data() + arena.size())->isArenaEnd());

  auto offsetFromArenaStart = [&arena](Header* header) {
    return reinterpret_cast<char*>(header) - arena.data();
  };

  // Returns the next non free block in the arena. Returns nullptr if there is
  // no. 'header' should be the header of a block within the arena. If 'header'
  // is nullptr, returns the first non free block in arena.
  auto nextNonFreeBlockInArena = [&](Header* header) {
    if (header != nullptr) {
      auto nextNonFreeBlock = nextBlock(false, header);
      if (nextNonFreeBlock == nullptr ||
          offsetFromArenaStart(nextNonFreeBlock) >= arena.size()) {
        return static_cast<Header*>(nullptr);
      }
      return nextNonFreeBlock;
    }

    header = reinterpret_cast<Header*>(arena.data());
    while (header) {
      if (!header->isFree()) {
        return header;
      }
      header = header->next();
    }
    return static_cast<Header*>(nullptr);
  };

  auto movedFrom = nextNonFreeBlockInArena(nullptr);
  int64_t movedToOffset{0};
  while (movedFrom != nullptr) {
    const auto movedFromOffset{offsetFromArenaStart(movedFrom)};
    if (movedFromOffset == movedToOffset) {
      movedToOffset = movedFrom->end() - arena.data();
      movedFrom = nextNonFreeBlockInArena(movedFrom);
      continue;
    }
    VELOX_CHECK_GT(movedFromOffset, movedToOffset);

    auto movedTo = reinterpret_cast<Header*>(arena.data() + movedToOffset);
    auto nextNonFreeBlock = nextNonFreeBlockInArena(movedFrom);
    updateMap(movedFrom, movedTo, multipartMap, movedBlocks);
    // Don't use memcpy here since 'movedTo' and 'movedFrom' might overlap.
    memmove(movedTo, movedFrom, sizeof(Header) + movedFrom->size());
    movedTo->clearPreviousFree();

    movedFrom = nextNonFreeBlock;
    movedToOffset = movedTo->end() - arena.data();
  }

  if (movedToOffset == arena.size()) {
    return nullptr;
  }

  const auto remainingSize{arena.size() - movedToOffset};
  VELOX_CHECK_GE(remainingSize, kMinBlockSize);

  auto freeBlock =
      new (arena.data() + movedToOffset) Header(remainingSize - sizeof(Header));
  freeBlock->setFree();
  // TODO: Size at the end is not set.
  return freeBlock;
}

std::vector<HashStringAllocator::Header*>
HashStringAllocator::AllocationCompactor::squeeze(
    HeaderMap& multipartMap,
    HeaderMap& movedBlocks) {
  std::vector<HashStringAllocator::Header*> freeBlocks;
  for (int64_t subRangeOffset = 0; subRangeOffset < size();
       subRangeOffset += kHugePageSize) {
    const auto arenaStart = range_.data() + subRangeOffset;
    const auto arenaSize =
        std::min<int64_t>(range_.size(), kHugePageSize) - simd::kPadding;
    const auto arena = folly::Range<char*>(arenaStart, arenaSize);

    auto freeBlock = squeezeArena(arena, multipartMap, movedBlocks);
    if (freeBlock != nullptr) {
      freeBlocks.push_back(freeBlock);
    }
  }
  return freeBlocks;
}
void HashStringAllocator::AllocationCompactor::updateMapAsNext(
    Header* from,
    Header* to,
    HeaderMap& multipartMap,
    HeaderMap& movedBlocks) {
  if (!multipartMap.contains(from)) {
    VELOX_CHECK(!movedBlocks.contains(from));
    movedBlocks[from] = to;
    return;
  }

  auto prevHeader = multipartMap[from];
  VELOX_CHECK(!prevHeader->isFree());
  VELOX_CHECK(!prevHeader->isArenaEnd());
  VELOX_CHECK(prevHeader->isContinued());
  VELOX_CHECK_EQ(
      reinterpret_cast<void*>(prevHeader->nextContinued()),
      reinterpret_cast<void*>(from));
  prevHeader->setNextContinued(to);
  multipartMap.erase(from);
  multipartMap[to] = prevHeader;
}

void HashStringAllocator::AllocationCompactor::updateMapAsPrevious(
    Header* from,
    Header* to,
    HeaderMap& multipartMap) {
  VELOX_CHECK(from->isContinued());
  auto nextHeader = from->nextContinued();
  VELOX_CHECK(!nextHeader->isFree());
  VELOX_CHECK(!nextHeader->isArenaEnd());
  VELOX_CHECK(multipartMap.contains(nextHeader));
  multipartMap[nextHeader] = to;
}

void HashStringAllocator::AllocationCompactor::updateMap(
    Header* from,
    Header* to,
    HeaderMap& multipartMap,
    HeaderMap& movedBlocks) {
  updateMapAsNext(from, to, multipartMap, movedBlocks);

  if (from->isContinued()) {
    updateMapAsPrevious(from, to, multipartMap);
  }
}

Header* HashStringAllocator::AllocationCompactor::nextBlock(
    bool isFree,
    Header* header) const {
  if (header != nullptr) {
    VELOX_CHECK_GE(reinterpret_cast<char*>(header), range_.data());
    VELOX_CHECK_LT(
        reinterpret_cast<char*>(header),
        range_.data() + range_.size() - simd::kPadding);
    header = reinterpret_cast<Header*>(header->end());
  } else {
    header = reinterpret_cast<Header*>(range_.data());
  }

  while (!header->isArenaEnd()) {
    if (isFree == header->isFree()) {
      return header;
    }
    header = reinterpret_cast<Header*>(header->end());
  }

  const auto boundary = reinterpret_cast<char*>(header) + simd::kPadding;
  VELOX_CHECK_LE(boundary, range_.data() + range_.size());
  if (boundary == range_.data() + range_.size()) {
    return nullptr;
  }

  auto arenaStart = boundary;
  while (arenaStart < range_.data() + range_.size()) {
    header = reinterpret_cast<Header*>(arenaStart);
    while (!header->isArenaEnd()) {
      if (isFree == header->isFree()) {
        return header;
      }
      header = reinterpret_cast<Header*>(header->end());
    }
    arenaStart += kHugePageSize;
  }
  return nullptr;
}

HashStringAllocator::AllocationCompactor::MoveResult
HashStringAllocator::AllocationCompactor::moveBlock(
    Header* srcBlock,
    int64_t srcOffset,
    Header** prevContPtr,
    Header* destBlock,
    HeaderMap& multipartMap,
    HeaderMap& movedBlocks) {
  // Sanity check.
  VELOX_CHECK(!srcBlock->isFree());
  VELOX_CHECK(destBlock->isFree());
  VELOX_CHECK(srcOffset < srcBlock->size());
  VELOX_CHECK((srcOffset > 0) == (prevContPtr != nullptr));
  // TODO: check srcBlock and destBlock is not overlapped

  // Determine move size.
  const auto bytesToMove{srcBlock->size() - srcOffset};
  auto movableSize =
      tryAccommodate(destBlock->size(), bytesToMove, srcBlock->isContinued());
  if (movableSize == 0) {
    VELOX_CHECK(srcBlock->isContinued());
    return {0, prevContPtr, nullptr, destBlock->size()};
  }

  VELOX_CHECK_LE(movableSize, bytesToMove);
  const auto movingRestOfSrc = (movableSize == bytesToMove);
  // When moving final part of non-continued block, 'bytesTomove' may be less
  // than kMinAlloc, which will need padding to make the result block valid.
  const auto needsPadding = (bytesToMove < HashStringAllocator::kMinAlloc);
  if (needsPadding) {
    VELOX_CHECK(!srcBlock->isContinued());
    VELOX_CHECK(movingRestOfSrc);
  }

  // Update map
  if (srcOffset == 0) {
    updateMapAsNext(srcBlock, destBlock, multipartMap, movedBlocks);
  }
  if (movingRestOfSrc && srcBlock->isContinued()) {
    // Moving last part of a continued 'srcBlock', and this is previous.
    // Update next->previous
    updateMapAsPrevious(srcBlock, destBlock, multipartMap);
  }

  auto destSizeNeeded = movableSize;
  if (needsPadding) {
    VELOX_CHECK_LE(movableSize, HashStringAllocator::kMinAlloc);
    destSizeNeeded = HashStringAllocator::kMinAlloc;
  } else if (!movingRestOfSrc) {
    destSizeNeeded += Header::kContinuedPtrSize;
  }
  VELOX_CHECK_LE(destSizeNeeded, destBlock->size());

  const auto remainingDestSize{destBlock->size() - destSizeNeeded};
  Header* remainingDestBlock{nullptr};
  auto destNewSize{destBlock->size()};
  if (remainingDestSize >= kMinBlockSize) {
    // Create a new free block at the end of moved block.
    // TODO: size at the end.
    remainingDestBlock = new (destBlock->begin() + destSizeNeeded)
        Header(destBlock->size() - destSizeNeeded - sizeof(Header));
    remainingDestBlock->setFree();
    destNewSize = destSizeNeeded;
  }

  // Actual move.
  memcpy(destBlock->begin(), srcBlock->begin() + srcOffset, movableSize);
  destBlock->clearFree();
  if (movingRestOfSrc && !srcBlock->isContinued()) {
    destBlock->clearContinued();
  } else {
    destBlock->setContinued();
  }
  destBlock->setSize(destNewSize);

  if (prevContPtr != nullptr) {
    *prevContPtr = destBlock;
  }

  Header** resultPrevContPtr{nullptr};
  if (!movingRestOfSrc) {
    resultPrevContPtr = reinterpret_cast<Header**>(
        destBlock->end() - Header::kContinuedPtrSize);
  }

  auto destDiscardedSize{0};
  // 'srcBlock' is splitted for this move and there is still 'destBlock'
  // remaining, means that 'srcBlock' is splitted to not break the invariance of
  // HSA: a block has at least kMinAlloc bytes as data.
  const auto discardRemaining = (!movingRestOfSrc) && (remainingDestSize > 0);
  if (discardRemaining) {
    // TODO: this does not compile
    // VELOX_CHECK_EQ(remainingDestSize, AllocationCompactor::kMinBlockSize);
    remainingDestBlock = nullptr;
    destDiscardedSize = remainingDestSize;
  }

  return {
      movableSize, resultPrevContPtr, remainingDestBlock, destDiscardedSize};
}

void HashStringAllocator::testCheckFreeInCurrentRange() const {
  for (auto i = 0; i < kNumFreeLists; ++i) {
    auto* item = free_[i].next();
    while (item != &free_[i]) {
      auto header = headerOf(item);
      VELOX_CHECK(pool_.isInCurrentRange(header));
      item = item->next();
    }
  }
}

} // namespace facebook::velox
