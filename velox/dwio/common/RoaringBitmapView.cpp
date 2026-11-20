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

#include "velox/dwio/common/RoaringBitmapView.h"

#include <folly/lang/Bits.h>

#include <algorithm>
#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {
namespace {

constexpr uint32_t kSerialCookieNoRun = 12'346;
constexpr uint32_t kSerialCookie = 12'347;
constexpr uint32_t kRunContainersNoOffsetThreshold = 4;
constexpr uint32_t kMaxContainers = 65'536;
constexpr uint32_t kMaxArrayContainerCardinality = 4'096;
constexpr size_t kBitsetContainerBytes = 8'192;
constexpr uint32_t kBitsetContainerWords = 1'024;
constexpr uint32_t kMaxContainerValue = 65'535;

uint16_t readLittleEndian16(const uint8_t* data) {
  uint16_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

uint32_t readLittleEndian32(const uint8_t* data) {
  uint32_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

uint64_t readLittleEndian64(const uint8_t* data) {
  uint64_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

// Sets bits [begin, end) of 'bitmap'. Writes only the bytes that hold those
// bits, so it never touches memory past a buffer sized for the batch.
void setBitRange(uint8_t* bitmap, uint64_t begin, uint64_t end) {
  while (begin < end && begin % 8 != 0) {
    bits::setBit(bitmap, begin++);
  }
  const uint64_t fullBytesEnd = end & ~uint64_t{7};
  if (begin < fullBytesEnd) {
    std::memset(bitmap + begin / 8, 0xFF, (fullBytesEnd - begin) / 8);
    begin = fullBytesEnd;
  }
  while (begin < end) {
    bits::setBit(bitmap, begin++);
  }
}

// The apply functions below set the bits of the container's positions that
// fall in [from, to). 'lower' is the row position of bit 0 in 'bitmap'. The
// caller guarantees base <= from < to <= base + 65536.

void applyArrayContainer(
    const uint8_t* values,
    uint32_t count,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  uint32_t first = 0;
  uint32_t last = count;
  while (first < last) {
    const uint32_t mid = first + (last - first) / 2;
    if (readLittleEndian16(values + 2 * mid) < low) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  for (uint32_t i = first; i < count; ++i) {
    const uint32_t value = readLittleEndian16(values + 2 * i);
    if (value >= high) {
      break;
    }
    bits::setBit(bitmap, static_cast<uint64_t>(base + value - lower));
  }
}

void applyRunContainer(
    const uint8_t* runs,
    uint32_t numRuns,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  // Runs are sorted and disjoint, so their last values increase as well.
  uint32_t first = 0;
  uint32_t last = numRuns;
  while (first < last) {
    const uint32_t mid = first + (last - first) / 2;
    const uint32_t runLast =
        static_cast<uint32_t>(readLittleEndian16(runs + 4 * mid)) +
        readLittleEndian16(runs + 4 * mid + 2);
    if (runLast < low) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  for (uint32_t i = first; i < numRuns; ++i) {
    const uint32_t start = readLittleEndian16(runs + 4 * i);
    if (start >= high) {
      break;
    }
    const uint32_t end = start + readLittleEndian16(runs + 4 * i + 2) + 1;
    setBitRange(
        bitmap,
        static_cast<uint64_t>(base + std::max(start, low) - lower),
        static_cast<uint64_t>(base + std::min(end, high) - lower));
  }
}

void applyBitsetContainer(
    const uint8_t* words,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  const uint32_t firstWord = low / 64;
  const uint32_t lastWord = (high - 1) / 64;
  for (uint32_t w = firstWord; w <= lastWord; ++w) {
    uint64_t word = readLittleEndian64(words + 8 * w);
    if (w == firstWord) {
      word &= ~uint64_t{0} << (low % 64);
    }
    if (w == lastWord && high % 64 != 0) {
      word &= (uint64_t{1} << (high % 64)) - 1;
    }
    const int64_t wordBase = base + static_cast<int64_t>(w) * 64 - lower;
    while (word != 0) {
      bits::setBit(
          bitmap, static_cast<uint64_t>(wordBase + __builtin_ctzll(word)));
      word &= word - 1;
    }
  }
}

void appendArrayPositions(
    const uint8_t* values,
    uint32_t count,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t i = 0; i < count; ++i) {
    positions.push_back(base + readLittleEndian16(values + 2 * i));
  }
}

void appendRunPositions(
    const uint8_t* runs,
    uint32_t numRuns,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t i = 0; i < numRuns; ++i) {
    const uint32_t start = readLittleEndian16(runs + 4 * i);
    const uint32_t last = start + readLittleEndian16(runs + 4 * i + 2);
    for (uint32_t value = start; value <= last; ++value) {
      positions.push_back(base + value);
    }
  }
}

void appendBitsetPositions(
    const uint8_t* words,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t w = 0; w < kBitsetContainerWords; ++w) {
    uint64_t word = readLittleEndian64(words + 8 * w);
    while (word != 0) {
      positions.push_back(
          base + static_cast<int64_t>(w) * 64 + __builtin_ctzll(word));
      word &= word - 1;
    }
  }
}

} // namespace

RoaringBitmapView::RoaringBitmapView(std::string_view data, Format format)
    : data_(data) {
  if (format == Format::k32Bit) {
    parse32Bit(0, 0);
  } else {
    parse64Bit();
  }

  // Writers emit containers in increasing order. Sort anyway so an out-of-order
  // bitmap is still applied correctly; there are few containers per file.
  const auto byBase = [](const Container& lhs, const Container& rhs) {
    return lhs.base < rhs.base;
  };
  if (!std::is_sorted(containers_.begin(), containers_.end(), byBase)) {
    std::sort(containers_.begin(), containers_.end(), byBase);
  }
  // A key with two containers would leave positions() unsorted and could count
  // a position twice.
  for (size_t i = 1; i < containers_.size(); ++i) {
    VELOX_CHECK_LT(
        containers_[i - 1].base,
        containers_[i].base,
        "Roaring bitmap has two containers for the same key.");
  }
}

void RoaringBitmapView::parse64Bit() {
  VELOX_CHECK_GE(
      data_.size(), sizeof(uint64_t), "Truncated Roaring64Bitmap group count.");
  const uint64_t numGroups = readLittleEndian64(data());
  static constexpr uint64_t kMaxGroups = 1'000'000;
  VELOX_CHECK_LE(
      numGroups,
      kMaxGroups,
      "Roaring64Bitmap group count exceeds sanity limit: {}",
      numGroups);

  size_t offset = sizeof(uint64_t);
  for (uint64_t group = 0; group < numGroups; ++group) {
    VELOX_CHECK_GE(
        data_.size() - offset,
        sizeof(uint32_t),
        "Truncated Roaring64Bitmap group header.");
    const uint32_t highBits = readLittleEndian32(data() + offset);
    offset += sizeof(uint32_t);
    VELOX_CHECK_LE(
        highBits,
        kMaxGroupKey,
        "Roaring64Bitmap group key exceeds the maximum the Iceberg "
        "deletion-vector format can represent: {}",
        highBits);
    offset = parse32Bit(offset, static_cast<int64_t>(highBits) << 32);
  }
}

size_t RoaringBitmapView::parse32Bit(size_t offset, int64_t highBitsOffset) {
  const auto* data = this->data();
  const auto remaining = [&]() { return data_.size() - offset; };
  VELOX_CHECK_GE(remaining(), 8, "32-bit bitmap too small.");

  const uint32_t cookie = readLittleEndian32(data + offset);
  offset += sizeof(uint32_t);

  bool hasRunContainers{false};
  uint32_t numContainers{0};
  if ((cookie & 0xFFFF) == kSerialCookie) {
    hasRunContainers = true;
    numContainers = (cookie >> 16) + 1;
  } else if (cookie == kSerialCookieNoRun) {
    numContainers = readLittleEndian32(data + offset);
    offset += sizeof(uint32_t);
  } else {
    VELOX_FAIL(
        "Unknown roaring bitmap cookie: {}. Expected {} or {}.",
        cookie,
        kSerialCookieNoRun,
        kSerialCookie);
  }

  if (numContainers == 0) {
    return offset;
  }
  VELOX_CHECK_LE(
      numContainers,
      kMaxContainers,
      "Roaring bitmap container count exceeds the 16-bit key space: {}",
      numContainers);

  const size_t runBitmapOffset = offset;
  if (hasRunContainers) {
    const size_t runBitmapBytes = bits::nbytes(numContainers);
    VELOX_CHECK_GE(remaining(), runBitmapBytes, "Truncated run bitmap.");
    offset += runBitmapBytes;
  }

  const size_t keysOffset = offset;
  VELOX_CHECK_GE(
      remaining(),
      static_cast<size_t>(numContainers) * 4,
      "Truncated container metadata.");
  offset += static_cast<size_t>(numContainers) * 4;

  const bool hasOffsetSection =
      !hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold;
  if (hasOffsetSection) {
    VELOX_CHECK_GE(
        remaining(),
        static_cast<size_t>(numContainers) * 4,
        "Truncated offset section.");
    offset += static_cast<size_t>(numContainers) * sizeof(uint32_t);
  }

  containers_.reserve(containers_.size() + numContainers);
  for (uint32_t i = 0; i < numContainers; ++i) {
    const uint16_t key = readLittleEndian16(data + keysOffset + 4 * i);
    const uint32_t cardinality = static_cast<uint32_t>(readLittleEndian16(
                                     data + keysOffset + 4 * i + 2)) +
        1;
    const bool isRun =
        hasRunContainers && bits::isBitSet(data + runBitmapOffset, i);
    const int64_t base = highBitsOffset | (static_cast<int64_t>(key) << 16);

    if (isRun) {
      VELOX_CHECK_GE(remaining(), 2, "Truncated run container.");
      const uint16_t numRuns = readLittleEndian16(data + offset);
      offset += sizeof(uint16_t);
      VELOX_CHECK_GE(
          remaining(),
          static_cast<size_t>(numRuns) * 4,
          "Truncated run container data.");
      // Applying a range binary-searches the runs, so they must be sorted,
      // disjoint and inside the container. A violation would otherwise drop
      // deletes silently.
      int64_t previousLast = -1;
      for (uint16_t r = 0; r < numRuns; ++r) {
        const uint32_t start = readLittleEndian16(data + offset + 4 * r);
        const uint32_t last =
            start + readLittleEndian16(data + offset + 4 * r + 2);
        VELOX_CHECK_LE(
            last,
            kMaxContainerValue,
            "Roaring run extends past its container.");
        VELOX_CHECK_GT(
            static_cast<int64_t>(start),
            previousLast,
            "Roaring runs are not sorted and disjoint.");
        previousLast = last;
      }
      containers_.push_back({
          .kind = ContainerKind::kRun,
          .count = numRuns,
          .base = base,
          .lastPosition = base + previousLast,
          .dataOffset = offset,
      });
      offset += static_cast<size_t>(numRuns) * 4;
    } else if (cardinality <= kMaxArrayContainerCardinality) {
      VELOX_CHECK_GE(
          remaining(),
          static_cast<size_t>(cardinality) * 2,
          "Truncated array container.");
      // Applying a range binary-searches the values, so they must be strictly
      // increasing. A violation would otherwise drop deletes silently.
      int32_t previous = -1;
      for (uint32_t j = 0; j < cardinality; ++j) {
        const int32_t value = readLittleEndian16(data + offset + 2 * j);
        VELOX_CHECK_GT(
            value,
            previous,
            "Roaring array container values are not strictly increasing.");
        previous = value;
      }
      containers_.push_back({
          .kind = ContainerKind::kArray,
          .count = cardinality,
          .base = base,
          .lastPosition = base + previous,
          .dataOffset = offset,
      });
      offset += static_cast<size_t>(cardinality) * 2;
    } else {
      VELOX_CHECK_GE(
          remaining(), kBitsetContainerBytes, "Truncated bitset container.");
      int64_t lastValue = -1;
      for (int32_t w = kBitsetContainerWords - 1; w >= 0; --w) {
        const uint64_t word = readLittleEndian64(data + offset + 8 * w);
        if (word != 0) {
          lastValue = static_cast<int64_t>(w) * 64 + 63 - __builtin_clzll(word);
          break;
        }
      }
      containers_.push_back({
          .kind = ContainerKind::kBitset,
          .count = 0,
          .base = base,
          .lastPosition = base + lastValue,
          .dataOffset = offset,
      });
      offset += kBitsetContainerBytes;
    }
  }
  return offset;
}

void RoaringBitmapView::apply(int64_t begin, int64_t end, uint8_t* bitmap) {
  while (containerIndex_ < containers_.size() &&
         containers_[containerIndex_].lastPosition < begin) {
    ++containerIndex_;
  }

  for (size_t i = containerIndex_;
       i < containers_.size() && containers_[i].base < end;
       ++i) {
    const auto& container = containers_[i];
    const int64_t from = std::max(begin, container.base);
    const int64_t to = std::min(end, container.lastPosition + 1);
    if (from >= to) {
      continue;
    }
    const auto* payload = data() + container.dataOffset;
    switch (container.kind) {
      case ContainerKind::kArray:
        applyArrayContainer(
            payload, container.count, container.base, from, to, begin, bitmap);
        break;
      case ContainerKind::kRun:
        applyRunContainer(
            payload, container.count, container.base, from, to, begin, bitmap);
        break;
      case ContainerKind::kBitset:
        applyBitsetContainer(payload, container.base, from, to, begin, bitmap);
        break;
    }
  }

  // A container whose last position is behind this range is fully consumed.
  while (containerIndex_ < containers_.size() &&
         containers_[containerIndex_].lastPosition < end) {
    ++containerIndex_;
  }
}

uint64_t RoaringBitmapView::cardinality() const {
  uint64_t cardinality{0};
  for (const auto& container : containers_) {
    const auto* payload = data() + container.dataOffset;
    switch (container.kind) {
      case ContainerKind::kArray:
        cardinality += container.count;
        break;
      case ContainerKind::kRun:
        for (uint32_t i = 0; i < container.count; ++i) {
          cardinality += readLittleEndian16(payload + 4 * i + 2) + uint64_t{1};
        }
        break;
      case ContainerKind::kBitset:
        for (uint32_t w = 0; w < kBitsetContainerWords; ++w) {
          cardinality +=
              __builtin_popcountll(readLittleEndian64(payload + 8 * w));
        }
        break;
    }
  }
  return cardinality;
}

std::vector<int64_t> RoaringBitmapView::positions() const {
  std::vector<int64_t> positions;
  for (const auto& container : containers_) {
    const auto* payload = data() + container.dataOffset;
    switch (container.kind) {
      case ContainerKind::kArray:
        appendArrayPositions(
            payload, container.count, container.base, positions);
        break;
      case ContainerKind::kRun:
        appendRunPositions(payload, container.count, container.base, positions);
        break;
      case ContainerKind::kBitset:
        appendBitsetPositions(payload, container.base, positions);
        break;
    }
  }
  return positions;
}

} // namespace facebook::velox::dwio::common
