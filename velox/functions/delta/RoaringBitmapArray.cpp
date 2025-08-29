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

#include "velox/functions/delta/RoaringBitmapArray.h"
#include "velox/common/base/IOUtils.h"

#include <common/base/Exceptions.h>
#include <folly/Likely.h>
#include <cstdint>
#include <limits>

namespace facebook::velox::functions::delta {

namespace {
int64_t composeFromHighLowBytes(int32_t high, int32_t low) {
  VELOX_CHECK_GE(high, 0);
  VELOX_CHECK_GE(low, 0);
  return static_cast<int64_t>(high) << 32 |
      (static_cast<int64_t>(low) & 0xFFFFFFFFLL);
}

const int64_t kMaxRepresentableValue = composeFromHighLowBytes(
    std::numeric_limits<int32_t>::max() - 1,
    std::numeric_limits<int32_t>::max());
} // namespace

void RoaringBitmapArray::deserialize(const char* serialized) {
  bitmaps_.clear();
  buckContexts_.clear();
  common::InputByteStream stream(serialized);
  const auto magicNumber = stream.read<int32_t>();
  switch (magicNumber) {
    case kPortableSerializationFormatMagicNumber: {
      const auto numberOfBitmaps = stream.read<int64_t>();
      VELOX_CHECK_GE(
          numberOfBitmaps,
          0,
          "Invalid RoaringBitmapArray length ({} < 0)",
          numberOfBitmaps);
      VELOX_CHECK_LE(
          numberOfBitmaps,
          std::numeric_limits<int32_t>::max(),
          "Invalid RoaringBitmapArray length ({} > {})",
          numberOfBitmaps,
          std::numeric_limits<int32_t>::max());
      // The sparse bitmap array format uses the number as an array size lower
      // bound.
      const auto minimumArraySize = static_cast<int32_t>(numberOfBitmaps);
      bitmaps_.reserve(minimumArraySize);
      buckContexts_.reserve(minimumArraySize);
      int32_t lastIndex = 0;
      for (int32_t i = 0; i < numberOfBitmaps; ++i) {
        const auto key = stream.read<int32_t>();
        VELOX_CHECK_GE(
            key, 0, "Invalid unsigned entry in RoaringBitmapArray ({})", key);
        VELOX_CHECK_GE(
            key,
            lastIndex,
            "Keys are required to be sorted in ascending order");
        // Fill gaps in sparse data.
        while (lastIndex < key) {
          bitmaps_.emplace_back(std::make_shared<roaring::Roaring>());
          buckContexts_.emplace_back(std::make_shared<roaring::BulkContext>());
          ++lastIndex;
        }
        roaring::api::roaring_bitmap_t* r =
            roaring::api::roaring_bitmap_portable_deserialize(
                serialized + stream.offset());
        VELOX_CHECK_NOT_NULL(r);
        bitmaps_.emplace_back(std::make_shared<roaring::Roaring>(r));
        buckContexts_.emplace_back(std::make_shared<roaring::BulkContext>());
        ++lastIndex;
        // Advances the stream for N bytes which is the serialized size of the
        // previous read bitmap.
        auto sizeToAdvance = bitmaps_.back()->getSizeInBytes();
        for (size_t j = 0; j < sizeToAdvance; j++) {
          stream.read<int8_t>();
        }
      }
    } break;
    default:
      VELOX_NYI(
          "Unexpected magic number in serialized roaring bitmap array: {}",
          magicNumber);
  }
}

void RoaringBitmapArray::serialize(char* buf) const {
  common::OutputByteStream stream(buf);
  stream.appendOne<int32_t>(kPortableSerializationFormatMagicNumber);
  stream.appendOne<int64_t>(bitmaps_.size());
  for (size_t i = 0; i < bitmaps_.size(); ++i) {
    stream.appendOne<int32_t>(i);
    const auto serializedSize = bitmaps_[i]->getSizeInBytes();
    // FIXME: The following code conducts an extra copy.
    std::vector<char> data(serializedSize);
    bitmaps_[i]->write(data.data(), data.size());
    stream.append(data.data(), data.size());
  }
}

void RoaringBitmapArray::add(int64_t value) {
  checkValue(value);
  const auto high = highBytesUnsafe(value);
  const auto low = lowBytesUnsafe(value);
  if (high >= bitmaps_.size()) {
    // Grows the bitmap array.
    for (int32_t i = bitmaps_.size(); i <= high; ++i) {
      bitmaps_.emplace_back(std::make_shared<roaring::Roaring>());
      buckContexts_.emplace_back(std::make_shared<roaring::BulkContext>());
    }
  }
  auto bitmap = bitmaps_[high];
  bitmap->add(low);
}

bool RoaringBitmapArray::contains(int64_t value) {
  checkValue(value);
  const auto high = highBytesUnsafe(value);
  if (high >= bitmaps_.size()) {
    return false;
  }
  if (FOLLY_LIKELY(high == lastHighBytes_)) {
    // Fast path for ordered input.
    const auto low = lowBytesUnsafe(value);
    return lastBitmap_->containsBulk(*lastContext_, low);
  }
  const auto highBitmap = bitmaps_[high];
  const auto highBuckContext = buckContexts_[high];
  const auto low = lowBytesUnsafe(value);
  lastHighBytes_ = high;
  lastBitmap_ = highBitmap.get();
  lastContext_ = highBuckContext.get();
  return highBitmap->containsBulk(*highBuckContext, low);
}

int64_t RoaringBitmapArray::serializedSizeInBytes() const {
  const int64_t magicNumberSize = 4L;
  const int64_t bitmapCountSize = 8L;
  const int64_t individualBitmapKeySize = 4L;

  int64_t sum = 0L;
  sum += magicNumberSize;
  sum += bitmapCountSize;

  for (const auto& bitmap : bitmaps_) {
    sum += individualBitmapKeySize;
    sum += bitmap->getSizeInBytes();
  }

  return sum;
}

void RoaringBitmapArray::checkValue(int64_t value) {
  VELOX_CHECK_GE(value, 0, "Invalid RoaringBitmapArray value ({}) >= 0", value);
  VELOX_CHECK_LE(
      value,
      kMaxRepresentableValue,
      "Invalid RoaringBitmapArray value ({}) <= {}",
      value,
      kMaxRepresentableValue);
}

int32_t RoaringBitmapArray::highBytesUnsafe(int64_t value) {
  return static_cast<int32_t>(value >> 32);
}

int32_t RoaringBitmapArray::lowBytesUnsafe(int64_t value) {
  return static_cast<int32_t>(value & 0xFFFFFFFFLL);
}
} // namespace facebook::velox::functions::delta
