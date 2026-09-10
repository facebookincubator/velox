/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <bit>
#include <cmath>
#include <span>
#include <unordered_map>
#include <vector>

#include "folly/container/F14Map.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Frequency partition encoding assigns shorter bit widths to more frequent
// values, similar to Huffman coding but with random access support. Values
// are partitioned by frequency into tiers (1-bit, 2-bit, 4-bit, etc.) and
// rows are reordered to group same-tier values together.
//
// When an index type other than NoIndex is chosen at encode time, a positional
// index is appended to the wire format so that materialize() and
// readWithVisitor() operate in original-row-index space, correctly
// reconstructing the original column order.
//
// Wire format (NoIndex — backward-compatible with legacy format):
//   Encoding::kPrefixSize bytes: standard Encoding prefix
//   4 bytes: number of partitions
//   4+XX bytes: partition offsets encoding (nested)
//   4+YY bytes: partition sizes encoding (nested)
//   For each non-empty tier (1-bit, 2-bit, 4-bit, 8-bit, 16-bit, 32-bit):
//     4+ZZ bytes: dictionary encoding (nested)
//     4+WW bytes: keys encoding (nested)
//   4+VV bytes: unencoded values (nested, if any)
//
// Wire format extension (any indexed mode — appended at end):
//   1 byte: formatVersion (= kFormatVersion = 1)
//   1 byte: indexType (FreqPartIndexType)
//   4 bytes: indexPayloadBytes
//   [indexPayloadBytes bytes]: index payload (see below)
//
// Index payload layout (PerTierBitmaps):
//   For each non-empty tier (same order as above):
//     4 bytes: bitmapByteCount = ceil(N/64)*8
//     [bitmapByteCount]: bitmap (uint64_t words, LSB-first;
//                        bit i set iff original position i belongs to tier)
//
// Index payload layout (TierTagArray):
//   1 byte: tagBits = ceilLog2(numTiers + 1), minimum 1
//   3 bytes: padding
//   4 bytes: tagArrayByteCount
//   [tagArrayByteCount]: packed tag array (LSB-first; tag = tier index
//                        0..numTiers-1, or numTiers for fallback)
//
// Index payload layout (EliasFano):
//   For each non-empty tier (same order as above):
//     1 byte: lowBits
//     3 bytes: padding
//     4 bytes: lowByteCount
//     4 bytes: highWordCount
//     [lowByteCount]: packed low array (LSB-first)
//     [highWordCount * 8]: high bitmap (uint64_t words)

namespace facebook::nimble {

/// Selects the positional index representation stored in the wire format.
enum class FreqPartIndexType : uint8_t {
  NoIndex = 0, ///< no index; tier-order output (backward-compatible)
  PerTierBitmaps = 1, ///< one N-bit bitmap per tier + Rank9 superblock
  TierTagArray = 2, ///< packed tag array + sampled rank index
  EliasFano = 3, ///< per-tier Elias-Fano positions (decoded at load time)
};

template <typename T>
class FrequencyPartitionEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static const int kNumPartitionsOffset = Encoding::kPrefixSize;
  static constexpr uint8_t kFormatVersion = 1;
  static constexpr uint32_t kRankSampleStride = 256;

  FrequencyPartitionEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory = nullptr,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;
  uint32_t getTierForRow(uint32_t rowIndex) const;

 private:
  struct TierInfo {
    uint32_t keyBits;
    uint32_t capacity;
    Vector<T> dictionary;
    // Materialized keys: indices[rank] → dictionary index.
    // For NoIndex: rank = sequential position in tier's encoded range.
    // For indexed modes: rank = count of tier elements before original pos u.
    Vector<uint32_t> indices;
    uint32_t startRow; // NoIndex only: offset in encoded stream
    uint32_t size; // NoIndex only: count in encoded stream

    // Index fields — populated based on indexType_:
    uint32_t tierCount{0}; // element count (indexed modes)
    Vector<uint64_t> bitmap; // PerTierBitmaps: N-bit bitmap
    Vector<uint32_t> rankSuperblock; // PerTierBitmaps: Rank9 cumulative counts
    Vector<uint32_t> efPositions; // EliasFano: decoded original positions

    TierInfo(velox::memory::MemoryPool* pool)
        : keyBits(0),
          capacity(0),
          dictionary(pool),
          indices(pool),
          startRow(0),
          size(0),
          bitmap(pool),
          rankSuperblock(pool),
          efPositions(pool) {}
  };

  // Get capacity for a given key bit width
  static constexpr uint32_t getCapacity(uint32_t keyBits) {
    return (keyBits == 1) ? 2
        : (keyBits == 2)  ? 4
        : (keyBits == 4)  ? 16
        : (keyBits == 8)  ? 256
        : (keyBits == 16) ? (65536 - 256)
        : (keyBits == 32) ? (4294967296ULL - 65536)
                          : 0;
  }

  static constexpr uint32_t getMaxKeyBits() {
    constexpr size_t valueSize =
        std::is_same_v<T, std::string_view> ? sizeof(int64_t) : sizeof(T);
    constexpr size_t valueBits = valueSize * 8;

    if constexpr (valueBits <= 8) {
      return 4;
    } else if constexpr (valueBits <= 16) {
      return 8;
    } else if constexpr (valueBits <= 32) {
      return 16;
    } else {
      return 32;
    }
  }

  // ---------------------------------------------------------------------------
  // Bit-packing utilities (ported from EncodingsPlayground)
  // ---------------------------------------------------------------------------

  static constexpr uint8_t ceilLog2WithMinOne(uint32_t x) {
    if (x <= 1u) {
      return 1u;
    }
    return static_cast<uint8_t>(std::bit_width(x - 1u));
  }

  static uint8_t chooseEliasFanoLowBits(uint64_t universe, uint64_t n) {
    if (n == 0 || universe <= n) {
      return 0;
    }
    const uint64_t ratio = universe / n;
    if (ratio <= 1) {
      return 0;
    }
    return static_cast<uint8_t>(
        std::min<uint64_t>(31, std::bit_width(ratio) - 1));
  }

  // Decode Elias-Fano encoded positions from raw bytes into a sorted vector.
  // Requires zero-padding past highBase for safe 8-byte reads.
  static void decodeEliasFanoPositions(
      const uint8_t* lowBase,
      uint8_t lowBits,
      const uint8_t* highBase,
      uint32_t highWords,
      uint32_t count,
      Vector<uint32_t>& out) {
    out.resize(count);
    uint32_t rank = 0;
    for (uint32_t w = 0; w < highWords && rank < count; ++w) {
      uint64_t bits = 0;
      std::memcpy(&bits, highBase + static_cast<size_t>(w) * 8, 8);
      while (bits && rank < count) {
        const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(bits));
        const uint32_t high = w * 64 + bit - rank;
        const uint32_t low = unpackBits(lowBase, rank, lowBits);
        out[rank++] = (high << lowBits) | low;
        bits &= bits - 1;
      }
    }
  }

  // Extract `keyBits`-wide value at zero-based rank `r` from a LSB-first
  // packed byte array. Safe bounded read (no padding required).
  static uint32_t unpackBits(const uint8_t* base, uint32_t r, uint8_t keyBits) {
    if (keyBits == 0) {
      return 0;
    }
    const size_t bitPos = static_cast<size_t>(r) * keyBits;
    const size_t byteIdx = bitPos >> 3;
    const uint32_t bitOff = static_cast<uint32_t>(bitPos & 7);
    const uint32_t mask = (keyBits == 32) ? ~0u : ((1u << keyBits) - 1u);
    // At most ceil((7+31)/8) = 5 bytes needed; read only those.
    const uint32_t bytesNeeded = (bitOff + keyBits + 7) / 8;
    uint64_t buf = 0;
    for (uint32_t b = 0; b < bytesNeeded; ++b) {
      buf |= static_cast<uint64_t>(base[byteIdx + b]) << (b * 8);
    }
    return static_cast<uint32_t>((buf >> bitOff) & mask);
  }

  // Pack `keys` into LSB-first format, appending bytes to `out`.
  static void packBitsInto(
      std::vector<char>& out,
      const uint32_t* keys,
      uint32_t count,
      uint8_t keyBits) {
    if (keyBits == 0 || count == 0) {
      return;
    }
    const size_t byteCount = (static_cast<size_t>(count) * keyBits + 7) / 8;
    const size_t base = out.size();
    out.resize(base + byteCount, '\0');
    for (uint32_t r = 0; r < count; ++r) {
      const uint32_t k = keys[r];
      size_t bitPos = static_cast<size_t>(r) * keyBits;
      uint32_t remaining = k;
      uint8_t bitsLeft = keyBits;
      while (bitsLeft > 0) {
        const size_t byteIdx = bitPos / 8;
        const size_t bitOff = bitPos % 8;
        const uint8_t chunk = static_cast<uint8_t>(
            bitsLeft > (8 - bitOff) ? (8 - bitOff) : bitsLeft);
        reinterpret_cast<uint8_t&>(out[base + byteIdx]) |=
            static_cast<uint8_t>((remaining & ((1u << chunk) - 1u)) << bitOff);
        remaining >>= chunk;
        bitPos += chunk;
        bitsLeft -= chunk;
      }
    }
  }

  // Extract tag at position `pos` from a LSB-first packed tag array.
  // tagBits must be ≤ 8.
  static uint8_t
  unpackTagAt(const uint8_t* base, uint32_t pos, uint8_t tagBits) {
    const size_t bitPos = static_cast<size_t>(pos) * tagBits;
    const size_t byteIdx = bitPos / 8;
    const size_t bitOff = bitPos % 8;
    const uint8_t mask = static_cast<uint8_t>((1u << tagBits) - 1u);
    uint16_t buf = static_cast<uint16_t>(base[byteIdx]);
    if (bitOff + tagBits > 8) {
      buf |= static_cast<uint16_t>(base[byteIdx + 1]) << 8;
    }
    return static_cast<uint8_t>((buf >> bitOff) & mask);
  }

  // ---------------------------------------------------------------------------
  // Rank / positional index helpers
  // ---------------------------------------------------------------------------

  // O(1) rank of set bits in tier.bitmap strictly before position `pos`,
  // using the precomputed Rank9 superblock in tier.rankSuperblock.
  static uint32_t popcountPrefixFast(const TierInfo& tier, uint32_t pos) {
    const uint32_t blk = pos / 512;
    const uint32_t wordOff = (pos % 512) / 64;
    const uint32_t bitOff = pos % 64;
    uint32_t rank = tier.rankSuperblock[blk];
    for (uint32_t w = blk * 8; w < blk * 8 + wordOff; ++w) {
      rank += static_cast<uint32_t>(__builtin_popcountll(tier.bitmap[w]));
    }
    if (bitOff) {
      rank += static_cast<uint32_t>(__builtin_popcountll(
          tier.bitmap[blk * 8 + wordOff] & ((uint64_t{1} << bitOff) - 1)));
    }
    return rank;
  }

  // Count fallback elements strictly before position `pos`.
  // Uses coveredBitmap_ and fallbackWordPrefix_.
  uint32_t fallbackRankAt(uint32_t pos) const {
    const uint32_t word = pos / 64;
    uint32_t rank = fallbackWordPrefix_[word];
    const uint64_t uncov = ~coveredBitmap_[word];
    rank += static_cast<uint32_t>(
        __builtin_popcountll(uncov & ((uint64_t{1} << (pos % 64)) - 1)));
    return rank;
  }

  // Count elements with tag == tierIdx strictly before position `pos`.
  // Uses tierRankSamples_ and tagArray_. tierIdx == tiers_.size() is fallback.
  uint32_t tierRankAtForTag(uint32_t tierIdx, uint32_t pos) const {
    const uint32_t sampleIdx = pos / kRankSampleStride;
    uint32_t rank = tierRankSamples_[tierIdx][sampleIdx];
    const uint32_t scanStart = sampleIdx * kRankSampleStride;
    const uint8_t* tagBase = tagArray_.data();
    const uint8_t numActiveTiers = static_cast<uint8_t>(tiers_.size());
    for (uint32_t j = scanStart; j < pos; ++j) {
      const uint8_t t = unpackTagAt(tagBase, j, tagBits_);
      const uint32_t b = (t < numActiveTiers) ? t : numActiveTiers;
      if (b == tierIdx) {
        ++rank;
      }
    }
    return rank;
  }

  // ---------------------------------------------------------------------------
  // Per-index-type decode helpers
  // ---------------------------------------------------------------------------

  template <FreqPartIndexType I>
  T decodeAtOriginalIndexImpl(uint32_t u) const;

  template <FreqPartIndexType I>
  void materializeImpl(T* dst, uint32_t start, uint32_t count) const;

  // ---------------------------------------------------------------------------
  // Member variables
  // ---------------------------------------------------------------------------

  std::vector<TierInfo> tiers_;

  Vector<T> unencodedValues_;
  uint32_t unencodedStartRow_;

  // NoIndex streaming state
  uint32_t currentTier_;
  uint32_t currentTierOffset_;

  // Index type and indexed-mode streaming state
  FreqPartIndexType indexType_;
  uint32_t totalRowCount_;
  uint32_t currentOriginalPos_;

  // PerTierBitmaps / EliasFano: covered bitmap and fallback prefix table
  Vector<uint64_t> coveredBitmap_; // bit i set ↔ position i in some tier
  Vector<uint32_t>
      fallbackWordPrefix_; // [w] = fallback count in positions [0, w*64)

  // TierTagArray fields
  uint8_t tagBits_;
  Vector<uint8_t> tagArray_;
  // tierRankSamples_[t][si] = count of tag==t in positions [0,
  // si*kRankSampleStride) Index tiers_.size() is used for the fallback bucket.
  std::vector<std::vector<uint32_t>> tierRankSamples_;
};

//
// End of public API. Implementation follows.
//

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <typename T>
FrequencyPartitionEncoding<T>::FrequencyPartitionEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      unencodedValues_{this->pool_},
      unencodedStartRow_(0),
      currentTier_(0),
      currentTierOffset_(0),
      indexType_(FreqPartIndexType::NoIndex),
      totalRowCount_(0),
      currentOriginalPos_(0),
      coveredBitmap_{this->pool_},
      fallbackWordPrefix_{this->pool_},
      tagBits_(0),
      tagArray_{this->pool_} {
  const auto* pos = data.data() + kNumPartitionsOffset;
  const uint32_t numPartitions = encoding::readUint32(pos);
  const EncodingFactory encodingFactory(options);

  const uint32_t partitionOffsetsSize = encoding::readUint32(pos);
  auto partitionOffsetsEncoding = encodingFactory.create(
      *this->pool_,
      std::string_view(pos, partitionOffsetsSize),
      stringBufferFactory);
  pos += partitionOffsetsSize;

  const uint32_t partitionSizesSize = encoding::readUint32(pos);
  auto partitionSizesEncoding = encodingFactory.create(
      *this->pool_,
      std::string_view(pos, partitionSizesSize),
      stringBufferFactory);
  pos += partitionSizesSize;

  Vector<uint32_t> partitionOffsets{this->pool_};
  Vector<uint32_t> partitionSizes{this->pool_};
  partitionOffsets.resize(numPartitions);
  partitionSizes.resize(numPartitions);
  partitionOffsetsEncoding->materialize(numPartitions, partitionOffsets.data());
  partitionSizesEncoding->materialize(numPartitions, partitionSizes.data());

  constexpr uint32_t keyBitOptions[] = {1, 2, 4, 8, 16, 32};

  const uint32_t numCodedTiers = (numPartitions > 0) ? numPartitions - 1 : 0;

  for (uint32_t i = 0; i < numPartitions; ++i) {
    if (i < numCodedTiers) {
      TierInfo tier(this->pool_);
      tier.keyBits = keyBitOptions[i];
      tier.capacity = getCapacity(tier.keyBits);
      tier.startRow = partitionOffsets[i];
      tier.size = partitionSizes[i];
      tier.tierCount = partitionSizes[i];

      if (tier.size > 0) {
        const uint32_t dictSize = encoding::readUint32(pos);
        auto dictEncoding = encodingFactory.create(
            *this->pool_, std::string_view(pos, dictSize), stringBufferFactory);
        pos += dictSize;

        const uint32_t dictCount = dictEncoding->rowCount();
        tier.dictionary.resize(dictCount);
        dictEncoding->materialize(dictCount, tier.dictionary.data());

        const uint32_t keysSize = encoding::readUint32(pos);
        auto keysEncoding = encodingFactory.create(
            *this->pool_, std::string_view(pos, keysSize), stringBufferFactory);
        pos += keysSize;

        tier.indices.resize(tier.size);
        keysEncoding->materialize(tier.size, tier.indices.data());
      }

      tiers_.push_back(std::move(tier));
    } else {
      unencodedStartRow_ = partitionOffsets[i];
      const uint32_t unencodedSize = partitionSizes[i];

      if (unencodedSize > 0) {
        const uint32_t valuesSize = encoding::readUint32(pos);
        auto valuesEncoding = encodingFactory.create(
            *this->pool_,
            std::string_view(pos, valuesSize),
            stringBufferFactory);
        pos += valuesSize;

        unencodedValues_.resize(unencodedSize);
        valuesEncoding->materialize(unencodedSize, unencodedValues_.data());
      }
    }
  }

  totalRowCount_ = this->rowCount();

  // Check for optional index extension block appended after all nested data.
  const char* dataEnd = data.data() + data.size();
  if (pos < dataEnd) {
    const uint8_t fmtVer = encoding::read<uint8_t>(pos);
    const auto idxType =
        static_cast<FreqPartIndexType>(encoding::read<uint8_t>(pos));
    const uint32_t payloadBytes = encoding::readUint32(pos);
    const char* payloadStart = pos;

    if (fmtVer > kFormatVersion) {
      // Unknown format version — skip payload gracefully.
      pos += payloadBytes;
    } else {
      indexType_ = idxType;
      const uint32_t numWords = (totalRowCount_ + 63) / 64;

      switch (indexType_) {
        case FreqPartIndexType::PerTierBitmaps: {
          coveredBitmap_.resize(numWords, 0);
          for (auto& tier : tiers_) {
            if (tier.size == 0) {
              continue;
            }
            const uint32_t bitmapByteCount = encoding::readUint32(pos);
            NIMBLE_CHECK(
                bitmapByteCount == numWords * 8,
                "PerTierBitmaps bitmap size mismatch");
            tier.bitmap.resize(numWords);
            std::memcpy(tier.bitmap.data(), pos, numWords * 8);
            pos += numWords * 8;

            // Build Rank9 superblock: one entry per 8-word (512-bit) block.
            const uint32_t numBlocks = (numWords + 7) / 8;
            tier.rankSuperblock.resize(numBlocks + 1, 0);
            for (uint32_t blk = 0; blk < numBlocks; ++blk) {
              tier.rankSuperblock[blk + 1] = tier.rankSuperblock[blk];
              const uint32_t wEnd = std::min((blk + 1) * 8u, numWords);
              for (uint32_t w = blk * 8; w < wEnd; ++w) {
                tier.rankSuperblock[blk + 1] +=
                    static_cast<uint32_t>(__builtin_popcountll(tier.bitmap[w]));
              }
            }

            // OR into covered bitmap.
            for (uint32_t w = 0; w < numWords; ++w) {
              coveredBitmap_[w] |= tier.bitmap[w];
            }
          }

          // Build fallback prefix-popcount table.
          fallbackWordPrefix_.resize(numWords + 1, 0);
          for (uint32_t w = 0; w < numWords; ++w) {
            uint64_t uncov = ~coveredBitmap_[w];
            if (w == numWords - 1 && (totalRowCount_ % 64) != 0) {
              uncov &= (uint64_t{1} << (totalRowCount_ % 64)) - 1;
            }
            fallbackWordPrefix_[w + 1] = fallbackWordPrefix_[w] +
                static_cast<uint32_t>(__builtin_popcountll(uncov));
          }
          break;
        }

        case FreqPartIndexType::TierTagArray: {
          tagBits_ = encoding::read<uint8_t>(pos);
          pos += 3; // skip alignment padding
          const uint32_t tagArrayByteCount = encoding::readUint32(pos);
          tagArray_.resize(tagArrayByteCount);
          std::memcpy(tagArray_.data(), pos, tagArrayByteCount);
          pos += tagArrayByteCount;

          // Single O(N) pass: build per-tier sampled rank index.
          const uint8_t numActiveTiers = static_cast<uint8_t>(tiers_.size());
          const uint32_t numBuckets = static_cast<uint32_t>(numActiveTiers) + 1;
          const uint32_t numSamples =
              (totalRowCount_ + kRankSampleStride - 1) / kRankSampleStride + 1;
          tierRankSamples_.assign(
              numBuckets, std::vector<uint32_t>(numSamples, 0));

          std::vector<uint32_t> counts(numBuckets, 0);
          for (uint32_t i = 0; i < totalRowCount_; ++i) {
            if (i % kRankSampleStride == 0) {
              const uint32_t si = i / kRankSampleStride;
              for (uint32_t t = 0; t < numBuckets; ++t) {
                tierRankSamples_[t][si] = counts[t];
              }
            }
            const uint8_t tag = unpackTagAt(tagArray_.data(), i, tagBits_);
            const uint32_t bucket =
                (tag < numActiveTiers) ? tag : numActiveTiers;
            ++counts[bucket];
          }
          // Sentinel sample past the last stride.
          const uint32_t lastSi =
              (totalRowCount_ + kRankSampleStride - 1) / kRankSampleStride;
          for (uint32_t t = 0; t < numBuckets; ++t) {
            tierRankSamples_[t][lastSi] = counts[t];
          }

          // Set tierCount from the scan.
          for (uint32_t t = 0; t < numActiveTiers; ++t) {
            tiers_[t].tierCount = counts[t];
          }
          break;
        }

        case FreqPartIndexType::EliasFano: {
          const bool hasFallback = !unencodedValues_.empty();
          if (hasFallback) {
            coveredBitmap_.resize(numWords, 0);
          }

          for (auto& tier : tiers_) {
            if (tier.size == 0) {
              continue;
            }
            const uint8_t lowBits = encoding::read<uint8_t>(pos);
            pos += 3; // padding
            const uint32_t lowByteCount = encoding::readUint32(pos);
            const uint32_t highWordCount = encoding::readUint32(pos);

            const uint8_t* lowBase = reinterpret_cast<const uint8_t*>(pos);
            pos += lowByteCount;
            const uint8_t* highBase = reinterpret_cast<const uint8_t*>(pos);
            pos += static_cast<size_t>(highWordCount) * 8;

            decodeEliasFanoPositions(
                lowBase,
                lowBits,
                highBase,
                highWordCount,
                tier.tierCount,
                tier.efPositions);

            if (hasFallback) {
              for (uint32_t p : tier.efPositions) {
                coveredBitmap_[p / 64] |= uint64_t{1} << (p % 64);
              }
            }
          }

          if (hasFallback) {
            fallbackWordPrefix_.resize(numWords + 1, 0);
            for (uint32_t w = 0; w < numWords; ++w) {
              uint64_t uncov = ~coveredBitmap_[w];
              if (w == numWords - 1 && (totalRowCount_ % 64) != 0) {
                uncov &= (uint64_t{1} << (totalRowCount_ % 64)) - 1;
              }
              fallbackWordPrefix_[w + 1] = fallbackWordPrefix_[w] +
                  static_cast<uint32_t>(__builtin_popcountll(uncov));
            }
          }
          break;
        }

        case FreqPartIndexType::NoIndex:
          // Nothing extra to parse.
          break;
      }

      NIMBLE_CHECK(
          pos == payloadStart + payloadBytes, "index payload size mismatch");
    }
  }
}

// ---------------------------------------------------------------------------
// reset / skip / getTierForRow
// ---------------------------------------------------------------------------

template <typename T>
void FrequencyPartitionEncoding<T>::reset() {
  currentTier_ = 0;
  currentTierOffset_ = 0;
  currentOriginalPos_ = 0;
}

template <typename T>
uint32_t FrequencyPartitionEncoding<T>::getTierForRow(uint32_t rowIndex) const {
  for (uint32_t i = 0; i < tiers_.size(); ++i) {
    if (rowIndex < tiers_[i].startRow) {
      break; // tiers are contiguous and ordered
    }
    if (rowIndex < tiers_[i].startRow + tiers_[i].size) {
      return i;
    }
  }
  return tiers_.size();
}

template <typename T>
void FrequencyPartitionEncoding<T>::skip(uint32_t rowCount) {
  if (indexType_ != FreqPartIndexType::NoIndex) {
    currentOriginalPos_ += rowCount;
    return;
  }

  uint32_t remaining = rowCount;
  while (remaining > 0 && currentTier_ <= tiers_.size()) {
    if (currentTier_ < tiers_.size()) {
      const auto& tier = tiers_[currentTier_];
      const uint32_t availableInTier = tier.size - currentTierOffset_;
      const uint32_t toSkip = std::min(remaining, availableInTier);
      currentTierOffset_ += toSkip;
      remaining -= toSkip;

      if (currentTierOffset_ >= tier.size) {
        ++currentTier_;
        currentTierOffset_ = 0;
      }
    } else {
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// decodeAtOriginalIndexImpl — one definition, branch-free via if constexpr
// ---------------------------------------------------------------------------

template <typename T>
template <FreqPartIndexType I>
T FrequencyPartitionEncoding<T>::decodeAtOriginalIndexImpl(uint32_t u) const {
  if constexpr (I == FreqPartIndexType::PerTierBitmaps) {
    for (const auto& tier : tiers_) {
      if (tier.bitmap.empty()) {
        continue;
      }
      if (tier.bitmap[u / 64] & (uint64_t{1} << (u % 64))) {
        const uint32_t rank = popcountPrefixFast(tier, u);
        return tier.dictionary[tier.indices[rank]];
      }
    }
    return unencodedValues_[fallbackRankAt(u)];

  } else if constexpr (I == FreqPartIndexType::TierTagArray) {
    const uint8_t numActiveTiers = static_cast<uint8_t>(tiers_.size());
    const uint8_t tag = unpackTagAt(tagArray_.data(), u, tagBits_);
    if (tag < numActiveTiers) {
      const uint32_t rank = tierRankAtForTag(tag, u);
      return tiers_[tag].dictionary[tiers_[tag].indices[rank]];
    }
    const uint32_t fallbackRank = tierRankAtForTag(numActiveTiers, u);
    return unencodedValues_[fallbackRank];

  } else if constexpr (I == FreqPartIndexType::EliasFano) {
    for (const auto& tier : tiers_) {
      if (tier.efPositions.empty()) {
        continue;
      }
      auto it =
          std::lower_bound(tier.efPositions.begin(), tier.efPositions.end(), u);
      if (it != tier.efPositions.end() && *it == u) {
        const uint32_t rank =
            static_cast<uint32_t>(it - tier.efPositions.begin());
        return tier.dictionary[tier.indices[rank]];
      }
    }
    return unencodedValues_[fallbackRankAt(u)];

  } else {
    // NoIndex must not reach this path — use the NoIndex materialize branch.
    NIMBLE_UNREACHABLE(
        "decodeAtOriginalIndexImpl called with unsupported index type");
  }
}

// ---------------------------------------------------------------------------
// materializeImpl
// ---------------------------------------------------------------------------

template <typename T>
template <FreqPartIndexType I>
void FrequencyPartitionEncoding<T>::materializeImpl(
    T* dst,
    uint32_t start,
    uint32_t count) const {
  for (uint32_t i = 0; i < count; ++i) {
    dst[i] = decodeAtOriginalIndexImpl<I>(start + i);
  }
}

// ---------------------------------------------------------------------------
// materialize / readWithVisitor
// ---------------------------------------------------------------------------

template <typename T>
void FrequencyPartitionEncoding<T>::materialize(
    uint32_t rowCount,
    void* buffer) {
  T* output = static_cast<T*>(buffer);

  if (indexType_ != FreqPartIndexType::NoIndex) {
    switch (indexType_) {
      case FreqPartIndexType::PerTierBitmaps:
        materializeImpl<FreqPartIndexType::PerTierBitmaps>(
            output, currentOriginalPos_, rowCount);
        break;
      case FreqPartIndexType::TierTagArray:
        materializeImpl<FreqPartIndexType::TierTagArray>(
            output, currentOriginalPos_, rowCount);
        break;
      case FreqPartIndexType::EliasFano:
        materializeImpl<FreqPartIndexType::EliasFano>(
            output, currentOriginalPos_, rowCount);
        break;
      default:
        NIMBLE_UNREACHABLE("unknown FreqPartIndexType");
    }
    currentOriginalPos_ += rowCount;
    return;
  }

  // NoIndex: existing tier-sequential path.
  uint32_t remaining = rowCount;
  uint32_t outputIdx = 0;

  while (remaining > 0) {
    if (currentTier_ < tiers_.size()) {
      const auto& tier = tiers_[currentTier_];
      const uint32_t availableInTier = tier.size - currentTierOffset_;
      const uint32_t toRead = std::min(remaining, availableInTier);

      for (uint32_t i = 0; i < toRead; ++i) {
        const uint32_t index = tier.indices[currentTierOffset_ + i];
        output[outputIdx++] = tier.dictionary[index];
      }

      currentTierOffset_ += toRead;
      remaining -= toRead;

      if (currentTierOffset_ >= tier.size) {
        ++currentTier_;
        currentTierOffset_ = 0;
      }
    } else if (currentTier_ == tiers_.size()) {
      const uint32_t availableUnencoded =
          static_cast<uint32_t>(unencodedValues_.size()) - currentTierOffset_;
      const uint32_t toRead = std::min(remaining, availableUnencoded);

      std::memcpy(
          output + outputIdx,
          unencodedValues_.data() + currentTierOffset_,
          toRead * sizeof(T));

      outputIdx += toRead;
      currentTierOffset_ += toRead;
      remaining -= toRead;

      if (currentTierOffset_ >= unencodedValues_.size()) {
        ++currentTier_;
        currentTierOffset_ = 0;
      }
    } else {
      break;
    }
  }
}

template <typename T>
template <typename V>
void FrequencyPartitionEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  // All paths below use the encoding's own streaming state (currentOriginalPos_
  // for indexed modes; currentTier_/currentTierOffset_ for NoIndex) so that
  // position is correct across multiple readWithVisitor calls and when
  // interleaved with skip()/materialize() calls. visitor.rowIndex() is NOT used
  // because it includes null-row positions from NullableEncoding, which would
  // produce wrong indices into FPE's non-null-only storage.
  //
  // A skip lambda is supplied so readWithVisitorSlow can advance the state past
  // non-selected non-null rows (e.g. during chunk-index seeks or filter-driven
  // row-group skipping).
  if (indexType_ != FreqPartIndexType::NoIndex) {
    auto skipFn = [&](auto n) { skip(n); };
    switch (indexType_) {
      case FreqPartIndexType::PerTierBitmaps:
        detail::readWithVisitorSlow(visitor, params, skipFn, [&] {
          return decodeAtOriginalIndexImpl<FreqPartIndexType::PerTierBitmaps>(
              currentOriginalPos_++);
        });
        break;
      case FreqPartIndexType::TierTagArray:
        detail::readWithVisitorSlow(visitor, params, skipFn, [&] {
          return decodeAtOriginalIndexImpl<FreqPartIndexType::TierTagArray>(
              currentOriginalPos_++);
        });
        break;
      case FreqPartIndexType::EliasFano:
        detail::readWithVisitorSlow(visitor, params, skipFn, [&] {
          return decodeAtOriginalIndexImpl<FreqPartIndexType::EliasFano>(
              currentOriginalPos_++);
        });
        break;
      default:
        NIMBLE_UNREACHABLE("unknown FreqPartIndexType");
    }
    return;
  }

  // NoIndex: values are in tier-reordered space; read one at a time using the
  // currentTier_/currentTierOffset_ streaming state (same as materialize).
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto n) { skip(n); },
      [&] {
        T value;
        if (currentTier_ < tiers_.size()) {
          const auto& tier = tiers_[currentTier_];
          value = tier.dictionary[tier.indices[currentTierOffset_]];
          if (++currentTierOffset_ >= tier.size) {
            ++currentTier_;
            currentTierOffset_ = 0;
          }
        } else {
          value = unencodedValues_[currentTierOffset_++];
        }
        return value;
      });
}

// ---------------------------------------------------------------------------
// encode()
// ---------------------------------------------------------------------------

template <typename T>
std::string_view FrequencyPartitionEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const auto indexType =
      static_cast<FreqPartIndexType>(options.frequencyPartitionIndex);
  const uint32_t valueCount = static_cast<uint32_t>(values.size());

  // Build frequency map
  folly::F14FastMap<physicalType, uint32_t> frequencyMap;
  for (const auto& value : values) {
    frequencyMap[value]++;
  }

  // Sort by frequency (descending)
  std::vector<std::pair<physicalType, uint32_t>> freqVec;
  freqVec.reserve(frequencyMap.size());
  for (const auto& [value, freq] : frequencyMap) {
    freqVec.emplace_back(value, freq);
  }
  std::sort(freqVec.begin(), freqVec.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });

  const uint32_t uniqueCount = static_cast<uint32_t>(freqVec.size());
  constexpr uint32_t maxKeyBits = getMaxKeyBits();

  if constexpr (maxKeyBits == 0) {
    return {};
  }

  // Tier assignments
  struct TierAssignment {
    uint32_t keyBits;
    uint32_t capacity;
    Vector<physicalType> dictionary;
    folly::F14FastMap<physicalType, uint32_t> valueToKey;

    explicit TierAssignment(velox::memory::MemoryPool& pool)
        : keyBits(0), capacity(0), dictionary(&pool) {}
  };

  auto* pool = &buffer.getMemoryPool();
  std::vector<TierAssignment> tierAssignments;
  tierAssignments.reserve(6);

  uint32_t valuesAssigned = 0;
  constexpr uint32_t keyBitOptions[] = {1, 2, 4, 8, 16, 32};

  for (uint32_t keyBits : keyBitOptions) {
    if (keyBits > maxKeyBits || valuesAssigned >= uniqueCount) {
      break;
    }

    TierAssignment tier{*pool};
    tier.keyBits = keyBits;
    tier.capacity = getCapacity(keyBits);

    const uint32_t numToAssign =
        std::min(tier.capacity, uniqueCount - valuesAssigned);
    tier.dictionary.reserve(numToAssign);

    for (uint32_t i = 0; i < numToAssign; ++i) {
      const auto& value = freqVec[valuesAssigned + i].first;
      tier.dictionary.push_back(value);
      tier.valueToKey[value] = i;
    }

    valuesAssigned += numToAssign;
    tierAssignments.push_back(std::move(tier));
  }

  // Build value-to-tier mapping
  folly::F14FastMap<physicalType, uint32_t> valueToTier;
  valueToTier.reserve(valuesAssigned);
  for (size_t tierIdx = 0; tierIdx < tierAssignments.size(); ++tierIdx) {
    for (const auto& [value, key] : tierAssignments[tierIdx].valueToKey) {
      valueToTier[value] = static_cast<uint32_t>(tierIdx);
    }
  }

  // Assign rows to tiers
  std::vector<std::vector<uint32_t>> tierRows(tierAssignments.size() + 1);
  for (auto& vec : tierRows) {
    vec.reserve(valueCount / tierRows.size());
  }

  for (uint32_t row = 0; row < valueCount; ++row) {
    const auto& value = values[row];
    auto tierIt = valueToTier.find(value);
    if (tierIt == valueToTier.end()) {
      tierRows.back().push_back(row);
    } else {
      tierRows[tierIt->second].push_back(row);
    }
  }

  // Partition offsets and sizes
  Vector<uint32_t> partitionOffsets(pool);
  Vector<uint32_t> partitionSizes(pool);
  partitionOffsets.reserve(tierRows.size());
  partitionSizes.reserve(tierRows.size());

  uint32_t offset = 0;
  for (const auto& rows : tierRows) {
    partitionOffsets.push_back(offset);
    partitionSizes.push_back(static_cast<uint32_t>(rows.size()));
    offset += static_cast<uint32_t>(rows.size());
  }

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view serializedOffsets =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::FrequencyPartition::PartitionOffsets,
          {partitionOffsets},
          scopedBuffer.get(),
          options);
  std::string_view serializedSizes = selection.template encodeNested<uint32_t>(
      EncodingIdentifiers::FrequencyPartition::PartitionSizes,
      {partitionSizes},
      scopedBuffer.get(),
      options);

  std::vector<std::string_view> serializedDicts(tierAssignments.size());
  std::vector<std::string_view> serializedKeys(tierAssignments.size());

  for (size_t tierIdx = 0; tierIdx < tierAssignments.size(); ++tierIdx) {
    const auto& tier = tierAssignments[tierIdx];
    const auto& rows = tierRows[tierIdx];

    if (rows.empty()) {
      continue;
    }

    serializedDicts[tierIdx] = selection.template encodeNested<physicalType>(
        EncodingIdentifiers::FrequencyPartition::Dict1Bit + tierIdx,
        {tier.dictionary},
        scopedBuffer.get(),
        options);

    Vector<uint32_t> keys(pool);
    keys.reserve(rows.size());
    for (uint32_t row : rows) {
      keys.push_back(tier.valueToKey.at(values[row]));
    }

    serializedKeys[tierIdx] = selection.template encodeNested<uint32_t>(
        EncodingIdentifiers::FrequencyPartition::Keys1Bit + tierIdx,
        {keys},
        scopedBuffer.get(),
        options);
  }

  std::string_view serializedUnencoded;
  if (!tierRows.back().empty()) {
    Vector<physicalType> unencodedValues(pool);
    unencodedValues.reserve(tierRows.back().size());
    for (uint32_t row : tierRows.back()) {
      unencodedValues.push_back(values[row]);
    }
    serializedUnencoded = selection.template encodeNested<physicalType>(
        EncodingIdentifiers::FrequencyPartition::UnencodedValues,
        {unencodedValues},
        scopedBuffer.get(),
        options);
  }

  // Build index payload (only for non-NoIndex modes)
  std::vector<char> indexPayload;
  const uint32_t numWords = (valueCount + 63) / 64;
  const uint32_t numTiers = static_cast<uint32_t>(tierAssignments.size());

  if (indexType == FreqPartIndexType::PerTierBitmaps) {
    // Build per-tier bitmaps from tierRows (already in ascending row order).
    for (uint32_t t = 0; t < numTiers; ++t) {
      if (tierRows[t].empty()) {
        continue;
      }
      std::vector<uint64_t> bitmap(numWords, 0);
      for (uint32_t row : tierRows[t]) {
        bitmap[row / 64] |= uint64_t{1} << (row % 64);
      }
      const uint32_t bitmapByteCount = numWords * 8;
      const char* byteCount4 = reinterpret_cast<const char*>(&bitmapByteCount);
      indexPayload.insert(indexPayload.end(), byteCount4, byteCount4 + 4);
      const char* bitmapBytes = reinterpret_cast<const char*>(bitmap.data());
      indexPayload.insert(
          indexPayload.end(), bitmapBytes, bitmapBytes + numWords * 8);
    }

  } else if (indexType == FreqPartIndexType::TierTagArray) {
    // Build tag array: tag[pos] = tier index (0..numTiers-1) or numTiers
    // (fallback).
    const uint8_t tagBits = ceilLog2WithMinOne(numTiers + 1);
    const size_t tagArrayByteCount =
        (static_cast<size_t>(valueCount) * tagBits + 7) / 8;

    // Reverse map: row → tag
    std::vector<uint8_t> rowToTag(valueCount, static_cast<uint8_t>(numTiers));
    for (uint32_t t = 0; t < numTiers; ++t) {
      for (uint32_t row : tierRows[t]) {
        rowToTag[row] = static_cast<uint8_t>(t);
      }
    }

    // Pack tags into byte array using accumulator pattern.
    std::vector<uint8_t> tagArray(tagArrayByteCount, 0);
    uint64_t tagAcc = 0;
    size_t tagAccBits = 0;
    size_t tagByteOut = 0;
    for (uint32_t pos = 0; pos < valueCount; ++pos) {
      tagAcc |= static_cast<uint64_t>(rowToTag[pos]) << tagAccBits;
      tagAccBits += tagBits;
      while (tagAccBits >= 8) {
        tagArray[tagByteOut++] = static_cast<uint8_t>(tagAcc & 0xFF);
        tagAcc >>= 8;
        tagAccBits -= 8;
      }
    }
    if (tagAccBits > 0) {
      tagArray[tagByteOut] = static_cast<uint8_t>(tagAcc & 0xFF);
    }

    // Write: tagBits (1) + padding (3) + tagArrayByteCount (4) + tag data
    indexPayload.push_back(static_cast<char>(tagBits));
    indexPayload.resize(indexPayload.size() + 3, '\0'); // 3 bytes padding
    const uint32_t tagByteCount32 = static_cast<uint32_t>(tagArrayByteCount);
    const char* bc = reinterpret_cast<const char*>(&tagByteCount32);
    indexPayload.insert(indexPayload.end(), bc, bc + 4);
    const char* tagBytes = reinterpret_cast<const char*>(tagArray.data());
    indexPayload.insert(
        indexPayload.end(), tagBytes, tagBytes + tagArrayByteCount);

  } else if (indexType == FreqPartIndexType::EliasFano) {
    // Build per-tier Elias-Fano position encodings.
    for (uint32_t t = 0; t < numTiers; ++t) {
      if (tierRows[t].empty()) {
        continue;
      }
      const std::vector<uint32_t>& positions = tierRows[t]; // ascending order
      const uint32_t tierCount = static_cast<uint32_t>(positions.size());

      const uint8_t lowBits = chooseEliasFanoLowBits(valueCount, tierCount);
      const uint32_t lowMask = (lowBits == 0) ? 0u
          : (lowBits == 32)                   ? 0xFFFFFFFFu
                                              : ((1u << lowBits) - 1u);

      // Build low array
      std::vector<uint32_t> lows;
      lows.reserve(tierCount);
      for (uint32_t pos : positions) {
        lows.push_back(pos & lowMask);
      }

      const size_t lowByteCount =
          (static_cast<size_t>(tierCount) * lowBits + 7) / 8;

      // Build high bitmap
      const size_t highBitsLen =
          static_cast<size_t>(valueCount >> (lowBits == 0 ? 0 : lowBits)) +
          tierCount + 1;
      const uint32_t highWordCount =
          static_cast<uint32_t>((highBitsLen + 63) / 64);
      std::vector<uint64_t> highBits(highWordCount, 0);
      for (uint32_t i = 0; i < tierCount; ++i) {
        const size_t hi = static_cast<size_t>(
            lowBits == 0 ? positions[i] : (positions[i] >> lowBits));
        const size_t bitPos = hi + i;
        highBits[bitPos / 64] |= uint64_t{1} << (bitPos % 64);
      }

      // Serialize: lowBits(1) + pad(3) + lowByteCount(4) + highWordCount(4) +
      //            lowArray + highBitmap
      indexPayload.push_back(static_cast<char>(lowBits));
      indexPayload.resize(indexPayload.size() + 3, '\0'); // 3 bytes padding
      const uint32_t lbc32 = static_cast<uint32_t>(lowByteCount);
      const char* lbcp = reinterpret_cast<const char*>(&lbc32);
      indexPayload.insert(indexPayload.end(), lbcp, lbcp + 4);
      const char* hwcp = reinterpret_cast<const char*>(&highWordCount);
      indexPayload.insert(indexPayload.end(), hwcp, hwcp + 4);

      // Pack low bits into payload
      packBitsInto(indexPayload, lows.data(), tierCount, lowBits);
      // Pad to byte boundary (already done by packBitsInto)

      // Append high bitmap
      const char* hbytes = reinterpret_cast<const char*>(highBits.data());
      indexPayload.insert(
          indexPayload.end(), hbytes, hbytes + highWordCount * 8);
    }
  }
  // NoIndex: indexPayload remains empty.

  // Compute total encoding size
  uint32_t encodingSize = Encoding::serializePrefixSize(valueCount, useVarint) +
      4 + // num partitions
      4 + static_cast<uint32_t>(serializedOffsets.size()) + 4 +
      static_cast<uint32_t>(serializedSizes.size());

  for (const auto& dict : serializedDicts) {
    if (!dict.empty()) {
      encodingSize += 4 + static_cast<uint32_t>(dict.size());
    }
  }
  for (const auto& keys : serializedKeys) {
    if (!keys.empty()) {
      encodingSize += 4 + static_cast<uint32_t>(keys.size());
    }
  }
  if (!serializedUnencoded.empty()) {
    encodingSize += 4 + static_cast<uint32_t>(serializedUnencoded.size());
  }

  // Add index extension block if any index is stored.
  const bool hasIndex =
      indexType != FreqPartIndexType::NoIndex && !indexPayload.empty();
  if (hasIndex) {
    encodingSize += 1 + 1 + 4 + static_cast<uint32_t>(indexPayload.size());
  }

  // Write encoded data
  char* reserved = buffer.reserve(encodingSize);
  char* wpos = reserved;

  Encoding::serializePrefix(
      EncodingType::FrequencyPartition,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      wpos);
  encoding::writeUint32(static_cast<uint32_t>(tierRows.size()), wpos);
  encoding::writeUint32(static_cast<uint32_t>(serializedOffsets.size()), wpos);
  encoding::writeBytes(serializedOffsets, wpos);
  encoding::writeUint32(static_cast<uint32_t>(serializedSizes.size()), wpos);
  encoding::writeBytes(serializedSizes, wpos);

  for (size_t i = 0; i < serializedDicts.size(); ++i) {
    if (!serializedDicts[i].empty()) {
      encoding::writeUint32(
          static_cast<uint32_t>(serializedDicts[i].size()), wpos);
      encoding::writeBytes(serializedDicts[i], wpos);
      encoding::writeUint32(
          static_cast<uint32_t>(serializedKeys[i].size()), wpos);
      encoding::writeBytes(serializedKeys[i], wpos);
    }
  }
  if (!serializedUnencoded.empty()) {
    encoding::writeUint32(
        static_cast<uint32_t>(serializedUnencoded.size()), wpos);
    encoding::writeBytes(serializedUnencoded, wpos);
  }

  if (hasIndex) {
    encoding::write<uint8_t>(kFormatVersion, wpos);
    encoding::write<uint8_t>(static_cast<uint8_t>(indexType), wpos);
    encoding::writeUint32(static_cast<uint32_t>(indexPayload.size()), wpos);
    std::memcpy(wpos, indexPayload.data(), indexPayload.size());
    wpos += indexPayload.size();
  }

  NIMBLE_DCHECK_EQ(wpos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

// ---------------------------------------------------------------------------
// debugString
// ---------------------------------------------------------------------------

template <typename T>
std::string FrequencyPartitionEncoding<T>::debugString(int offset) const {
  const char* idxName = indexType_ == FreqPartIndexType::NoIndex ? "NoIndex"
      : indexType_ == FreqPartIndexType::PerTierBitmaps ? "PerTierBitmaps"
      : indexType_ == FreqPartIndexType::TierTagArray   ? "TierTagArray"
      : indexType_ == FreqPartIndexType::EliasFano      ? "EliasFano"
                                                        : "Unknown";
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}tiers={}, unencoded_rows={}, index={}",
      std::string(offset, ' '),
      tiers_.size(),
      unencodedValues_.size(),
      idxName);

  for (size_t i = 0; i < tiers_.size(); ++i) {
    const auto& tier = tiers_[i];
    log += fmt::format(
        "\n{}tier[{}]: {}bit codes, {} unique values, {} rows",
        std::string(offset + 2, ' '),
        i,
        tier.keyBits,
        tier.dictionary.size(),
        tier.size);
  }

  return log;
}

} // namespace facebook::nimble
