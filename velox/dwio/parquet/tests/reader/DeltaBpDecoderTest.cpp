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

#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

using facebook::velox::parquet::DeltaBpDecoder;

namespace {

// Minimal, spec-compliant DELTA_BINARY_PACKED page builder used to drive the
// decoder with hand-chosen per-miniblock bit widths. Unlike the Arrow writer,
// it does not collapse constant miniblocks to bit width 0, so a test can force
// a specific wide bit width (e.g. 63) and exercise the skip path against it.
class DeltaPageBuilder {
 public:
  DeltaPageBuilder(uint32_t valuesPerMiniBlock, uint32_t miniBlocksPerBlock)
      : valuesPerMiniBlock_(valuesPerMiniBlock),
        miniBlocksPerBlock_(miniBlocksPerBlock),
        valuesPerBlock_(valuesPerMiniBlock * miniBlocksPerBlock) {}

  // Builds a single-block page. 'deltas' holds the valuesPerBlock deltas that
  // follow the first value. 'bitWidths' holds one bit width per miniblock.
  // 'firstValue' is the value stored in the page header. Returns the encoded
  // page bytes alone and fills 'expected' with the reconstructed values
  // (firstValue + prefix sums). Callers supply the decoder's trailing
  // padding, since how much of it is readable is what some tests vary.
  std::vector<uint8_t> build(
      int64_t firstValue,
      int64_t minDelta,
      const std::vector<int64_t>& deltas,
      const std::vector<uint32_t>& bitWidths,
      std::vector<int64_t>& expected) {
    EXPECT_EQ(deltas.size(), valuesPerBlock_);
    EXPECT_EQ(bitWidths.size(), miniBlocksPerBlock_);

    // Reconstruct the values with unsigned modular arithmetic, matching the
    // decoder. The total value count includes the header value.
    expected.clear();
    expected.push_back(firstValue);
    for (uint32_t i = 0; i < valuesPerBlock_; ++i) {
      expected.push_back(
          static_cast<int64_t>(
              static_cast<uint64_t>(expected.back()) +
              static_cast<uint64_t>(deltas[i])));
    }

    std::vector<uint8_t> buf;
    putUleb(buf, valuesPerBlock_);
    putUleb(buf, miniBlocksPerBlock_);
    putUleb(buf, static_cast<uint64_t>(expected.size()));
    putZigZag(buf, firstValue);

    // Single block.
    putZigZag(buf, minDelta);
    for (uint32_t w : bitWidths) {
      buf.push_back(static_cast<uint8_t>(w));
    }
    for (uint32_t mb = 0; mb < miniBlocksPerBlock_; ++mb) {
      const uint32_t bitWidth = bitWidths[mb];
      std::vector<uint64_t> stored(valuesPerMiniBlock_);
      for (uint32_t j = 0; j < valuesPerMiniBlock_; ++j) {
        const int64_t delta = deltas[mb * valuesPerMiniBlock_ + j];
        stored[j] = static_cast<uint64_t>(delta - minDelta);
        if (bitWidth < 64) {
          EXPECT_LT(stored[j], 1ULL << bitWidth)
              << "stored delta does not fit in bit width " << bitWidth;
        }
      }
      appendPackedMiniBlock(buf, stored, bitWidth);
    }

    return buf;
  }

 private:
  static void putUleb(std::vector<uint8_t>& buf, uint64_t value) {
    do {
      uint8_t byte = value & 0x7f;
      value >>= 7;
      if (value != 0) {
        byte |= 0x80;
      }
      buf.push_back(byte);
    } while (value != 0);
  }

  static void putZigZag(std::vector<uint8_t>& buf, int64_t value) {
    const uint64_t zigzag = (static_cast<uint64_t>(value) << 1) ^
        static_cast<uint64_t>(value >> 63);
    putUleb(buf, zigzag);
  }

  // Packs 'values' little-endian, LSB-first and contiguous, matching the
  // layout read by bits::detail::loadBits in the decoder.
  static void appendPackedMiniBlock(
      std::vector<uint8_t>& buf,
      const std::vector<uint64_t>& values,
      uint32_t bitWidth) {
    if (bitWidth == 0) {
      return;
    }
    const size_t numBits = values.size() * bitWidth;
    std::vector<uint8_t> packed((numBits + 7) / 8, 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const uint64_t value = values[i];
      const size_t bitPos = i * bitWidth;
      for (uint32_t b = 0; b < bitWidth; ++b) {
        if ((value >> b) & 1ULL) {
          packed[(bitPos + b) >> 3] |=
              static_cast<uint8_t>(1u << ((bitPos + b) & 7));
        }
      }
    }
    buf.insert(buf.end(), packed.begin(), packed.end());
  }

  const uint32_t valuesPerMiniBlock_;
  const uint32_t miniBlocksPerBlock_;
  const uint32_t valuesPerBlock_;
};

// True when the packed delta at position 'index' with the given bit width
// straddles a 64-bit load boundary: its top (bitInByte + bitWidth - 64) bits
// live in the next 64-bit word. This is exactly the case the buggy single-load
// path mishandled.
bool crossesLoadBoundary(uint32_t index, uint32_t bitWidth) {
  return ((index * bitWidth) & 7) + bitWidth > 64;
}

// Builds a page whose first miniblock is forced to 'bitWidth', places a single
// wide delta at the first boundary-crossing position, and leaves the remaining
// miniblocks constant. Skips past the wide first miniblock (routing through
// sumMiniBlockDeltas) and verifies the values decoded afterwards. If the sum
// drops the crossing value's high bits, lastValue_ is left off by a non-zero
// amount below 2^64, so the tail diverges from the expected sequence.
//
// A single crossing value (rather than many) is deliberate: identical wide
// values sum to a multiple of 2^64 for several bit widths, which would cancel
// the dropped bits and hide the bug. One value drops a non-zero amount that
// cannot cancel.
void testSkipOverWideMiniBlock(uint32_t bitWidth) {
  SCOPED_TRACE("bitWidth=" + std::to_string(bitWidth));
  constexpr uint32_t kValuesPerMiniBlock = 32;
  constexpr uint32_t kMiniBlocksPerBlock = 4;
  constexpr uint32_t kValuesPerBlock =
      kValuesPerMiniBlock * kMiniBlocksPerBlock;

  // A value with all 'bitWidth' bits set needs exactly 'bitWidth' bits and, at
  // a crossing position, loses its high bits under the old single-load read.
  const uint64_t wideDelta =
      (bitWidth == 64) ? ~0ULL : ((1ULL << bitWidth) - 1);
  std::vector<int64_t> deltas(kValuesPerBlock, 0);
  int crossingIndex = -1;
  for (uint32_t j = 0; j < kValuesPerMiniBlock; ++j) {
    if (crossesLoadBoundary(j, bitWidth)) {
      crossingIndex = static_cast<int>(j);
      break;
    }
  }
  if (crossingIndex >= 0) {
    deltas[crossingIndex] = static_cast<int64_t>(wideDelta);
  } else {
    // Bit widths that never cross act as controls; still force the width with a
    // value whose high bit is set.
    deltas[0] = static_cast<int64_t>(
        (bitWidth == 64) ? ~0ULL : ((1ULL << (bitWidth - 1)) | 1ULL));
  }
  std::vector<uint32_t> bitWidths{bitWidth, 0, 0, 0};

  DeltaPageBuilder builder(kValuesPerMiniBlock, kMiniBlocksPerBlock);
  std::vector<int64_t> expected;
  auto page = builder.build(
      /*firstValue=*/0,
      /*minDelta=*/0,
      deltas,
      bitWidths,
      expected);
  page.insert(page.end(), DeltaBpDecoder::kRequiredTrailingPadding, 0);
  const auto* start = reinterpret_cast<const char*>(page.data());

  // Full decode is the trusted reference path and must round-trip the input.
  {
    DeltaBpDecoder decoder(start);
    std::vector<int64_t> out(expected.size());
    decoder.readValues(out.data(), static_cast<int32_t>(out.size()));
    EXPECT_EQ(out, expected);
  }

  // Skip the header value plus the whole wide miniblock, then decode the rest.
  const int32_t skipCount = 1 + static_cast<int32_t>(kValuesPerMiniBlock);
  DeltaBpDecoder decoder(start);
  decoder.skip(skipCount);
  const int32_t remaining = static_cast<int32_t>(expected.size()) - skipCount;
  ASSERT_GT(remaining, 0);
  std::vector<int64_t> out(remaining);
  decoder.readValues(out.data(), remaining);
  for (int32_t i = 0; i < remaining; ++i) {
    EXPECT_EQ(out[i], expected[skipCount + i]) << "at index " << i;
  }
}

// Regression test for the DELTA_BINARY_PACKED skip fast path.
// sumMiniBlockDeltas previously read each packed delta with a single 64-bit
// load, dropping the high bits of any INT64 delta that straddled the load
// boundary. Bit widths 59, 61, 62, and 63 exercise the crossing; 57, 58, 60,
// and 64 do not and act as controls.
TEST(DeltaBpDecoderTest, skipWideMiniBlock) {
  for (uint32_t bitWidth = 57; bitWidth <= 64; ++bitWidth) {
    testSkipOverWideMiniBlock(bitWidth);
  }
}

// Owns a mapping whose last readable byte is immediately followed by an
// inaccessible guard page. Placing a page here bounds the decoder's over-read
// by hardware: reading further than the mapping faults instead of quietly
// succeeding, on every build and with no sanitizer required.
class GuardedBuffer {
 public:
  // Maps 'size' readable bytes ending flush against the guard page.
  explicit GuardedBuffer(size_t size) : size_(size) {
    const size_t pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    mappedSize_ = ((size + pageSize - 1) / pageSize) * pageSize + pageSize;
    mapping_ = ::mmap(
        nullptr,
        mappedSize_,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0);
    VELOX_CHECK_NE(mapping_, MAP_FAILED, "Failed to map guarded buffer");
    auto* guard = static_cast<uint8_t*>(mapping_) + mappedSize_ - pageSize;
    VELOX_CHECK_EQ(::mprotect(guard, pageSize, PROT_NONE), 0);
    data_ = guard - size;
  }

  ~GuardedBuffer() {
    ::munmap(mapping_, mappedSize_);
  }

  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;

  uint8_t* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  const size_t size_;
  size_t mappedSize_{0};
  void* mapping_{nullptr};
  uint8_t* data_{nullptr};
};

// Decodes a whole page whose readable trailing padding is exactly
// DeltaBpDecoder::kRequiredTrailingPadding bytes, with a guard page beyond it.
// All four miniblocks carry 'bitWidth', so the last one ends at the page end
// and its final SIMD load reaches the furthest past the encoded data.
void testDecodeAgainstPaddingLimit(uint32_t bitWidth) {
  SCOPED_TRACE("bitWidth=" + std::to_string(bitWidth));
  constexpr uint32_t kValuesPerMiniBlock = 32;
  constexpr uint32_t kMiniBlocksPerBlock = 4;
  constexpr uint32_t kValuesPerBlock =
      kValuesPerMiniBlock * kMiniBlocksPerBlock;

  // Set the high bit so every delta genuinely needs all 'bitWidth' bits, and
  // vary the low bits so a dropped or misplaced value is visible.
  const uint64_t highBit = 1ULL << (bitWidth - 1);
  std::vector<int64_t> deltas(kValuesPerBlock);
  for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
    deltas[i] = static_cast<int64_t>(highBit | (i % highBit));
  }
  const std::vector<uint32_t> bitWidths(kMiniBlocksPerBlock, bitWidth);

  DeltaPageBuilder builder(kValuesPerMiniBlock, kMiniBlocksPerBlock);
  std::vector<int64_t> expected;
  const auto page = builder.build(
      /*firstValue=*/0,
      /*minDelta=*/0,
      deltas,
      bitWidths,
      expected);

  GuardedBuffer buffer(page.size() + DeltaBpDecoder::kRequiredTrailingPadding);
  std::memcpy(buffer.data(), page.data(), page.size());
  std::memset(buffer.data() + page.size(), 0, buffer.size() - page.size());

  const auto* start = reinterpret_cast<const char*>(buffer.data());
  DeltaBpDecoder decoder(start);
  std::vector<int64_t> actual(expected.size());
  decoder.readValues(actual.data(), static_cast<int32_t>(actual.size()));

  EXPECT_EQ(actual, expected);
  EXPECT_EQ(decoder.validValuesCount(), 0);
  EXPECT_EQ(decoder.bufferStart(), start + page.size());
}

// Pins the trailing-padding contract the SIMD decode path depends on. The
// bitWidth > 16 path builds its 128-bit window from loads at 'byteOff' and
// 'byteOff + 8', so it touches up to 16 - ceil(bitWidth / 4) bytes past the
// miniblock -- 11 at bit widths 17 through 20, the widest over-read the
// decoder performs. Running every SIMD-eligible width against a guard page
// placed exactly kRequiredTrailingPadding bytes out fails if that bound ever
// grows, or if the constant is lowered to match it.
//
// Note that this covers the padding half of the contract only. The path also
// requires unaligned loads, and reading an unaligned address through a
// 'uint64_t*' is undefined behavior that x86-64 and ARM64 both execute
// correctly, so only -fsanitize=alignment observes it. That check does run in
// the ASAN/UBSAN build: the -fno-sanitize=alignment added for issue #15811
// sits on CMAKE_EXE_LINKER_FLAGS, and UBSan instruments at compile time, so
// the flag does not suppress it. It reports and recovers rather than
// aborting, which is why a misaligned load shows up as a diagnostic without
// failing the job. This test drives unaligned payload offsets so a regression
// surfaces there.
TEST(DeltaBpDecoderTest, decodeAgainstPaddingLimit) {
  for (uint32_t bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    testDecodeAgainstPaddingLimit(bitWidth);
  }
}

} // namespace
