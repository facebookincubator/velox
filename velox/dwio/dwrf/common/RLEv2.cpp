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

#include "velox/dwio/dwrf/common/RLEv2.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/dwrf/common/Common.h"

namespace facebook::velox::dwrf {

using memory::MemoryPool;

struct FixedBitSizes {
  enum FBS {
    ONE = 0,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    ELEVEN,
    TWELVE,
    THIRTEEN,
    FOURTEEN,
    FIFTEEN,
    SIXTEEN,
    SEVENTEEN,
    EIGHTEEN,
    NINETEEN,
    TWENTY,
    TWENTYONE,
    TWENTYTWO,
    TWENTYTHREE,
    TWENTYFOUR,
    TWENTYSIX,
    TWENTYEIGHT,
    THIRTY,
    THIRTYTWO,
    FORTY,
    FORTYEIGHT,
    FIFTYSIX,
    SIXTYFOUR,
    SIZE
  };
};

// Map FBS enum to bit width value.
const uint8_t FBSToBitWidthMap[FixedBitSizes::SIZE] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32, 40, 48, 56, 64};

// Map bit length i to closest fixed bit width that can contain i bits.
const uint8_t ClosestFixedBitsMap[65] = {
    1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 26, 26, 28, 28, 30, 30, 32, 32, 40,
    40, 40, 40, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 56, 56,
    56, 56, 56, 56, 56, 56, 64, 64, 64, 64, 64, 64, 64, 64};

// Map bit length i to closest aligned fixed bit width that can contain i bits.
const uint8_t ClosestAlignedFixedBitsMap[65] = {
    1,  1,  2,  4,  4,  8,  8,  8,  8,  16, 16, 16, 16, 16, 16, 16, 16,
    24, 24, 24, 24, 24, 24, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 40,
    40, 40, 40, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 56, 56,
    56, 56, 56, 56, 56, 56, 64, 64, 64, 64, 64, 64, 64, 64};

// Map bit width to FBS enum.
const uint8_t BitWidthToFBSMap[65] = {
    FixedBitSizes::ONE,         FixedBitSizes::ONE,
    FixedBitSizes::TWO,         FixedBitSizes::THREE,
    FixedBitSizes::FOUR,        FixedBitSizes::FIVE,
    FixedBitSizes::SIX,         FixedBitSizes::SEVEN,
    FixedBitSizes::EIGHT,       FixedBitSizes::NINE,
    FixedBitSizes::TEN,         FixedBitSizes::ELEVEN,
    FixedBitSizes::TWELVE,      FixedBitSizes::THIRTEEN,
    FixedBitSizes::FOURTEEN,    FixedBitSizes::FIFTEEN,
    FixedBitSizes::SIXTEEN,     FixedBitSizes::SEVENTEEN,
    FixedBitSizes::EIGHTEEN,    FixedBitSizes::NINETEEN,
    FixedBitSizes::TWENTY,      FixedBitSizes::TWENTYONE,
    FixedBitSizes::TWENTYTWO,   FixedBitSizes::TWENTYTHREE,
    FixedBitSizes::TWENTYFOUR,  FixedBitSizes::TWENTYSIX,
    FixedBitSizes::TWENTYSIX,   FixedBitSizes::TWENTYEIGHT,
    FixedBitSizes::TWENTYEIGHT, FixedBitSizes::THIRTY,
    FixedBitSizes::THIRTY,      FixedBitSizes::THIRTYTWO,
    FixedBitSizes::THIRTYTWO,   FixedBitSizes::FORTY,
    FixedBitSizes::FORTY,       FixedBitSizes::FORTY,
    FixedBitSizes::FORTY,       FixedBitSizes::FORTY,
    FixedBitSizes::FORTY,       FixedBitSizes::FORTY,
    FixedBitSizes::FORTY,       FixedBitSizes::FORTYEIGHT,
    FixedBitSizes::FORTYEIGHT,  FixedBitSizes::FORTYEIGHT,
    FixedBitSizes::FORTYEIGHT,  FixedBitSizes::FORTYEIGHT,
    FixedBitSizes::FORTYEIGHT,  FixedBitSizes::FORTYEIGHT,
    FixedBitSizes::FORTYEIGHT,  FixedBitSizes::FIFTYSIX,
    FixedBitSizes::FIFTYSIX,    FixedBitSizes::FIFTYSIX,
    FixedBitSizes::FIFTYSIX,    FixedBitSizes::FIFTYSIX,
    FixedBitSizes::FIFTYSIX,    FixedBitSizes::FIFTYSIX,
    FixedBitSizes::FIFTYSIX,    FixedBitSizes::SIXTYFOUR,
    FixedBitSizes::SIXTYFOUR,   FixedBitSizes::SIXTYFOUR,
    FixedBitSizes::SIXTYFOUR,   FixedBitSizes::SIXTYFOUR,
    FixedBitSizes::SIXTYFOUR,   FixedBitSizes::SIXTYFOUR,
    FixedBitSizes::SIXTYFOUR};

// The input n must be less than FixedBitSizes::SIZE.
inline uint32_t decodeBitWidth(uint32_t n) {
  return FBSToBitWidthMap[n];
}

inline uint32_t getClosestFixedBits(uint32_t n) {
  if (n <= 64) {
    return ClosestFixedBitsMap[n];
  } else {
    return 64;
  }
}

inline uint32_t getClosestAlignedFixedBits(uint32_t n) {
  if (n <= 64) {
    return ClosestAlignedFixedBitsMap[n];
  } else {
    return 64;
  }
}

inline uint32_t encodeBitWidth(uint32_t n) {
  if (n <= 64) {
    return BitWidthToFBSMap[n];
  } else {
    return FixedBitSizes::SIXTYFOUR;
  }
}

inline uint32_t findClosestNumBits(int64_t value) {
  if (value < 0) {
    return getClosestFixedBits(64);
  }

  uint32_t count = 0;
  while (value != 0) {
    count++;
    value = value >> 1;
  }
  return getClosestFixedBits(count);
}

inline bool isSafeSubtract(int64_t left, int64_t right) {
  return ((left ^ right) >= 0) || ((left ^ (left - right)) >= 0);
}

template <bool isSigned>
inline uint32_t RleEncoderV2<isSigned>::getOpCode(EncodingType encoding) {
  return static_cast<uint32_t>(encoding << 6);
}

template uint32_t RleEncoderV2<true>::getOpCode(EncodingType encoding);
template uint32_t RleEncoderV2<false>::getOpCode(EncodingType encoding);

/**
 * Prepare for Direct or PatchedBase encoding
 * compute zigZagLiterals and zzBits100p (Max number of encoding bits required)
 * @return zigzagLiterals
 */
template <bool isSigned>
int64_t* RleEncoderV2<isSigned>::prepareForDirectOrPatchedBase(
    EncodingOption& option) {
  if (isSigned) {
    computeZigZagLiterals(option);
  }
  int64_t* currentZigzagLiterals = isSigned ? zigzagLiterals : literals;
  option.zzBits100p =
      percentileBits(currentZigzagLiterals, 0, numLiterals, 1.0);
  return currentZigzagLiterals;
}

template int64_t* RleEncoderV2<true>::prepareForDirectOrPatchedBase(
    EncodingOption& option);
template int64_t* RleEncoderV2<false>::prepareForDirectOrPatchedBase(
    EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::determineEncoding(EncodingOption& option) {
  // We need to compute zigzag values for DIRECT and PATCHED_BASE encodings,
  // but not for SHORT_REPEAT or DELTA. So we only perform the zigzag
  // computation when it's determined to be necessary.

  // not a big win for shorter runs to determine encoding
  if (numLiterals <= MIN_REPEAT) {
    // we need to compute zigzag values for DIRECT encoding if we decide to
    // break early for delta overflows or for shorter runs
    prepareForDirectOrPatchedBase(option);
    option.encoding = DIRECT;
    return;
  }

  // DELTA encoding check

  // for identifying monotonic sequences
  bool isIncreasing = true;
  bool isDecreasing = true;
  option.isFixedDelta = true;

  option.min = literals[0];
  int64_t max = literals[0];
  int64_t initialDelta = literals[1] - literals[0];
  int64_t currDelta = 0;
  int64_t deltaMax = 0;
  adjDeltas[option.adjDeltasCount++] = initialDelta;

  for (size_t i = 1; i < numLiterals; i++) {
    const int64_t l1 = literals[i];
    const int64_t l0 = literals[i - 1];
    currDelta = l1 - l0;
    option.min = std::min(option.min, l1);
    max = std::max(max, l1);

    isIncreasing &= (l0 <= l1);
    isDecreasing &= (l0 >= l1);

    option.isFixedDelta &= (currDelta == initialDelta);
    if (i > 1) {
      adjDeltas[option.adjDeltasCount++] = std::abs(currDelta);
      deltaMax = std::max(deltaMax, adjDeltas[i - 1]);
    }
  }

  // it's faster to exit under delta overflow condition without checking for
  // PATCHED_BASE condition as encoding using DIRECT is faster and has less
  // overhead than PATCHED_BASE
  if (!isSafeSubtract(max, option.min)) {
    prepareForDirectOrPatchedBase(option);
    option.encoding = DIRECT;
    return;
  }

  // invariant - subtracting any number from any other in the literals after
  // option point won't overflow

  // if min is equal to max then the delta is 0, option condition happens for
  // fixed values run >10 which cannot be encoded with SHORT_REPEAT
  if (option.min == max) {
    if (!option.isFixedDelta) {
      throw std::invalid_argument(
          std::to_string(option.min) + "==" + std::to_string(max) +
          ", isFixedDelta cannot be false");
    }

    if (currDelta != 0) {
      throw std::invalid_argument(
          std::to_string(option.min) + "==" + std::to_string(max) +
          ", currDelta should be zero");
    }
    option.fixedDelta = 0;
    option.encoding = DELTA;
    return;
  }

  if (option.isFixedDelta) {
    if (currDelta != initialDelta) {
      throw std::invalid_argument(
          "currDelta should be equal to initialDelta for fixed delta encoding");
    }

    option.encoding = DELTA;
    option.fixedDelta = currDelta;
    return;
  }

  // if initialDelta is 0 then we cannot delta encode as we cannot identify
  // the sign of deltas (increasing or decreasing)
  if (initialDelta != 0) {
    // stores the number of bits required for packing delta blob in
    // delta encoding
    option.bitsDeltaMax = findClosestNumBits(deltaMax);

    // monotonic condition
    if (isIncreasing || isDecreasing) {
      option.encoding = DELTA;
      return;
    }
  }

  // PATCHED_BASE encoding check

  // percentile values are computed for the zigzag encoded values. if the
  // number of bit requirement between 90th and 100th percentile varies
  // beyond a threshold then we need to patch the values. if the variation
  // is not significant then we can use direct encoding

  int64_t* currentZigzagLiterals = prepareForDirectOrPatchedBase(option);
  option.zzBits90p =
      percentileBits(currentZigzagLiterals, 0, numLiterals, 0.9, true);
  uint32_t diffBitsLH = option.zzBits100p - option.zzBits90p;

  // if the difference between 90th percentile and 100th percentile fixed
  // bits is > 1 then we need patch the values
  if (diffBitsLH > 1) {
    // patching is done only on base reduced values.
    // remove base from literals
    for (size_t i = 0; i < numLiterals; i++) {
      baseRedLiterals[option.baseRedLiteralsCount++] =
          (literals[i] - option.min);
    }

    // 95th percentile width is used to determine max allowed value
    // after which patching will be done
    option.brBits95p = percentileBits(baseRedLiterals, 0, numLiterals, 0.95);

    // 100th percentile is used to compute the max patch width
    option.brBits100p =
        percentileBits(baseRedLiterals, 0, numLiterals, 1.0, true);

    // after base reducing the values, if the difference in bits between
    // 95th percentile and 100th percentile value is zero then there
    // is no point in patching the values, in which case we will
    // fallback to DIRECT encoding.
    // The decision to use patched base was based on zigzag values, but the
    // actual patching is done on base reduced literals.
    if ((option.brBits100p - option.brBits95p) != 0) {
      option.encoding = PATCHED_BASE;
      preparePatchedBlob(option);
      return;
    } else {
      option.encoding = DIRECT;
      return;
    }
  } else {
    // if difference in bits between 95th percentile and 100th percentile is
    // 0, then patch length will become 0. Hence we will fallback to direct
    option.encoding = DIRECT;
    return;
  }
}

template void RleEncoderV2<true>::determineEncoding(EncodingOption& option);
template void RleEncoderV2<false>::determineEncoding(EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::computeZigZagLiterals(EncodingOption& option) {
  assert(isSigned);
  for (size_t i = 0; i < numLiterals; i++) {
    zigzagLiterals[option.zigzagLiteralsCount++] = ZigZag::encode(literals[i]);
  }
}

template void RleEncoderV2<true>::computeZigZagLiterals(EncodingOption& option);
template void RleEncoderV2<false>::computeZigZagLiterals(
    EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::preparePatchedBlob(EncodingOption& option) {
  // mask will be max value beyond which patch will be generated
  int64_t mask =
      static_cast<int64_t>(static_cast<uint64_t>(1) << option.brBits95p) - 1;

  // since we are considering only 95 percentile, the size of gap and
  // patch array can contain only be 5% values
  option.patchLength = static_cast<uint32_t>(std::ceil((numLiterals / 20)));

  // #bit for patch
  option.patchWidth = option.brBits100p - option.brBits95p;
  option.patchWidth = getClosestFixedBits(option.patchWidth);

  // if patch bit requirement is 64 then it will not possible to pack
  // gap and patch together in a long. To make sure gap and patch can be
  // packed together adjust the patch width
  if (option.patchWidth == 64) {
    option.patchWidth = 56;
    option.brBits95p = 8;
    mask =
        static_cast<int64_t>(static_cast<uint64_t>(1) << option.brBits95p) - 1;
  }

  uint32_t gapIdx = 0;
  uint32_t patchIdx = 0;
  size_t prev = 0;
  size_t maxGap = 0;

  std::vector<int64_t> gapList;
  std::vector<int64_t> patchList;

  for (size_t i = 0; i < numLiterals; i++) {
    // if value is above mask then create the patch and record the gap
    if (baseRedLiterals[i] > mask) {
      size_t gap = i - prev;
      if (gap > maxGap) {
        maxGap = gap;
      }

      // gaps are relative, so store the previous patched value index
      prev = i;
      gapList.push_back(static_cast<int64_t>(gap));
      gapIdx++;

      // extract the most significant bits that are over mask bits
      int64_t patch = baseRedLiterals[i] >> option.brBits95p;
      patchList.push_back(patch);
      patchIdx++;

      // strip off the MSB to enable safe bit packing
      baseRedLiterals[i] &= mask;
    }
  }

  // adjust the patch length to number of entries in gap list
  option.patchLength = gapIdx;

  // if the element to be patched is the first and only element then
  // max gap will be 0, but to store the gap as 0 we need atleast 1 bit
  if (maxGap == 0 && option.patchLength != 0) {
    option.patchGapWidth = 1;
  } else {
    option.patchGapWidth = findClosestNumBits(static_cast<int64_t>(maxGap));
  }

  // special case: if the patch gap width is greater than 256, then
  // we need 9 bits to encode the gap width. But we only have 3 bits in
  // header to record the gap width. To deal with this case, we will save
  // two entries in patch list in the following way
  // 256 gap width => 0 for patch value
  // actual gap - 256 => actual patch value
  // We will do the same for gap width = 511. If the element to be patched is
  // the last element in the scope then gap width will be 511. In this case we
  // will have 3 entries in the patch list in the following way
  // 255 gap width => 0 for patch value
  // 255 gap width => 0 for patch value
  // 1 gap width => actual patch value
  if (option.patchGapWidth > 8) {
    option.patchGapWidth = 8;
    // for gap = 511, we need two additional entries in patch list
    if (maxGap == 511) {
      option.patchLength += 2;
    } else {
      option.patchLength += 1;
    }
  }

  // create gap vs patch list
  gapIdx = 0;
  patchIdx = 0;
  for (size_t i = 0; i < option.patchLength; i++) {
    int64_t g = gapList[gapIdx++];
    int64_t p = patchList[patchIdx++];
    while (g > 255) {
      gapVsPatchList[option.gapVsPatchListCount++] =
          (255L << option.patchWidth);
      i++;
      g -= 255;
    }

    // store patch value in LSBs and gap in MSBs
    gapVsPatchList[option.gapVsPatchListCount++] =
        ((g << option.patchWidth) | p);
  }
}

template void RleEncoderV2<true>::preparePatchedBlob(EncodingOption& option);
template void RleEncoderV2<false>::preparePatchedBlob(EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeInts(
    int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize) {
  if (input == nullptr || len < 1 || bitSize < 1) {
    return;
  }

  if (getClosestAlignedFixedBits(bitSize) == bitSize) {
    uint32_t numBytes;
    uint32_t endOffSet = static_cast<uint32_t>(offset + len);
    if (bitSize < 8) {
      char bitMask = static_cast<char>((1 << bitSize) - 1);
      uint32_t numHops = 8 / bitSize;
      uint32_t remainder = static_cast<uint32_t>(len % numHops);
      uint32_t endUnroll = endOffSet - remainder;
      for (uint32_t i = offset; i < endUnroll; i += numHops) {
        char toWrite = 0;
        for (uint32_t j = 0; j < numHops; ++j) {
          toWrite |= static_cast<char>(
              (input[i + j] & bitMask) << (8 - (j + 1) * bitSize));
        }
        IntEncoder<isSigned>::writeByte(toWrite);
      }

      if (remainder > 0) {
        uint32_t startShift = 8 - bitSize;
        char toWrite = 0;
        for (uint32_t i = endUnroll; i < endOffSet; ++i) {
          toWrite |= static_cast<char>((input[i] & bitMask) << startShift);
          startShift -= bitSize;
        }
        IntEncoder<isSigned>::writeByte(toWrite);
      }

    } else {
      numBytes = bitSize / 8;

      for (uint32_t i = offset; i < endOffSet; ++i) {
        for (uint32_t j = 0; j < numBytes; ++j) {
          char toWrite =
              static_cast<char>((input[i] >> (8 * (numBytes - j - 1))) & 255);
          IntEncoder<isSigned>::writeByte(toWrite);
        }
      }
    }

    return;
  }

  // write for unaligned bit size
  uint32_t bitsLeft = 8;
  char current = 0;
  for (uint32_t i = offset; i < (offset + len); i++) {
    int64_t value = input[i];
    uint32_t bitsToWrite = bitSize;
    while (bitsToWrite > bitsLeft) {
      // add the bits to the bottom of the current word
      current |= static_cast<char>(value >> (bitsToWrite - bitsLeft));
      // subtract out the bits we just added
      bitsToWrite -= bitsLeft;
      // zero out the bits above bitsToWrite
      value &= (static_cast<uint64_t>(1) << bitsToWrite) - 1;
      IntEncoder<isSigned>::writeByte(current);
      current = 0;
      bitsLeft = 8;
    }
    bitsLeft -= bitsToWrite;
    current |= static_cast<char>(value << bitsLeft);
    if (bitsLeft == 0) {
      IntEncoder<isSigned>::writeByte(current);
      current = 0;
      bitsLeft = 8;
    }
  }

  // flush
  if (bitsLeft != 8) {
    IntEncoder<isSigned>::writeByte(current);
  }
}

template void RleEncoderV2<true>::writeInts(
    int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize);
template void RleEncoderV2<false>::writeInts(
    int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize);

template <bool isSigned>
void RleEncoderV2<isSigned>::initializeLiterals(int64_t val) {
  literals[numLiterals++] = val;
  fixedRunLength = 1;
  variableRunLength = 1;
}

template void RleEncoderV2<true>::initializeLiterals(int64_t val);
template void RleEncoderV2<false>::initializeLiterals(int64_t val);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeValues(EncodingOption& option) {
  if (numLiterals != 0) {
    switch (option.encoding) {
      case SHORT_REPEAT:
        writeShortRepeatValues(option);
        break;
      case DIRECT:
        writeDirectValues(option);
        break;
      case PATCHED_BASE:
        writePatchedBasedValues(option);
        break;
      case DELTA:
        writeDeltaValues(option);
        break;
      default:
        throw std::runtime_error("Not implemented yet");
    }

    numLiterals = 0;
    prevDelta = 0;
  }
}

template void RleEncoderV2<true>::writeValues(EncodingOption& option);
template void RleEncoderV2<false>::writeValues(EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeShortRepeatValues(EncodingOption&) {
  int64_t repeatVal;
  if (isSigned) {
    repeatVal = ZigZag::encode(literals[0]);
  } else {
    repeatVal = literals[0];
  }

  const uint32_t numBitsRepeatVal = findClosestNumBits(repeatVal);
  const uint32_t numBytesRepeatVal = numBitsRepeatVal % 8 == 0
      ? (numBitsRepeatVal >> 3)
      : ((numBitsRepeatVal >> 3) + 1);

  uint32_t header = getOpCode(SHORT_REPEAT);

  fixedRunLength -= MIN_REPEAT;
  header |= fixedRunLength;
  header |= ((numBytesRepeatVal - 1) << 3);

  IntEncoder<isSigned>::writeByte(static_cast<char>(header));

  for (int32_t i = static_cast<int32_t>(numBytesRepeatVal - 1); i >= 0; i--) {
    int64_t b = ((repeatVal >> (i * 8)) & 0xff);
    IntEncoder<isSigned>::writeByte(static_cast<char>(b));
  }

  fixedRunLength = 0;
}

template void RleEncoderV2<true>::writeShortRepeatValues(EncodingOption&);
template void RleEncoderV2<false>::writeShortRepeatValues(EncodingOption&);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeDirectValues(EncodingOption& option) {
  // write the number of fixed bits required in next 5 bits
  uint32_t fb = option.zzBits100p;
  if (alignedBitPacking) {
    fb = getClosestAlignedFixedBits(fb);
  }

  const uint32_t efb = encodeBitWidth(fb) << 1;

  // adjust variable run length
  variableRunLength -= 1;

  // extract the 9th bit of run length
  const uint32_t tailBits = (variableRunLength & 0x100) >> 8;

  // create first byte of the header
  const char headerFirstByte =
      static_cast<char>(getOpCode(DIRECT) | efb | tailBits);

  // second byte of the header stores the remaining 8 bits of runlength
  const char headerSecondByte = static_cast<char>(variableRunLength & 0xff);

  // write header
  IntEncoder<isSigned>::writeByte(headerFirstByte);
  IntEncoder<isSigned>::writeByte(headerSecondByte);

  // bit packing the zigzag encoded literals
  int64_t* currentZigzagLiterals = isSigned ? zigzagLiterals : literals;
  writeInts(currentZigzagLiterals, 0, numLiterals, fb);

  // reset run length
  variableRunLength = 0;
}

template void RleEncoderV2<true>::writeDirectValues(EncodingOption& option);
template void RleEncoderV2<false>::writeDirectValues(EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::writePatchedBasedValues(EncodingOption& option) {
  // NOTE: Aligned bit packing cannot be applied for PATCHED_BASE encoding
  // because patch is applied to MSB bits. For example: If fixed bit width of
  // base value is 7 bits and if patch is 3 bits, the actual value is
  // constructed by shifting the patch to left by 7 positions.
  // actual_value = patch << 7 | base_value
  // So, if we align base_value then actual_value can not be reconstructed.

  // write the number of fixed bits required in next 5 bits
  const uint32_t efb = encodeBitWidth(option.brBits95p) << 1;

  // adjust variable run length, they are one off
  variableRunLength -= 1;

  // extract the 9th bit of run length
  const uint32_t tailBits = (variableRunLength & 0x100) >> 8;

  // create first byte of the header
  const char headerFirstByte =
      static_cast<char>(getOpCode(PATCHED_BASE) | efb | tailBits);

  // second byte of the header stores the remaining 8 bits of runlength
  const char headerSecondByte = static_cast<char>(variableRunLength & 0xff);

  // if the min value is negative toggle the sign
  const bool isNegative = (option.min < 0);
  if (isNegative) {
    option.min = -option.min;
  }

  // find the number of bytes required for base and shift it by 5 bits
  // to accommodate patch width. The additional bit is used to store the sign
  // of the base value.
  const uint32_t baseWidth = findClosestNumBits(option.min) + 1;
  const uint32_t baseBytes =
      baseWidth % 8 == 0 ? baseWidth / 8 : (baseWidth / 8) + 1;
  const uint32_t bb = (baseBytes - 1) << 5;

  // if the base value is negative then set MSB to 1
  if (isNegative) {
    option.min |= (1LL << ((baseBytes * 8) - 1));
  }

  // third byte contains 3 bits for number of bytes occupied by base
  // and 5 bits for patchWidth
  const char headerThirdByte =
      static_cast<char>(bb | encodeBitWidth(option.patchWidth));

  // fourth byte contains 3 bits for page gap width and 5 bits for
  // patch length
  const char headerFourthByte =
      static_cast<char>((option.patchGapWidth - 1) << 5 | option.patchLength);

  // write header
  IntEncoder<isSigned>::writeByte(headerFirstByte);
  IntEncoder<isSigned>::writeByte(headerSecondByte);
  IntEncoder<isSigned>::writeByte(headerThirdByte);
  IntEncoder<isSigned>::writeByte(headerFourthByte);

  // write the base value using fixed bytes in big endian order
  for (int32_t i = static_cast<int32_t>(baseBytes - 1); i >= 0; i--) {
    char b = static_cast<char>(((option.min >> (i * 8)) & 0xff));
    IntEncoder<isSigned>::writeByte(b);
  }

  // base reduced literals are bit packed
  uint32_t closestFixedBits = getClosestFixedBits(option.brBits95p);

  writeInts(baseRedLiterals, 0, numLiterals, closestFixedBits);

  // write patch list
  closestFixedBits =
      getClosestFixedBits(option.patchGapWidth + option.patchWidth);

  writeInts(gapVsPatchList, 0, option.patchLength, closestFixedBits);

  // reset run length
  variableRunLength = 0;
}

template void RleEncoderV2<true>::writePatchedBasedValues(
    EncodingOption& option);
template void RleEncoderV2<false>::writePatchedBasedValues(
    EncodingOption& option);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeDeltaValues(EncodingOption& option) {
  uint32_t len = 0;
  uint32_t fb = option.bitsDeltaMax;
  uint32_t efb = 0;

  if (alignedBitPacking) {
    fb = getClosestAlignedFixedBits(fb);
  }

  if (option.isFixedDelta) {
    // if fixed run length is greater than threshold then it will be fixed
    // delta sequence with delta value 0 else fixed delta sequence with
    // non-zero delta value
    if (fixedRunLength > MIN_REPEAT) {
      // ex. sequence: 2 2 2 2 2 2 2 2
      len = fixedRunLength - 1;
      fixedRunLength = 0;
    } else {
      // ex. sequence: 4 6 8 10 12 14 16
      len = variableRunLength - 1;
      variableRunLength = 0;
    }
  } else {
    // fixed width 0 is used for long repeating values.
    // sequences that require only 1 bit to encode will have an additional bit
    if (fb == 1) {
      fb = 2;
    }
    efb = encodeBitWidth(fb) << 1;
    len = variableRunLength - 1;
    variableRunLength = 0;
  }

  // extract the 9th bit of run length
  const uint32_t tailBits = (len & 0x100) >> 8;

  // create first byte of the header
  const char headerFirstByte =
      static_cast<char>(getOpCode(DELTA) | efb | tailBits);

  // second byte of the header stores the remaining 8 bits of runlength
  const char headerSecondByte = static_cast<char>(len & 0xff);

  // write header
  IntEncoder<isSigned>::writeByte(headerFirstByte);
  IntEncoder<isSigned>::writeByte(headerSecondByte);

  // store the first value from zigzag literal array
  if (isSigned) {
    IntEncoder<isSigned>::writeVslong(literals[0]);
  } else {
    IntEncoder<isSigned>::writeVulong(literals[0]);
  }

  if (option.isFixedDelta) {
    // if delta is fixed then we don't need to store delta blob
    IntEncoder<isSigned>::writeVslong(option.fixedDelta);
  } else {
    // store the first value as delta value using zigzag encoding
    IntEncoder<isSigned>::writeVslong(adjDeltas[0]);

    // adjacent delta values are bit packed. The length of adjDeltas array is
    // always one less than the number of literals (delta difference for n
    // elements is n-1). We have already written one element, write the
    // remaining numLiterals - 2 elements here
    writeInts(adjDeltas, 1, numLiterals - 2, fb);
  }
}

template void RleEncoderV2<true>::writeDeltaValues(EncodingOption& option);
template void RleEncoderV2<false>::writeDeltaValues(EncodingOption& option);

/**
 * Compute the bits required to represent pth percentile value
 * @param data - array
 * @param p - percentile value (>=0.0 to <=1.0)
 * @return pth percentile bits
 */
template <bool isSigned>
uint32_t RleEncoderV2<isSigned>::percentileBits(
    int64_t* data,
    size_t offset,
    size_t length,
    double p,
    bool reuseHist) {
  if ((p > 1.0) || (p <= 0.0)) {
    throw std::invalid_argument("Invalid p value: " + std::to_string(p));
  }

  if (!reuseHist) {
    // histogram that store the encoded bit requirement for each values.
    // maximum number of bits that can encoded is 32 (refer FixedBitSizes)
    memset(histgram, 0, FixedBitSizes::SIZE * sizeof(int32_t));
    // compute the histogram
    for (size_t i = offset; i < (offset + length); i++) {
      uint32_t idx = encodeBitWidth(findClosestNumBits(data[i]));
      histgram[idx] += 1;
    }
  }

  int32_t perLen =
      static_cast<int32_t>(static_cast<double>(length) * (1.0 - p));

  // return the bits required by pth percentile length
  for (int32_t i = HIST_LEN - 1; i >= 0; i--) {
    perLen -= histgram[i];
    if (perLen < 0) {
      return decodeBitWidth(static_cast<uint32_t>(i));
    }
  }
  return 0;
}

template uint32_t RleEncoderV2<true>::percentileBits(
    int64_t* data,
    size_t offset,
    size_t length,
    double p,
    bool reuseHist);
template uint32_t RleEncoderV2<false>::percentileBits(
    int64_t* data,
    size_t offset,
    size_t length,
    double p,
    bool reuseHist);

template <bool isSigned>
int64_t RleDecoderV2<isSigned>::readLongBE(uint64_t bsz) {
  int64_t ret = 0, val;
  uint64_t n = bsz;
  while (n > 0) {
    n--;
    val = readByte();
    ret |= (val << (n * 8));
  }
  return ret;
}

template <bool isSigned>
RleDecoderV2<isSigned>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool)
    : dwio::common::IntDecoder<isSigned>{std::move(input), false, 0},
      firstByte(0),
      runLength(0),
      runRead(0),
      deltaBase(0),
      byteSize(0),
      firstValue(0),
      prevValue(0),
      bitSize(0),
      bitsLeft(0),
      curByte(0),
      patchBitSize(0),
      unpackedIdx(0),
      patchIdx(0),
      base(0),
      curGap(0),
      curPatch(0),
      patchMask(0),
      actualGap(0),
      unpacked(pool, 0),
      unpackedPatch(pool, 0) {
  // PASS
}

template RleDecoderV2<true>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool);
template RleDecoderV2<false>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool);

template <bool isSigned>
void RleDecoderV2<isSigned>::seekToRowGroup(
    dwio::common::PositionProvider& location) {
  // move the input stream
  dwio::common::IntDecoder<isSigned>::inputStream->seekToPosition(location);
  // clear state
  dwio::common::IntDecoder<isSigned>::bufferEnd =
      dwio::common::IntDecoder<isSigned>::bufferStart = 0;
  runRead = runLength = 0;
  // skip ahead the given number of records
  skip(location.next());
}

template void RleDecoderV2<true>::seekToRowGroup(
    dwio::common::PositionProvider& location);
template void RleDecoderV2<false>::seekToRowGroup(
    dwio::common::PositionProvider& location);

template <bool isSigned>
void RleDecoderV2<isSigned>::skip(uint64_t numValues) {
  // simple for now, until perf tests indicate something encoding specific is
  // needed
  const uint64_t N = 64;
  int64_t dummy[N];

  while (numValues) {
    uint64_t nRead = std::min(N, numValues);
    next(dummy, nRead, nullptr);
    numValues -= nRead;
  }
}

template void RleDecoderV2<true>::skip(uint64_t numValues);
template void RleDecoderV2<false>::skip(uint64_t numValues);

template <bool isSigned>
void RleDecoderV2<isSigned>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls) {
  uint64_t nRead = 0;

  while (nRead < numValues) {
    // Skip any nulls before attempting to read first byte.
    while (nulls && bits::isBitNull(nulls, nRead)) {
      if (++nRead == numValues) {
        return; // ended with null values
      }
    }

    if (runRead == runLength) {
      resetRun();
      firstByte = readByte();
    }

    uint64_t offset = nRead, length = numValues - nRead;

    EncodingType enc = static_cast<EncodingType>((firstByte >> 6) & 0x03);
    switch (static_cast<int64_t>(enc)) {
      case SHORT_REPEAT:
        nRead += nextShortRepeats(data, offset, length, nulls);
        break;
      case DIRECT:
        nRead += nextDirect(data, offset, length, nulls);
        break;
      case PATCHED_BASE:
        nRead += nextPatched(data, offset, length, nulls);
        break;
      case DELTA:
        nRead += nextDelta(data, offset, length, nulls);
        break;
      default:
        DWIO_RAISE("unknown encoding");
    }
  }
}

template void RleDecoderV2<true>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls);
template void RleDecoderV2<false>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls);

template <bool isSigned>
uint64_t RleDecoderV2<isSigned>::nextShortRepeats(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls) {
  if (runRead == runLength) {
    // extract the number of fixed bytes
    byteSize = (firstByte >> 3) & 0x07;
    byteSize += 1;

    runLength = firstByte & 0x07;
    // run lengths values are stored only after MIN_REPEAT value is met
    runLength += RLE_MINIMUM_REPEAT;
    runRead = 0;

    // read the repeated value which is store using fixed bytes
    firstValue = readLongBE(byteSize);

    if (isSigned) {
      firstValue = ZigZag::decode(static_cast<uint64_t>(firstValue));
    }
  }

  uint64_t nRead = std::min(runLength - runRead, numValues);

  if (nulls) {
    for (uint64_t pos = offset; pos < offset + nRead; ++pos) {
      if (!bits::isBitNull(nulls, pos)) {
        data[pos] = firstValue;
        ++runRead;
      }
    }
  } else {
    for (uint64_t pos = offset; pos < offset + nRead; ++pos) {
      data[pos] = firstValue;
      ++runRead;
    }
  }

  return nRead;
}

template uint64_t RleDecoderV2<true>::nextShortRepeats(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);
template uint64_t RleDecoderV2<false>::nextShortRepeats(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

template <bool isSigned>
uint64_t RleDecoderV2<isSigned>::nextDirect(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls) {
  if (runRead == runLength) {
    // extract the number of fixed bits
    unsigned char fbo = (firstByte >> 1) & 0x1f;
    bitSize = decodeBitWidth(fbo);

    // extract the run length
    runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
    runLength |= readByte();
    // runs are one off
    runLength += 1;
    runRead = 0;
  }

  uint64_t nRead = std::min(runLength - runRead, numValues);

  runRead += readLongs(data, offset, nRead, bitSize, nulls);

  if (isSigned) {
    if (nulls) {
      for (uint64_t pos = offset; pos < offset + nRead; ++pos) {
        if (!bits::isBitNull(nulls, pos)) {
          data[pos] = ZigZag::decode(static_cast<uint64_t>(data[pos]));
        }
      }
    } else {
      for (uint64_t pos = offset; pos < offset + nRead; ++pos) {
        data[pos] = ZigZag::decode(static_cast<uint64_t>(data[pos]));
      }
    }
  }

  return nRead;
}

template uint64_t RleDecoderV2<true>::nextDirect(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);
template uint64_t RleDecoderV2<false>::nextDirect(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

template <bool isSigned>
uint64_t RleDecoderV2<isSigned>::nextPatched(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls) {
  if (runRead == runLength) {
    // extract the number of fixed bits
    unsigned char fbo = (firstByte >> 1) & 0x1f;
    bitSize = decodeBitWidth(fbo);

    // extract the run length
    runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
    runLength |= readByte();
    // runs are one off
    runLength += 1;
    runRead = 0;

    // extract the number of bytes occupied by base
    uint64_t thirdByte = readByte();
    byteSize = (thirdByte >> 5) & 0x07;
    // base width is one off
    byteSize += 1;

    // extract patch width
    uint32_t pwo = thirdByte & 0x1f;
    patchBitSize = decodeBitWidth(pwo);

    // read fourth byte and extract patch gap width
    uint64_t fourthByte = readByte();
    uint32_t pgw = (fourthByte >> 5) & 0x07;
    // patch gap width is one off
    pgw += 1;

    // extract the length of the patch list
    size_t pl = fourthByte & 0x1f;
    DWIO_ENSURE_NE(
        pl,
        0,
        "Corrupt PATCHED_BASE encoded data (pl==0)! ",
        dwio::common::IntDecoder<isSigned>::inputStream->getName());

    // read the next base width number of bytes to extract base value
    base = readLongBE(byteSize);
    int64_t mask = (static_cast<int64_t>(1) << ((byteSize * 8) - 1));
    // if mask of base value is 1 then base is negative value else positive
    if ((base & mask) != 0) {
      base = base & ~mask;
      base = -base;
    }

    // TODO: something more efficient than resize
    unpacked.resize(runLength);
    unpackedIdx = 0;
    readLongs(unpacked.data(), 0, runLength, bitSize);
    // any remaining bits are thrown out
    resetReadLongs();

    // TODO: something more efficient than resize
    unpackedPatch.resize(pl);
    patchIdx = 0;
    // TODO: Skip corrupt?
    //    if ((patchBitSize + pgw) > 64 && !skipCorrupt) {
    DWIO_ENSURE_LE(
        (patchBitSize + pgw),
        64,
        "Corrupt PATCHED_BASE encoded data (patchBitSize + pgw > 64)! ",
        dwio::common::IntDecoder<isSigned>::inputStream->getName());
    uint32_t cfb = getClosestFixedBits(patchBitSize + pgw);
    readLongs(unpackedPatch.data(), 0, pl, cfb);
    // any remaining bits are thrown out
    resetReadLongs();

    // apply the patch directly when decoding the packed data
    patchMask = ((static_cast<int64_t>(1) << patchBitSize) - 1);

    adjustGapAndPatch();
  }

  uint64_t nRead = std::min(runLength - runRead, numValues);

  for (uint64_t pos = offset; pos < offset + nRead; ++pos) {
    // skip null positions
    if (nulls && bits::isBitNull(nulls, pos)) {
      continue;
    }
    if (static_cast<int64_t>(unpackedIdx) != actualGap) {
      // no patching required. add base to unpacked value to get final value
      data[pos] = base + unpacked[unpackedIdx];
    } else {
      // extract the patch value
      int64_t patchedVal = unpacked[unpackedIdx] | (curPatch << bitSize);

      // add base to patched value
      data[pos] = base + patchedVal;

      // increment the patch to point to next entry in patch list
      ++patchIdx;

      if (patchIdx < unpackedPatch.size()) {
        adjustGapAndPatch();

        // next gap is relative to the current gap
        actualGap += unpackedIdx;
      }
    }

    ++runRead;
    ++unpackedIdx;
  }

  return nRead;
}

template uint64_t RleDecoderV2<true>::nextPatched(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

template uint64_t RleDecoderV2<false>::nextPatched(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

template <bool isSigned>
uint64_t RleDecoderV2<isSigned>::nextDelta(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls) {
  if (runRead == runLength) {
    // extract the number of fixed bits
    unsigned char fbo = (firstByte >> 1) & 0x1f;
    if (fbo != 0) {
      bitSize = decodeBitWidth(fbo);
    } else {
      bitSize = 0;
    }

    // extract the run length
    runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
    runLength |= readByte();
    ++runLength; // account for first value
    runRead = deltaBase = 0;

    // read the first value stored as vint
    if constexpr (isSigned) {
      firstValue = dwio::common::IntDecoder<isSigned>::readVsLong();
    } else {
      firstValue = static_cast<int64_t>(
          dwio::common::IntDecoder<isSigned>::readVuLong());
    }

    prevValue = firstValue;

    // read the fixed delta value stored as vint (deltas can be negative even
    // if all number are positive)
    deltaBase = dwio::common::IntDecoder<isSigned>::readVsLong();
  }

  uint64_t nRead = std::min(runLength - runRead, numValues);

  uint64_t pos = offset;
  for (; pos < offset + nRead; ++pos) {
    // skip null positions
    if (!nulls || !bits::isBitNull(nulls, pos)) {
      break;
    }
  }
  if (runRead == 0 && pos < offset + nRead) {
    data[pos++] = firstValue;
    ++runRead;
  }

  if (bitSize == 0) {
    // add fixed deltas to adjacent values
    for (; pos < offset + nRead; ++pos) {
      // skip null positions
      if (nulls && bits::isBitNull(nulls, pos)) {
        continue;
      }
      prevValue = data[pos] = prevValue + deltaBase;
      ++runRead;
    }
  } else {
    for (; pos < offset + nRead; ++pos) {
      // skip null positions
      if (!nulls || !bits::isBitNull(nulls, pos)) {
        break;
      }
    }
    if (runRead < 2 && pos < offset + nRead) {
      // add delta base and first value
      prevValue = data[pos++] = firstValue + deltaBase;
      ++runRead;
    }

    // write the unpacked values, add it to previous value and store final
    // value to result buffer. if the delta base value is negative then it
    // is a decreasing sequence else an increasing sequence
    uint64_t remaining = (offset + nRead) - pos;
    runRead += readLongs(data, pos, remaining, bitSize, nulls);

    if (deltaBase < 0) {
      for (; pos < offset + nRead; ++pos) {
        // skip null positions
        if (nulls && bits::isBitNull(nulls, pos)) {
          continue;
        }
        prevValue = data[pos] = prevValue - data[pos];
      }
    } else {
      for (; pos < offset + nRead; ++pos) {
        // skip null positions
        if (nulls && bits::isBitNull(nulls, pos)) {
          continue;
        }
        prevValue = data[pos] = prevValue + data[pos];
      }
    }
  }
  return nRead;
}

template uint64_t RleDecoderV2<true>::nextDelta(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

template uint64_t RleDecoderV2<false>::nextDelta(
    int64_t* const data,
    uint64_t offset,
    uint64_t numValues,
    const uint64_t* const nulls);

} // namespace facebook::velox::dwrf
