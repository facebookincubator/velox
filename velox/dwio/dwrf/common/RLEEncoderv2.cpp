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
#include "velox/dwio/dwrf/common/RLEv2Util.h"

namespace facebook::velox::dwrf {

template <bool isSigned>
RleEncoderV2<isSigned>::RleEncoderV2(
    std::unique_ptr<BufferedOutputStream> outStream,
    bool alignBitPacking)
    : IntEncoder<isSigned>{std::move(outStream), false, 0},
      alignedBitPacking_(alignBitPacking),
      prevDelta_(0),
      numLiterals_(0) {}

template RleEncoderV2<true>::RleEncoderV2(
    std::unique_ptr<BufferedOutputStream> outStream,
    bool alignBitPacking);
template RleEncoderV2<false>::RleEncoderV2(
    std::unique_ptr<BufferedOutputStream> outStream,
    bool alignBitPacking);

template <bool isSigned>
inline uint32_t RleEncoderV2<isSigned>::getOpCode(EncodingType encoding) {
  return static_cast<uint32_t>(encoding << 6);
}

template uint32_t RleEncoderV2<true>::getOpCode(EncodingType encoding);
template uint32_t RleEncoderV2<false>::getOpCode(EncodingType encoding);

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
  VELOX_CHECK((p > 0.0) && (p <= 1.0), fmt::format("Invalid p value: {}", p));

  if (!reuseHist) {
    // histogram that store the encoded bit requirement for each values.
    // maximum number of bits that can encoded is 32 (refer FixedBitSizes)
    memset(histgram_, 0, FixedBitSizes::SIZE * sizeof(int32_t));
    // compute the histogram
    for (size_t i = offset; i < (offset + length); i++) {
      uint32_t idx = encodeBitWidth(findClosestNumBits(data[i]));
      histgram_[idx] += 1;
    }
  }

  auto perLen = static_cast<int32_t>(static_cast<double>(length) * (1.0 - p));

  // return the bits required by pth percentile length
  for (int32_t i = HIST_LEN - 1; i >= 0; i--) {
    perLen -= histgram_[i];
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
void RleEncoderV2<isSigned>::writeValue(const int64_t val) {
  if (numLiterals_ == 0) {
    initializeLiterals(val);
    return;
  }

  if (numLiterals_ == 1) {
    prevDelta_ = val - literals_[0];
    literals_[numLiterals_++] = val;

    if (val == literals_[0]) {
      fixedRunLength_ = 2;
      variableRunLength_ = 0;
    } else {
      fixedRunLength_ = 0;
      variableRunLength_ = 2;
    }
    return;
  }

  auto currentDelta = val - literals_[numLiterals_ - 1];
  EncodingOption option = {};
  if (prevDelta_ == 0 && currentDelta == 0) {
    // case 1: fixed delta run
    literals_[numLiterals_++] = val;

    if (variableRunLength_ > 0) {
      // if variable run is non-zero then we are seeing repeating
      // values at the end of variable run in which case fixed Run
      // length is 2
      fixedRunLength_ = 2;
    }
    fixedRunLength_++;

    // if fixed run met the minimum condition and if variable
    // run is non-zero then flush the variable run and shift the
    // tail fixed runs to start of the buffer
    if (fixedRunLength_ >= MIN_REPEAT && variableRunLength_ > 0) {
      numLiterals_ -= MIN_REPEAT;
      variableRunLength_ -= (MIN_REPEAT - 1);

      determineEncoding(option);
      writeValues(option);

      // shift tail fixed runs to beginning of the buffer
      for (size_t i = 0; i < MIN_REPEAT; ++i) {
        literals_[i] = val;
      }
      numLiterals_ = MIN_REPEAT;
    }

    if (fixedRunLength_ == MAX_LITERAL_SIZE) {
      option.encoding = DELTA;
      option.isFixedDelta = true;
      writeValues(option);
    }
    return;
  }

  // case 2: variable delta run

  // if fixed run length is non-zero and if it satisfies the
  // short repeat conditions then write the values as short repeats
  // else use delta encoding
  if (fixedRunLength_ >= MIN_REPEAT) {
    if (fixedRunLength_ <= MAX_SHORT_REPEAT_LENGTH) {
      option.encoding = SHORT_REPEAT;
    } else {
      option.encoding = DELTA;
      option.isFixedDelta = true;
    }
    writeValues(option);
  }

  // if fixed run length is <MIN_REPEAT and current value is
  // different from previous then treat it as variable run
  if (fixedRunLength_ > 0 && fixedRunLength_ < MIN_REPEAT &&
      val != literals_[numLiterals_ - 1]) {
    variableRunLength_ = fixedRunLength_;
    fixedRunLength_ = 0;
  }

  // after writing values re-initialize the variables
  if (numLiterals_ == 0) {
    initializeLiterals(val);
  } else {
    prevDelta_ = val - literals_[numLiterals_ - 1];
    literals_[numLiterals_++] = val;
    variableRunLength_++;

    if (variableRunLength_ == MAX_LITERAL_SIZE) {
      determineEncoding(option);
      writeValues(option);
    }
  }
}

template void RleEncoderV2<true>::writeValue(const int64_t val);
template void RleEncoderV2<false>::writeValue(const int64_t val);

template <bool isSigned>
void RleEncoderV2<isSigned>::computeZigZagLiterals(EncodingOption& option) {
  assert(isSigned);
  for (size_t i = 0; i < numLiterals_; i++) {
    zigzagLiterals_[option.zigzagLiteralsCount++] =
        ZigZag::encode(literals_[i]);
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
  option.patchLength = static_cast<uint32_t>(std::ceil((numLiterals_ / 20)));

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

  for (size_t i = 0; i < numLiterals_; i++) {
    // if value is above mask then create the patch and record the gap
    if (baseRedLiterals_[i] > mask) {
      size_t gap = i - prev;
      if (gap > maxGap) {
        maxGap = gap;
      }

      // gaps are relative, so store the previous patched value index
      prev = i;
      gapList.push_back(static_cast<int64_t>(gap));
      gapIdx++;

      // extract the most significant bits that are over mask bits
      int64_t patch = baseRedLiterals_[i] >> option.brBits95p;
      patchList.push_back(patch);
      patchIdx++;

      // strip off the MSB to enable safe bit packing
      baseRedLiterals_[i] &= mask;
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
      gapVsPatchList_[option.gapVsPatchListCount++] =
          (255L << option.patchWidth);
      i++;
      g -= 255;
    }

    // store patch value in LSBs and gap in MSBs
    gapVsPatchList_[option.gapVsPatchListCount++] =
        ((g << option.patchWidth) | p);
  }
}

template void RleEncoderV2<true>::preparePatchedBlob(EncodingOption& option);
template void RleEncoderV2<false>::preparePatchedBlob(EncodingOption& option);

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
  int64_t* currentZigzagLiterals =
      isSigned ? zigzagLiterals_.data() : literals_.data();
  option.zzBits100p =
      percentileBits(currentZigzagLiterals, 0, numLiterals_, 1.0);
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
  if (numLiterals_ <= MIN_REPEAT) {
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

  option.min = literals_[0];
  int64_t max = literals_[0];
  int64_t initialDelta = literals_[1] - literals_[0];
  int64_t currDelta = 0;
  int64_t deltaMax = 0;
  adjDeltas_[option.adjDeltasCount++] = initialDelta;

  for (size_t i = 1; i < numLiterals_; i++) {
    const int64_t l1 = literals_[i];
    const int64_t l0 = literals_[i - 1];
    currDelta = l1 - l0;
    option.min = std::min(option.min, l1);
    max = std::max(max, l1);

    isIncreasing &= (l0 <= l1);
    isDecreasing &= (l0 >= l1);

    option.isFixedDelta &= (currDelta == initialDelta);
    if (i > 1) {
      adjDeltas_[option.adjDeltasCount++] = std::abs(currDelta);
      deltaMax = std::max(deltaMax, adjDeltas_[i - 1]);
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

  // invariant - subtracting any number from any other in the literals_ after
  // option point won't overflow

  // if min is equal to max then the delta is 0, option condition happens for
  // fixed values run >10 which cannot be encoded with SHORT_REPEAT
  if (option.min == max) {
    VELOX_CHECK(
        option.isFixedDelta,
        fmt::format("{}=={}, isFixedDelta cannot be false", option.min, max));

    VELOX_CHECK(
        currDelta == 0,
        fmt::format("{}=={}, currDelta should be zero", option.min, max));

    option.fixedDelta = 0;
    option.encoding = DELTA;
    return;
  }

  if (option.isFixedDelta) {
    VELOX_CHECK(
        currDelta == initialDelta,
        "currDelta should be equal to initialDelta for fixed delta encoding");

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
      percentileBits(currentZigzagLiterals, 0, numLiterals_, 0.9, true);
  uint32_t diffBitsLH = option.zzBits100p - option.zzBits90p;

  // if the difference between 90th percentile and 100th percentile fixed
  // bits is > 1 then we need patch the values
  if (diffBitsLH > 1) {
    // patching is done only on base reduced values.
    // remove base from literals_
    for (size_t i = 0; i < numLiterals_; i++) {
      baseRedLiterals_[option.baseRedLiteralsCount++] =
          (literals_[i] - option.min);
    }

    // 95th percentile width is used to determine max allowed value
    // after which patching will be done
    option.brBits95p =
        percentileBits(baseRedLiterals_.data(), 0, numLiterals_, 0.95);

    // 100th percentile is used to compute the max patch width
    option.brBits100p =
        percentileBits(baseRedLiterals_.data(), 0, numLiterals_, 1.0, true);

    // after base reducing the values, if the difference in bits between
    // 95th percentile and 100th percentile value is zero then there
    // is no point in patching the values, in which case we will
    // fallback to DIRECT encoding.
    // The decision to use patched base was based on zigzag values, but the
    // actual patching is done on base reduced literals_.
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
template <typename T>
uint64_t RleEncoderV2<isSigned>::addImpl(
    const T* data,
    const common::Ranges& ranges,
    const uint64_t* nulls) {
  uint64_t count = 0;
  if (nulls) {
    for (auto& pos : ranges) {
      if (!bits::isBitNull(nulls, pos)) {
        writeValue(data[pos]);
        ++count;
      }
    }
  } else {
    for (auto& pos : ranges) {
      writeValue(data[pos]);
      ++count;
    }
  }
  return count;
}

template <bool isSigned>
void RleEncoderV2<isSigned>::writeValues(EncodingOption& option) {
  if (numLiterals_ != 0) {
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
        VELOX_UNREACHABLE();
    }

    numLiterals_ = 0;
    prevDelta_ = 0;
  }
}

template void RleEncoderV2<true>::writeValues(EncodingOption& option);
template void RleEncoderV2<false>::writeValues(EncodingOption& option);

template <bool isSigned>
uint64_t RleEncoderV2<isSigned>::flush() {
  if (numLiterals_ != 0) {
    EncodingOption option = {};
    if (variableRunLength_ != 0) {
      determineEncoding(option);
      writeValues(option);
    } else if (fixedRunLength_ != 0) {
      if (fixedRunLength_ < MIN_REPEAT) {
        variableRunLength_ = fixedRunLength_;
        fixedRunLength_ = 0;
        determineEncoding(option);
        writeValues(option);
      } else if (
          fixedRunLength_ >= MIN_REPEAT &&
          fixedRunLength_ <= MAX_SHORT_REPEAT_LENGTH) {
        option.encoding = SHORT_REPEAT;
        writeValues(option);
      } else {
        option.encoding = DELTA;
        option.isFixedDelta = true;
        writeValues(option);
      }
    }
  }

  IntEncoder<isSigned>::output_->BackUp(static_cast<int>(
      IntEncoder<isSigned>::bufferLength_ -
      IntEncoder<isSigned>::bufferPosition_));
  uint64_t dataSize = IntEncoder<isSigned>::output_->flush();
  IntEncoder<isSigned>::bufferLength_ = 0;
  IntEncoder<isSigned>::bufferPosition_ = 0;
  return dataSize;
}

template uint64_t RleEncoderV2<true>::flush();
template uint64_t RleEncoderV2<false>::flush();

template <bool isSigned>
void RleEncoderV2<isSigned>::writeShortRepeatValues(EncodingOption&) {
  int64_t repeatVal;
  if (isSigned) {
    repeatVal = ZigZag::encode(literals_[0]);
  } else {
    repeatVal = literals_[0];
  }

  const uint32_t numBitsRepeatVal = findClosestNumBits(repeatVal);
  const uint32_t numBytesRepeatVal = numBitsRepeatVal % 8 == 0
      ? (numBitsRepeatVal >> 3)
      : ((numBitsRepeatVal >> 3) + 1);

  uint32_t header = getOpCode(SHORT_REPEAT);

  fixedRunLength_ -= MIN_REPEAT;
  header |= fixedRunLength_;
  header |= ((numBytesRepeatVal - 1) << 3);

  IntEncoder<isSigned>::writeByte(static_cast<char>(header));

  for (auto i = static_cast<int32_t>(numBytesRepeatVal - 1); i >= 0; i--) {
    int64_t b = ((repeatVal >> (i * 8)) & 0xff);
    IntEncoder<isSigned>::writeByte(static_cast<char>(b));
  }

  fixedRunLength_ = 0;
}

template void RleEncoderV2<true>::writeShortRepeatValues(EncodingOption&);
template void RleEncoderV2<false>::writeShortRepeatValues(EncodingOption&);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeDirectValues(EncodingOption& option) {
  // write the number of fixed bits required in next 5 bits
  uint32_t fb = option.zzBits100p;
  if (alignedBitPacking_) {
    fb = getClosestAlignedFixedBits(fb);
  }

  const uint32_t efb = encodeBitWidth(fb) << 1;

  // adjust variable run length
  variableRunLength_ -= 1;

  // extract the 9th bit of run length
  const uint32_t tailBits = (variableRunLength_ & 0x100) >> 8;

  // create first byte of the header
  const char headerFirstByte =
      static_cast<char>(getOpCode(DIRECT) | efb | tailBits);

  // second byte of the header stores the remaining 8 bits of runlength
  const char headerSecondByte = static_cast<char>(variableRunLength_ & 0xff);

  // write header
  IntEncoder<isSigned>::writeByte(headerFirstByte);
  IntEncoder<isSigned>::writeByte(headerSecondByte);

  // bit packing the zigzag encoded literals_
  int64_t* currentZigzagLiterals =
      isSigned ? zigzagLiterals_.data() : literals_.data();
  writeInts(currentZigzagLiterals, 0, numLiterals_, fb);

  // reset run length
  variableRunLength_ = 0;
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
  variableRunLength_ -= 1;

  // extract the 9th bit of run length
  const uint32_t tailBits = (variableRunLength_ & 0x100) >> 8;

  // create first byte of the header
  const char headerFirstByte =
      static_cast<char>(getOpCode(PATCHED_BASE) | efb | tailBits);

  // second byte of the header stores the remaining 8 bits of runlength
  const char headerSecondByte = static_cast<char>(variableRunLength_ & 0xff);

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
  for (auto i = static_cast<int32_t>(baseBytes - 1); i >= 0; i--) {
    char b = static_cast<char>(((option.min >> (i * 8)) & 0xff));
    IntEncoder<isSigned>::writeByte(b);
  }

  // base reduced literals_ are bit packed
  uint32_t closestFixedBits = getClosestFixedBits(option.brBits95p);

  writeInts(baseRedLiterals_.data(), 0, numLiterals_, closestFixedBits);

  // write patch list
  closestFixedBits =
      getClosestFixedBits(option.patchGapWidth + option.patchWidth);

  writeInts(gapVsPatchList_.data(), 0, option.patchLength, closestFixedBits);

  // reset run length
  variableRunLength_ = 0;
}

template void RleEncoderV2<true>::writePatchedBasedValues(EncodingOption&);
template void RleEncoderV2<false>::writePatchedBasedValues(EncodingOption&);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeDeltaValues(EncodingOption& option) {
  uint32_t len = 0;
  uint32_t fb = option.bitsDeltaMax;
  uint32_t efb = 0;

  if (alignedBitPacking_) {
    fb = getClosestAlignedFixedBits(fb);
  }

  if (option.isFixedDelta) {
    // if fixed run length is greater than threshold then it will be fixed
    // delta sequence with delta value 0 else fixed delta sequence with
    // non-zero delta value
    if (fixedRunLength_ > MIN_REPEAT) {
      // ex. sequence: 2 2 2 2 2 2 2 2
      len = fixedRunLength_ - 1;
      fixedRunLength_ = 0;
    } else {
      // ex. sequence: 4 6 8 10 12 14 16
      len = variableRunLength_ - 1;
      variableRunLength_ = 0;
    }
  } else {
    // fixed width 0 is used for long repeating values.
    // sequences that require only 1 bit to encode will have an additional bit
    if (fb == 1) {
      fb = 2;
    }
    efb = encodeBitWidth(fb) << 1;
    len = variableRunLength_ - 1;
    variableRunLength_ = 0;
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
    IntEncoder<isSigned>::writeVslong(literals_[0]);
  } else {
    IntEncoder<isSigned>::writeVulong(literals_[0]);
  }

  if (option.isFixedDelta) {
    // if delta is fixed then we don't need to store delta blob
    IntEncoder<isSigned>::writeVslong(option.fixedDelta);
  } else {
    // store the first value as delta value using zigzag encoding
    IntEncoder<isSigned>::writeVslong(adjDeltas_[0]);

    // adjacent delta values are bit packed. The length of adjDeltas_ array is
    // always one less than the number of literals_ (delta difference for n
    // elements is n-1). We have already written one element, write the
    // remaining numLiterals_ - 2 elements here
    writeInts(adjDeltas_.data(), 1, numLiterals_ - 2, fb);
  }
}

template void RleEncoderV2<true>::writeDeltaValues(EncodingOption&);
template void RleEncoderV2<false>::writeDeltaValues(EncodingOption&);

template <bool isSigned>
void RleEncoderV2<isSigned>::writeInts(
    const int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize) {
  if (input == nullptr || len < 1 || bitSize < 1) {
    return;
  }

  if (getClosestAlignedFixedBits(bitSize) == bitSize) {
    uint32_t numBytes;
    auto endOffSet = static_cast<uint32_t>(offset + len);
    if (bitSize < 8) {
      char bitMask = static_cast<char>((1 << bitSize) - 1);
      uint32_t numHops = 8 / bitSize;
      auto remainder = static_cast<uint32_t>(len % numHops);
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
    const int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize);
template void RleEncoderV2<false>::writeInts(
    const int64_t* input,
    uint32_t offset,
    size_t len,
    uint32_t bitSize);

template <bool isSigned>
void RleEncoderV2<isSigned>::initializeLiterals(int64_t val) {
  literals_[numLiterals_++] = val;
  fixedRunLength_ = 1;
  variableRunLength_ = 1;
}

template void RleEncoderV2<true>::initializeLiterals(int64_t val);
template void RleEncoderV2<false>::initializeLiterals(int64_t val);
} // namespace facebook::velox::dwrf
