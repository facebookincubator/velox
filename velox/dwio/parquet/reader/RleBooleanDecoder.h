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

#pragma once

#include "velox/dwio/parquet/reader/RleBpDecoder.h"
#include "arrow/util/rle_encoding.h" // need for arrow decoder stuff

namespace facebook::velox::parquet {

class RleBooleanDecoder : public facebook::velox::parquet::RleBpDecoder {
 public:
  using super = facebook::velox::parquet::RleBpDecoder;
  RleBooleanDecoder(const char* start, const char* end, BufferPtr decompressedData_, const short* repetitionLevels_, int32_t& len, int32_t& numNulls)
      : super::RleBpDecoder{start+4, end, 1} {
            // const uint8_t* data = decompressedData_->asMutable<uint8_t>();
            // int64_t size = decompressedData_->size();

            // Print each byte in the buffer
            // for (int64_t i = 0; i < size; ++i) {
            //     std::cout << "Byte " << i << ": " << static_cast<int>(data[i]) << std::endl;

            //     // iterate through bits
            //     for (int bit = 0; bit < 8; ++bit) {
            //         bool value = (data[i] & (1 << bit)) != 0;
            //         std::cout << "Bit " << (i * 8 + bit) << ": " << (value ? "true" : "false") << " ";
            //     }
            //     std::cout << std::endl;
            // }
        // const uint8_t* rleData = reinterpret_cast<const uint8_t*>(bufferStart_);
      // You would need the number of bytes in the RLE encoded data.
      //size_t numBytes = end - start;  // Adjust this based on actual data.
      // You might need to calculate the bit width as needed for booleans.
      int bitWidth = 1;  // For boolean values.

      // ARROW IMPLEMNTATION/SETUP BELOW SET DATA
      // length is the minimum length of the whole data including the first 4 so we know it should be 4 to be valid
      if (len < 4) {
        VELOX_FAIL("Received invalid length : " + std::to_string(len) +
                             " (corrupt data page?)");
      }
      // num_bytes will be the first 4 bytes that tell us the length of encoded data
      num_bytes = ::arrow::bit_util::FromLittleEndian(::arrow::util::SafeLoadAs<uint32_t>(reinterpret_cast<const uint8_t*>(start)));
      if (num_bytes < 0 || num_bytes > static_cast<uint32_t>(len - 4)) {
        VELOX_FAIL("Received invalid number of bytes : " +
                             std::to_string(num_bytes) + " (corrupt data page?)");
      }

      nullCount = numNulls;

      start += 4;

      decoder_ = std::make_shared<::arrow::util::RleDecoder>(reinterpret_cast<const uint8_t*>(start), num_bytes, 1);
      // std::vector<uint8_t> outputValues_;
      // outputValues_.resize(numValues_, 0);
      // uint8_t* output = outputValues_.data();
      // decoder_->GetBatch(output, numValues_);

      // ARROW IMPLEMENTATION COMPLETE USING ARROW METHODS FOR DECODING...

    // start should be the value at byte 4...
      bool rleOrPacked = (start[0] & 1) == 0 ? true : false;
      if (rleOrPacked) {
        // rle case
        currentCount_ = static_cast<unsigned int>(start[0]) >> 1;
        //currentValue = BytesUtils.readIntLittleEndianPaddedOnBitWidth(in, bitWidth);
        currentValue_ = start[1] & 1;
      }
      else {
        // bit packed case
        uint8_t numGroups = static_cast<uint8_t>(start[0]) >> 1;
        currentCount_ = numGroups * 8;
        //LOG.debug("reading {} values BIT PACKED", currentCount);
        //currentBuffer = new int[currentCount]; // TODO: reuse a buffer
        std::vector<uint64_t> currentBuffer(currentCount_);
        //byte[] bytes = new byte[numGroups * bitWidth];
        std::vector<unsigned char> bytes(numGroups * bitWidth);
        // At the end of the file RLE data though, there might not be that many bytes left.
        // bytesToRead = Math.min(bytesToRead, in.available());
        int bytesToRead = static_cast<int>(std::ceil(currentCount_ * bitWidth / 8.0));
        // new DataInputStream(in).readFully(bytes, 0, bytesToRead);
        //readFully(bytes, 0, bytesToRead, repetitionLevels_);
        for (int valueIndex = 0, byteIndex = 0; valueIndex < currentCount_; valueIndex += 8, byteIndex += bitWidth) {
          // packer.unpack8Values(bytes, byteIndex, currentBuffer, valueIndex);
          unpack8Values(bytes, byteIndex, currentBuffer, valueIndex);
        }
      }
      }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }

    super::skip(numValues);
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();

    skip<hasNulls>(current, 0, nulls);
    int32_t toSkip;
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();
    std::vector<uint64_t> outputBuffer(20);
    bool* b = nullptr;
    for (;;) {
      // comment this out and the ending else if we want to try out
      // the arrow decoder approach with the null count placing as null
      if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
        toSkip = visitor.processNull(atEnd);
      } else {
        if (hasNulls && !allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) {
            return;
          }
        }

        // We are at a non-null value on a row to visit.
        // if (nullCount == visitor.numRows()) {
        //   for (int i = 0; i < nullCount; i++) {
        //     visitor.processNull(atEnd);
        //   }
        //   return;
        // }

        std::vector<uint8_t> outputValues_;
        outputValues_.resize(visitor.numRows(), 0);
        std::cout<<"numRows: " << visitor.numRows()<<std::endl;
        uint8_t* output = outputValues_.data();
        //int count = decoder_->GetBatch(output, visitor.numRows());
        // std::cout<<"cout decoded:" << count << "num nulls " <<nullCount <<std::endl;
        std::cout << "num nulls : "<< nullCount << std::endl;
        // for (int i = 0; i < visitor.numRows(); i++) {
        //   std::cout << output[i] << std::endl;
        //   if (i == visitor.numRows()-1) {
        //     atEnd = true;
        //   }
        //   if (nullCount > 0 && output[i] == 0) {
        //     visitor.processNull(atEnd);
        //     nullCount--;
        //   }
        //   else {
        //     toSkip = visitor.process(output[i], atEnd);
        //   }
        //   //toSkip = visitor.process(output[i], atEnd);
        //   if (toSkip) {
        //     skip<hasNulls>(toSkip, i, nulls);
        //     i += toSkip;
        //   }
        // }
        // if (atEnd) {
        //   return;
        // }
        if (!remainingValues_) {
          readHeader();
        }
        if (repeating_) {
          toSkip = visitor.process(value_, atEnd);
        } else {
          value_ = readBitField();
          toSkip = visitor.process(value_, atEnd);
        }
        if (toSkip) {
          skip<hasNulls>(toSkip, current, nulls);
          current += toSkip;
        }
        if(remainingValues_ == 0){
          return;
        }
        --remainingValues_;
      }
      ++current;
      if (toSkip) {
        skip<hasNulls>(toSkip, current, nulls);
        current += toSkip;
      }
      if (atEnd) {
        return;
      }
    }
  }

 private:

  int64_t readBitField() {
    auto value =
        dwio::common::safeLoadBits(
            super::bufferStart_, bitOffset_, bitWidth_, lastSafeWord_) &
        bitMask_;
    bitOffset_ += bitWidth_;
    super::bufferStart_ += bitOffset_ >> 3;
    bitOffset_ &= 7;
    return value;
  }
  bool readBoolean() {

      std::vector<uint8_t> outputValues_;
      outputValues_.resize(numValues_, 0);
      uint8_t* output = outputValues_.data();
      decoder_->GetBatch(output, numValues_);
      return true;

    // if (remainingBits_ == 0) {
    //   remainingBits_ = 7;
    //   reversedLastByte_ = *reinterpret_cast<const uint8_t*>(bufferStart_);
    //   bufferStart_++;
    //   return reversedLastByte_ & 0x1;
    // } else {
    //   return reversedLastByte_ & (1 << (8 - (remainingBits_--)));
    // }
  }

  // void decode() {
  //   if (bufferStart_ >= bufferEnd_) {
  //     remainingCount_ = 0; // No more runs
  //     return;
  //   }

  //   // Read the run length (first byte)
  //   uint8_t runLength = *bufferStart_++;
  //   std::cout << "Read runLength: " << static_cast<int>(runLength) << std::endl;

    
  //   // Read the boolean value (next byte)
  //   if (bufferStart_ >= bufferEnd_) {
  //     remainingCount_ = 0; // Not enough data for value
  //     return;
  //   }

  //   currentValue_ = (*bufferStart_++ != 0); // 0 for false, non-zero for true
  //   std::cout << "Read currentValue: " << currentValue_ << std::endl;
  //   remainingCount_ = runLength; // Set the remaining count
  // }

  void readFully(const std::vector<unsigned char>& b, int off, int len, const uint8_t* repetitionLevels_) {
    if (len < 0) {
      VELOX_FAIL("Index Out of bounds exception");
    }
    int n = 0;
    while (n < len) {
      // int count = in.read(b, off + n, len - n);
      int count = repetitionLevels_[0];
      if (count < 0) {
        VELOX_FAIL("End of file Exception");
      }
      n += count;
    }
  }

  void unpack8Values(const std::vector<unsigned char>& in, int inPos, std::vector<uint64_t>& out, int outPos) {
          out[ 0 + outPos] =
          //           [_______0]
          //                  [0]
            (static_cast<int>(in[0 + inPos]) & 1);
      out[ 1 + outPos] =
          //           [______1_]
          //                 [0]
            (static_cast<int>(in[0 + inPos] >> 1) & 1);
      out[ 2 + outPos] =
          //           [_____2__]
          //                [0]
            (static_cast<int>(in[0 + inPos] >> 2) & 1);
      out[ 3 + outPos] =
          //           [____3___]
          //               [0]
            (static_cast<int>(in[0 + inPos] >> 3) & 1);
      out[ 4 + outPos] =
          //           [___4____]
          //              [0]
            (static_cast<int>(in[0 + inPos] >> 4) & 1);
      out[ 5 + outPos] =
          //           [__5_____]
          //             [0]
            (static_cast<int>(in[0 + inPos] >> 5) & 1);
      out[ 6 + outPos] =
          //           [_6______]
          //            [0]
            (static_cast<int>(in[0 + inPos] >> 6) & 1);
      out[ 7 + outPos] =
          //           [7_______]
          //           [0]
            (static_cast<int>(in[0 + inPos] >> 7) & 1);
  }

  size_t remainingCount_{0};
  bool currentValue_{true};
  const char* bufferStart_;
  const char* bufferEnd_;
  int currentCount_ = 0;
  uint32_t num_bytes = 0;
  uint32_t numValues_ = 1024;
  int32_t nullCount = 0;
  // int32_t num_values_;
  std::shared_ptr<::arrow::util::RleDecoder> decoder_;
};

} // namespace facebook::velox::parquet