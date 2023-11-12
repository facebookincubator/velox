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

#include "velox/common/compression/v2/HadoopCompressionFormat.h"
#include "velox/common/base/Exceptions.h"

#include <folly/lang/Bits.h>

namespace facebook::velox::common {

bool HadoopCompressionFormat::tryDecompressHadoop(
    uint64_t inputLength,
    const uint8_t* input,
    uint64_t outputLength,
    uint8_t* output,
    uint64_t& actualDecompressedSize) {
  // Parquet files written with the Hadoop Lz4RawCodec use their own framing.
  // The input buffer can contain an arbitrary number of "frames", each
  // with the following structure:
  // - bytes 0..3: big-endian uint32_t representing the frame decompressed
  // size
  // - bytes 4..7: big-endian uint32_t representing the frame compressed size
  // - bytes 8...: frame compressed data
  //
  // The Hadoop Lz4Codec source code can be found here:
  // https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-nativetask/src/main/native/src/codec/Lz4Codec.cc
  uint64_t totalDecompressedSize = 0;

  while (inputLength >= kPrefixLength) {
    const uint32_t expectedDecompressedSize =
        folly::Endian::big(folly::loadUnaligned<uint32_t>(input));
    const uint32_t expectedCompressedSize = folly::Endian::big(
        folly::loadUnaligned<uint32_t>(input + sizeof(uint32_t)));
    input += kPrefixLength;
    inputLength -= kPrefixLength;

    if (inputLength < expectedCompressedSize) {
      // Not enough bytes for Hadoop "frame"
      return false;
    }
    if (outputLength < expectedDecompressedSize) {
      // Not enough bytes to hold advertised output => probably not Hadoop
      return false;
    }
    // Try decompressing and compare with expected decompressed length
    try {
      auto decompressedSize = decompressInternal(
          expectedCompressedSize, input, outputLength, output);
      if (decompressedSize != expectedDecompressedSize) {
        return false;
      }
    } catch (const VeloxException& e) {
      return false;
    }
    input += expectedCompressedSize;
    inputLength -= expectedCompressedSize;
    output += expectedDecompressedSize;
    outputLength -= expectedDecompressedSize;
    totalDecompressedSize += expectedDecompressedSize;
  }

  if (inputLength == 0) {
    actualDecompressedSize = totalDecompressedSize;
    return true;
  }
  return false;
}
} // namespace facebook::velox::common
