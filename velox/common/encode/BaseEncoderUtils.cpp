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
#include "velox/common/encode/BaseEncoderUtils.h"

#include <folly/Expected.h>

namespace facebook::velox::encoding {

// static
Expected<size_t> BaseEncoderUtils::calculateDecodedSize(
    std::string_view input,
    size_t& inputSize,
    const CodecBlockSizes& blockSizes) {
  if (inputSize == 0) {
    return 0;
  }

  auto binaryBlockByteSize = blockSizes.binaryBlockByteSize;
  auto encodedBlockByteSize = blockSizes.encodedBlockByteSize;

  // Check if the input string is padded.
  if (isPadded(input)) {
    // If padded, ensure that the string length is a multiple of the encoded
    // block size.
    if (inputSize % encodedBlockByteSize != 0) {
      return folly::makeUnexpected(
          Status::UserError("decode() - invalid input string length."));
    }

    auto decodedSize = (inputSize * binaryBlockByteSize) / encodedBlockByteSize;
    auto paddingCount = numPadding(input);
    inputSize -= paddingCount;

    // Adjust the needed size by deducting the bytes corresponding to the
    // padding from the calculated size.
    return decodedSize -
        ((paddingCount * binaryBlockByteSize) + (encodedBlockByteSize - 1)) /
        encodedBlockByteSize;
  }

  // If not padded, calculate extra bytes, if any.
  auto extraBytes = inputSize % encodedBlockByteSize;
  auto decodedSize = (inputSize / encodedBlockByteSize) * binaryBlockByteSize;

  // Adjust the needed size for extra bytes, if present.
  if (extraBytes) {
    if (extraBytes == 1) {
      return folly::makeUnexpected(
          Status::UserError("decode() - invalid input string length."));
    }
    decodedSize += (extraBytes * binaryBlockByteSize) / encodedBlockByteSize;
  }

  return decodedSize;
}

// static
size_t BaseEncoderUtils::calculateEncodedSize(
    size_t inputSize,
    bool includePadding,
    const CodecBlockSizes& blockSizes) {
  if (inputSize == 0) {
    return 0;
  }

  auto binaryBlockByteSize = blockSizes.binaryBlockByteSize;
  auto encodedBlockByteSize = blockSizes.encodedBlockByteSize;

  // Calculate the output size assuming that we are including padding.
  size_t encodedSize =
      ((inputSize + binaryBlockByteSize - 1) / binaryBlockByteSize) *
      encodedBlockByteSize;

  if (!includePadding) {
    // If padding was not requested, compute the exact number of encoded
    // characters needed: ceil(inputSize * encodedBlockByteSize /
    // binaryBlockByteSize).
    return (inputSize * encodedBlockByteSize + binaryBlockByteSize - 1) /
        binaryBlockByteSize;
  }
  return encodedSize;
}

} // namespace facebook::velox::encoding
