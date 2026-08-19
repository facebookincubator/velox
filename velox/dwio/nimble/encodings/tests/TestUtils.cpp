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

#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"

namespace facebook::nimble::test {

static constexpr int kRowCountOffset = 2;
static constexpr int kPrefixSize = 6;
static constexpr int kCompressionTypeSize = 1;

uint64_t TestUtils::getRawDataSize(
    velox::memory::MemoryPool& memoryPool,
    std::string_view encodingStr) {
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, &memoryPool));
    return buffer->asMutable<void>();
  };
  auto encoding =
      EncodingFactory().create(memoryPool, encodingStr, stringBufferFactory);
  EncodingType encodingType = encoding->encodingType();
  DataType dataType = encoding->dataType();
  uint32_t rowCount = encoding->rowCount();

  if (encodingType == EncodingType::Sentinel) {
    NIMBLE_UNSUPPORTED("Sentinel encoding is not supported");
  }

  if (encodingType == EncodingType::Nullable) {
    auto pos = encodingStr.data() + kPrefixSize;
    auto nonNullsSize = encoding::readUint32(pos);
    // Sum of the  nulls count and size of the non-null child encoding.
    return getRawDataSize(memoryPool, {pos, nonNullsSize}) + rowCount;
  } else {
    if (dataType != DataType::String) {
      auto typeSize = nimble::detail::dataTypeSize(dataType);
      auto result = typeSize * rowCount;
      return result;
    } else {
      auto pos = encodingStr.data() + kPrefixSize; // Skip the prefix.
      uint64_t result = 0;

      switch (encodingType) {
        case EncodingType::Trivial: {
          pos += kCompressionTypeSize;
          auto lengthsSize = encoding::readUint32(pos);
          auto lengths = EncodingFactory().create(
              memoryPool, {pos, lengthsSize}, stringBufferFactory);
          std::vector<uint32_t> buffer(rowCount);
          lengths->materialize(rowCount, buffer.data());
          result += std::accumulate(buffer.begin(), buffer.end(), 0u);
          break;
        }

        case EncodingType::Constant: {
          auto valueSize = encoding::readUint32(pos);
          result += rowCount * valueSize;
          break;
        }

        case EncodingType::MainlyConstant: {
          auto isCommonSize = encoding::readUint32(pos);
          pos += isCommonSize;
          auto otherValuesSize = encoding::readUint32(pos);
          auto otherValuesOffset = pos;
          auto otherValuesCount =
              encoding::peek<uint32_t>(pos + kRowCountOffset);
          pos += otherValuesSize;
          auto constantValueSize = encoding::readUint32(pos);
          result += (rowCount - otherValuesCount) * constantValueSize;
          result +=
              getRawDataSize(memoryPool, {otherValuesOffset, otherValuesSize});
          break;
        }

        case EncodingType::Dictionary: {
          auto alphabetSize = encoding::readUint32(pos);
          auto alphabetCount = encoding::peek<uint32_t>(pos + kRowCountOffset);
          auto alphabet = EncodingFactory().create(
              memoryPool, {pos, alphabetSize}, stringBufferFactory);
          std::vector<std::string_view> alphabetBuffer(alphabetCount);
          alphabet->materialize(alphabetCount, alphabetBuffer.data());

          pos += alphabetSize;
          auto indicesSize = encodingStr.length() - (pos - encodingStr.data());
          auto indices = EncodingFactory().create(
              memoryPool, {pos, indicesSize}, stringBufferFactory);
          std::vector<uint32_t> indicesBuffer(rowCount);
          indices->materialize(rowCount, indicesBuffer.data());
          for (int i = 0; i < rowCount; ++i) {
            result += alphabetBuffer[indicesBuffer[i]].size();
          }
          break;
        }

        case EncodingType::RLE: {
          auto runLengthsSize = encoding::readUint32(pos);
          auto runLengthsCount =
              encoding::peek<uint32_t>(pos + kRowCountOffset);
          auto runLengths = EncodingFactory().create(
              memoryPool, {pos, runLengthsSize}, stringBufferFactory);
          std::vector<uint32_t> runLengthsBuffer(runLengthsCount);
          runLengths->materialize(runLengthsCount, runLengthsBuffer.data());

          pos += runLengthsSize;
          auto runValuesSize =
              encodingStr.length() - (pos - encodingStr.data());
          auto runValues = EncodingFactory().create(
              memoryPool, {pos, runValuesSize}, stringBufferFactory);
          std::vector<std::string_view> runValuesBuffer(runLengthsCount);
          runValues->materialize(runLengthsCount, runValuesBuffer.data());

          for (int i = 0; i < runLengthsCount; ++i) {
            result += runLengthsBuffer[i] * runValuesBuffer[i].size();
          }
          break;
        }
        default:
          NIMBLE_UNSUPPORTED("Encoding type does not support strings.");
      }
      return result;
    }
  }
}
} // namespace facebook::nimble::test
