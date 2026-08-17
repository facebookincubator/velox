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
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

namespace {
constexpr uint32_t kMinEncodingLayoutBufferSize = 5;

void captureChild(
    std::vector<std::optional<const EncodingLayout>>& children,
    const char*& cursor,
    uint32_t size,
    const Encoding::Options& childOptions) {
  children.emplace_back(
      size > 0
          ? std::optional<const EncodingLayout>(
                EncodingLayoutCapture::capture({cursor, size}, childOptions))
          : std::nullopt);
  cursor += size;
}

} // namespace

std::optional<std::string> EncodingLayout::Config::get(
    const std::string& key) const {
  auto it = configs_.find(key);
  if (it == configs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

EncodingLayout::EncodingLayout(
    EncodingType encodingType,
    Config encodingConfig,
    CompressionType compressionType,
    std::vector<std::optional<const EncodingLayout>> children)
    : encodingType_{encodingType},
      encodingConfig_{std::move(encodingConfig)},
      compressionType_{compressionType},
      children_{std::move(children)} {}

int32_t EncodingLayout::serialize(std::span<char> output) const {
  // Serialized encoding layout is as follows:
  // 1 byte - Encoding Type
  // 1 byte - CompressionType
  // 1 byte - Children (nested encoding) Count
  // 2 bytes - Extra data size (currently always set to zero. Reserved for futre
  //           exra args for compression or encoding)
  // Extra data size bytes - Extra data (currently not used)
  // 1 bytes - 1st (nested encoding) child exists
  // X bytes - 1st (nested encoding) child
  // 1 bytes - 2nd (nested encoding) child exists
  // Y bytes - 2nd (nested encoding) child
  // ...

  // We store at least kMinEncodingLayoutBufferSize bytes: encoding type,
  // compression type, children count and extra data size (2 bytes), plus one
  // byte per child.
  NIMBLE_CHECK_GE(
      output.size(),
      kMinEncodingLayoutBufferSize + children_.size(),
      "Captured encoding layout buffer too small.");

  output[0] = static_cast<char>(encodingType_);
  output[1] = static_cast<char>(compressionType_);
  output[2] = static_cast<char>(children_.size());
  // Currently, extra data is not used and always set to zero.
  output[3] = output[4] = 0;

  int32_t size = kMinEncodingLayoutBufferSize;

  for (auto i = 0; i < children_.size(); ++i) {
    const auto& child = children_[i];
    if (child.has_value()) {
      output[size++] = 1;
      size += child->serialize(output.subspan(size));
    } else {
      // Set child size to 0
      output[size++] = 0;
    }
  }

  return size;
}

std::pair<EncodingLayout, uint32_t> EncodingLayout::create(
    std::string_view encoding) {
  NIMBLE_CHECK_GE(
      encoding.size(),
      kMinEncodingLayoutBufferSize,
      "Invalid captured encoding layout. Buffer too small.");

  auto pos = encoding.data();
  const auto encodingType = encoding::read<uint8_t, EncodingType>(pos);
  const auto compressionType = encoding::read<uint8_t, CompressionType>(pos);
  const auto childrenCount = encoding::read<uint8_t>(pos);
  [[maybe_unused]] const auto extraDataSize = encoding::read<uint16_t>(pos);

  NIMBLE_DCHECK_EQ(extraDataSize, 0, "Extra data currently not supported.");

  uint32_t offset = kMinEncodingLayoutBufferSize;
  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(childrenCount);
  for (auto i = 0; i < childrenCount; ++i) {
    auto childExists = encoding::peek<uint8_t>(encoding.data() + offset);
    ++offset;
    if (childExists > 0) {
      auto encodingLayout = EncodingLayout::create(encoding.substr(offset));
      offset += encodingLayout.second;
      children.emplace_back(std::move(encodingLayout.first));
    } else {
      children.emplace_back(std::nullopt);
    }
  }

  return {{encodingType, {}, compressionType, std::move(children)}, offset};
}

EncodingType EncodingLayout::encodingType() const {
  return encodingType_;
}

CompressionType EncodingLayout::compressionType() const {
  return compressionType_;
}

uint8_t EncodingLayout::childrenCount() const {
  return children_.size();
}

const std::optional<const EncodingLayout>& EncodingLayout::child(
    NestedEncodingIdentifier identifier) const {
  NIMBLE_DCHECK_LT(
      identifier,
      children_.size(),
      "Encoding layout identifier is out of range.");

  return children_[identifier];
}

const EncodingLayout::Config& EncodingLayout::config() const {
  return encodingConfig_;
}

EncodingLayout EncodingLayoutCapture::capture(
    std::string_view encoding,
    const Encoding::Options& options) {
  NIMBLE_CHECK_GE(
      encoding.size(),
      EncodingPrefix::kRowCountOffset,
      "Encoding size too small.");
  const auto encodingType = EncodingPrefix::encodingType(encoding);
  const auto prefixSize =
      EncodingPrefix::prefixSize(encoding, options.useVarintRowCount);
  CompressionType compressionType = CompressionType::Uncompressed;

  if (encodingType == EncodingType::FixedBitWidth ||
      encodingType == EncodingType::Trivial ||
      encodingType == EncodingType::BlockBitPacking ||
      encodingType == EncodingType::FOR) {
    compressionType =
        encoding::peek<uint8_t, CompressionType>(encoding.data() + prefixSize);
  }

  std::vector<std::optional<const EncodingLayout>> children;
  switch (encodingType) {
    case EncodingType::FixedBitWidth:
    case EncodingType::Varint:
    case EncodingType::Constant:
    case EncodingType::Prefix:
    case EncodingType::DeltaBlock:
    case EncodingType::SimdForBitpack:
    case EncodingType::FrequencyPartition:
    case EncodingType::Huffman:
      // Non nested encodings have zero children
      break;
    case EncodingType::Slice:
      // The wrapped encoding is carried verbatim rather than as a nested
      // stream, and the layout tree describes how data is encoded, not how a
      // slice was deferred. Reported as childless.
      break;
    case EncodingType::SubIntSplit: {
      // SubIntSplit decomposes its input into per-section bit-range
      // sub-streams. Read the section headers, recursively capture each
      // section's nested encoding as a child, and preserve the recovered
      // bit boundaries in the encoding config so the same split layout can
      // be replayed (see ReplayedEncodingSelectionPolicy).
      const char* pos = encoding.data() + prefixSize;
      const uint8_t splitCount = encoding::read<uint8_t>(pos);
      encoding::read<uint8_t>(pos); // reserved

      struct SectionMeta {
        uint8_t bitStart;
        uint8_t bitEnd;
        uint32_t encodedSize;
      };

      std::vector<SectionMeta> sectionMeta;
      sectionMeta.reserve(splitCount);
      for (uint8_t s = 0; s < splitCount; ++s) {
        SectionMeta meta{};
        meta.bitStart = encoding::read<uint8_t>(pos);
        meta.bitEnd = encoding::read<uint8_t>(pos);
        meta.encodedSize = encoding::readUint32(pos);
        sectionMeta.push_back(meta);
      }

      children.reserve(splitCount);
      for (uint8_t s = 0; s < splitCount; ++s) {
        children.emplace_back(EncodingLayoutCapture::capture(
            {pos, sectionMeta[s].encodedSize}, options));
        pos += sectionMeta[s].encodedSize;
      }

      std::vector<detail::subintsplit::SegmentPlan> boundaryPlans;
      boundaryPlans.reserve(splitCount);
      for (uint8_t s = 0; s < splitCount; ++s) {
        detail::subintsplit::SegmentPlan segment{};
        segment.bitStart = static_cast<int>(sectionMeta[s].bitStart);
        segment.bitEnd = static_cast<int>(sectionMeta[s].bitEnd);
        boundaryPlans.push_back(segment);
      }

      return {
          EncodingType::SubIntSplit,
          EncodingLayout::Config{
              detail::subintsplit::makePreserveSplitConfig(boundaryPlans)},
          compressionType,
          std::move(children)};
    }
    case EncodingType::ALP: {
      const char* pos = encoding.data() + prefixSize;
      const auto header = detail::alp::readHeader(pos);
      if (header.hasExceptions) {
        varint::readVarint32(&pos); // exceptionCount
      }
      const uint32_t encodedValuesBytes = varint::readVarint32(&pos);

      children.reserve(header.hasExceptions ? 3 : 1);
      captureChild(children, pos, encodedValuesBytes, options);
      if (header.hasExceptions) {
        const auto positionsBytes = varint::readVarint32(&pos);
        captureChild(children, pos, positionsBytes, options);
        const auto valuesBytes = varint::readVarint32(&pos);
        captureChild(children, pos, valuesBytes, options);
      }
      break;
    }
    case EncodingType::PFOR: {
      const auto dataType =
          encoding::peek<uint8_t, DataType>(encoding.data() + 1);
      const char* pos = encoding.data() + prefixSize;
      pos += detail::dataTypeSize(dataType); // baseline
      encoding::readChar(pos); // baseBitWidth
      varint::readVarint32(&pos); // numExceptions

      children.reserve(2);
      const auto positionsBytes = varint::readVarint32(&pos);
      captureChild(children, pos, positionsBytes, options);
      const auto valuesBytes = varint::readVarint32(&pos);
      captureChild(children, pos, valuesBytes, options);
      break;
    }
    case EncodingType::BlockBitPacking: {
      // BlockBitPacking nests three per-block metadata sub-streams: baselines,
      // bit widths, and data offsets.
      const char* pos = encoding.data() + prefixSize;
      encoding::readChar(pos); // compressionType
      varint::readVarint32(&pos); // blockSize
      varint::readVarint32(&pos); // numBlocks

      children.reserve(3);
      const auto baselinesBytes = varint::readVarint32(&pos);
      captureChild(children, pos, baselinesBytes, options);
      const auto bitWidthsBytes = varint::readVarint32(&pos);
      captureChild(children, pos, bitWidthsBytes, options);
      const auto dataOffsetsBytes = varint::readVarint32(&pos);
      captureChild(children, pos, dataOffsetsBytes, options);
      break;
    }
    case EncodingType::FOR: {
      const char* pos = encoding.data() + prefixSize;
      encoding::readChar(pos); // compressionType
      varint::readVarint32(&pos); // frameSize
      varint::readVarint32(&pos); // numFrames
      varint::readVarint32(&pos); // firstFrameRows

      children.reserve(3);
      const auto bitWidthsBytes = varint::readVarint32(&pos);
      captureChild(children, pos, bitWidthsBytes, options);
      const auto referencesBytes = varint::readVarint32(&pos);
      captureChild(children, pos, referencesBytes, options);
      const auto bitOffsetsBytes = varint::readVarint32(&pos);
      captureChild(children, pos, bitOffsetsBytes, options);
      break;
    }
    case EncodingType::Trivial: {
      const auto dataType =
          encoding::peek<uint8_t, DataType>(encoding.data() + 1);
      if (dataType == DataType::String) {
        const char* pos = encoding.data() + prefixSize + 1;
        const uint32_t lengthsBytes = encoding::readUint32(pos);

        children.reserve(1);
        children.emplace_back(
            EncodingLayoutCapture::capture({pos, lengthsBytes}, options));
      }
      break;
    }
    case EncodingType::Fsst: {
      FsstEncoding::captureNestedEncoding(encoding, children, options);
      break;
    }
    case EncodingType::SparseBool: {
      children.reserve(1);
      children.emplace_back(
          EncodingLayoutCapture::capture(
              encoding.substr(prefixSize + 1), options));
      break;
    }
    case EncodingType::MainlyConstant: {
      children.reserve(2);

      const char* pos = encoding.data() + prefixSize;
      const uint32_t isCommonBytes = encoding::readUint32(pos);

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, isCommonBytes}, options));

      pos += isCommonBytes;
      const uint32_t otherValuesBytes = encoding::readUint32(pos);

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, otherValuesBytes}, options));
      break;
    }
    case EncodingType::Dictionary: {
      children.reserve(2);
      const char* pos = encoding.data() + prefixSize;
      const uint32_t alphabetBytes = encoding::readUint32(pos);

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, alphabetBytes}, options));

      pos += alphabetBytes;

      children.emplace_back(
          EncodingLayoutCapture::capture(
              {pos, encoding.size() - (pos - encoding.data())}, options));
      break;
    }
    case EncodingType::SharedDictionary: {
      children.reserve(1);
      const char* pos = encoding.data() + prefixSize;
      const auto indicesOffset = static_cast<size_t>(pos - encoding.data());
      NIMBLE_CHECK_LT(
          indicesOffset,
          encoding.size(),
          "Shared dictionary encoding is missing its indices.");
      children.emplace_back(
          EncodingLayoutCapture::capture(
              {pos, encoding.size() - indicesOffset}, options));
      break;
    }
    case EncodingType::RLE: {
      const auto dataType =
          encoding::peek<uint8_t, DataType>(encoding.data() + 1);

      children.reserve(dataType == DataType::Bool ? 1 : 2);

      const char* pos = encoding.data() + prefixSize;
      const uint32_t runLengthBytes = encoding::readUint32(pos);

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, runLengthBytes}, options));

      if (dataType != DataType::Bool) {
        pos += runLengthBytes;

        children.emplace_back(
            EncodingLayoutCapture::capture(
                {pos, encoding.size() - (pos - encoding.data())}, options));
      }
      break;
    }
    case EncodingType::Delta: {
      children.reserve(3);

      const char* pos = encoding.data() + prefixSize;
      const uint32_t deltaBytes = encoding::readUint32(pos);
      const uint32_t restatementBytes = encoding::readUint32(pos);

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, deltaBytes}, options));

      pos += deltaBytes;

      children.emplace_back(
          EncodingLayoutCapture::capture({pos, restatementBytes}, options));

      pos += restatementBytes;

      children.emplace_back(
          EncodingLayoutCapture::capture(
              {pos, encoding.size() - (pos - encoding.data())}, options));
      break;
    }
    case EncodingType::Nullable: {
      const char* pos = encoding.data() + prefixSize;
      const uint32_t dataBytes = encoding::readUint32(pos);

      // For nullable encodings we only capture the data encoding part, so we
      // are "overwriting" the current captured node with the nested data node.
      return EncodingLayoutCapture::capture({pos, dataBytes}, options);
    }
    case EncodingType::Sentinel: {
      // For sentinel encodings we only capture the data encoding part, so we
      // are "overwriting" the current captured node with the nested data node.
      return EncodingLayoutCapture::capture(
          encoding.substr(prefixSize + 8), options);
    }
  }

  return {
      encodingType,
      /*encodingConfig=*/
      {},
      compressionType,
      std::move(children)};
}

} // namespace facebook::nimble
