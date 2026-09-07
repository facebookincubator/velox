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
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <span>
#include <type_traits>
#include <vector>

#include "folly/container/F14Map.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Canonical Huffman encoding for integral values. The bitstream preserves row
/// order and stores a bit offset every 256 rows for bounded random access.
///
/// Wire format:
///   EncodingPrefix:
///     1 byte: EncodingType::Huffman
///     1 byte: DataType
///     4 bytes or varint: row count
///   varint: symbol count (N)
///   1 byte: maximum code length / decode-table log
///   N * sizeof(physicalType) bytes: symbol alphabet
///   N bytes: canonical Huffman code lengths, indexed by alphabet position
///   varint: checkpoint count (C), ceil(row count / 256)
///   C varints: delta-encoded bit offsets into the encoded symbol stream
///   varint: encoded symbol stream size, including decoder padding
///   variable bytes: LSB-first encoded symbol stream
///   4 zero bytes: decoder lookahead padding, included in the stream size
///
/// Codes are reconstructed canonically from the code lengths. Decoding a code
/// produces an alphabet index; alphabet[index] is the original integral value.
template <typename T>
class HuffmanEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static constexpr uint32_t kCheckpointStride{256};
  // OpenZL's standard Huffman codec supports 256 symbols with a maximum
  // 12-bit table. Its Large Huffman codec supports 65,536 symbols with a
  // maximum 20-bit table and length-limits deeper trees. Nimble keeps the
  // 12-bit table but supports up to 4,096 integral symbols, falling back when
  // a skewed distribution would require a deeper tree.
  static constexpr uint32_t kMaxSymbols{4096};
  static constexpr uint8_t kMaxCodeBits{12};

  HuffmanEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& /*stringBufferFactory*/ = nullptr,
      const Encoding::Options& options = {});

  void reset() final {
    currentRow_ = 0;
  }

  void skip(uint32_t rowCount) final {
    NIMBLE_CHECK_LE(rowCount, this->rowCount_ - currentRow_);
    currentRow_ += rowCount;
  }

  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params) {
    detail::readWithVisitorSlow(
        visitor,
        params,
        [&](auto count) { skip(count); },
        [&] { return decodeValue(currentRow_++); });
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {});

 private:
  // Temporary encoder tree. Child fields index nodes_; leaves have no children
  // and internal nodes have no symbol.
  struct TreeNode {
    uint64_t frequency;
    int32_t left;
    int32_t right;
    int32_t symbol;

    // Declaring the explicit constructor below removes the implicit default
    // constructor; keep TreeNode default-constructible because
    // Vector<TreeNode>'s debug-only placeholder_ member (see Vector.h) odr-uses
    // it under mode/dev.
    TreeNode() = default;

    // Explicit constructor so emplace_back() works under compilers that do not
    // apply parenthesized aggregate initialization (P0960) inside construct_at
    // / placement-new (e.g. the OSS GCC build).
    TreeNode(uint64_t frequency, int32_t left, int32_t right, int32_t symbol)
        : frequency(frequency), left(left), right(right), symbol(symbol) {}
  };

  struct QueueEntry {
    uint64_t frequency;
    int32_t node;

    // Explicit constructor keeps priority_queue::emplace() working on OSS GCC
    // versions that do not apply parenthesized aggregate initialization here.
    QueueEntry(uint64_t frequency, int32_t node)
        : frequency(frequency), node(node) {}

    bool operator>(const QueueEntry& other) const {
      // Node order makes equal-frequency trees deterministic.
      return frequency != other.frequency ? frequency > other.frequency
                                          : node > other.node;
    }
  };

  // Maps tableLog_ lookahead bits to an alphabet index and the number of bits
  // consumed for that symbol.
  struct DecodeEntry {
    uint16_t symbol;
    uint8_t bits;
  };

  // Convert canonical MSB-first codes to Nimble's LSB-first bitstream order.
  static uint16_t reverseBits(uint16_t value, uint8_t bits) {
    uint16_t reversed = 0;
    for (uint8_t i = 0; i < bits; ++i) {
      reversed = static_cast<uint16_t>((reversed << 1) | (value & 1));
      value >>= 1;
    }
    return reversed;
  }

  template <typename NodeContainer, typename LengthContainer>
  static bool assignCodeLengths(
      const NodeContainer& nodes,
      int32_t node,
      uint8_t depth,
      LengthContainer& lengths) {
    if (nodes[node].symbol >= 0) {
      if (depth > kMaxCodeBits) {
        return false;
      }
      lengths[nodes[node].symbol] = std::max<uint8_t>(1, depth);
      return true;
    }
    return assignCodeLengths(nodes, nodes[node].left, depth + 1, lengths) &&
        assignCodeLengths(nodes, nodes[node].right, depth + 1, lengths);
  }

  template <
      typename FrequencyContainer,
      typename NodeContainer,
      typename LengthContainer>
  static bool buildCodeLengths(
      const FrequencyContainer& frequencies,
      NodeContainer& nodes,
      LengthContainer& lengths) {
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>>
        queue;
    for (uint32_t symbol = 0; symbol < frequencies.size(); ++symbol) {
      nodes.emplace_back(
          frequencies[symbol], -1, -1, static_cast<int32_t>(symbol));
      queue.emplace(frequencies[symbol], static_cast<int32_t>(symbol));
    }
    while (queue.size() > 1) {
      const auto left = queue.top();
      queue.pop();
      const auto right = queue.top();
      queue.pop();
      const auto node = static_cast<int32_t>(nodes.size());
      nodes.emplace_back(
          left.frequency + right.frequency, left.node, right.node, -1);
      queue.emplace(left.frequency + right.frequency, node);
    }
    return assignCodeLengths(nodes, queue.top().node, 0, lengths);
  }

  physicalType decodeValue(uint32_t row) const;

  Vector<physicalType> alphabet_;
  Vector<DecodeEntry> decodeTable_;
  Vector<uint32_t> checkpoints_;
  const uint8_t* bitstream_{nullptr};
  uint32_t bitstreamBytes_{0};
  uint8_t tableLog_{0};
  uint32_t currentRow_{0};
};

template <typename T>
HuffmanEncoding<T>::HuffmanEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>&,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      alphabet_{&pool},
      decodeTable_{&pool},
      checkpoints_{&pool} {
  static_assert(
      std::is_integral_v<physicalType> && !std::is_same_v<physicalType, bool>);

  const char* pos = data.data() + this->dataOffset();
  const uint32_t symbolCount = varint::readVarint32(&pos);
  tableLog_ = static_cast<uint8_t>(encoding::readChar(pos));
  NIMBLE_CHECK_GE(symbolCount, 2);
  NIMBLE_CHECK_LE(symbolCount, kMaxSymbols);
  NIMBLE_CHECK_LE(tableLog_, kMaxCodeBits);

  alphabet_.resize(symbolCount);
  std::memcpy(alphabet_.data(), pos, symbolCount * sizeof(physicalType));
  pos += symbolCount * sizeof(physicalType);

  const auto* lengths = reinterpret_cast<const uint8_t*>(pos);
  pos += symbolCount;
  decodeTable_.resize(1u << tableLog_);
  std::fill(decodeTable_.begin(), decodeTable_.end(), DecodeEntry{0, 0});

  // Number of symbols assigned to each code length, indexed by bit length.
  std::array<uint16_t, kMaxCodeBits + 1> counts{};
  for (uint32_t i = 0; i < symbolCount; ++i) {
    NIMBLE_CHECK_GT(lengths[i], 0);
    NIMBLE_CHECK_LE(lengths[i], tableLog_);
    ++counts[lengths[i]];
  }
  std::array<uint16_t, kMaxCodeBits + 1> nextCode{};
  uint16_t code = 0;
  for (uint8_t bits = 1; bits <= tableLog_; ++bits) {
    code = static_cast<uint16_t>((code + counts[bits - 1]) << 1);
    nextCode[bits] = code;
  }
  for (uint32_t symbol = 0; symbol < symbolCount; ++symbol) {
    const uint8_t bits = lengths[symbol];
    const uint16_t reversed = reverseBits(nextCode[bits]++, bits);
    for (uint32_t slot = reversed; slot < decodeTable_.size();
         slot += 1u << bits) {
      decodeTable_[slot] = DecodeEntry{static_cast<uint16_t>(symbol), bits};
    }
  }

  const uint32_t checkpointCount = varint::readVarint32(&pos);
  NIMBLE_CHECK_EQ(
      checkpointCount,
      velox::bits::divRoundUp(this->rowCount_, kCheckpointStride));
  checkpoints_.resize(checkpointCount);
  uint32_t bitOffset = 0;
  for (uint32_t i = 0; i < checkpointCount; ++i) {
    bitOffset += varint::readVarint32(&pos);
    checkpoints_[i] = bitOffset;
  }
  bitstreamBytes_ = varint::readVarint32(&pos);
  bitstream_ = reinterpret_cast<const uint8_t*>(pos);
}

template <typename T>
typename HuffmanEncoding<T>::physicalType HuffmanEncoding<T>::decodeValue(
    uint32_t row) const {
  const uint32_t checkpoint = row / kCheckpointStride;
  uint32_t bitOffset = checkpoints_[checkpoint];
  uint32_t current = checkpoint * kCheckpointStride;
  while (current <= row) {
    uint32_t bits = 0;
    const uint32_t byteOffset = bitOffset >> 3;
    const uint32_t available =
        std::min<uint32_t>(4, bitstreamBytes_ - byteOffset);
    std::memcpy(&bits, bitstream_ + byteOffset, available);
    bits >>= bitOffset & 7;
    const auto entry = decodeTable_[bits & ((1u << tableLog_) - 1)];
    NIMBLE_CHECK_GT(entry.bits, 0);
    bitOffset += entry.bits;
    if (current++ == row) {
      return alphabet_[entry.symbol];
    }
  }
  NIMBLE_UNREACHABLE("Invalid Huffman row {}", row);
}

template <typename T>
void HuffmanEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_DCHECK_LE(currentRow_ + rowCount, this->rowCount_);
  auto* output = static_cast<physicalType*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = decodeValue(currentRow_ + i);
  }
  currentRow_ += rowCount;
}

template <typename T>
std::optional<uint64_t> HuffmanEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (values.size() < 2) {
    return std::nullopt;
  }
  const auto& uniqueCounts = statistics.uniqueCounts();
  if (!uniqueCounts.has_value() || uniqueCounts->size() < 2 ||
      uniqueCounts->size() > kMaxSymbols) {
    return std::nullopt;
  }

  folly::F14FastMap<physicalType, uint32_t> symbolByValue;
  std::vector<uint32_t> frequencies;
  frequencies.reserve(uniqueCounts->size());
  for (const auto value : values) {
    auto [it, inserted] =
        symbolByValue.emplace(value, static_cast<uint32_t>(frequencies.size()));
    if (inserted) {
      frequencies.push_back(0);
    }
    ++frequencies[it->second];
  }
  std::vector<TreeNode> nodes;
  nodes.reserve(2 * frequencies.size() - 1);
  std::vector<uint8_t> lengths(frequencies.size());
  if (!buildCodeLengths(frequencies, nodes, lengths)) {
    return std::nullopt;
  }

  uint64_t encodedBits = 0;
  const uint64_t rowsMinusOne = values.size() - 1;
  for (const auto& [value, count] : uniqueCounts.value()) {
    (void)value;
    encodedBits += count * velox::bits::bitsRequired(rowsMinusOne / count);
  }
  const uint64_t checkpoints =
      velox::bits::divRoundUp(values.size(), kCheckpointStride);
  const uint64_t bitstreamBytes = (encodedBits + 7) / 8 + 4;
  return EncodingPrefix::serializedSize(
             values.size(), options.useVarintRowCount) +
      varint::varintSize(uniqueCounts->size()) + 1 +
      uniqueCounts->size() * sizeof(physicalType) + uniqueCounts->size() +
      varint::varintSize(checkpoints) + checkpoints * 2 +
      varint::varintSize(bitstreamBytes) + bitstreamBytes;
}

template <typename T>
std::string_view HuffmanEncoding<T>::encode(
    EncodingSelection<physicalType>&,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(
      std::is_integral_v<physicalType> && !std::is_same_v<physicalType, bool>);
  if (values.size() < 2) {
    NIMBLE_INCOMPATIBLE_ENCODING("Huffman encoding requires at least two rows");
  }

  auto* pool = &buffer.getMemoryPool();
  folly::F14FastMap<physicalType, uint32_t> symbolByValue;
  Vector<physicalType> alphabet{pool};
  Vector<uint32_t> frequencies{pool};
  for (const auto value : values) {
    auto [it, inserted] =
        symbolByValue.emplace(value, static_cast<uint32_t>(alphabet.size()));
    if (inserted) {
      if (alphabet.size() >= kMaxSymbols) {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "Huffman encoding supports at most {} symbols", kMaxSymbols);
      }
      alphabet.push_back(value);
      frequencies.push_back(0);
    }
    ++frequencies[it->second];
  }
  if (alphabet.size() < 2) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "Huffman encoding requires at least two symbols");
  }

  Vector<TreeNode> nodes{pool};
  Vector<uint8_t> lengths{pool, alphabet.size()};
  if (!buildCodeLengths(frequencies, nodes, lengths)) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "Huffman tree exceeds the {}-bit limit", kMaxCodeBits);
  }
  const uint8_t tableLog = *std::max_element(lengths.begin(), lengths.end());

  // Number of symbols assigned to each code length, indexed by bit length.
  std::array<uint16_t, kMaxCodeBits + 1> counts{};
  for (const auto length : lengths) {
    ++counts[length];
  }
  std::array<uint16_t, kMaxCodeBits + 1> nextCode{};
  uint16_t code = 0;
  for (uint8_t bits = 1; bits <= tableLog; ++bits) {
    code = static_cast<uint16_t>((code + counts[bits - 1]) << 1);
    nextCode[bits] = code;
  }
  Vector<uint16_t> codes{pool, alphabet.size()};
  for (size_t symbol = 0; symbol < alphabet.size(); ++symbol) {
    codes[symbol] = reverseBits(nextCode[lengths[symbol]]++, lengths[symbol]);
  }

  Vector<uint32_t> checkpoints{pool};
  Vector<uint8_t> bitstream{pool};
  uint64_t bitBuffer = 0;
  uint32_t bufferedBits = 0;
  uint32_t totalBits = 0;
  for (size_t row = 0; row < values.size(); ++row) {
    if (row % kCheckpointStride == 0) {
      checkpoints.push_back(totalBits);
    }
    const auto symbol = symbolByValue.at(values[row]);
    bitBuffer |= static_cast<uint64_t>(codes[symbol]) << bufferedBits;
    bufferedBits += lengths[symbol];
    totalBits += lengths[symbol];
    while (bufferedBits >= 8) {
      bitstream.push_back(static_cast<uint8_t>(bitBuffer));
      bitBuffer >>= 8;
      bufferedBits -= 8;
    }
  }
  if (bufferedBits != 0) {
    bitstream.push_back(static_cast<uint8_t>(bitBuffer));
  }
  bitstream.resize(bitstream.size() + 4, 0);

  uint32_t checkpointBytes = 0;
  uint32_t previousCheckpoint = 0;
  for (const auto checkpoint : checkpoints) {
    checkpointBytes += varint::varintSize(checkpoint - previousCheckpoint);
    previousCheckpoint = checkpoint;
  }
  const uint64_t encodedSize =
      Encoding::serializePrefixSize(values.size(), options.useVarintRowCount) +
      varint::varintSize(alphabet.size()) + 1 +
      alphabet.size() * sizeof(physicalType) + lengths.size() +
      varint::varintSize(checkpoints.size()) + checkpointBytes +
      varint::varintSize(bitstream.size()) + bitstream.size();
  NIMBLE_CHECK_LE(encodedSize, std::numeric_limits<uint32_t>::max());
  const auto size = static_cast<uint32_t>(encodedSize);
  char* start = buffer.reserve(size);
  char* pos = start;
  Encoding::serializePrefix(
      EncodingType::Huffman,
      TypeTraits<T>::dataType,
      values.size(),
      options.useVarintRowCount,
      pos);
  varint::writeVarint(alphabet.size(), &pos);
  encoding::writeChar(tableLog, pos);
  std::memcpy(pos, alphabet.data(), alphabet.size() * sizeof(physicalType));
  pos += alphabet.size() * sizeof(physicalType);
  std::memcpy(pos, lengths.data(), lengths.size());
  pos += lengths.size();
  varint::writeVarint(checkpoints.size(), &pos);
  previousCheckpoint = 0;
  for (const auto checkpoint : checkpoints) {
    varint::writeVarint(checkpoint - previousCheckpoint, &pos);
    previousCheckpoint = checkpoint;
  }
  varint::writeVarint(bitstream.size(), &pos);
  std::memcpy(pos, bitstream.data(), bitstream.size());
  pos += bitstream.size();
  NIMBLE_DCHECK_EQ(pos - start, size);
  return {start, size};
}

} // namespace facebook::nimble
