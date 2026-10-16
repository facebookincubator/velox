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

#include "velox/connectors/hive/hudi/HudiDeleteBlockDecoder.h"

#include <cstring>

#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::hudi {

namespace {

// Log block version required for delete blocks.
constexpr uint32_t kDeleteBlockVersion{3};

// Union branch ordinals of the HoodieDeleteRecord.orderingVal field, matching
// the HoodieDeleteRecordList Avro schema. Only integral branches are decoded.
enum class OrderingBranch : int64_t {
  kNull = 0,
  kInt = 1,
  kLong = 2,
  kFloat = 3,
  kDouble = 4,
  kBytes = 5,
  kString = 6,
  kDecimal = 7,
  kDate = 8,
  kTimeMillis = 9,
  kTimeMicros = 10,
  kTimestampMillis = 11,
  kTimestampMicros = 12,
};

// Minimal reader for the subset of the Avro binary encoding used by the
// HoodieDeleteRecordList datum: variable-length integers, zig-zag longs,
// strings, arrays and unions.
class AvroBinaryReader {
 public:
  explicit AvroBinaryReader(std::string_view data) : data_{data} {}

  // Reads a zig-zag encoded long (also used for union indices and array block
  // counts).
  int64_t readLong() {
    uint64_t value{0};
    int shift{0};
    while (true) {
      VELOX_CHECK_LT(pos_, data_.size(), "Truncated Avro datum");
      const auto byte = static_cast<uint8_t>(data_[pos_++]);
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) {
        break;
      }
      shift += 7;
      VELOX_CHECK_LT(shift, 64, "Malformed Avro varint");
    }
    return static_cast<int64_t>(value >> 1) ^ -static_cast<int64_t>(value & 1);
  }

  // Reads a length-prefixed UTF-8 string.
  std::string readString() {
    const auto length = readLong();
    VELOX_CHECK_GE(length, 0, "Negative Avro string length");
    VELOX_CHECK_LE(
        pos_ + static_cast<size_t>(length),
        data_.size(),
        "Truncated Avro string");
    auto result = std::string{data_.substr(pos_, length)};
    pos_ += length;
    return result;
  }

 private:
  const std::string_view data_;
  size_t pos_{0};
};

// Reads a big-endian uint32 at `offset` within `data`.
uint32_t readBigEndian32(std::string_view data, size_t offset) {
  VELOX_CHECK_LE(
      offset + sizeof(uint32_t), data.size(), "Truncated delete block");
  uint32_t value;
  std::memcpy(&value, data.data() + offset, sizeof(uint32_t));
  return folly::Endian::big(value);
}

// Reads a HoodieDeleteRecord.orderingVal union and returns its value when the
// selected branch is an integral type.
std::optional<int64_t> readOrderingValue(AvroBinaryReader& reader) {
  const auto branch = static_cast<OrderingBranch>(reader.readLong());
  switch (branch) {
    case OrderingBranch::kNull:
      return std::nullopt;
    case OrderingBranch::kInt:
    case OrderingBranch::kLong:
    case OrderingBranch::kDate:
    case OrderingBranch::kTimeMillis:
    case OrderingBranch::kTimeMicros:
    case OrderingBranch::kTimestampMillis:
    case OrderingBranch::kTimestampMicros:
      return reader.readLong();
    default:
      VELOX_NYI(
          "Unsupported Hudi delete ordering value type: {}",
          static_cast<int64_t>(branch));
  }
}

// Reads a nullable string union ([null, string]).
std::string readNullableString(AvroBinaryReader& reader) {
  const auto branch = reader.readLong();
  VELOX_CHECK(
      branch == 0 || branch == 1,
      "Invalid Avro union branch for nullable string: {}",
      branch);
  if (branch == 0) {
    return {};
  }
  return reader.readString();
}

} // namespace

std::vector<HudiDeleteRecord> decodeHudiDeleteBlock(const HudiLogBlock& block) {
  VELOX_CHECK(block.isDeleteBlock(), "Expected a Hudi delete log block");

  const std::string_view content{block.content};
  const auto version = readBigEndian32(content, 0);
  VELOX_CHECK_EQ(
      version, kDeleteBlockVersion, "Unsupported Hudi delete block version");
  // The 4 bytes after the version hold the datum length; the datum spans the
  // remainder of the content window.
  const auto datum = content.substr(2 * sizeof(uint32_t));

  AvroBinaryReader reader{datum};
  std::vector<HudiDeleteRecord> records;
  // The datum is a record with a single array field, so it encodes directly as
  // an Avro array: a sequence of blocks, each a (possibly negative) item count
  // followed by that many items, terminated by a zero count.
  while (true) {
    auto count = reader.readLong();
    if (count == 0) {
      break;
    }
    if (count < 0) {
      count = -count;
      // Negative counts are followed by the block's byte size, which we ignore.
      reader.readLong();
    }
    for (int64_t i = 0; i < count; ++i) {
      HudiDeleteRecord record;
      record.recordKey = readNullableString(reader);
      record.partitionPath = readNullableString(reader);
      record.orderingValue = readOrderingValue(reader);
      records.push_back(std::move(record));
    }
  }
  return records;
}

} // namespace facebook::velox::connector::hive::hudi
