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

#include "velox/connectors/hive/iceberg/IcebergDwrfStatsCollector.h"

#include <array>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/dwio/dwrf/common/FileMetadata.h"
#include "velox/dwio/dwrf/common/Statistics.h"
#include "velox/dwio/dwrf/writer/Writer.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Recursively inserts 'field' and all of its descendant field ids.
void addAllRecursive(
    const parquet::ParquetFieldId& field,
    const TypePtr& type,
    folly::F14FastSet<int32_t>& fieldIds) {
  fieldIds.insert(field.fieldId);
  const auto numChildren =
      std::min<size_t>(field.children.size(), type->size());
  for (size_t i = 0; i < numChildren; ++i) {
    addAllRecursive(
        field.children.at(i),
        type->childAt(static_cast<uint32_t>(i)),
        fieldIds);
  }
}

// Collects the field ids that should skip bounds collection: MAP and ARRAY
// types and all of their descendants (Iceberg does not store bounds for
// repeated types). Mirrors
// IcebergParquetStatsCollector::collectSkipBoundsFieldIds.
void collectSkipBoundsFieldIds(
    const parquet::ParquetFieldId& field,
    const TypePtr& type,
    folly::F14FastSet<int32_t>& fieldIds) {
  VELOX_CHECK_NOT_NULL(type, "Input column type cannot be null.");

  if (type->isMap() || type->isArray()) {
    addAllRecursive(field, type, fieldIds);
    return;
  }

  const auto numChildren =
      std::min<size_t>(field.children.size(), type->size());
  for (size_t i = 0; i < numChildren; ++i) {
    collectSkipBoundsFieldIds(
        field.children.at(i),
        type->childAt(static_cast<uint32_t>(i)),
        fieldIds);
  }
}

// Serializes an integer as 'numBytes' little-endian two's-complement bytes.
std::string encodeLittleEndian(int64_t value, int32_t numBytes) {
  std::string out(numBytes, '\0');
  auto bits = static_cast<uint64_t>(value);
  for (int32_t i = 0; i < numBytes; ++i) {
    out[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
  }
  return out;
}

// Serializes a float as 4-byte little-endian IEEE-754.
std::string encodeFloatLittleEndian(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  std::string out(4, '\0');
  for (int32_t i = 0; i < 4; ++i) {
    out[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
  }
  return out;
}

// Serializes a double as 8-byte little-endian IEEE-754.
std::string encodeDoubleLittleEndian(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  std::string out(8, '\0');
  for (int32_t i = 0; i < 8; ++i) {
    out[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
  }
  return out;
}

// Serializes an unscaled decimal as the minimal-length big-endian two's
// complement byte array required by the Iceberg single-value spec.
std::string encodeDecimalBigEndian(int128_t value) {
  std::array<uint8_t, 16> bytes{};
  auto bits = static_cast<__uint128_t>(value);
  for (int32_t i = 15; i >= 0; --i) {
    bytes[i] = static_cast<uint8_t>(bits & 0xFF);
    bits >>= 8;
  }

  const bool negative = value < 0;
  const uint8_t pad = negative ? 0xFF : 0x00;
  // Trim redundant leading sign bytes while preserving the sign bit.
  size_t start = 0;
  while (start < 15) {
    const bool nextSignBitSet = (bytes[start + 1] & 0x80) != 0;
    if (bytes[start] == pad && nextSignBitSet == negative) {
      ++start;
    } else {
      break;
    }
  }
  return std::string(
      reinterpret_cast<const char*>(bytes.data() + start), 16 - start);
}

// Decodes a UTF-8 string into Unicode code points.
std::vector<char32_t> utf8Decode(std::string_view input) {
  std::vector<char32_t> codePoints;
  size_t i = 0;
  while (i < input.size()) {
    const auto byte = static_cast<unsigned char>(input[i]);
    size_t length = 1;
    char32_t codePoint = byte;
    if (byte >= 0xF0) {
      length = 4;
      codePoint = byte & 0x07;
    } else if (byte >= 0xE0) {
      length = 3;
      codePoint = byte & 0x0F;
    } else if (byte >= 0xC0) {
      length = 2;
      codePoint = byte & 0x1F;
    }
    for (size_t j = 1; j < length && i + j < input.size(); ++j) {
      codePoint =
          (codePoint << 6) | (static_cast<unsigned char>(input[i + j]) & 0x3F);
    }
    codePoints.push_back(codePoint);
    i += length;
  }
  return codePoints;
}

// Encodes Unicode code points into a UTF-8 string.
std::string utf8Encode(const std::vector<char32_t>& codePoints) {
  std::string out;
  for (const auto codePoint : codePoints) {
    if (codePoint < 0x80) {
      out.push_back(static_cast<char>(codePoint));
    } else if (codePoint < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
      out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
      out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
      out.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
  }
  return out;
}

constexpr char32_t kMaxCodePoint{0x10FFFF};
constexpr char32_t kSurrogateLow{0xD800};
constexpr char32_t kSurrogateHigh{0xDFFF};

// Truncates a lower bound to at most 'maxCodePoints' code points. A prefix of a
// string sorts before the full string, so simple truncation yields a valid
// (<= actual minimum) lower bound.
std::string truncateLowerBound(std::string_view value, int32_t maxCodePoints) {
  auto codePoints = utf8Decode(value);
  if (static_cast<int32_t>(codePoints.size()) <= maxCodePoints) {
    return std::string(value);
  }
  codePoints.resize(maxCodePoints);
  return utf8Encode(codePoints);
}

// Truncates an upper bound to at most 'maxCodePoints' code points, then rounds
// up so the result remains >= the actual maximum. Returns nullopt when the
// value cannot be rounded up (all leading code points are the max code point),
// in which case the upper bound is omitted. Matches Iceberg's
// UnicodeUtil.truncateStringMax behavior.
std::optional<std::string> truncateUpperBound(
    std::string_view value,
    int32_t maxCodePoints) {
  auto codePoints = utf8Decode(value);
  if (static_cast<int32_t>(codePoints.size()) <= maxCodePoints) {
    return std::string(value);
  }
  codePoints.resize(maxCodePoints);
  for (int32_t i = maxCodePoints - 1; i >= 0; --i) {
    if (codePoints[i] < kMaxCodePoint) {
      ++codePoints[i];
      if (codePoints[i] >= kSurrogateLow && codePoints[i] <= kSurrogateHigh) {
        codePoints[i] = kSurrogateHigh + 1;
      }
      codePoints.resize(i + 1);
      return utf8Encode(codePoints);
    }
  }
  return std::nullopt;
}

std::string base64Encode(std::string_view bytes) {
  return encoding::Base64::encode(bytes.data(), bytes.size());
}

// Serializes the integer min/max for a column to Iceberg single-value binary,
// choosing width/encoding from the logical type. Returns nullopt for types
// without a defined integer serialization.
std::optional<std::pair<std::string, std::string>>
serializeIntegerBounds(const TypePtr& type, int64_t min, int64_t max) {
  if (type->isDate()) {
    return std::pair{encodeLittleEndian(min, 4), encodeLittleEndian(max, 4)};
  }
  if (type->isShortDecimal()) {
    return std::pair{
        encodeDecimalBigEndian(static_cast<int128_t>(min)),
        encodeDecimalBigEndian(static_cast<int128_t>(max))};
  }
  const auto kind = type->kind();
  if (kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
      kind == TypeKind::INTEGER) {
    return std::pair{encodeLittleEndian(min, 4), encodeLittleEndian(max, 4)};
  }
  if (kind == TypeKind::BIGINT || kind == TypeKind::TIMESTAMP) {
    return std::pair{encodeLittleEndian(min, 8), encodeLittleEndian(max, 8)};
  }
  return std::nullopt;
}

// Populates lower/upper bounds for a column from its converted DWRF statistics.
// Skips types whose typed stats do not expose scalar min/max (binary, boolean,
// long decimal) or whose min/max is absent (all-null columns).
void setBounds(
    const dwio::common::ColumnStatistics& columnStatistics,
    const TypePtr& type,
    int32_t truncateLength,
    IcebergDataFileStatistics::ColumnStats& out) {
  if (const auto* intStats =
          dynamic_cast<const dwio::common::IntegerColumnStatistics<int64_t>*>(
              &columnStatistics)) {
    const auto min = intStats->getMinimum();
    const auto max = intStats->getMaximum();
    if (!min.has_value() || !max.has_value()) {
      return;
    }
    if (auto bounds = serializeIntegerBounds(type, min.value(), max.value())) {
      out.lowerBound = base64Encode(bounds->first);
      out.upperBound = base64Encode(bounds->second);
    }
    return;
  }

  if (const auto* doubleStats =
          dynamic_cast<const dwio::common::DoubleColumnStatistics*>(
              &columnStatistics)) {
    const auto min = doubleStats->getMinimum();
    const auto max = doubleStats->getMaximum();
    if (!min.has_value() || !max.has_value()) {
      return;
    }
    if (type->isReal()) {
      out.lowerBound = base64Encode(
          encodeFloatLittleEndian(static_cast<float>(min.value())));
      out.upperBound = base64Encode(
          encodeFloatLittleEndian(static_cast<float>(max.value())));
    } else {
      out.lowerBound = base64Encode(encodeDoubleLittleEndian(min.value()));
      out.upperBound = base64Encode(encodeDoubleLittleEndian(max.value()));
    }
    return;
  }

  if (const auto* stringStats =
          dynamic_cast<const dwio::common::StringColumnStatistics*>(
              &columnStatistics)) {
    const auto& min = stringStats->getMinimum();
    const auto& max = stringStats->getMaximum();
    if (!min.has_value() || !max.has_value()) {
      return;
    }
    out.lowerBound =
        base64Encode(truncateLowerBound(min.value(), truncateLength));
    if (auto upper = truncateUpperBound(max.value(), truncateLength)) {
      out.upperBound = base64Encode(upper.value());
    }
  }
}

// Recursively records the "iceberg.id" attribute for 'node' and its
// descendants, keyed by pre-order schema node id (matching the DWRF/ORC
// footer node ids). 'fieldId' is the field-id tree aligned to 'node'.
void stampFieldIdAttributes(
    const dwio::common::TypeWithId& node,
    const parquet::ParquetFieldId& fieldId,
    std::unordered_map<
        uint32_t,
        std::vector<std::pair<std::string, std::string>>>& attributes) {
  attributes[node.id()] = {{"iceberg.id", std::to_string(fieldId.fieldId)}};
  const auto numChildren =
      std::min<size_t>(node.size(), fieldId.children.size());
  for (size_t i = 0; i < numChildren; ++i) {
    stampFieldIdAttributes(
        *node.childAt(static_cast<uint32_t>(i)),
        fieldId.children.at(i),
        attributes);
  }
}

// Builds the per-node "iceberg.id" attribute map for the DWRF/ORC writer from
// the Iceberg input column handles, aligned to 'schema' (the written row
// type).
std::unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
buildDwrfSchemaAttributes(
    const RowTypePtr& schema,
    const std::vector<IcebergColumnHandlePtr>& inputColumns) {
  std::unordered_map<uint32_t, std::vector<std::pair<std::string, std::string>>>
      attributes;
  if (schema == nullptr) {
    return attributes;
  }
  const auto schemaWithId = dwio::common::TypeWithId::create(schema);
  const auto numColumns =
      std::min<size_t>(schemaWithId->size(), inputColumns.size());
  for (size_t i = 0; i < numColumns; ++i) {
    stampFieldIdAttributes(
        *schemaWithId->childAt(static_cast<uint32_t>(i)),
        inputColumns[i]->field(),
        attributes);
  }
  return attributes;
}

} // namespace

IcebergDwrfStatsCollector::IcebergDwrfStatsCollector(
    const std::vector<IcebergColumnHandlePtr>& inputColumns,
    const RowTypePtr& schema)
    : inputColumns_(inputColumns) {
  VELOX_CHECK_NOT_NULL(schema, "Schema cannot be null.");
  const auto schemaWithId = dwio::common::TypeWithId::create(schema);
  const auto numColumns =
      std::min<size_t>(schemaWithId->size(), inputColumns.size());
  for (size_t i = 0; i < numColumns; ++i) {
    const auto& field = inputColumns[i]->field();
    buildNodeInfo(
        *schemaWithId->childAt(static_cast<uint32_t>(i)),
        field,
        /*topLevel=*/true);
    collectSkipBoundsFieldIds(
        field, inputColumns[i]->dataType(), skipBoundsFieldIds_);
  }
}

void IcebergDwrfStatsCollector::buildNodeInfo(
    const dwio::common::TypeWithId& node,
    const parquet::ParquetFieldId& field,
    bool topLevel) {
  const auto& type = node.type();
  const bool flat =
      topLevel && !type->isRow() && !type->isArray() && !type->isMap();
  nodeInfo_[node.id()] = NodeInfo{field.fieldId, type, flat};

  const auto numChildren = std::min<size_t>(node.size(), field.children.size());
  for (size_t i = 0; i < numChildren; ++i) {
    buildNodeInfo(
        *node.childAt(static_cast<uint32_t>(i)),
        field.children.at(i),
        /*topLevel=*/false);
  }
}

IcebergDataFileStatisticsPtr IcebergDwrfStatsCollector::aggregate(
    const dwrf::FooterWrapper& footer,
    const dwrf::StatsContext& statsContext) const {
  auto dataFileStats = std::make_shared<IcebergDataFileStatistics>();
  const int64_t numRecords = static_cast<int64_t>(footer.numberOfRows());
  dataFileStats->numRecords = numRecords;

  const auto statisticsSize = footer.statisticsSize();
  for (const auto& [nodeId, info] : nodeInfo_) {
    if (nodeId >= static_cast<uint32_t>(statisticsSize)) {
      continue;
    }

    const auto columnStatistics = dwrf::buildColumnStatisticsFromProto(
        footer.statistics(static_cast<int>(nodeId)), statsContext);
    if (columnStatistics == nullptr) {
      continue;
    }

    auto& stats = dataFileStats->columnStats[info.fieldId];

    if (columnStatistics->getSize().has_value()) {
      stats.columnSize =
          static_cast<int64_t>(columnStatistics->getSize().value());
    }

    const auto numValues = columnStatistics->getNumberOfValues();
    if (info.topLevelFlat) {
      // For a primitive top-level column the Iceberg value count equals the
      // row count (non-null + null), and DWRF's non-null count yields the null
      // count directly.
      stats.valueCount = numRecords;
      if (numValues.has_value()) {
        stats.nullValueCount =
            numRecords - static_cast<int64_t>(numValues.value());
      }
    } else if (numValues.has_value()) {
      // For nested columns DWRF reports the non-null occurrence count. Use it
      // as the value count and leave the null count unset, since it is not 1:1
      // with the row count.
      stats.valueCount = static_cast<int64_t>(numValues.value());
    }

    if (shouldStoreBounds(info.fieldId)) {
      setBounds(*columnStatistics, info.type, kDefaultTruncateLength, stats);
    }
  }

  return dataFileStats;
}

void IcebergDwrfStatsCollector::configureWriterOptions(
    dwio::common::WriterOptions& options) const {
  auto* dwrfOptions = dynamic_cast<dwrf::WriterOptions*>(&options);
  if (dwrfOptions == nullptr) {
    return;
  }
  dwrfOptions->schemaAttributes = buildDwrfSchemaAttributes(
      std::dynamic_pointer_cast<const RowType>(dwrfOptions->schema),
      inputColumns_);
}

IcebergDataFileStatisticsPtr IcebergDwrfStatsCollector::collect(
    dwio::common::Writer& writer,
    std::unique_ptr<dwio::common::FileMetadata>& /* closeMetadata */) const {
  // dwrf::Writer::close() returns an empty placeholder, so the column
  // statistics are read from the still-alive writer's footer proto.
  auto* dwrfWriter = dynamic_cast<dwrf::Writer*>(&writer);
  if (dwrfWriter == nullptr) {
    return nullptr;
  }
  const auto& footerWriteWrapper = dwrfWriter->getFooter();
  if (footerWriteWrapper == nullptr) {
    return nullptr;
  }
  const auto footer = footerWriteWrapper->format() == dwrf::DwrfFormat::kDwrf
      ? dwrf::FooterWrapper(footerWriteWrapper->getDwrfPtr())
      : dwrf::FooterWrapper(footerWriteWrapper->getOrcPtr());
  const auto writerVersion =
      dwrfWriter->getContext().getConfig(dwrf::Config::WRITER_VERSION);
  const dwrf::StatsContext statsContext(writerVersion);
  return aggregate(footer, statsContext);
}

} // namespace facebook::velox::connector::hive::iceberg
