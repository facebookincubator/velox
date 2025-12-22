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

#include "velox/dwio/avro/reader/AvroReader.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>
#include <avro/LogicalType.hh>
#include <avro/Node.hh>
#include <avro/Schema.hh>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/expression/VectorWriters.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::avro {

enum class AvroLogicalType {
  kNone,
  kDate,
  kTimeMillis,
  kTimeMicros,
  kTimestampMillis,
  kTimestampMicros,
  kDecimal,
  kUuid,
};

enum class AvroUnionKind {
  kNone,
  kSimple,
  kNumericPromotion,
  kStruct,
};

struct AvroTypeInfo {
  TypePtr veloxType;
  bool nullable{false};
  std::vector<std::string> fieldNames;
  std::vector<std::shared_ptr<AvroTypeInfo>> children;
  AvroLogicalType logicalType{AvroLogicalType::kNone};
  uint8_t decimalPrecision{0};
  uint8_t decimalScale{0};
  AvroUnionKind unionKind{AvroUnionKind::kNone};
  std::optional<size_t> nullUnionBranchIndex;
};

namespace {

using dwio::common::BufferedInput;
using dwio::common::LogType;
using dwio::common::ReaderOptions;
using dwio::common::TypeWithId;
using exec::GenericWriter;

constexpr std::string_view kAvroSchemaLiteralKey = "avro.schema.literal";
constexpr std::string_view kAvroScanBatchBytesKey = "avro.scan.batch.bytes";
constexpr uint64_t kDefaultAvroScanBatchBytes = 100UL << 20; // 100MB

std::optional<::avro::ValidSchema> loadOverrideSchema(
    const ReaderOptions& options) {
  const auto& serdeParameters = options.serDeOptions().parameters;

  const auto schemaIt =
      serdeParameters.find(std::string(kAvroSchemaLiteralKey));
  if (schemaIt == serdeParameters.end() || schemaIt->second.empty()) {
    return std::nullopt;
  }

  ::avro::ValidSchema schema;
  std::istringstream schemaStream(schemaIt->second);
  try {
    ::avro::compileJsonSchema(schemaStream, schema);
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Failed to parse Avro schema override from '{}': {}",
        kAvroSchemaLiteralKey,
        e.what());
  }

  return schema;
}

uint64_t loadAvroScanBatchBytes(const ReaderOptions& options) {
  const auto& serdeParameters = options.serDeOptions().parameters;
  const auto it = serdeParameters.find(std::string(kAvroScanBatchBytesKey));
  if (it == serdeParameters.end() || it->second.empty()) {
    return kDefaultAvroScanBatchBytes;
  }

  uint64_t parsedValue = 0;
  try {
    parsedValue = folly::to<uint64_t>(it->second);
  } catch (const folly::ConversionError& e) {
    VELOX_USER_FAIL(
        "Failed to parse '{}' value '{}': {}",
        kAvroSchemaLiteralKey,
        it->second,
        e.what());
  }
  VELOX_USER_CHECK_GT(
      parsedValue,
      0,
      "{} must be a positive integer, got {}",
      kAvroSchemaLiteralKey,
      parsedValue);
  return parsedValue;
}

std::shared_ptr<AvroTypeInfo> buildTypeInfo(
    const ::avro::NodePtr& node,
    const ReaderOptions& options);

void applyDecimalLogicalType(
    const ::avro::LogicalType& logical,
    AvroTypeInfo& info) {
  const auto precisionValue = logical.precision();
  const auto scaleValue = logical.scale();
  VELOX_CHECK_LE(
      precisionValue,
      static_cast<decltype(precisionValue)>(
          std::numeric_limits<uint8_t>::max()));
  VELOX_CHECK_LE(
      scaleValue,
      static_cast<decltype(scaleValue)>(std::numeric_limits<uint8_t>::max()));
  auto precision = static_cast<uint8_t>(precisionValue);
  auto scale = static_cast<uint8_t>(scaleValue);

  VELOX_CHECK_GE(precision, 1);
  VELOX_CHECK_LE(precision, LongDecimalType::kMaxPrecision);
  VELOX_CHECK_GE(scale, 0);
  VELOX_CHECK_LE(scale, precisionValue);
  info.logicalType = AvroLogicalType::kDecimal;
  info.decimalPrecision = precision;
  info.decimalScale = scale;
  info.veloxType = DECIMAL(precision, scale);
}

::avro::NodePtr resolveIfSymbolic(const ::avro::NodePtr& node) {
  if (node->type() == ::avro::Type::AVRO_SYMBOLIC) {
    return ::avro::resolveSymbol(node);
  }
  return node;
}

std::shared_ptr<AvroTypeInfo> buildUnionType(
    const ::avro::NodePtr& node,
    const ReaderOptions& options) {
  const auto resolvedNode = resolveIfSymbolic(node);
  auto info = std::make_shared<AvroTypeInfo>();

  std::vector<std::shared_ptr<AvroTypeInfo>> nonNullInfos;
  nonNullInfos.reserve(resolvedNode->leaves());
  bool allIntsOrLongs = true;
  bool allFloatsOrDoubles = true;

  for (size_t i = 0; i < resolvedNode->leaves(); ++i) {
    const auto branchNode = resolveIfSymbolic(resolvedNode->leafAt(i));
    if (branchNode->type() == ::avro::Type::AVRO_NULL) {
      info->nullUnionBranchIndex = i;
      continue;
    }

    auto childInfo = buildTypeInfo(branchNode, options);
    nonNullInfos.push_back(childInfo);

    switch (branchNode->type()) {
      case ::avro::Type::AVRO_INT:
      case ::avro::Type::AVRO_LONG:
        allFloatsOrDoubles = false;
        break;
      case ::avro::Type::AVRO_FLOAT:
      case ::avro::Type::AVRO_DOUBLE:
        allIntsOrLongs = false;
        break;
      default:
        allIntsOrLongs = false;
        allFloatsOrDoubles = false;
        break;
    }
  }

  info->nullable = info->nullUnionBranchIndex.has_value();

  VELOX_CHECK(
      !nonNullInfos.empty(), "Unsupported Avro union without non-null branch.");

  info->children = std::move(nonNullInfos);

  if (info->children.size() == 1) {
    info->unionKind = AvroUnionKind::kSimple;
    const auto& childInfo = info->children.front();
    info->veloxType = childInfo->veloxType;
    info->logicalType = childInfo->logicalType;
    info->decimalPrecision = childInfo->decimalPrecision;
    info->decimalScale = childInfo->decimalScale;
    return info;
  }

  if (allIntsOrLongs) {
    info->unionKind = AvroUnionKind::kNumericPromotion;
    info->veloxType = BIGINT();
    return info;
  }

  if (allFloatsOrDoubles) {
    info->unionKind = AvroUnionKind::kNumericPromotion;
    info->veloxType = DOUBLE();
    return info;
  }

  info->unionKind = AvroUnionKind::kStruct;
  std::vector<std::string> fieldNames;
  std::vector<TypePtr> childTypes;
  fieldNames.reserve(info->children.size());
  childTypes.reserve(info->children.size());
  for (size_t index = 0; index < info->children.size(); ++index) {
    fieldNames.push_back("member" + std::to_string(index));
    childTypes.push_back(info->children[index]->veloxType);
  }
  info->fieldNames = fieldNames;
  info->veloxType = ROW(std::move(fieldNames), std::move(childTypes));
  return info;
}

std::shared_ptr<AvroTypeInfo> buildTypeInfo(
    const ::avro::NodePtr& node,
    const ReaderOptions& options) {
  const auto resolvedNode = resolveIfSymbolic(node);
  auto info = std::make_shared<AvroTypeInfo>();
  const auto logical = resolvedNode->logicalType();
  const auto logicalType = logical.type();
  switch (resolvedNode->type()) {
    case ::avro::Type::AVRO_NULL:
      info->veloxType = UNKNOWN();
      info->nullable = true;
      break;
    case ::avro::Type::AVRO_BOOL:
      info->veloxType = BOOLEAN();
      break;
    case ::avro::Type::AVRO_INT:
      if (logicalType == ::avro::LogicalType::Type::DATE) {
        info->logicalType = AvroLogicalType::kDate;
        info->veloxType = DATE();
      } else if (logicalType == ::avro::LogicalType::Type::TIME_MILLIS) {
        info->logicalType = AvroLogicalType::kTimeMillis;
        info->veloxType = TIME();
      } else {
        info->veloxType = INTEGER();
      }
      break;
    case ::avro::Type::AVRO_LONG:
      if (logicalType == ::avro::LogicalType::Type::TIME_MICROS) {
        info->logicalType = AvroLogicalType::kTimeMicros;
        info->veloxType = TIME();
      } else if (logicalType == ::avro::LogicalType::Type::TIMESTAMP_MILLIS) {
        info->logicalType = AvroLogicalType::kTimestampMillis;
        info->veloxType = TIMESTAMP();
      } else if (logicalType == ::avro::LogicalType::Type::TIMESTAMP_MICROS) {
        info->logicalType = AvroLogicalType::kTimestampMicros;
        info->veloxType = TIMESTAMP();
      } else {
        info->veloxType = BIGINT();
      }
      break;
    case ::avro::Type::AVRO_FLOAT:
      info->veloxType = REAL();
      break;
    case ::avro::Type::AVRO_DOUBLE:
      info->veloxType = DOUBLE();
      break;
    case ::avro::Type::AVRO_STRING:
      if (logicalType == ::avro::LogicalType::Type::UUID) {
        info->logicalType = AvroLogicalType::kUuid;
      }
      info->veloxType = VARCHAR();
      break;
    case ::avro::Type::AVRO_BYTES:
      if (logicalType == ::avro::LogicalType::Type::DECIMAL) {
        applyDecimalLogicalType(logical, *info);
      } else {
        info->veloxType = VARBINARY();
      }
      break;
    case ::avro::Type::AVRO_FIXED:
      if (logicalType == ::avro::LogicalType::Type::DECIMAL) {
        applyDecimalLogicalType(logical, *info);
      } else {
        info->veloxType = VARBINARY();
      }
      break;
    case ::avro::Type::AVRO_ENUM:
      info->veloxType = VARCHAR();
      break;
    case ::avro::Type::AVRO_ARRAY: {
      auto elementInfo = buildTypeInfo(resolvedNode->leafAt(0), options);
      info->veloxType = ARRAY(elementInfo->veloxType);
      info->children.push_back(elementInfo);
      break;
    }
    case ::avro::Type::AVRO_MAP: {
      auto valueInfo = buildTypeInfo(resolvedNode->leafAt(1), options);
      info->veloxType = MAP(VARCHAR(), valueInfo->veloxType);
      info->children.push_back(valueInfo);
      break;
    }
    case ::avro::Type::AVRO_RECORD: {
      std::vector<std::string> names;
      std::vector<TypePtr> children;
      auto count = resolvedNode->leaves();
      names.reserve(count);
      children.reserve(count);
      info->children.reserve(count);
      std::unordered_set<std::string> fieldNameSeen;
      for (size_t i = 0; i < count; ++i) {
        auto fieldName = resolvedNode->nameAt(i);
        if (options.fileColumnNamesReadAsLowerCase()) {
          boost::algorithm::to_lower(fieldName);
        }
        VELOX_CHECK(
            fieldNameSeen.insert(fieldName).second,
            "Avro schema found duplicated field: {}",
            fieldName);
        names.push_back(fieldName);
        auto childInfo = buildTypeInfo(resolvedNode->leafAt(i), options);
        children.push_back(childInfo->veloxType);
        info->children.push_back(childInfo);
      }
      info->fieldNames = names;
      info->veloxType = ROW(std::move(names), std::move(children));
      break;
    }
    case ::avro::Type::AVRO_UNION:
      return buildUnionType(resolvedNode, options);
    default:
      VELOX_UNSUPPORTED(
          "Unsupported Avro type {}", static_cast<int>(resolvedNode->type()));
  }
  return info;
}

class ReadFileAvroInputStream : public ::avro::SeekableInputStream {
 public:
  ReadFileAvroInputStream(
      std::shared_ptr<dwio::common::ReadFileInputStream> input,
      uint64_t start,
      uint64_t length,
      memory::MemoryPool& pool) {
    const auto blockSize = std::max<uint64_t>(
        input->getNaturalReadSize(), static_cast<uint64_t>(128 << 10));
    stream_ = std::make_unique<dwio::common::SeekableFileInputStream>(
        std::move(input), start, length, pool, LogType::FILE, blockSize);
  }

  bool next(const uint8_t** data, size_t* len) override {
    const void* rawData = nullptr;
    int32_t size = 0;
    if (!stream_->Next(&rawData, &size)) {
      *data = nullptr;
      *len = 0;
      return false;
    }
    *data = static_cast<const uint8_t*>(rawData);
    *len = static_cast<size_t>(size);
    pushback_ = 0;
    return true;
  }

  void backup(size_t len) override {
    if (pushback_ > 0) {
      const void* rawData = nullptr;
      int32_t size = 0;
      stream_->Next(&rawData, &size);
    }
    pushback_ += len;
    stream_->BackUp(static_cast<int32_t>(pushback_));
  }

  void skip(size_t len) override {
    stream_->SkipInt64(static_cast<int64_t>(len));
    pushback_ = 0;
  }

  size_t byteCount() const override {
    return static_cast<size_t>(stream_->ByteCount());
  }

  void seek(int64_t position) override {
    std::vector<uint64_t> positions{static_cast<uint64_t>(position)};
    dwio::common::PositionProvider provider(positions);
    stream_->seekToPosition(provider);
    pushback_ = 0;
  }

 private:
  std::unique_ptr<dwio::common::SeekableFileInputStream> stream_;
  size_t pushback_ = 0;
};

int128_t readDecimalUnscaledValue(const uint8_t* data, size_t size) {
  if (size == 0) {
    return 0;
  }

  VELOX_CHECK_LE(
      size,
      sizeof(int128_t),
      "Decimal value encoded with {} bytes exceeds supported precision.",
      size);
  uint128_t acc = 0;
  for (size_t i = 0; i < size; ++i) {
    acc = (acc << 8) | static_cast<uint128_t>(data[i]);
  }
  const bool neg = (data[0] & 0x80) != 0;
  if (neg && size < 16) {
    const unsigned missingBits = static_cast<unsigned>(128 - size * 8);
    const uint128_t mask = (~static_cast<uint128_t>(0)) << (128 - missingBits);
    acc |= mask;
  }
  return static_cast<int128_t>(acc);
}

void writeDecimalValue(
    const AvroTypeInfo& info,
    int128_t unscaledValue,
    GenericWriter& writer) {
  VELOX_CHECK(
      DecimalUtil::valueInPrecisionRange<int128_t>(
          unscaledValue, info.decimalPrecision),
      "Decimal value {} exceeds precision {}.",
      unscaledValue,
      static_cast<int32_t>(info.decimalPrecision));

  if (info.veloxType->isShortDecimal()) {
    writer.castTo<int64_t>() = static_cast<int64_t>(unscaledValue);
  } else {
    writer.castTo<int128_t>() = unscaledValue;
  }
}

void writeDecimal(
    const AvroTypeInfo& info,
    const uint8_t* data,
    size_t size,
    GenericWriter& writer) {
  const auto value = readDecimalUnscaledValue(data, size);
  writeDecimalValue(info, value, writer);
}

struct ResolvedDatum {
  const AvroTypeInfo* info;
  const ::avro::GenericDatum* datum;
  std::optional<size_t> unionChildIndex;
};

std::optional<ResolvedDatum> resolveDatum(
    const AvroTypeInfo& info,
    const ::avro::GenericDatum& datum) {
  if (datum.type() == ::avro::Type::AVRO_NULL) {
    VELOX_CHECK(
        info.nullable, "Encountered null value for non-nullable Avro schema.");
    return std::nullopt;
  }

  if (!datum.isUnion()) {
    return ResolvedDatum{&info, &datum, std::nullopt};
  }

  size_t childIndex = datum.unionBranch();
  if (info.nullable) {
    const auto nullIndex = info.nullUnionBranchIndex.value();
    if (childIndex > nullIndex) {
      childIndex -= 1;
    }
  }

  switch (info.unionKind) {
    case AvroUnionKind::kSimple:
      return ResolvedDatum{&*info.children[childIndex], &datum, childIndex};
    case AvroUnionKind::kNumericPromotion:
    case AvroUnionKind::kStruct:
      return ResolvedDatum{&info, &datum, childIndex};
    case AvroUnionKind::kNone:
      VELOX_UNSUPPORTED("Encountered union datum without union schema.");
  }
  VELOX_UNREACHABLE();
}

void writeDatum(const ResolvedDatum& resolved, GenericWriter& writer) {
  const auto& resolvedInfo = *resolved.info;
  const auto& resolvedDatum = *resolved.datum;

  if (resolved.unionChildIndex.has_value()) {
    switch (resolvedInfo.unionKind) {
      case AvroUnionKind::kNumericPromotion: {
        const auto branchType = resolvedDatum.type();
        switch (resolvedInfo.veloxType->kind()) {
          case TypeKind::BIGINT: {
            int64_t value = 0;
            if (branchType == ::avro::Type::AVRO_INT) {
              value = static_cast<int64_t>(resolvedDatum.value<int32_t>());
            } else if (branchType == ::avro::Type::AVRO_LONG) {
              value = resolvedDatum.value<int64_t>();
            } else {
              VELOX_UNSUPPORTED(
                  "Unsupported Avro union branch {} for BIGINT promotion.",
                  static_cast<int>(branchType));
            }
            writer.castTo<int64_t>() = value;
            return;
          }
          case TypeKind::DOUBLE: {
            double value = 0;
            if (branchType == ::avro::Type::AVRO_FLOAT) {
              value = static_cast<double>(resolvedDatum.value<float>());
            } else if (branchType == ::avro::Type::AVRO_DOUBLE) {
              value = resolvedDatum.value<double>();
            } else {
              VELOX_UNSUPPORTED(
                  "Unsupported Avro union branch {} for DOUBLE promotion.",
                  static_cast<int>(branchType));
            }
            writer.castTo<double>() = value;
            return;
          }
          default:
            VELOX_UNSUPPORTED(
                "Unsupported numeric promotion target {}.",
                resolvedInfo.veloxType->toString());
        }
      }
      case AvroUnionKind::kStruct: {
        const auto selectedIndex = resolved.unionChildIndex.value();
        auto& rowWriter = writer.castTo<DynamicRow>();
        for (size_t i = 0; i < resolvedInfo.children.size(); ++i) {
          if (i == selectedIndex) {
            continue;
          }
          rowWriter.set_null_at(static_cast<int32_t>(i));
        }
        auto& childWriter =
            rowWriter.get_writer_at(static_cast<int32_t>(selectedIndex));
        ResolvedDatum childResolved{
            resolvedInfo.children[selectedIndex].get(),
            resolved.datum,
            std::nullopt};
        writeDatum(childResolved, childWriter);
        return;
      }
      case AvroUnionKind::kSimple:
      case AvroUnionKind::kNone:
        break;
    }
  }

  switch (resolvedDatum.type()) {
    case ::avro::Type::AVRO_BOOL:
      writer.castTo<bool>() = resolvedDatum.value<bool>();
      return;

    case ::avro::Type::AVRO_INT: {
      const auto value = resolvedDatum.value<int32_t>();
      switch (resolvedInfo.logicalType) {
        case AvroLogicalType::kDate:
          writer.castTo<int32_t>() = value;
          return;
        case AvroLogicalType::kTimeMillis:
          writer.castTo<int64_t>() = static_cast<int64_t>(value) * 1000;
          return;
        default:
          writer.castTo<int32_t>() = value;
          return;
      }
    }

    case ::avro::Type::AVRO_LONG: {
      const auto value = resolvedDatum.value<int64_t>();
      switch (resolvedInfo.logicalType) {
        case AvroLogicalType::kTimeMicros:
          writer.castTo<int64_t>() = value;
          return;
        case AvroLogicalType::kTimestampMillis:
          writer.castTo<Timestamp>() = Timestamp::fromMillis(value);
          return;
        case AvroLogicalType::kTimestampMicros:
          writer.castTo<Timestamp>() = Timestamp::fromMicros(value);
          return;
        default:
          writer.castTo<int64_t>() = value;
          return;
      }
    }

    case ::avro::Type::AVRO_FLOAT:
      writer.castTo<float>() = resolvedDatum.value<float>();
      return;

    case ::avro::Type::AVRO_DOUBLE:
      writer.castTo<double>() = resolvedDatum.value<double>();
      return;

    case ::avro::Type::AVRO_STRING: {
      auto& value = resolvedDatum.value<std::string>();
      writer.castTo<Varchar>().copy_from(value);
      return;
    }

    case ::avro::Type::AVRO_BYTES: {
      auto& value = resolvedDatum.value<std::vector<uint8_t>>();
      if (resolvedInfo.logicalType == AvroLogicalType::kDecimal) {
        writeDecimal(resolvedInfo, value.data(), value.size(), writer);
      } else {
        writer.castTo<Varbinary>().copy_from(value);
      }
      return;
    }

    case ::avro::Type::AVRO_FIXED: {
      const auto& fixed = resolvedDatum.value<::avro::GenericFixed>().value();
      if (resolvedInfo.logicalType == AvroLogicalType::kDecimal) {
        writeDecimal(resolvedInfo, fixed.data(), fixed.size(), writer);
      } else {
        writer.castTo<Varbinary>().copy_from(fixed);
      }
      return;
    }

    case ::avro::Type::AVRO_ENUM: {
      auto& value = resolvedDatum.value<::avro::GenericEnum>().symbol();
      writer.castTo<Varchar>().copy_from(value);
      return;
    }

    case ::avro::Type::AVRO_ARRAY: {
      auto& arrayWriter = writer.castTo<Array<Any>>();
      const auto& elements =
          resolvedDatum.value<::avro::GenericArray>().value();
      arrayWriter.reserve(static_cast<vector_size_t>(elements.size()));
      for (const auto& element : elements) {
        auto resolvedElement =
            resolveDatum(*resolvedInfo.children.front(), element);
        if (!resolvedElement.has_value()) {
          arrayWriter.add_null();
          continue;
        }

        auto& elementWriter = arrayWriter.add_item();
        writeDatum(*resolvedElement, elementWriter);
      }
      return;
    }

    case ::avro::Type::AVRO_MAP: {
      VELOX_CHECK_EQ(
          resolvedInfo.children.size(),
          1,
          "Avro map expects exactly one value type definition.");
      auto& mapWriter = writer.castTo<Map<Varchar, Any>>();
      const auto& entries = resolvedDatum.value<::avro::GenericMap>().value();
      mapWriter.reserve(static_cast<vector_size_t>(entries.size()));
      for (const auto& [key, valueDatum] : entries) {
        auto resolvedValue =
            resolveDatum(*resolvedInfo.children.front(), valueDatum);
        if (!resolvedValue.has_value()) {
          auto& keyWriter = mapWriter.add_null();
          keyWriter.copy_from(key);
          continue;
        }

        auto [keyWriter, valueWriter] = mapWriter.add_item();
        keyWriter.copy_from(key);
        writeDatum(*resolvedValue, valueWriter);
      }
      return;
    }

    case ::avro::Type::AVRO_RECORD: {
      auto& rowWriter = writer.castTo<DynamicRow>();
      const auto& record = resolvedDatum.value<::avro::GenericRecord>();
      VELOX_CHECK_EQ(
          resolvedInfo.children.size(),
          record.fieldCount(),
          "Mismatch between Avro record fields and schema information.");
      for (size_t i = 0; i < resolvedInfo.children.size(); ++i) {
        auto resolvedField =
            resolveDatum(*resolvedInfo.children[i], record.fieldAt(i));
        if (!resolvedField.has_value()) {
          rowWriter.set_null_at(static_cast<int32_t>(i));
          continue;
        }

        auto& childWriter = rowWriter.get_writer_at(i);
        writeDatum(*resolvedField, childWriter);
      }
      return;
    }

    default:
      VELOX_UNSUPPORTED(
          "Unsupported Avro datum type {}",
          static_cast<int>(resolvedDatum.type()));
  }
}

} // namespace

struct AvroFileContents {
  AvroFileContents(
      std::unique_ptr<BufferedInput> inputIn,
      const ReaderOptions& options,
      ::avro::ValidSchema avroSchemaIn,
      std::shared_ptr<AvroTypeInfo> typeInfoIn,
      std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> readerIn,
      uint64_t avroScanBatchBytesIn)
      : input(std::move(inputIn)),
        pool(options.memoryPool()),
        avroSchema(std::move(avroSchemaIn)),
        typeInfo(std::move(typeInfoIn)),
        avroScanBatchBytes(avroScanBatchBytesIn),
        cachedReader(std::move(readerIn)) {
    readFileInput = input->getInputStream();
    fileLength = readFileInput->getLength();
    rowType = std::dynamic_pointer_cast<const RowType>(typeInfo->veloxType);
    VELOX_CHECK_NOT_NULL(rowType, "Avro root schema must be a record");
    schemaWithId = TypeWithId::create(rowType);
  }

  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> makeReader()
      const {
    auto stream = std::make_unique<ReadFileAvroInputStream>(
        readFileInput, 0, fileLength, pool);
    return std::make_unique<::avro::DataFileReader<::avro::GenericDatum>>(
        std::move(stream));
  }

  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>>
  acquireReader() {
    if (cachedReader) {
      return std::move(cachedReader);
    }
    return makeReader();
  }

  std::unique_ptr<BufferedInput> input;
  memory::MemoryPool& pool;
  ::avro::ValidSchema avroSchema;
  std::shared_ptr<const RowType> rowType;
  std::shared_ptr<const TypeWithId> schemaWithId;
  std::shared_ptr<AvroTypeInfo> typeInfo;
  std::shared_ptr<dwio::common::ReadFileInputStream> readFileInput;
  uint64_t fileLength;
  uint64_t avroScanBatchBytes;
  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> cachedReader;
};

AvroReader::AvroReader(
    std::unique_ptr<BufferedInput> input,
    const ReaderOptions& options) {
  auto readFileInput = input->getInputStream();
  auto length = readFileInput->getLength();
  auto stream = std::make_unique<ReadFileAvroInputStream>(
      readFileInput, 0, length, options.memoryPool());
  auto overrideSchema = loadOverrideSchema(options);
  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> reader;
  ::avro::ValidSchema schema;
  if (overrideSchema.has_value()) {
    reader = std::make_unique<::avro::DataFileReader<::avro::GenericDatum>>(
        std::move(stream), *overrideSchema);
    schema = *overrideSchema;
  } else {
    reader = std::make_unique<::avro::DataFileReader<::avro::GenericDatum>>(
        std::move(stream));
    schema = reader->readerSchema();
  }

  auto typeInfo = buildTypeInfo(schema.root(), options);
  auto avroScanBatchBytes = loadAvroScanBatchBytes(options);
  contents_ = std::make_shared<AvroFileContents>(
      std::move(input),
      options,
      std::move(schema),
      std::move(typeInfo),
      std::move(reader),
      avroScanBatchBytes);
}

std::optional<uint64_t> AvroReader::numberOfRows() const {
  return std::nullopt;
}

std::unique_ptr<dwio::common::ColumnStatistics> AvroReader::columnStatistics(
    uint32_t /*index*/) const {
  return nullptr;
}

const RowTypePtr& AvroReader::rowType() const {
  return contents_->rowType;
}

const std::shared_ptr<const TypeWithId>& AvroReader::typeWithId() const {
  return contents_->schemaWithId;
}

std::unique_ptr<dwio::common::RowReader> AvroReader::createRowReader(
    const dwio::common::RowReaderOptions& options) const {
  return std::make_unique<AvroRowReader>(contents_, options);
}

AvroRowReader::AvroRowReader(
    std::shared_ptr<AvroFileContents> contents,
    const dwio::common::RowReaderOptions& options)
    : contents_(std::move(contents)),
      options_(options),
      currentRow_(0),
      atEnd_(false),
      rowSize_(0),
      estimatedRowVectorSize_(0) {
  reader_ = contents_->acquireReader();
  datum_ = std::make_unique<::avro::GenericDatum>(reader_->readerSchema());
  if (options.limit() >=
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    splitLimit_ = std::nullopt;
  } else {
    splitLimit_ = static_cast<int64_t>(options.limit());
  }
  if (options.offset() > 0) {
    reader_->sync(static_cast<int64_t>(options.offset()));
  }
  uint64_t skip = options_.skipRows();
  while (skip > 0) {
    if (readerPastSync() || !reader_->read(*datum_)) {
      atEnd_ = true;
      break;
    }
    --skip;
    ++currentRow_;
  }
  if (skip > 0) {
    atEnd_ = true;
  }
}

int64_t AvroRowReader::nextRowNumber() {
  return atEnd_ ? kAtEnd : static_cast<int64_t>(currentRow_);
}

int64_t AvroRowReader::nextReadSize(const uint64_t size) {
  if (atEnd_) {
    return kAtEnd;
  }
  const auto rowSize = estimatedRowSize();
  if (!rowSize.has_value()) {
    return static_cast<int64_t>(size);
  }
  const auto rowsByBytes =
      std::max<uint64_t>(1, contents_->avroScanBatchBytes / rowSize.value());

  return static_cast<int64_t>(std::min<uint64_t>(size, rowsByBytes));
}

uint64_t AvroRowReader::next(
    const uint64_t size,
    VectorPtr& result,
    const dwio::common::Mutation* mutation) {
  if (atEnd_ || size == 0) {
    return 0;
  }
  const auto rowsToRead = nextReadSize(size);
  SelectivityVector rows(rowsToRead);
  if (result &&
      (!result->type()->equivalent(*contents_->rowType) ||
       result->size() != size)) {
    result.reset();
  }
  BaseVector::ensureWritable(
      rows, contents_->rowType, &contents_->pool, result);
  auto rowVector = std::static_pointer_cast<RowVector>(result);
  VELOX_CHECK_NOT_NULL(rowVector);
  rowVector->resize(rowsToRead);
  exec::VectorWriter<Any> writer;
  writer.init(*rowVector);

  VELOX_CHECK_EQ(
      contents_->typeInfo->veloxType->kind(),
      TypeKind::ROW,
      "Avro root schema must be a record.");
  const auto* rootInfo = contents_->typeInfo.get();

  vector_size_t numRead = 0;
  while (numRead < rowsToRead) {
    if (readerPastSync() || !reader_->read(*datum_)) {
      atEnd_ = true;
      break;
    }
    writer.setOffset(numRead);
    VELOX_CHECK_EQ(
        datum_->type(),
        ::avro::Type::AVRO_RECORD,
        "Avro root schema must be a record; found type {} instead.",
        static_cast<int>(datum_->type()));

    const ResolvedDatum resolved{rootInfo, datum_.get(), std::nullopt};
    writeDatum(resolved, writer.current());
    writer.commit(true);
    ++numRead;
    ++currentRow_;
  }
  writer.finish();
  rowVector->resize(numRead);
  if (numRead > 0) {
    const auto batchBytes = rowVector->estimateFlatSize();
    rowSize_ += numRead;
    estimatedRowVectorSize_ += batchBytes;
  }
  std::shared_ptr<const common::ScanSpec> scanSpec = options_.scanSpec();
  if (scanSpec) {
    result = projectColumns(rowVector, *scanSpec, mutation);
  } else {
    result = rowVector;
  }
  return numRead;
}

void AvroRowReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& /*stats*/) const {}

void AvroRowReader::resetFilterCaches() {}

std::optional<size_t> AvroRowReader::estimatedRowSize() const {
  if (rowSize_ == 0 || estimatedRowVectorSize_ == 0) {
    return std::nullopt;
  }
  return std::max<size_t>(1, estimatedRowVectorSize_ / rowSize_);
}

bool AvroRowReader::readerPastSync() const {
  return splitLimit_.has_value() && reader_->pastSync(splitLimit_.value());
}

} // namespace facebook::velox::avro
