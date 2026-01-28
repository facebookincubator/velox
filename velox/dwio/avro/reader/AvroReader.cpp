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

#include <boost/algorithm/string.hpp>
#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>
#include <avro/Schema.hh>
#include <avro/Stream.hh>

#include "velox/dwio/common/Options.h"
#include "velox/expression/VectorWriters.h"

namespace facebook::velox::avro {

namespace {
using dwio::common::BufferedInput;
using dwio::common::LogType;
using dwio::common::ReaderOptions;
using dwio::common::TypeWithId;
using exec::GenericWriter;

// ::avro::DataFileReaderBase::pastSync(position) internally evaluates
// `position + SyncSize`. To avoid signed int64 overflow (UB),
// the maximum safe value for `position` is
// std::numeric_limits<int64_t>::max() - SyncSize.
constexpr int64_t kMaxSafeAvroReaderPosition =
    std::numeric_limits<int64_t>::max() - ::avro::SyncSize;

constexpr std::string_view kAvroScanBatchBytesKey = "avro.scan.batch.bytes";
constexpr uint64_t kDefaultAvroScanBatchBytes = 100UL << 20; // 100MB

uint64_t loadAvroScanBatchBytes(const dwio::common::RowReaderOptions& options) {
  const auto& params = options.serdeParameters();
  const auto it = params.find(std::string(kAvroScanBatchBytesKey));
  if (it == params.end() || it->second.empty()) {
    return kDefaultAvroScanBatchBytes;
  }

  uint64_t bytes = 0;
  try {
    bytes = folly::to<uint64_t>(it->second);
  } catch (const folly::ConversionError& e) {
    VELOX_USER_FAIL(
        "Invalid value for '{}': '{}'. Details: {}.",
        kAvroScanBatchBytesKey,
        it->second,
        e.what());
  }
  VELOX_USER_CHECK_GT(
      bytes,
      0,
      "Invalid value for '{}': '{}'. Expected a positive integer number of bytes.",
      kAvroScanBatchBytesKey);
  return bytes;
}

enum class AvroLogicalType {
  kNone,
  kDate,
  kTimeMillis,
  kTimeMicros,
  kTimestampMillis,
  kTimestampMicros,
  kTimestampNanos,
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

/// See the Avro schema specification for details:
/// https://avro.apache.org/docs/1.12.0/specification/
///
/// Avro schema to Velox type mapping rules.
///
/// Primitive types (left: Avro physical type, Avro logical type is NONE):
///   null        -> UNKNOWN() (marks nullable)
///   boolean     -> BOOLEAN()
///   int         -> INTEGER()
///   long        -> BIGINT()
///   float       -> REAL()
///   double      -> DOUBLE()
///   string      -> VARCHAR()
///   bytes       -> VARBINARY()
///
/// Complex types (left: Avro physical type, Avro logical type is NONE):
///   enum           -> VARCHAR()
///   fixed          -> VARBINARY()
///   array<T>       -> ARRAY(veloxType(T))
///   map<string, V> -> MAP(VARCHAR(), veloxType(V))
///                       Avro spec mandates string keys
///   record{f1..fn} -> ROW(fieldNames, fieldTypes)
///   union          -> see "Union handling" below
///
/// Logical types (left: Avro physical type + logical type):
///   int  + date                    -> DATE()
///   int  + time-millis             -> TIME()
///   long + time-micros             -> TIME()
///   long + timestamp-millis        -> TIMESTAMP()
///   long + timestamp-micros        -> TIMESTAMP()
///   long + timestamp-nanos         -> TIMESTAMP()
///   string + uuid                  -> VARCHAR()
///   fixed  + uuid                  -> VARBINARY()
///   bytes/fixed + decimal          -> DECIMAL(precision, scale)
///   long + local-timestamp-millis  -> (no corresponding Velox type)
///   long + local-timestamp-micros  -> (no corresponding Velox type)
///   long + local-timestamp-nanos   -> (no corresponding Velox type)
///   fixed(size=12) + duration      -> (no corresponding Velox type)
///
/// Union handling:
///   - union with "null"          -> nullable = true
///   - single non-null branch     -> passthrough
///   - unions of {int, long}      -> BIGINT() (numeric promotion)
///   - unions of {float, double}  -> DOUBLE() (numeric promotion)
///   - other multi-branch unions
///       -> ROW(member0..memberN,
///              veloxType(branch0)..veloxType(branchN))
///          Field names are auto-generated as "member<i>" by branch index.
///
/// Notes:
///   - Symbolic nodes are resolved via avro::resolveSymbol() before mapping.
///   - Record field names can be lower-cased via
///     options.fileColumnNamesReadAsLowerCase(), duplicated field names
///     are rejected.
///   - Unsupported types fall through to VELOX_UNSUPPORTED().
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
  const auto precision = static_cast<uint8_t>(precisionValue);
  const auto scale = static_cast<uint8_t>(scaleValue);

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
  VELOX_CHECK(
      node->leaves() > 0,
      "Invalid Avro union: a union must contain at least one schema branch.");

  auto info = std::make_shared<AvroTypeInfo>();
  std::vector<std::shared_ptr<AvroTypeInfo>> nonNullInfos;
  nonNullInfos.reserve(node->leaves());
  bool allIntsOrLongs = true;
  bool allFloatsOrDoubles = true;

  for (size_t i = 0; i < node->leaves(); ++i) {
    const auto branchNode = resolveIfSymbolic(node->leafAt(i));
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

  // For a union schema of ["null"],
  // NULL represents the type itself, not nullability.
  if (nonNullInfos.empty()) {
    nonNullInfos.push_back(buildTypeInfo(node->leafAt(0), options));
  }

  info->children = std::move(nonNullInfos);

  if (info->children.size() == 1) {
    auto childInfo = info->children.front();
    childInfo->unionKind = AvroUnionKind::kSimple;
    childInfo->nullUnionBranchIndex = info->nullUnionBranchIndex;
    childInfo->nullable = info->nullable;
    return childInfo;
  }

  // info->children.size() > 1
  if (allIntsOrLongs) {
    info->unionKind = AvroUnionKind::kNumericPromotion;
    info->veloxType = BIGINT();
    info->children.clear();
    return info;
  }
  if (allFloatsOrDoubles) {
    info->unionKind = AvroUnionKind::kNumericPromotion;
    info->veloxType = DOUBLE();
    info->children.clear();
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
      } else if (logicalType == ::avro::LogicalType::Type::TIMESTAMP_NANOS) {
        info->logicalType = AvroLogicalType::kTimestampNanos;
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
      } else if (logicalType == ::avro::LogicalType::Type::UUID) {
        info->logicalType = AvroLogicalType::kUuid;
        info->veloxType = VARBINARY();
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
      // Avro spec mandates that map keys are strings
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
          "Unsupported Avro type (enum value = {}). "
          "Please refer to avro::Type in "
          "https://github.com/apache/avro/blob/main/lang/c%2B%2B/include/avro/Types.hh "
          "to find the corresponding type.",
          static_cast<int>(resolvedNode->type()));
  }
  if (logicalType != ::avro::LogicalType::Type::NONE &&
      info->logicalType == AvroLogicalType::kNone) {
    VELOX_UNSUPPORTED(
        "Unsupported Avro logical type (enum value = {}). "
        "Please refer to avro::LogicalType::Type in "
        "https://github.com/apache/avro/blob/main/lang/c%2B%2B/include/avro/LogicalType.hh "
        "to find the corresponding logical type.",
        static_cast<int>(logicalType));
  }
  return info;
}

// Adapts dwio::common::ReadFileInputStream to avro-cpp's
// ::avro::SeekableInputStream, with support for multiple backup() calls
// after a single next().
class ReadFileAvroInputStream : public ::avro::SeekableInputStream {
 public:
  ReadFileAvroInputStream(
      std::shared_ptr<dwio::common::ReadFileInputStream> input,
      uint64_t start,
      uint64_t length,
      memory::MemoryPool& pool) {
    stream_ = std::make_unique<dwio::common::SeekableFileInputStream>(
        std::move(input),
        start,
        length,
        pool,
        LogType::FILE,
        input->getNaturalReadSize());
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
    const std::vector<uint64_t> positions{static_cast<uint64_t>(position)};
    dwio::common::PositionProvider provider(positions);
    stream_->seekToPosition(provider);
    pushback_ = 0;
  }

 private:
  std::unique_ptr<dwio::common::SeekableFileInputStream> stream_;
  size_t pushback_ = 0;
};

struct ResolvedDatum {
  const AvroTypeInfo* info;
  const ::avro::GenericDatum* datum;
  std::optional<size_t> unionChildIndex;
};

void writeDecimal(
    const AvroTypeInfo& info,
    const uint8_t* data,
    size_t size,
    GenericWriter& writer) {
  // Decode Avro decimal bytes (big-endian, two's complement) into int128.
  int128_t unscaledValue = 0;
  if (size != 0) {
    VELOX_CHECK(
        size <= sizeof(int128_t),
        "Decimal value encoded with {} bytes exceeds supported precision.",
        size);

    uint128_t acc = 0;
    for (size_t i = 0; i < size; ++i) {
      acc = (acc << 8) | static_cast<uint128_t>(data[i]);
    }

    // Sign-extend negative values when encoded with fewer than 16 bytes.
    if ((data[0] & 0x80) != 0 && size < sizeof(int128_t)) {
      const uint32_t missingBits = static_cast<uint32_t>(128 - size * 8);
      acc |= (~static_cast<uint128_t>(0)) << (128 - missingBits);
    }

    unscaledValue = static_cast<int128_t>(acc);
  }

  VELOX_CHECK(
      DecimalUtil::valueInPrecisionRange<int128_t>(
          unscaledValue, info.decimalPrecision),
      "Decimal value {} exceeds precision {}.",
      unscaledValue,
      static_cast<int32_t>(info.decimalPrecision));

  // Write using the physical width required by the Velox decimal type.
  if (info.veloxType->isShortDecimal()) {
    writer.castTo<int64_t>() = static_cast<int64_t>(unscaledValue);
  } else {
    writer.castTo<int128_t>() = unscaledValue;
  }
}

std::optional<ResolvedDatum> resolveUnionAndNull(
    const AvroTypeInfo& info,
    const ::avro::GenericDatum& datum) {
  // For unions, datum.type() reflects the active branch,
  // matching both AVRO_NULL values and unions with NULL as the selected branch.
  if (datum.type() == ::avro::Type::AVRO_NULL) {
    VELOX_CHECK(
        info.nullable, "Encountered null value for non-nullable Avro schema.");
    return std::nullopt;
  }

  if (info.unionKind == AvroUnionKind::kNone) {
    // already validated during avro->velox schema mapping.
    VELOX_CHECK(
        !datum.isUnion(), "Encountered union datum without union schema.");
    return ResolvedDatum{&info, &datum, std::nullopt};
  }

  // In AvroTypeInfo, the AVRO_NULL union branch represents nullability
  // rather than a child type, so we remap the union branch index
  // accordingly.
  size_t childIndex = datum.unionBranch();
  if (info.nullable) {
    const auto nullIndex = info.nullUnionBranchIndex.value();
    if (childIndex > nullIndex) {
      childIndex -= 1;
    }
  }
  return ResolvedDatum{&info, &datum, childIndex};
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
              // Unreachable: already validated during avro->velox schema
              // mapping.
              VELOX_UNREACHABLE(
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
              // Unreachable: already validated during avro->velox schema
              // mapping.
              VELOX_UNREACHABLE(
                  "Unsupported Avro union branch {} for DOUBLE promotion.",
                  static_cast<int>(branchType));
            }
            writer.castTo<double>() = value;
            return;
          }
          default:
            // Unreachable: already validated during avro->velox schema mapping.
            VELOX_UNREACHABLE(
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
      default:
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
        case AvroLogicalType::kTimestampNanos:
          writer.castTo<Timestamp>() = Timestamp::fromNanos(value);
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
            resolveUnionAndNull(*resolvedInfo.children.front(), element);
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
            resolveUnionAndNull(*resolvedInfo.children.front(), valueDatum);
        if (!resolvedValue.has_value()) {
          auto& keyWriter = mapWriter.add_null();
          keyWriter.copy_from(key);
          continue;
        }

        auto&& [keyWriter, valueWriter] = mapWriter.add_item();
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
            resolveUnionAndNull(*resolvedInfo.children[i], record.fieldAt(i));
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
      // Unreachable: already validated during avro->velox schema mapping.
      VELOX_UNREACHABLE(
          "Unsupported Avro datum type reached at runtime (enum value = {}).",
          static_cast<int>(resolvedDatum.type()));
  }
}
} // namespace

struct AvroFileContents {
  AvroFileContents(
      std::shared_ptr<AvroTypeInfo> typeInfoIn,
      std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> readerIn,
      std::shared_ptr<const RowType> rowTypeIn,
      std::shared_ptr<const TypeWithId> schemaWithIdIn,
      memory::MemoryPool& poolIn)
      : typeInfo(std::move(typeInfoIn)),
        avroReader(std::move(readerIn)),
        rowType(std::move(rowTypeIn)),
        schemaWithId(std::move(schemaWithIdIn)),
        pool(poolIn) {}

  std::shared_ptr<AvroTypeInfo> typeInfo;
  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> avroReader;
  std::shared_ptr<const RowType> rowType;
  std::shared_ptr<const TypeWithId> schemaWithId;
  memory::MemoryPool& pool;
};

AvroReader::AvroReader(
    const std::unique_ptr<BufferedInput>& input,
    const ReaderOptions& options) {
  auto readFileInput = input->getInputStream();
  auto length = readFileInput->getLength();
  auto stream = std::make_unique<ReadFileAvroInputStream>(
      readFileInput, 0, length, options.memoryPool());

  // Reader schema precedence: user-configured `avro.schema.literal`,
  // otherwise the schema embedded in the Avro file.
  ::avro::ValidSchema avroSchema;
  std::unique_ptr<::avro::DataFileReader<::avro::GenericDatum>> avroReader;
  if (options.serDeOptions().avroSchema.has_value()) {
    std::istringstream schemaStream(options.serDeOptions().avroSchema.value());
    try {
      ::avro::compileJsonSchema(schemaStream, avroSchema);
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed to parse Avro schema override from '{}': {}",
          options.serDeOptions().kAvroSchema,
          e.what());
    }
    avroReader = std::make_unique<::avro::DataFileReader<::avro::GenericDatum>>(
        std::move(stream), avroSchema);
  } else {
    avroReader = std::make_unique<::avro::DataFileReader<::avro::GenericDatum>>(
        std::move(stream));
    avroSchema = avroReader->readerSchema();
  }

  auto avroSchemaRoot = avroSchema.root();
  VELOX_CHECK_EQ(
      avroSchemaRoot->type(),
      ::avro::Type::AVRO_RECORD,
      "Avro root schema must be of type RECORD, but got Avro type enum value {}. "
      "Please refer to avro::Type in "
      "https://github.com/apache/avro/blob/main/lang/c%2B%2B/include/avro/Types.hh "
      "to find the corresponding type.",
      static_cast<int>(avroSchemaRoot->type()));
  auto typeInfo = buildTypeInfo(avroSchemaRoot, options);
  auto rowType = std::static_pointer_cast<const RowType>(typeInfo->veloxType);
  std::shared_ptr<const TypeWithId> schemaWithId = TypeWithId::create(rowType);

  contents_ = std::make_shared<AvroFileContents>(
      std::move(typeInfo),
      std::move(avroReader),
      std::move(rowType),
      std::move(schemaWithId),
      options.memoryPool());
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
      reader_(std::move(contents_->avroReader)),
      datum_(std::make_unique<::avro::GenericDatum>(reader_->readerSchema())),
      splitLimit_(
          options.limit() >= static_cast<uint64_t>(kMaxSafeAvroReaderPosition)
              ? kMaxSafeAvroReaderPosition
              : static_cast<int64_t>(options.limit())),
      avroScanBatchBytes_(loadAvroScanBatchBytes(options)),
      options_(options),
      atEnd_(false),
      rowSize_(0),
      estimatedRowVectorSize_(0) {
  if (options.offset() > 0) {
    reader_->sync(static_cast<int64_t>(options.offset()));
  }
  uint64_t skip = options.skipRows();
  while (skip > 0) {
    if (reader_->pastSync(splitLimit_) || !reader_->read(*datum_)) {
      atEnd_ = true;
      break;
    }
    --skip;
  }
  if (skip > 0) {
    atEnd_ = true;
  }
}

int64_t AvroRowReader::nextRowNumber() {
  return atEnd_ ? kAtEnd : static_cast<int64_t>(rowSize_);
}

std::optional<size_t> AvroRowReader::estimatedRowSize() const {
  if (rowSize_ == 0 || estimatedRowVectorSize_ == 0) {
    return std::nullopt;
  }
  return std::max<size_t>(1, estimatedRowVectorSize_ / rowSize_);
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
      std::max<uint64_t>(1, avroScanBatchBytes_ / rowSize.value());

  return static_cast<int64_t>(std::min<uint64_t>(size, rowsByBytes));
}

void AvroRowReader::updateRuntimeStats(
    dwio::common::RuntimeStatistics& /*stats*/) const {}

void AvroRowReader::resetFilterCaches() {}

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
  rowVector->resize(rowsToRead);
  exec::VectorWriter<Any> writer;
  writer.init(*rowVector);
  const auto* rootInfo = contents_->typeInfo.get();

  vector_size_t numRead = 0;
  while (numRead < rowsToRead) {
    if (reader_->pastSync(splitLimit_) || !reader_->read(*datum_)) {
      atEnd_ = true;
      break;
    }
    writer.setOffset(numRead);

    const ResolvedDatum resolved{rootInfo, datum_.get(), std::nullopt};
    writeDatum(resolved, writer.current());
    writer.commit(true);
    ++numRead;
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

} // namespace facebook::velox::avro
