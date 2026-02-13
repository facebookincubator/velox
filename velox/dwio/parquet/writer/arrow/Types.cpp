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

// Adapted from Apache Arrow.

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "arrow/util/checked_cast.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

using arrow::internal::checked_cast;

namespace facebook::velox::parquet::arrow {

fmt::underlying_t<LogicalType::TimeUnit::Unit> formatAs(
    LogicalType::TimeUnit::Unit unit) {
  return fmt::underlying(unit);
}

bool isCodecSupported(Compression::type codec) {
  switch (codec) {
    case Compression::UNCOMPRESSED:
    case Compression::SNAPPY:
    case Compression::GZIP:
    case Compression::BROTLI:
    case Compression::ZSTD:
    case Compression::LZ4:
    case Compression::LZ4_HADOOP:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<util::Codec> getCodec(Compression::type codec) {
  return getCodec(codec, util::CodecOptions());
}

std::unique_ptr<util::Codec> getCodec(
    Compression::type codec,
    const util::CodecOptions& codecOptions) {
  std::unique_ptr<util::Codec> result;
  if (codec == Compression::LZO) {
    throw ParquetException(
        "While LZO compression is supported by the Parquet format in "
        "general, it is currently not supported by the C++ implementation.");
  }

  if (!isCodecSupported(codec)) {
    std::stringstream ss;
    ss << "Codec type " << util::Codec::getCodecAsString(codec)
       << " not supported in Parquet format";
    throw ParquetException(ss.str());
  }

  PARQUET_ASSIGN_OR_THROW(result, util::Codec::create(codec, codecOptions));
  return result;
}

// Use compression level to create Codec.
std::unique_ptr<util::Codec> getCodec(
    Compression::type codec,
    int compressionLevel) {
  return getCodec(codec, util::CodecOptions{compressionLevel});
}

bool pageCanUseChecksum(PageType::type pageType) {
  switch (pageType) {
    case PageType::type::kDataPage:
    case PageType::type::kDataPageV2:
    case PageType::type::kDictionaryPage:
      return true;
    default:
      return false;
  }
}

std::string formatStatValue(Type::type parquetType, ::std::string_view val) {
  std::stringstream result;

  const char* bytes = val.data();
  switch (parquetType) {
    case Type::kBoolean:
      result << reinterpret_cast<const bool*>(bytes)[0];
      break;
    case Type::kInt32:
      result << reinterpret_cast<const int32_t*>(bytes)[0];
      break;
    case Type::kInt64:
      result << reinterpret_cast<const int64_t*>(bytes)[0];
      break;
    case Type::kDouble:
      result << reinterpret_cast<const double*>(bytes)[0];
      break;
    case Type::kFloat:
      result << reinterpret_cast<const float*>(bytes)[0];
      break;
    case Type::kInt96: {
      auto const i32Val = reinterpret_cast<const int32_t*>(bytes);
      result << i32Val[0] << " " << i32Val[1] << " " << i32Val[2];
      break;
    }
    case Type::kByteArray: {
      return std::string(val);
    }
    case Type::kFixedLenByteArray: {
      return std::string(val);
    }
    case Type::kUndefined:
    default:
      break;
  }
  return result.str();
}

std::string encodingToString(Encoding::type t) {
  switch (t) {
    case Encoding::kPlain:
      return "PLAIN";
    case Encoding::kPlainDictionary:
      return "PLAIN_DICTIONARY";
    case Encoding::kRle:
      return "RLE";
    case Encoding::kBitPacked:
      return "BIT_PACKED";
    case Encoding::kDeltaBinaryPacked:
      return "DELTA_BINARY_PACKED";
    case Encoding::kDeltaLengthByteArray:
      return "DELTA_LENGTH_BYTE_ARRAY";
    case Encoding::kDeltaByteArray:
      return "DELTA_BYTE_ARRAY";
    case Encoding::kRleDictionary:
      return "RLE_DICTIONARY";
    case Encoding::kByteStreamSplit:
      return "BYTE_STREAM_SPLIT";
    default:
      return "UNKNOWN";
  }
}

std::string typeToString(Type::type t) {
  switch (t) {
    case Type::kBoolean:
      return "BOOLEAN";
    case Type::kInt32:
      return "INT32";
    case Type::kInt64:
      return "INT64";
    case Type::kInt96:
      return "INT96";
    case Type::kFloat:
      return "FLOAT";
    case Type::kDouble:
      return "DOUBLE";
    case Type::kByteArray:
      return "BYTE_ARRAY";
    case Type::kFixedLenByteArray:
      return "FIXED_LEN_BYTE_ARRAY";
    case Type::kUndefined:
    default:
      return "UNKNOWN";
  }
}

std::string convertedTypeToString(ConvertedType::type t) {
  switch (t) {
    case ConvertedType::kNone:
      return "NONE";
    case ConvertedType::kUtf8:
      return "UTF8";
    case ConvertedType::kMap:
      return "MAP";
    case ConvertedType::kMapKeyValue:
      return "MAP_KEY_VALUE";
    case ConvertedType::kList:
      return "LIST";
    case ConvertedType::kEnum:
      return "ENUM";
    case ConvertedType::kDecimal:
      return "DECIMAL";
    case ConvertedType::kDate:
      return "DATE";
    case ConvertedType::kTimeMillis:
      return "TIME_MILLIS";
    case ConvertedType::kTimeMicros:
      return "TIME_MICROS";
    case ConvertedType::kTimestampMillis:
      return "TIMESTAMP_MILLIS";
    case ConvertedType::kTimestampMicros:
      return "TIMESTAMP_MICROS";
    case ConvertedType::kUint8:
      return "UINT_8";
    case ConvertedType::kUint16:
      return "UINT_16";
    case ConvertedType::kUint32:
      return "UINT_32";
    case ConvertedType::kUint64:
      return "UINT_64";
    case ConvertedType::kInt8:
      return "INT_8";
    case ConvertedType::kInt16:
      return "INT_16";
    case ConvertedType::kInt32:
      return "INT_32";
    case ConvertedType::kInt64:
      return "INT_64";
    case ConvertedType::kJson:
      return "JSON";
    case ConvertedType::kBson:
      return "BSON";
    case ConvertedType::kInterval:
      return "INTERVAL";
    case ConvertedType::kUndefined:
    default:
      return "UNKNOWN";
  }
}

int getTypeByteSize(Type::type parquetType) {
  switch (parquetType) {
    case Type::kBoolean:
      return TypeTraits<BooleanType::typeNum>::valueByteSize;
    case Type::kInt32:
      return TypeTraits<Int32Type::typeNum>::valueByteSize;
    case Type::kInt64:
      return TypeTraits<Int64Type::typeNum>::valueByteSize;
    case Type::kInt96:
      return TypeTraits<Int96Type::typeNum>::valueByteSize;
    case Type::kDouble:
      return TypeTraits<DoubleType::typeNum>::valueByteSize;
    case Type::kFloat:
      return TypeTraits<FloatType::typeNum>::valueByteSize;
    case Type::kByteArray:
      return TypeTraits<ByteArrayType::typeNum>::valueByteSize;
    case Type::kFixedLenByteArray:
      return TypeTraits<FLBAType::typeNum>::valueByteSize;
    case Type::kUndefined:
    default:
      return 0;
  }
  return 0;
}

// Return the Sort Order of the Parquet Physical Types.
SortOrder::type defaultSortOrder(Type::type primitive) {
  switch (primitive) {
    case Type::kBoolean:
    case Type::kInt32:
    case Type::kInt64:
    case Type::kFloat:
    case Type::kDouble:
      return SortOrder::kSigned;
    case Type::kByteArray:
    case Type::kFixedLenByteArray:
      return SortOrder::kUnsigned;
    case Type::kInt96:
    case Type::kUndefined:
      return SortOrder::kUnknown;
  }
  return SortOrder::kUnknown;
}

// Return the SortOrder of the Parquet Types using Logical or Physical Types.
SortOrder::type getSortOrder(
    ConvertedType::type converted,
    Type::type primitive) {
  if (converted == ConvertedType::kNone)
    return defaultSortOrder(primitive);
  switch (converted) {
    case ConvertedType::kInt8:
    case ConvertedType::kInt16:
    case ConvertedType::kInt32:
    case ConvertedType::kInt64:
    case ConvertedType::kDate:
    case ConvertedType::kTimeMicros:
    case ConvertedType::kTimeMillis:
    case ConvertedType::kTimestampMicros:
    case ConvertedType::kTimestampMillis:
      return SortOrder::kSigned;
    case ConvertedType::kUint8:
    case ConvertedType::kUint16:
    case ConvertedType::kUint32:
    case ConvertedType::kUint64:
    case ConvertedType::kEnum:
    case ConvertedType::kUtf8:
    case ConvertedType::kBson:
    case ConvertedType::kJson:
      return SortOrder::kUnsigned;
    case ConvertedType::kDecimal:
    case ConvertedType::kList:
    case ConvertedType::kMap:
    case ConvertedType::kMapKeyValue:
    case ConvertedType::kInterval:
    case ConvertedType::kNone: // required instead of default
    case ConvertedType::kNa: // required instead of default
    case ConvertedType::kUndefined:
      return SortOrder::kUnknown;
  }
  return SortOrder::kUnknown;
}

SortOrder::type getSortOrder(
    const std::shared_ptr<const LogicalType>& logicalType,
    Type::type primitive) {
  SortOrder::type o = SortOrder::kUnknown;
  if (logicalType && logicalType->isValid()) {
    o =
        (logicalType->isNone() ? defaultSortOrder(primitive)
                               : logicalType->sortOrder());
  }
  return o;
}

ColumnOrder ColumnOrder::undefined_ = ColumnOrder(ColumnOrder::kUndefined);
ColumnOrder ColumnOrder::typeDefined_ =
    ColumnOrder(ColumnOrder::kTypeDefinedOrder);

// Static methods for LogicalType class.

std::shared_ptr<const LogicalType> LogicalType::fromConvertedType(
    const ConvertedType::type convertedType,
    const schema::DecimalMetadata convertedDecimalMetadata) {
  switch (convertedType) {
    case ConvertedType::kUtf8:
      return StringLogicalType::make();
    case ConvertedType::kMapKeyValue:
    case ConvertedType::kMap:
      return MapLogicalType::make();
    case ConvertedType::kList:
      return ListLogicalType::make();
    case ConvertedType::kEnum:
      return EnumLogicalType::make();
    case ConvertedType::kDecimal:
      return DecimalLogicalType::make(
          convertedDecimalMetadata.precision, convertedDecimalMetadata.scale);
    case ConvertedType::kDate:
      return DateLogicalType::make();
    case ConvertedType::kTimeMillis:
      return TimeLogicalType::make(true, LogicalType::TimeUnit::kMillis);
    case ConvertedType::kTimeMicros:
      return TimeLogicalType::make(true, LogicalType::TimeUnit::kMicros);
    case ConvertedType::kTimestampMillis:
      return TimestampLogicalType::make(
          true, LogicalType::TimeUnit::kMillis, true, false);
    case ConvertedType::kTimestampMicros:
      return TimestampLogicalType::make(
          true, LogicalType::TimeUnit::kMicros, true, false);
    case ConvertedType::kInterval:
      return IntervalLogicalType::make();
    case ConvertedType::kInt8:
      return IntLogicalType::make(8, true);
    case ConvertedType::kInt16:
      return IntLogicalType::make(16, true);
    case ConvertedType::kInt32:
      return IntLogicalType::make(32, true);
    case ConvertedType::kInt64:
      return IntLogicalType::make(64, true);
    case ConvertedType::kUint8:
      return IntLogicalType::make(8, false);
    case ConvertedType::kUint16:
      return IntLogicalType::make(16, false);
    case ConvertedType::kUint32:
      return IntLogicalType::make(32, false);
    case ConvertedType::kUint64:
      return IntLogicalType::make(64, false);
    case ConvertedType::kJson:
      return JsonLogicalType::make();
    case ConvertedType::kBson:
      return BsonLogicalType::make();
    case ConvertedType::kNa:
      return NullLogicalType::make();
    case ConvertedType::kNone:
      return NoLogicalType::make();
    case ConvertedType::kUndefined:
      return UndefinedLogicalType::make();
  }
  return UndefinedLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::fromThrift(
    const facebook::velox::parquet::thrift::LogicalType& type) {
  if (type.__isset.STRING) {
    return StringLogicalType::make();
  } else if (type.__isset.MAP) {
    return MapLogicalType::make();
  } else if (type.__isset.LIST) {
    return ListLogicalType::make();
  } else if (type.__isset.ENUM) {
    return EnumLogicalType::make();
  } else if (type.__isset.DECIMAL) {
    return DecimalLogicalType::make(type.DECIMAL.precision, type.DECIMAL.scale);
  } else if (type.__isset.DATE) {
    return DateLogicalType::make();
  } else if (type.__isset.TIME) {
    LogicalType::TimeUnit::Unit unit;
    if (type.TIME.unit.__isset.MILLIS) {
      unit = LogicalType::TimeUnit::kMillis;
    } else if (type.TIME.unit.__isset.MICROS) {
      unit = LogicalType::TimeUnit::kMicros;
    } else if (type.TIME.unit.__isset.NANOS) {
      unit = LogicalType::TimeUnit::kNanos;
    } else {
      unit = LogicalType::TimeUnit::kUnknown;
    }
    return TimeLogicalType::make(type.TIME.isAdjustedToUTC, unit);
  } else if (type.__isset.TIMESTAMP) {
    LogicalType::TimeUnit::Unit unit;
    if (type.TIMESTAMP.unit.__isset.MILLIS) {
      unit = LogicalType::TimeUnit::kMillis;
    } else if (type.TIMESTAMP.unit.__isset.MICROS) {
      unit = LogicalType::TimeUnit::kMicros;
    } else if (type.TIMESTAMP.unit.__isset.NANOS) {
      unit = LogicalType::TimeUnit::kNanos;
    } else {
      unit = LogicalType::TimeUnit::kUnknown;
    }
    return TimestampLogicalType::make(type.TIMESTAMP.isAdjustedToUTC, unit);
    // TODO(tpboudreau): activate the commented code after parquet.thrift
    // recognizes IntervalType as a LogicalType.
    // } else if (type.__isset.INTERVAL) {
    //   return IntervalLogicalType::make();
  } else if (type.__isset.INTEGER) {
    return IntLogicalType::make(
        static_cast<int>(type.INTEGER.bitWidth), type.INTEGER.isSigned);
  } else if (type.__isset.UNKNOWN) {
    return NullLogicalType::make();
  } else if (type.__isset.JSON) {
    return JsonLogicalType::make();
  } else if (type.__isset.BSON) {
    return BsonLogicalType::make();
  } else if (type.__isset.UUID) {
    return UuidLogicalType::make();
  } else {
    throw ParquetException(
        "Metadata contains Thrift LogicalType that is not recognized");
  }
}

std::shared_ptr<const LogicalType> LogicalType::string() {
  return StringLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::map() {
  return MapLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::list() {
  return ListLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::enumType() {
  return EnumLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::decimal(
    int32_t precision,
    int32_t scale) {
  return DecimalLogicalType::make(precision, scale);
}

std::shared_ptr<const LogicalType> LogicalType::date() {
  return DateLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::time(
    bool isAdjustedToUtc,
    LogicalType::TimeUnit::Unit timeUnit) {
  VELOX_DCHECK_NE(
      static_cast<int>(timeUnit),
      static_cast<int>(LogicalType::TimeUnit::kUnknown));
  return TimeLogicalType::make(isAdjustedToUtc, timeUnit);
}

std::shared_ptr<const LogicalType> LogicalType::timestamp(
    bool isAdjustedToUtc,
    LogicalType::TimeUnit::Unit timeUnit,
    bool isFromConvertedType,
    bool forceSetConvertedType) {
  VELOX_DCHECK_NE(
      static_cast<int>(timeUnit),
      static_cast<int>(LogicalType::TimeUnit::kUnknown));
  return TimestampLogicalType::make(
      isAdjustedToUtc, timeUnit, isFromConvertedType, forceSetConvertedType);
}

std::shared_ptr<const LogicalType> LogicalType::interval() {
  return IntervalLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::intType(
    int bitWidth,
    bool isSigned) {
  VELOX_DCHECK(
      bitWidth == 64 || bitWidth == 32 || bitWidth == 16 || bitWidth == 8);
  return IntLogicalType::make(bitWidth, isSigned);
}

std::shared_ptr<const LogicalType> LogicalType::nullType() {
  return NullLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::json() {
  return JsonLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::bson() {
  return BsonLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::uuid() {
  return UuidLogicalType::make();
}

std::shared_ptr<const LogicalType> LogicalType::none() {
  return NoLogicalType::make();
}

/*
 * The logical type implementation classes are built in four layers: (1) the
 * base layer, which establishes the interface and provides generally reusable
 * implementations for the ToJSON() and Equals() methods; (2) an intermediate
 * derived layer for the "compatibility" methods, which provides implementations
 * for is_compatible() and ToConvertedType(); (3) another intermediate layer for
 * the "applicability" methods that provides several implementations for the
 * is_applicable() method; and (4) the final derived classes, one for each
 * logical type, which supply implementations for those methods that remain
 * virtual (usually just ToString() and ToThrift()) or otherwise need to be
 * overridden.
 */

// LogicalTypeImpl base class.

class LogicalType::Impl {
 public:
  virtual bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const = 0;

  virtual bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata = {false, -1, -1})
      const = 0;

  virtual ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const = 0;

  virtual std::string toString() const = 0;

  virtual bool isSerialized() const {
    return !(
        type_ == LogicalType::Type::kNone ||
        type_ == LogicalType::Type::kUndefined);
  }

  virtual std::string toJson() const {
    std::stringstream json;
    json << R"({"Type": ")" << toString() << R"("})";
    return json.str();
  }

  virtual facebook::velox::parquet::thrift::LogicalType toThrift() const {
    // Logical types inheriting this method should never be serialized.
    std::stringstream ss;
    ss << "Logical type " << toString() << " should not be serialized";
    throw ParquetException(ss.str());
  }

  virtual bool equals(const LogicalType& other) const {
    return other.type() == type_;
  }

  LogicalType::Type::type type() const {
    return type_;
  }

  SortOrder::type sortOrder() const {
    return order_;
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  virtual ~Impl() noexcept {}

  class Compatible;
  class SimpleCompatible;
  class Incompatible;

  class Applicable;
  class SimpleApplicable;
  class TypeLengthApplicable;
  class UniversalApplicable;
  class Inapplicable;

  class String;
  class Map;
  class List;
  class Enum;
  class Decimal;
  class Date;
  class Time;
  class Timestamp;
  class Interval;
  class Int;
  class Null;
  class Json;
  class Bson;
  class Uuid;
  class No;
  class Undefined;

 protected:
  Impl(LogicalType::Type::type t, SortOrder::type o) : type_(t), order_(o) {}
  Impl() = default;

 private:
  LogicalType::Type::type type_ = LogicalType::Type::kUndefined;
  SortOrder::type order_ = SortOrder::kUnknown;
};

// Special methods for public LogicalType class.

LogicalType::LogicalType() = default;
LogicalType::~LogicalType() noexcept = default;

// Delegating methods for public LogicalType class.

bool LogicalType::isApplicable(
    parquet::Type::type primitiveType,
    int32_t primitiveLength) const {
  return impl_->isApplicable(primitiveType, primitiveLength);
}

bool LogicalType::isCompatible(
    ConvertedType::type convertedType,
    schema::DecimalMetadata convertedDecimalMetadata) const {
  return impl_->isCompatible(convertedType, convertedDecimalMetadata);
}

ConvertedType::type LogicalType::toConvertedType(
    schema::DecimalMetadata* outDecimalMetadata) const {
  return impl_->toConvertedType(outDecimalMetadata);
}

std::string LogicalType::toString() const {
  return impl_->toString();
}

std::string LogicalType::toJson() const {
  return impl_->toJson();
}

facebook::velox::parquet::thrift::LogicalType LogicalType::toThrift() const {
  return impl_->toThrift();
}

bool LogicalType::equals(const LogicalType& other) const {
  return impl_->equals(other);
}

LogicalType::Type::type LogicalType::type() const {
  return impl_->type();
}

SortOrder::type LogicalType::sortOrder() const {
  return impl_->sortOrder();
}

// Type checks for public LogicalType class.

bool LogicalType::isString() const {
  return impl_->type() == LogicalType::Type::kString;
}
bool LogicalType::isMap() const {
  return impl_->type() == LogicalType::Type::kMap;
}
bool LogicalType::isList() const {
  return impl_->type() == LogicalType::Type::kList;
}
bool LogicalType::isEnum() const {
  return impl_->type() == LogicalType::Type::kEnum;
}
bool LogicalType::isDecimal() const {
  return impl_->type() == LogicalType::Type::kDecimal;
}
bool LogicalType::isDate() const {
  return impl_->type() == LogicalType::Type::kDate;
}
bool LogicalType::isTime() const {
  return impl_->type() == LogicalType::Type::kTime;
}
bool LogicalType::isTimestamp() const {
  return impl_->type() == LogicalType::Type::kTimestamp;
}
bool LogicalType::isInterval() const {
  return impl_->type() == LogicalType::Type::kInterval;
}
bool LogicalType::isInt() const {
  return impl_->type() == LogicalType::Type::kInt;
}
bool LogicalType::isNull() const {
  return impl_->type() == LogicalType::Type::kNil;
}
bool LogicalType::isJson() const {
  return impl_->type() == LogicalType::Type::kJson;
}
bool LogicalType::isBson() const {
  return impl_->type() == LogicalType::Type::kBson;
}
bool LogicalType::isUuid() const {
  return impl_->type() == LogicalType::Type::kUuid;
}
bool LogicalType::isNone() const {
  return impl_->type() == LogicalType::Type::kNone;
}
bool LogicalType::isValid() const {
  return impl_->type() != LogicalType::Type::kUndefined;
}
bool LogicalType::isInvalid() const {
  return !isValid();
}
bool LogicalType::isNested() const {
  return (impl_->type() == LogicalType::Type::kList) ||
      (impl_->type() == LogicalType::Type::kMap);
}
bool LogicalType::isNonnested() const {
  return !isNested();
}
bool LogicalType::isSerialized() const {
  return impl_->isSerialized();
}

// LogicalTypeImpl intermediate "compatibility" classes.

class LogicalType::Impl::Compatible : public virtual LogicalType::Impl {
 protected:
  Compatible() = default;
};

#define setDecimalMetadata(m___, i___, p___, s___) \
  {                                                \
    if (m___) {                                    \
      (m___)->isset = (i___);                      \
      (m___)->scale = (s___);                      \
      (m___)->precision = (p___);                  \
    }                                              \
  }

#define resetDecimalMetadata(m___)           \
  {                                          \
    setDecimalMetadata(m___, false, -1, -1); \
  }

// For logical types that always translate to the same converted type.
class LogicalType::Impl::SimpleCompatible
    : public virtual LogicalType::Impl::Compatible {
 public:
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override {
    return (convertedType == convertedType_) && !convertedDecimalMetadata.isset;
  }

  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override {
    resetDecimalMetadata(outDecimalMetadata);
    return convertedType_;
  }

 protected:
  explicit SimpleCompatible(ConvertedType::type c) : convertedType_(c) {}

 private:
  ConvertedType::type convertedType_ = ConvertedType::kNa;
};

// For logical types that have no corresponding converted type.
class LogicalType::Impl::Incompatible : public virtual LogicalType::Impl {
 public:
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override {
    return (convertedType == ConvertedType::kNone ||
            convertedType == ConvertedType::kNa) &&
        !convertedDecimalMetadata.isset;
  }

  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override {
    resetDecimalMetadata(outDecimalMetadata);
    return ConvertedType::kNone;
  }

 protected:
  Incompatible() = default;
};

// LogicalTypeImpl intermediate "applicability" classes.

class LogicalType::Impl::Applicable : public virtual LogicalType::Impl {
 protected:
  Applicable() = default;
};

// For logical types that can apply only to a single.
// Physical type.
class LogicalType::Impl::SimpleApplicable
    : public virtual LogicalType::Impl::Applicable {
 public:
  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override {
    return primitiveType == type_;
  }

 protected:
  explicit SimpleApplicable(parquet::Type::type t) : type_(t) {}

 private:
  parquet::Type::type type_;
};

// For logical types that can apply only to a particular.
// Physical type and physical length combination.
class LogicalType::Impl::TypeLengthApplicable
    : public virtual LogicalType::Impl::Applicable {
 public:
  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override {
    return primitiveType == type_ && primitiveLength == length_;
  }

 protected:
  TypeLengthApplicable(parquet::Type::type t, int32_t l)
      : type_(t), length_(l) {}

 private:
  parquet::Type::type type_;
  int32_t length_;
};

// For logical types that can apply to any physical type.
class LogicalType::Impl::UniversalApplicable
    : public virtual LogicalType::Impl::Applicable {
 public:
  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override {
    return true;
  }

 protected:
  UniversalApplicable() = default;
};

// For logical types that can never apply to any primitive.
// Physical type.
class LogicalType::Impl::Inapplicable : public virtual LogicalType::Impl {
 public:
  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override {
    return false;
  }

 protected:
  Inapplicable() = default;
};

// LogicalType implementation final classes.

#define OVERRIDE_TOSTRING(n___)           \
  std::string toString() const override { \
    return #n___;                         \
  }

#define OVERRIDE_TOTHRIFT(t___, s___)                                       \
  facebook::velox::parquet::thrift::LogicalType toThrift() const override { \
    facebook::velox::parquet::thrift::LogicalType type;                     \
    facebook::velox::parquet::thrift::t___ subtype;                         \
    type.__set_##s___(subtype);                                             \
    return type;                                                            \
  }

class LogicalType::Impl::String final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class StringLogicalType;

  OVERRIDE_TOSTRING(String)
  OVERRIDE_TOTHRIFT(StringType, STRING)

 private:
  String()
      : LogicalType::Impl(LogicalType::Type::kString, SortOrder::kUnsigned),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kUtf8),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kByteArray) {}
};

// Each public logical type class's Make() creation method instantiates a.
// Corresponding LogicalType::Impl::* object and installs that implementation
// in. The logical type it returns.

#define GENERATE_MAKE(a___)                                      \
  std::shared_ptr<const LogicalType> a___##LogicalType::make() { \
    auto* logicalType = new a___##LogicalType();                 \
    logicalType->impl_.reset(new LogicalType::Impl::a___());     \
    return std::shared_ptr<const LogicalType>(logicalType);      \
  }

GENERATE_MAKE(String)

class LogicalType::Impl::Map final : public LogicalType::Impl::SimpleCompatible,
                                     public LogicalType::Impl::Inapplicable {
 public:
  friend class MapLogicalType;

  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override {
    return (convertedType == ConvertedType::kMap ||
            convertedType == ConvertedType::kMapKeyValue) &&
        !convertedDecimalMetadata.isset;
  }

  OVERRIDE_TOSTRING(Map)
  OVERRIDE_TOTHRIFT(MapType, MAP)

 private:
  Map()
      : LogicalType::Impl(LogicalType::Type::kMap, SortOrder::kUnknown),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kMap) {}
};

GENERATE_MAKE(Map)

class LogicalType::Impl::List final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::Inapplicable {
 public:
  friend class ListLogicalType;

  OVERRIDE_TOSTRING(List)
  OVERRIDE_TOTHRIFT(ListType, LIST)

 private:
  List()
      : LogicalType::Impl(LogicalType::Type::kList, SortOrder::kUnknown),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kList) {}
};

GENERATE_MAKE(List)

class LogicalType::Impl::Enum final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class EnumLogicalType;

  OVERRIDE_TOSTRING(Enum)
  OVERRIDE_TOTHRIFT(EnumType, ENUM)

 private:
  Enum()
      : LogicalType::Impl(LogicalType::Type::kEnum, SortOrder::kUnsigned),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kEnum),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kByteArray) {}
};

GENERATE_MAKE(Enum)

// The parameterized logical types (currently Decimal, Time, Timestamp, and Int)
// Generally can't reuse the simple method implementations available in the
// base. And intermediate classes and must (re)implement them all.

class LogicalType::Impl::Decimal final : public LogicalType::Impl::Compatible,
                                         public LogicalType::Impl::Applicable {
 public:
  friend class DecimalLogicalType;

  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override;
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override;
  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override;
  std::string toString() const override;
  std::string toJson() const override;
  facebook::velox::parquet::thrift::LogicalType toThrift() const override;
  bool equals(const LogicalType& other) const override;

  int32_t precision() const {
    return precision_;
  }
  int32_t scale() const {
    return scale_;
  }

 private:
  Decimal(int32_t p, int32_t s)
      : LogicalType::Impl(LogicalType::Type::kDecimal, SortOrder::kSigned),
        precision_(p),
        scale_(s) {}
  int32_t precision_ = -1;
  int32_t scale_ = -1;
};

bool LogicalType::Impl::Decimal::isApplicable(
    parquet::Type::type primitiveType,
    int32_t primitiveLength) const {
  bool ok = false;
  switch (primitiveType) {
    case parquet::Type::kInt32: {
      ok = (1 <= precision_) && (precision_ <= 9);
    } break;
    case parquet::Type::kInt64: {
      ok = (1 <= precision_) && (precision_ <= 18);
      if (precision_ < 10) {
        // FIXME(tpb): warn that INT32 could be used.
      }
    } break;
    case parquet::Type::kFixedLenByteArray: {
      // If the primitive length is larger than this we will overflow int32
      // when. Calculating precision.
      if (primitiveLength <= 0 || primitiveLength > 891723282) {
        ok = false;
        break;
      }
      ok = precision_ <= static_cast<int32_t>(std::floor(
                             std::log10(2) * ((8.0 * primitiveLength) - 1.0)));
    } break;
    case parquet::Type::kByteArray: {
      ok = true;
    } break;
    default: {
    } break;
  }
  return ok;
}

bool LogicalType::Impl::Decimal::isCompatible(
    ConvertedType::type convertedType,
    schema::DecimalMetadata convertedDecimalMetadata) const {
  return convertedType == ConvertedType::kDecimal &&
      (convertedDecimalMetadata.isset &&
       convertedDecimalMetadata.scale == scale_ &&
       convertedDecimalMetadata.precision == precision_);
}

ConvertedType::type LogicalType::Impl::Decimal::toConvertedType(
    schema::DecimalMetadata* outDecimalMetadata) const {
  setDecimalMetadata(outDecimalMetadata, true, precision_, scale_);
  return ConvertedType::kDecimal;
}

std::string LogicalType::Impl::Decimal::toString() const {
  std::stringstream type;
  type << "Decimal(precision=" << precision_ << ", scale=" << scale_ << ")";
  return type.str();
}

std::string LogicalType::Impl::Decimal::toJson() const {
  std::stringstream json;
  json << R"({"Type": "Decimal", "precision": )" << precision_
       << R"(, "scale": )" << scale_ << "}";
  return json.str();
}

facebook::velox::parquet::thrift::LogicalType
LogicalType::Impl::Decimal::toThrift() const {
  facebook::velox::parquet::thrift::LogicalType type;
  facebook::velox::parquet::thrift::DecimalType decimalType;
  decimalType.__set_precision(precision_);
  decimalType.__set_scale(scale_);
  type.__set_DECIMAL(decimalType);
  return type;
}

bool LogicalType::Impl::Decimal::equals(const LogicalType& other) const {
  bool eq = false;
  if (other.isDecimal()) {
    const auto& otherDecimal = checked_cast<const DecimalLogicalType&>(other);
    eq =
        (precision_ == otherDecimal.precision() &&
         scale_ == otherDecimal.scale());
  }
  return eq;
}

std::shared_ptr<const LogicalType> DecimalLogicalType::make(
    int32_t precision,
    int32_t scale) {
  if (precision < 1) {
    throw ParquetException(
        "Precision must be greater than or equal to 1 for Decimal logical type");
  }
  if (scale < 0 || scale > precision) {
    throw ParquetException(
        "Scale must be a non-negative integer that does not exceed precision for "
        "Decimal logical type");
  }
  auto* logicalType = new DecimalLogicalType();
  logicalType->impl_.reset(new LogicalType::Impl::Decimal(precision, scale));
  return std::shared_ptr<const LogicalType>(logicalType);
}

int32_t DecimalLogicalType::precision() const {
  return (dynamic_cast<const LogicalType::Impl::Decimal&>(*impl_)).precision();
}

int32_t DecimalLogicalType::scale() const {
  return (dynamic_cast<const LogicalType::Impl::Decimal&>(*impl_)).scale();
}

class LogicalType::Impl::Date final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class DateLogicalType;

  OVERRIDE_TOSTRING(Date)
  OVERRIDE_TOTHRIFT(DateType, DATE)

 private:
  Date()
      : LogicalType::Impl(LogicalType::Type::kDate, SortOrder::kSigned),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kDate),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kInt32) {}
};

GENERATE_MAKE(Date)

#define timeUnitString(u___)                                             \
  ((u___) == LogicalType::TimeUnit::kMillis                              \
       ? "milliseconds"                                                  \
       : ((u___) == LogicalType::TimeUnit::kMicros                       \
              ? "microseconds"                                           \
              : ((u___) == LogicalType::TimeUnit::kNanos ? "nanoseconds" \
                                                         : "unknown")))

class LogicalType::Impl::Time final : public LogicalType::Impl::Compatible,
                                      public LogicalType::Impl::Applicable {
 public:
  friend class TimeLogicalType;

  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override;
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override;
  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override;
  std::string toString() const override;
  std::string toJson() const override;
  facebook::velox::parquet::thrift::LogicalType toThrift() const override;
  bool equals(const LogicalType& other) const override;

  bool isAdjustedToUtc() const {
    return adjusted_;
  }
  LogicalType::TimeUnit::Unit timeUnit() const {
    return unit_;
  }

 private:
  Time(bool a, LogicalType::TimeUnit::Unit u)
      : LogicalType::Impl(LogicalType::Type::kTime, SortOrder::kSigned),
        adjusted_(a),
        unit_(u) {}
  bool adjusted_ = false;
  LogicalType::TimeUnit::Unit unit_;
};

bool LogicalType::Impl::Time::isApplicable(
    parquet::Type::type primitiveType,
    int32_t primitiveLength) const {
  return (primitiveType == parquet::Type::kInt32 &&
          unit_ == LogicalType::TimeUnit::kMillis) ||
      (primitiveType == parquet::Type::kInt64 &&
       (unit_ == LogicalType::TimeUnit::kMicros ||
        unit_ == LogicalType::TimeUnit::kNanos));
}

bool LogicalType::Impl::Time::isCompatible(
    ConvertedType::type convertedType,
    schema::DecimalMetadata convertedDecimalMetadata) const {
  if (convertedDecimalMetadata.isset) {
    return false;
  } else if (adjusted_ && unit_ == LogicalType::TimeUnit::kMillis) {
    return convertedType == ConvertedType::kTimeMillis;
  } else if (adjusted_ && unit_ == LogicalType::TimeUnit::kMicros) {
    return convertedType == ConvertedType::kTimeMicros;
  } else {
    return (convertedType == ConvertedType::kNone) ||
        (convertedType == ConvertedType::kNa);
  }
}

ConvertedType::type LogicalType::Impl::Time::toConvertedType(
    schema::DecimalMetadata* outDecimalMetadata) const {
  resetDecimalMetadata(outDecimalMetadata);
  if (adjusted_) {
    if (unit_ == LogicalType::TimeUnit::kMillis) {
      return ConvertedType::kTimeMillis;
    } else if (unit_ == LogicalType::TimeUnit::kMicros) {
      return ConvertedType::kTimeMicros;
    }
  }
  return ConvertedType::kNone;
}

std::string LogicalType::Impl::Time::toString() const {
  std::stringstream type;
  type << "Time(isAdjustedToUTC=" << std::boolalpha << adjusted_
       << ", timeUnit=" << timeUnitString(unit_) << ")";
  return type.str();
}

std::string LogicalType::Impl::Time::toJson() const {
  std::stringstream json;
  json << R"({"Type": "Time", "isAdjustedToUTC": )" << std::boolalpha
       << adjusted_ << R"(, "timeUnit": ")" << timeUnitString(unit_) << R"("})";
  return json.str();
}

facebook::velox::parquet::thrift::LogicalType
LogicalType::Impl::Time::toThrift() const {
  facebook::velox::parquet::thrift::LogicalType type;
  facebook::velox::parquet::thrift::TimeType timeType;
  facebook::velox::parquet::thrift::TimeUnit timeUnit;
  VELOX_DCHECK_NE(
      static_cast<int>(unit_),
      static_cast<int>(LogicalType::TimeUnit::kUnknown));
  if (unit_ == LogicalType::TimeUnit::kMillis) {
    facebook::velox::parquet::thrift::MilliSeconds millis;
    timeUnit.__set_MILLIS(millis);
  } else if (unit_ == LogicalType::TimeUnit::kMicros) {
    facebook::velox::parquet::thrift::MicroSeconds micros;
    timeUnit.__set_MICROS(micros);
  } else if (unit_ == LogicalType::TimeUnit::kNanos) {
    facebook::velox::parquet::thrift::NanoSeconds nanos;
    timeUnit.__set_NANOS(nanos);
  }
  timeType.__set_isAdjustedToUTC(adjusted_);
  timeType.__set_unit(timeUnit);
  type.__set_TIME(timeType);
  return type;
}

bool LogicalType::Impl::Time::equals(const LogicalType& other) const {
  bool eq = false;
  if (other.isTime()) {
    const auto& otherTime = checked_cast<const TimeLogicalType&>(other);
    eq =
        (adjusted_ == otherTime.isAdjustedToUtc() &&
         unit_ == otherTime.timeUnit());
  }
  return eq;
}

std::shared_ptr<const LogicalType> TimeLogicalType::make(
    bool isAdjustedToUtc,
    LogicalType::TimeUnit::Unit timeUnit) {
  if (timeUnit == LogicalType::TimeUnit::kMillis ||
      timeUnit == LogicalType::TimeUnit::kMicros ||
      timeUnit == LogicalType::TimeUnit::kNanos) {
    auto* logicalType = new TimeLogicalType();
    logicalType->impl_.reset(
        new LogicalType::Impl::Time(isAdjustedToUtc, timeUnit));
    return std::shared_ptr<const LogicalType>(logicalType);
  } else {
    throw ParquetException(
        "TimeUnit must be one of MILLIS, MICROS, or NANOS for Time logical type");
  }
}

bool TimeLogicalType::isAdjustedToUtc() const {
  return (dynamic_cast<const LogicalType::Impl::Time&>(*impl_))
      .isAdjustedToUtc();
}

LogicalType::TimeUnit::Unit TimeLogicalType::timeUnit() const {
  return (dynamic_cast<const LogicalType::Impl::Time&>(*impl_)).timeUnit();
}

class LogicalType::Impl::Timestamp final
    : public LogicalType::Impl::Compatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class TimestampLogicalType;

  bool isSerialized() const override;
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override;
  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override;
  std::string toString() const override;
  std::string toJson() const override;
  facebook::velox::parquet::thrift::LogicalType toThrift() const override;
  bool equals(const LogicalType& other) const override;

  bool isAdjustedToUtc() const {
    return adjusted_;
  }
  LogicalType::TimeUnit::Unit timeUnit() const {
    return unit_;
  }

  bool isFromConvertedType() const {
    return isFromConvertedType_;
  }
  bool forceSetConvertedType() const {
    return forceSetConvertedType_;
  }

 private:
  Timestamp(
      bool adjusted,
      LogicalType::TimeUnit::Unit Unit,
      bool isFromConvertedType,
      bool forceSetConvertedType)
      : LogicalType::Impl(LogicalType::Type::kTimestamp, SortOrder::kSigned),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kInt64),
        adjusted_(adjusted),
        unit_(Unit),
        isFromConvertedType_(isFromConvertedType),
        forceSetConvertedType_(forceSetConvertedType) {}
  bool adjusted_ = false;
  LogicalType::TimeUnit::Unit unit_;
  bool isFromConvertedType_ = false;
  bool forceSetConvertedType_ = false;
};

bool LogicalType::Impl::Timestamp::isSerialized() const {
  return !isFromConvertedType_;
}

bool LogicalType::Impl::Timestamp::isCompatible(
    ConvertedType::type convertedType,
    schema::DecimalMetadata convertedDecimalMetadata) const {
  if (convertedDecimalMetadata.isset) {
    return false;
  } else if (unit_ == LogicalType::TimeUnit::kMillis) {
    if (adjusted_ || forceSetConvertedType_) {
      return convertedType == ConvertedType::kTimestampMillis;
    } else {
      return (convertedType == ConvertedType::kNone) ||
          (convertedType == ConvertedType::kNa);
    }
  } else if (unit_ == LogicalType::TimeUnit::kMicros) {
    if (adjusted_ || forceSetConvertedType_) {
      return convertedType == ConvertedType::kTimestampMicros;
    } else {
      return (convertedType == ConvertedType::kNone) ||
          (convertedType == ConvertedType::kNa);
    }
  } else {
    return (convertedType == ConvertedType::kNone) ||
        (convertedType == ConvertedType::kNa);
  }
}

ConvertedType::type LogicalType::Impl::Timestamp::toConvertedType(
    schema::DecimalMetadata* outDecimalMetadata) const {
  resetDecimalMetadata(outDecimalMetadata);
  if (adjusted_ || forceSetConvertedType_) {
    if (unit_ == LogicalType::TimeUnit::kMillis) {
      return ConvertedType::kTimestampMillis;
    } else if (unit_ == LogicalType::TimeUnit::kMicros) {
      return ConvertedType::kTimestampMicros;
    }
  }
  return ConvertedType::kNone;
}

std::string LogicalType::Impl::Timestamp::toString() const {
  std::stringstream type;
  type << "Timestamp(isAdjustedToUTC=" << std::boolalpha << adjusted_
       << ", timeUnit=" << timeUnitString(unit_)
       << ", is_from_converted_type=" << isFromConvertedType_
       << ", force_set_converted_type=" << forceSetConvertedType_ << ")";
  return type.str();
}

std::string LogicalType::Impl::Timestamp::toJson() const {
  std::stringstream json;
  json << R"({"Type": "Timestamp", "isAdjustedToUTC": )" << std::boolalpha
       << adjusted_ << R"(, "timeUnit": ")" << timeUnitString(unit_) << R"(")"
       << R"(, "isFromConvertedType": )" << isFromConvertedType_
       << R"(, "forceSetConvertedType": )" << forceSetConvertedType_ << R"(})";
  return json.str();
}

facebook::velox::parquet::thrift::LogicalType
LogicalType::Impl::Timestamp::toThrift() const {
  facebook::velox::parquet::thrift::LogicalType type;
  facebook::velox::parquet::thrift::TimestampType timestampType;
  facebook::velox::parquet::thrift::TimeUnit timeUnit;
  VELOX_DCHECK_NE(
      static_cast<int>(unit_),
      static_cast<int>(LogicalType::TimeUnit::kUnknown));
  if (unit_ == LogicalType::TimeUnit::kMillis) {
    facebook::velox::parquet::thrift::MilliSeconds millis;
    timeUnit.__set_MILLIS(millis);
  } else if (unit_ == LogicalType::TimeUnit::kMicros) {
    facebook::velox::parquet::thrift::MicroSeconds micros;
    timeUnit.__set_MICROS(micros);
  } else if (unit_ == LogicalType::TimeUnit::kNanos) {
    facebook::velox::parquet::thrift::NanoSeconds nanos;
    timeUnit.__set_NANOS(nanos);
  }
  timestampType.__set_isAdjustedToUTC(adjusted_);
  timestampType.__set_unit(timeUnit);
  type.__set_TIMESTAMP(timestampType);
  return type;
}

bool LogicalType::Impl::Timestamp::equals(const LogicalType& other) const {
  bool eq = false;
  if (other.isTimestamp()) {
    const auto& otherTimestamp =
        checked_cast<const TimestampLogicalType&>(other);
    eq =
        (adjusted_ == otherTimestamp.isAdjustedToUtc() &&
         unit_ == otherTimestamp.timeUnit());
  }
  return eq;
}

std::shared_ptr<const LogicalType> TimestampLogicalType::make(
    bool isAdjustedToUtc,
    LogicalType::TimeUnit::Unit timeUnit,
    bool isFromConvertedType,
    bool forceSetConvertedType) {
  if (timeUnit == LogicalType::TimeUnit::kMillis ||
      timeUnit == LogicalType::TimeUnit::kMicros ||
      timeUnit == LogicalType::TimeUnit::kNanos) {
    auto* logicalType = new TimestampLogicalType();
    logicalType->impl_.reset(new LogicalType::Impl::Timestamp(
        isAdjustedToUtc, timeUnit, isFromConvertedType, forceSetConvertedType));
    return std::shared_ptr<const LogicalType>(logicalType);
  } else {
    throw ParquetException(
        "TimeUnit must be one of MILLIS, MICROS, or NANOS for Timestamp logical type");
  }
}

bool TimestampLogicalType::isAdjustedToUtc() const {
  return (dynamic_cast<const LogicalType::Impl::Timestamp&>(*impl_))
      .isAdjustedToUtc();
}

LogicalType::TimeUnit::Unit TimestampLogicalType::timeUnit() const {
  return (dynamic_cast<const LogicalType::Impl::Timestamp&>(*impl_)).timeUnit();
}

bool TimestampLogicalType::isFromConvertedType() const {
  return (dynamic_cast<const LogicalType::Impl::Timestamp&>(*impl_))
      .isFromConvertedType();
}

bool TimestampLogicalType::forceSetConvertedType() const {
  return (dynamic_cast<const LogicalType::Impl::Timestamp&>(*impl_))
      .forceSetConvertedType();
}

class LogicalType::Impl::Interval final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::TypeLengthApplicable {
 public:
  friend class IntervalLogicalType;

  OVERRIDE_TOSTRING(Interval)
  // TODO(tpboudreau): uncomment the following line to enable serialization.
  // After parquet.thrift recognizes IntervalType as a ConvertedType.
  // OVERRIDE_TOTHRIFT(IntervalType, INTERVAL)

 private:
  Interval()
      : LogicalType::Impl(LogicalType::Type::kInterval, SortOrder::kUnknown),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kInterval),
        LogicalType::Impl::TypeLengthApplicable(
            parquet::Type::kFixedLenByteArray,
            12) {}
};

GENERATE_MAKE(Interval)

class LogicalType::Impl::Int final : public LogicalType::Impl::Compatible,
                                     public LogicalType::Impl::Applicable {
 public:
  friend class IntLogicalType;

  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const override;
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata) const override;
  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const override;
  std::string toString() const override;
  std::string toJson() const override;
  facebook::velox::parquet::thrift::LogicalType toThrift() const override;
  bool equals(const LogicalType& other) const override;

  int bitWidth() const {
    return width_;
  }
  bool isSigned() const {
    return signed_;
  }

 private:
  Int(int width, bool isSigned)
      : LogicalType::Impl(
            LogicalType::Type::kInt,
            (isSigned ? SortOrder::kSigned : SortOrder::kUnsigned)),
        width_(width),
        signed_(isSigned) {}
  int width_ = 0;
  bool signed_ = false;
};

bool LogicalType::Impl::Int::isApplicable(
    parquet::Type::type primitiveType,
    int32_t primitiveLength) const {
  return (primitiveType == parquet::Type::kInt32 && width_ <= 32) ||
      (primitiveType == parquet::Type::kInt64 && width_ == 64);
}

bool LogicalType::Impl::Int::isCompatible(
    ConvertedType::type convertedType,
    schema::DecimalMetadata convertedDecimalMetadata) const {
  if (convertedDecimalMetadata.isset) {
    return false;
  } else if (signed_ && width_ == 8) {
    return convertedType == ConvertedType::kInt8;
  } else if (signed_ && width_ == 16) {
    return convertedType == ConvertedType::kInt16;
  } else if (signed_ && width_ == 32) {
    return convertedType == ConvertedType::kInt32;
  } else if (signed_ && width_ == 64) {
    return convertedType == ConvertedType::kInt64;
  } else if (!signed_ && width_ == 8) {
    return convertedType == ConvertedType::kUint8;
  } else if (!signed_ && width_ == 16) {
    return convertedType == ConvertedType::kUint16;
  } else if (!signed_ && width_ == 32) {
    return convertedType == ConvertedType::kUint32;
  } else if (!signed_ && width_ == 64) {
    return convertedType == ConvertedType::kUint64;
  } else {
    return false;
  }
}

ConvertedType::type LogicalType::Impl::Int::toConvertedType(
    schema::DecimalMetadata* outDecimalMetadata) const {
  resetDecimalMetadata(outDecimalMetadata);
  if (signed_) {
    switch (width_) {
      case 8:
        return ConvertedType::kInt8;
      case 16:
        return ConvertedType::kInt16;
      case 32:
        return ConvertedType::kInt32;
      case 64:
        return ConvertedType::kInt64;
    }
  } else { // unsigned
    switch (width_) {
      case 8:
        return ConvertedType::kUint8;
      case 16:
        return ConvertedType::kUint16;
      case 32:
        return ConvertedType::kUint32;
      case 64:
        return ConvertedType::kUint64;
    }
  }
  return ConvertedType::kNone;
}

std::string LogicalType::Impl::Int::toString() const {
  std::stringstream type;
  type << "Int(bitWidth=" << width_ << ", isSigned=" << std::boolalpha
       << signed_ << ")";
  return type.str();
}

std::string LogicalType::Impl::Int::toJson() const {
  std::stringstream json;
  json << R"({"Type": "int", "bitWidth": )" << width_ << R"(, "isSigned": )"
       << std::boolalpha << signed_ << "}";
  return json.str();
}

facebook::velox::parquet::thrift::LogicalType LogicalType::Impl::Int::toThrift()
    const {
  facebook::velox::parquet::thrift::LogicalType type;
  facebook::velox::parquet::thrift::IntType intType;
  VELOX_DCHECK(width_ == 64 || width_ == 32 || width_ == 16 || width_ == 8);
  intType.__set_bitWidth(static_cast<int8_t>(width_));
  intType.__set_isSigned(signed_);
  type.__set_INTEGER(intType);
  return type;
}

bool LogicalType::Impl::Int::equals(const LogicalType& other) const {
  bool eq = false;
  if (other.isInt()) {
    const auto& otherInt = checked_cast<const IntLogicalType&>(other);
    eq = (width_ == otherInt.bitWidth() && signed_ == otherInt.isSigned());
  }
  return eq;
}

std::shared_ptr<const LogicalType> IntLogicalType::make(
    int bitWidth,
    bool isSigned) {
  if (bitWidth == 8 || bitWidth == 16 || bitWidth == 32 || bitWidth == 64) {
    auto* logicalType = new IntLogicalType();
    logicalType->impl_.reset(new LogicalType::Impl::Int(bitWidth, isSigned));
    return std::shared_ptr<const LogicalType>(logicalType);
  } else {
    throw ParquetException(
        "Bit width must be exactly 8, 16, 32, or 64 for Int logical type");
  }
}

int IntLogicalType::bitWidth() const {
  return (dynamic_cast<const LogicalType::Impl::Int&>(*impl_)).bitWidth();
}

bool IntLogicalType::isSigned() const {
  return (dynamic_cast<const LogicalType::Impl::Int&>(*impl_)).isSigned();
}

class LogicalType::Impl::Null final
    : public LogicalType::Impl::Incompatible,
      public LogicalType::Impl::UniversalApplicable {
 public:
  friend class NullLogicalType;

  OVERRIDE_TOSTRING(Null)
  OVERRIDE_TOTHRIFT(NullType, UNKNOWN)

 private:
  Null() : LogicalType::Impl(LogicalType::Type::kNil, SortOrder::kUnknown) {}
};

GENERATE_MAKE(Null)

class LogicalType::Impl::Json final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class JsonLogicalType;

  OVERRIDE_TOSTRING(JSON)
  OVERRIDE_TOTHRIFT(JsonType, JSON)

 private:
  Json()
      : LogicalType::Impl(LogicalType::Type::kJson, SortOrder::kUnsigned),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kJson),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kByteArray) {}
};

GENERATE_MAKE(Json)

class LogicalType::Impl::Bson final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::SimpleApplicable {
 public:
  friend class BsonLogicalType;

  OVERRIDE_TOSTRING(BSON)
  OVERRIDE_TOTHRIFT(BsonType, BSON)

 private:
  Bson()
      : LogicalType::Impl(LogicalType::Type::kBson, SortOrder::kUnsigned),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kBson),
        LogicalType::Impl::SimpleApplicable(parquet::Type::kByteArray) {}
};

GENERATE_MAKE(Bson)

class LogicalType::Impl::Uuid final
    : public LogicalType::Impl::Incompatible,
      public LogicalType::Impl::TypeLengthApplicable {
 public:
  friend class UuidLogicalType;

  OVERRIDE_TOSTRING(UUID)
  OVERRIDE_TOTHRIFT(UUIDType, UUID)

 private:
  Uuid()
      : LogicalType::Impl(LogicalType::Type::kUuid, SortOrder::kUnsigned),
        LogicalType::Impl::TypeLengthApplicable(
            parquet::Type::kFixedLenByteArray,
            16) {}
};

GENERATE_MAKE(Uuid)

class LogicalType::Impl::No final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::UniversalApplicable {
 public:
  friend class NoLogicalType;

  OVERRIDE_TOSTRING(None)

 private:
  No()
      : LogicalType::Impl(LogicalType::Type::kNone, SortOrder::kUnknown),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kNone) {}
};

GENERATE_MAKE(No)

class LogicalType::Impl::Undefined final
    : public LogicalType::Impl::SimpleCompatible,
      public LogicalType::Impl::UniversalApplicable {
 public:
  friend class UndefinedLogicalType;

  OVERRIDE_TOSTRING(Undefined)

 private:
  Undefined()
      : LogicalType::Impl(LogicalType::Type::kUndefined, SortOrder::kUnknown),
        LogicalType::Impl::SimpleCompatible(ConvertedType::kUndefined) {}
};

GENERATE_MAKE(Undefined)

} // namespace facebook::velox::parquet::arrow
