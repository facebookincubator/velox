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

// Adapted from Apache Arrow.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/util/Compression.h"

namespace facebook::velox::parquet::arrow::util {

class CodecOptions;
class Codec;

} // namespace facebook::velox::parquet::arrow::util

namespace facebook::velox::parquet::arrow {

// ----------------------------------------------------------------------.
// Metadata enums to match Thrift metadata.
//
// The reason we maintain our own enums is to avoid transitive dependency on
// the compiled Thrift headers (and thus thrift/Thrift.h) for users of the
// public API. After building parquet-cpp, you should not need to include
// Thrift headers in your application. This means some boilerplate to convert
// between our types and Parquet's Thrift types.
//
// We can also add special values like NONE to distinguish between metadata
// values being set and not set. As an example consider ConvertedType and
// CompressionCodec.

// Mirrors parquet::Type.
struct Type {
  enum type {
    kBoolean = 0,
    kInt32 = 1,
    kInt64 = 2,
    kInt96 = 3,
    kFloat = 4,
    kDouble = 5,
    kByteArray = 6,
    kFixedLenByteArray = 7,
    // Should always be last element.
    kUndefined = 8
  };
};

// TODO: For compatibility for now.
namespace parquet {
using Type = facebook::velox::parquet::arrow::Type;
}

// Mirrors parquet::ConvertedType.
struct ConvertedType {
  enum type {
    kNone, // Not a real converted type, but means no converted type is
           // specified
    kUtf8,
    kMap,
    kMapKeyValue,
    kList,
    kEnum,
    kDecimal,
    kDate,
    kTimeMillis,
    kTimeMicros,
    kTimestampMillis,
    kTimestampMicros,
    kUint8,
    kUint16,
    kUint32,
    kUint64,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kJson,
    kBson,
    kInterval,
    // DEPRECATED INVALID ConvertedType for all-null data.
    // Only useful for reading legacy files written out by interim Parquet
    // C++ releases. For writing, always emit LogicalType::nullType instead.
    // See PARQUET-1990.
    kNa = 25,
    kUndefined = 26 // Not a real converted type; should always be last element
  };
};

// Forward declaration.
namespace format {

class LogicalType;

}

// Mirrors parquet::FieldRepetitionType.
struct Repetition {
  enum type {
    kRequired = 0,
    kOptional = 1,
    kRepeated = 2,
    /*Always last*/ kUndefined = 3
  };
};

// Reference:
// Parquet-mr/parquet-hadoop/src/main/java/org/apache/parquet/.
//                            Format/converter/ParquetMetadataConverter.java.
// Sort order for page and column statistics. Types are associated with sort.
// Orders (e.g., UTF8 columns should use UNSIGNED) and column stats are.
// Aggregated using a sort order. As of parquet-format version 2.3.1, the.
// Order used to aggregate stats is always SIGNED and is not stored in the.
// Parquet file. These stats are discarded for types that need unsigned.
// See PARQUET-686.
struct SortOrder {
  enum type { kSigned, kUnsigned, kUnknown };
};

namespace schema {

struct DecimalMetadata {
  bool isset;
  int32_t scale;
  int32_t precision;
};

} // namespace schema

/// \brief Implementation of parquet.thrift LogicalType types.
class PARQUET_EXPORT LogicalType {
 public:
  struct Type {
    enum type {
      kUndefined = 0, // Not a real logical type
      kString = 1,
      kMap,
      kList,
      kEnum,
      kDecimal,
      kDate,
      kTime,
      kTimestamp,
      kInterval,
      kInt,
      kNil, // Thrift NullType: annotates data that is always null
      kJson,
      kBson,
      kUuid,
      kNone // Not a real logical type; should always be last element
    };
  };

  struct TimeUnit {
    enum Unit { kUnknown = 0, kMillis = 1, kMicros, kNanos };
  };

  /// \brief If possible, return a logical type equivalent to the given legacy
  /// converted type (and decimal metadata if applicable).
  static std::shared_ptr<const LogicalType> fromConvertedType(
      const ConvertedType::type convertedType,
      const schema::DecimalMetadata convertedDecimalMetadata = {false, -1, -1});

  /// \brief Return the logical type represented by the Thrift intermediary
  /// object.
  static std::shared_ptr<const LogicalType> fromThrift(
      const facebook::velox::parquet::thrift::LogicalType& thriftLogicalType);

  /// \brief Return the explicitly requested logical type.
  static std::shared_ptr<const LogicalType> string();
  static std::shared_ptr<const LogicalType> map();
  static std::shared_ptr<const LogicalType> list();
  static std::shared_ptr<const LogicalType> enumType();
  static std::shared_ptr<const LogicalType> decimal(
      int32_t precision,
      int32_t scale = 0);
  static std::shared_ptr<const LogicalType> date();
  static std::shared_ptr<const LogicalType> time(
      bool isAdjustedToUtc,
      LogicalType::TimeUnit::Unit timeUnit);

  /// \brief Create a Timestamp logical type.
  /// \param[in] is_adjusted_to_utc Set true if the data is UTC-normalized.
  /// \param[in] time_unit The resolution of the timestamp.
  /// \param[in] is_from_converted_type If true, the timestamp was generated.
  /// By translating a legacy converted type of TIMESTAMP_MILLIS or
  /// TIMESTAMP_MICROS. Default is false.
  /// \param[in] force_set_converted_type If true, always set the
  /// legacy ConvertedType TIMESTAMP_MICROS and TIMESTAMP_MILLIS.
  /// metadata. Default is false.
  static std::shared_ptr<const LogicalType> timestamp(
      bool isAdjustedToUtc,
      LogicalType::TimeUnit::Unit timeUnit,
      bool isFromConvertedType = false,
      bool forceSetConvertedType = false);

  static std::shared_ptr<const LogicalType> interval();
  static std::shared_ptr<const LogicalType> intType(
      int bitWidth,
      bool isSigned);

  /// \brief Create a logical type for data that's always null.
  ///
  /// Any physical type can be annotated with this logical type.
  static std::shared_ptr<const LogicalType> nullType();

  static std::shared_ptr<const LogicalType> json();
  static std::shared_ptr<const LogicalType> bson();
  static std::shared_ptr<const LogicalType> uuid();

  /// \brief Create a placeholder for when no logical type is specified.
  static std::shared_ptr<const LogicalType> none();

  /// \brief Return true if this logical type is consistent with the given.
  /// Underlying physical type.
  bool isApplicable(
      parquet::Type::type primitiveType,
      int32_t primitiveLength = -1) const;

  /// \brief Return true if this logical type is equivalent to the given legacy
  /// converted type (and decimal metadata if applicable).
  bool isCompatible(
      ConvertedType::type convertedType,
      schema::DecimalMetadata convertedDecimalMetadata = {false, -1, -1}) const;

  /// \brief If possible, return the legacy converted type (and decimal
  /// metadata if applicable) equivalent to this logical type.
  ConvertedType::type toConvertedType(
      schema::DecimalMetadata* outDecimalMetadata) const;

  /// \brief Return a printable representation of this logical type.
  std::string toString() const;

  /// \brief Return a JSON representation of this logical type.
  std::string toJson() const;

  /// \brief Return a serializable Thrift object for this logical type.
  facebook::velox::parquet::thrift::LogicalType toThrift() const;

  /// \brief Return true if the given logical type is equivalent to this
  /// logical type.
  bool equals(const LogicalType& other) const;

  /// \brief Return the enumerated type of this logical type.
  LogicalType::Type::type type() const;

  /// \brief Return the appropriate sort order for this logical type.
  SortOrder::type sortOrder() const;

  // Type checks ...
  bool isString() const;
  bool isMap() const;
  bool isList() const;
  bool isEnum() const;
  bool isDecimal() const;
  bool isDate() const;
  bool isTime() const;
  bool isTimestamp() const;
  bool isInterval() const;
  bool isInt() const;
  bool isNull() const;
  bool isJson() const;
  bool isBson() const;
  bool isUuid() const;
  bool isNone() const;
  /// \brief Return true if this logical type is of a known type.
  bool isValid() const;
  bool isInvalid() const;
  /// \brief Return true if this logical type is suitable for a schema.
  /// GroupNode.
  bool isNested() const;
  bool isNonnested() const;
  /// \brief Return true if this logical type is included in the Thrift output.
  /// For its node.
  bool isSerialized() const;

  LogicalType(const LogicalType&) = delete;
  LogicalType& operator=(const LogicalType&) = delete;
  virtual ~LogicalType() noexcept;

 protected:
  LogicalType();

  class Impl;
  std::unique_ptr<const Impl> impl_;
};

/// \brief Allowed for physical type BYTE_ARRAY, must be encoded as UTF-8.
class PARQUET_EXPORT StringLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  StringLogicalType() = default;
};

/// \brief Allowed for group nodes only.
class PARQUET_EXPORT MapLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  MapLogicalType() = default;
};

/// \brief Allowed for group nodes only.
class PARQUET_EXPORT ListLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  ListLogicalType() = default;
};

/// \brief Allowed for physical type BYTE_ARRAY, must be encoded as UTF-8.
class PARQUET_EXPORT EnumLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  EnumLogicalType() = default;
};

/// \brief Allowed for physical type INT32, INT64, FIXED_LEN_BYTE_ARRAY, or.
/// BYTE_ARRAY, depending on the precision.
class PARQUET_EXPORT DecimalLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make(
      int32_t precision,
      int32_t scale = 0);
  int32_t precision() const;
  int32_t scale() const;

 private:
  DecimalLogicalType() = default;
};

/// \brief Allowed for physical type INT32.
class PARQUET_EXPORT DateLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  DateLogicalType() = default;
};

/// \brief Allowed for physical type INT32 (for MILLIS) or INT64 (for MICROS
/// and NANOS).
class PARQUET_EXPORT TimeLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make(
      bool isAdjustedToUtc,
      LogicalType::TimeUnit::Unit timeUnit);
  bool isAdjustedToUtc() const;
  LogicalType::TimeUnit::Unit timeUnit() const;

 private:
  TimeLogicalType() = default;
};

/// \brief Allowed for physical type INT64.
class PARQUET_EXPORT TimestampLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make(
      bool isAdjustedToUtc,
      LogicalType::TimeUnit::Unit timeUnit,
      bool isFromConvertedType = false,
      bool forceSetConvertedType = false);
  bool isAdjustedToUtc() const;
  LogicalType::TimeUnit::Unit timeUnit() const;

  /// \brief If true, will not set LogicalType in Thrift metadata.
  bool isFromConvertedType() const;

  /// \brief If true, will set ConvertedType for micros and millis.
  /// Resolution in legacy ConvertedType Thrift metadata.
  bool forceSetConvertedType() const;

 private:
  TimestampLogicalType() = default;
};

/// \brief Allowed for physical type FIXED_LEN_BYTE_ARRAY with length 12.
class PARQUET_EXPORT IntervalLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  IntervalLogicalType() = default;
};

/// \brief Allowed for physical type INT32 (for bit widths 8, 16, and 32) and.
/// INT64 (for bit width 64).
class PARQUET_EXPORT IntLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make(int bitWidth, bool isSigned);
  int bitWidth() const;
  bool isSigned() const;

 private:
  IntLogicalType() = default;
};

/// \brief Allowed for any physical type.
class PARQUET_EXPORT NullLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  NullLogicalType() = default;
};

/// \brief Allowed for physical type BYTE_ARRAY.
class PARQUET_EXPORT JsonLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  JsonLogicalType() = default;
};

/// \brief Allowed for physical type BYTE_ARRAY.
class PARQUET_EXPORT BsonLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  BsonLogicalType() = default;
};

/// \brief Allowed for physical type FIXED_LEN_BYTE_ARRAY with length 16,.
/// Must encode raw UUID bytes.
class PARQUET_EXPORT UuidLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  UuidLogicalType() = default;
};

/// \brief Allowed for any physical type.
class PARQUET_EXPORT NoLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  NoLogicalType() = default;
};

// Internal API, for unrecognized logical types.
class PARQUET_EXPORT UndefinedLogicalType : public LogicalType {
 public:
  static std::shared_ptr<const LogicalType> make();

 private:
  UndefinedLogicalType() = default;
};

// Data encodings. Mirrors parquet::Encoding.
struct Encoding {
  enum type {
    kPlain = 0,
    kPlainDictionary = 2,
    kRle = 3,
    kBitPacked = 4,
    kDeltaBinaryPacked = 5,
    kDeltaLengthByteArray = 6,
    kDeltaByteArray = 7,
    kRleDictionary = 8,
    kByteStreamSplit = 9,
    // Should always be last element (except UNKNOWN).
    kUndefined = 10,
    kUnknown = 999
  };
};

// Exposed data encodings. It is the encoding of the data read from the file,.
// Rather than the encoding of the data in the file. E.g., the data encoded as.
// RLE_DICTIONARY in the file can be read as dictionary indices by RLE.
// Decoding, in which case the data read from the file is DICTIONARY encoded.
enum class ExposedEncoding {
  kNoEncoding = 0, // data is not encoded, i.e. already decoded during reading
  kDictionary = 1
};

/// \brief Return true if Parquet supports indicated compression type.
PARQUET_EXPORT
bool isCodecSupported(Compression::type codec);

PARQUET_EXPORT
std::unique_ptr<util::Codec> getCodec(Compression::type codec);

PARQUET_EXPORT
std::unique_ptr<util::Codec> getCodec(
    Compression::type codec,
    const util::CodecOptions& codecOptions);

PARQUET_EXPORT
std::unique_ptr<util::Codec> getCodec(
    Compression::type codec,
    int compressionLevel);

struct ParquetCipher {
  enum type { kAesGcmV1 = 0, kAesGcmCtrV1 = 1 };
};

struct AadMetadata {
  std::string aadPrefix;
  std::string aadFileUnique;
  bool supplyAadPrefix;
};

struct EncryptionAlgorithm {
  ParquetCipher::type algorithm;
  AadMetadata aad;
};

// Parquet::PageType.
struct PageType {
  enum type {
    kDataPage,
    kIndexPage,
    kDictionaryPage,
    kDataPageV2,
    // Should always be last element.
    kUndefined
  };
};

bool pageCanUseChecksum(PageType::type pageType);

class ColumnOrder {
 public:
  enum type { kUndefined, kTypeDefinedOrder };
  explicit ColumnOrder(ColumnOrder::type order) : columnOrder_(order) {}
  // Default to Type Defined Order.
  ColumnOrder() : columnOrder_(type::kTypeDefinedOrder) {}
  ColumnOrder::type order() const {
    return columnOrder_;
  }

  static ColumnOrder undefined_;
  static ColumnOrder typeDefined_;

 private:
  ColumnOrder::type columnOrder_;
};

/// \brief BoundaryOrder is a proxy around.
/// Facebook::velox::parquet::thrift::BoundaryOrder.
struct BoundaryOrder {
  enum type {
    kUnordered = 0,
    kAscending = 1,
    kDescending = 2,
    // Should always be last element.
    kUndefined = 3
  };
};

/// \brief SortingColumn is a proxy around.
/// Facebook::velox::parquet::thrift::SortingColumn.
struct PARQUET_EXPORT SortingColumn {
  // The column index (in this row group)
  int32_t columnIdx;

  // If true, indicates this column is sorted in descending order.
  bool descending;

  // If true, nulls will come before non-null values, otherwise, nulls go at
  // the. End.
  bool nullsFirst;
};

inline bool operator==(const SortingColumn& left, const SortingColumn& right) {
  return left.nullsFirst == right.nullsFirst &&
      left.descending == right.descending && left.columnIdx == right.columnIdx;
}

// ----------------------------------------------------------------------.

struct ByteArray {
  ByteArray() : len(0), ptr(NULLPTR) {}
  ByteArray(uint32_t len, const uint8_t* ptr) : len(len), ptr(ptr) {}

  ByteArray(::std::string_view view) // NOLINT implicit conversion
      : ByteArray(
            static_cast<uint32_t>(view.size()),
            reinterpret_cast<const uint8_t*>(view.data())) {}

  explicit operator std::string_view() const {
    return std::string_view{reinterpret_cast<const char*>(ptr), len};
  }

  uint32_t len;
  const uint8_t* ptr;
};

inline bool operator==(const ByteArray& left, const ByteArray& right) {
  return left.len == right.len &&
      (left.len == 0 || std::memcmp(left.ptr, right.ptr, left.len) == 0);
}

struct FixedLenByteArray {
  FixedLenByteArray() : ptr(NULLPTR) {}
  explicit FixedLenByteArray(const uint8_t* ptr) : ptr(ptr) {}
  const uint8_t* ptr;
};

using FLBA = FixedLenByteArray;

// Julian day at unix epoch.
//
// The Julian Day Number (JDN) is the integer assigned to a whole solar day in.
// The Julian day count starting from noon Universal time, with Julian day.
// Number 0 assigned to the day starting at noon on Monday, January 1, 4713 BC,.
// Proleptic Julian calendar (November 24, 4714 BC, in the proleptic Gregorian.
// Calendar),.
constexpr int64_t kJulianToUnixEpochDays = INT64_C(2440588);
constexpr int64_t kSecondsPerDay = INT64_C(60 * 60 * 24);
constexpr int64_t kMillisecondsPerDay = kSecondsPerDay * INT64_C(1000);
constexpr int64_t kMicrosecondsPerDay = kMillisecondsPerDay * INT64_C(1000);
constexpr int64_t kNanosecondsPerDay = kMicrosecondsPerDay * INT64_C(1000);

MANUALLY_ALIGNED_STRUCT(1) Int96 {
  uint32_t value[3];
};
STRUCT_END(Int96, 12);

inline bool operator==(const Int96& left, const Int96& right) {
  return std::equal(left.value, left.value + 3, right.value);
}

static inline std::string byteArrayToString(const ByteArray& a) {
  return std::string(reinterpret_cast<const char*>(a.ptr), a.len);
}

static inline void int96SetNanoSeconds(Int96& i96, int64_t nanoseconds) {
  std::memcpy(&i96.value, &nanoseconds, sizeof(nanoseconds));
}

struct DecodedInt96 {
  uint64_t daysSinceEpoch;
  uint64_t nanoseconds;
};

static inline DecodedInt96 decodeInt96Timestamp(const Int96& i96) {
  // We do the computations in the unsigned domain to avoid unsigned behaviour.
  // On overflow.
  DecodedInt96 result;
  result.daysSinceEpoch =
      i96.value[2] - static_cast<uint64_t>(kJulianToUnixEpochDays);
  result.nanoseconds = 0;

  memcpy(&result.nanoseconds, &i96.value, sizeof(uint64_t));
  return result;
}

static inline int64_t int96GetNanoSeconds(const Int96& i96) {
  const auto decoded = decodeInt96Timestamp(i96);
  return static_cast<int64_t>(
      decoded.daysSinceEpoch * kNanosecondsPerDay + decoded.nanoseconds);
}

static inline int64_t int96GetMicroSeconds(const Int96& i96) {
  const auto decoded = decodeInt96Timestamp(i96);
  uint64_t microseconds = decoded.nanoseconds / static_cast<uint64_t>(1000);
  return static_cast<int64_t>(
      decoded.daysSinceEpoch * kMicrosecondsPerDay + microseconds);
}

static inline int64_t int96GetMilliSeconds(const Int96& i96) {
  const auto decoded = decodeInt96Timestamp(i96);
  uint64_t milliseconds = decoded.nanoseconds / static_cast<uint64_t>(1000000);
  return static_cast<int64_t>(
      decoded.daysSinceEpoch * kMillisecondsPerDay + milliseconds);
}

static inline int64_t int96GetSeconds(const Int96& i96) {
  const auto decoded = decodeInt96Timestamp(i96);
  uint64_t seconds = decoded.nanoseconds / static_cast<uint64_t>(1000000000);
  return static_cast<int64_t>(
      decoded.daysSinceEpoch * kSecondsPerDay + seconds);
}

static inline std::string int96ToString(const Int96& a) {
  std::ostringstream result;
  std::copy(a.value, a.value + 3, std::ostream_iterator<uint32_t>(result, " "));
  return result.str();
}

static inline std::string fixedLenByteArrayToString(
    const FixedLenByteArray& a,
    int len) {
  std::ostringstream result;
  std::copy(a.ptr, a.ptr + len, std::ostream_iterator<uint32_t>(result, " "));
  return result.str();
}

template <Type::type TYPE>
struct TypeTraits {};

template <>
struct TypeTraits<Type::kBoolean> {
  using ValueType = bool;

  static constexpr int valueByteSize = 1;
  static constexpr const char* printfCode = "d";
};

template <>
struct TypeTraits<Type::kInt32> {
  using ValueType = int32_t;

  static constexpr int valueByteSize = 4;
  static constexpr const char* printfCode = "d";
};

template <>
struct TypeTraits<Type::kInt64> {
  using ValueType = int64_t;

  static constexpr int valueByteSize = 8;
  static constexpr const char* printfCode =
      (sizeof(long) == 64) ? "ld" : "lld"; // NOLINT: runtime/int
};

template <>
struct TypeTraits<Type::kInt96> {
  using ValueType = Int96;

  static constexpr int valueByteSize = 12;
  static constexpr const char* printfCode = "s";
};

template <>
struct TypeTraits<Type::kFloat> {
  using ValueType = float;

  static constexpr int valueByteSize = 4;
  static constexpr const char* printfCode = "f";
};

template <>
struct TypeTraits<Type::kDouble> {
  using ValueType = double;

  static constexpr int valueByteSize = 8;
  static constexpr const char* printfCode = "lf";
};

template <>
struct TypeTraits<Type::kByteArray> {
  using ValueType = ByteArray;

  static constexpr int valueByteSize = sizeof(ByteArray);
  static constexpr const char* printfCode = "s";
};

template <>
struct TypeTraits<Type::kFixedLenByteArray> {
  using ValueType = FixedLenByteArray;

  static constexpr int valueByteSize = sizeof(FixedLenByteArray);
  static constexpr const char* printfCode = "s";
};

template <Type::type TYPE>
struct PhysicalType {
  using CType = typename TypeTraits<TYPE>::ValueType;
  static constexpr Type::type typeNum = TYPE;
};

using BooleanType = PhysicalType<Type::kBoolean>;
using Int32Type = PhysicalType<Type::kInt32>;
using Int64Type = PhysicalType<Type::kInt64>;
using Int96Type = PhysicalType<Type::kInt96>;
using FloatType = PhysicalType<Type::kFloat>;
using DoubleType = PhysicalType<Type::kDouble>;
using ByteArrayType = PhysicalType<Type::kByteArray>;
using FLBAType = PhysicalType<Type::kFixedLenByteArray>;

template <typename Type>
inline std::string formatFwf(int width) {
  std::stringstream ss;
  ss << "%-" << width << TypeTraits<Type::typeNum>::printfCode;
  return ss.str();
}

PARQUET_EXPORT std::string encodingToString(Encoding::type t);

PARQUET_EXPORT std::string convertedTypeToString(ConvertedType::type t);

PARQUET_EXPORT std::string typeToString(Type::type t);

PARQUET_EXPORT std::string formatStatValue(
    Type::type parquetType,
    ::std::string_view val);

PARQUET_EXPORT int getTypeByteSize(Type::type t);

PARQUET_EXPORT SortOrder::type defaultSortOrder(Type::type primitive);

PARQUET_EXPORT SortOrder::type getSortOrder(
    ConvertedType::type converted,
    Type::type primitive);

PARQUET_EXPORT SortOrder::type getSortOrder(
    const std::shared_ptr<const LogicalType>& logicalType,
    Type::type primitive);

} // namespace facebook::velox::parquet::arrow
