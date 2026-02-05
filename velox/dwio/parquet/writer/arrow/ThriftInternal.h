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

#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// TCompactProtocol requires some #defines to work right.
#define SIGNED_RIGHT_SHIFT_IS 1
#define ARITHMETIC_RIGHT_SHIFT 1
#include <thrift/TApplicationException.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileDecryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/FileEncryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"

#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

// ----------------------------------------------------------------------.
// Convert Thrift enums to Parquet enums.

// Unsafe enum converters (input is not checked for validity)

static inline Type::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::Type::type type) {
  return static_cast<Type::type>(type);
}

static inline ConvertedType::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::ConvertedType::type type) {
  // Item 0 is NONE.
  return static_cast<ConvertedType::type>(static_cast<int>(type) + 1);
}

static inline Repetition::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::FieldRepetitionType::type type) {
  return static_cast<Repetition::type>(type);
}

static inline Encoding::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::Encoding::type type) {
  return static_cast<Encoding::type>(type);
}

static inline PageType::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::PageType::type type) {
  return static_cast<PageType::type>(type);
}

static inline Compression::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::CompressionCodec::type type) {
  switch (type) {
    case facebook::velox::parquet::thrift::CompressionCodec::UNCOMPRESSED:
      return Compression::UNCOMPRESSED;
    case facebook::velox::parquet::thrift::CompressionCodec::SNAPPY:
      return Compression::SNAPPY;
    case facebook::velox::parquet::thrift::CompressionCodec::GZIP:
      return Compression::GZIP;
    case facebook::velox::parquet::thrift::CompressionCodec::LZO:
      return Compression::LZO;
    case facebook::velox::parquet::thrift::CompressionCodec::BROTLI:
      return Compression::BROTLI;
    case facebook::velox::parquet::thrift::CompressionCodec::LZ4:
      return Compression::LZ4_HADOOP;
    case facebook::velox::parquet::thrift::CompressionCodec::LZ4_RAW:
      return Compression::LZ4;
    case facebook::velox::parquet::thrift::CompressionCodec::ZSTD:
      return Compression::ZSTD;
    default:
      VELOX_DCHECK(false, "Cannot reach here");
      return Compression::UNCOMPRESSED;
  }
}

static inline BoundaryOrder::type fromThriftUnsafe(
    facebook::velox::parquet::thrift::BoundaryOrder::type type) {
  return static_cast<BoundaryOrder::type>(type);
}

namespace internal {

template <typename T>
struct ThriftenumTypeTraits {};

template <>
struct ThriftenumTypeTraits<::facebook::velox::parquet::thrift::Type::type> {
  using Parquetenum = Type;
};

template <>
struct ThriftenumTypeTraits<
    ::facebook::velox::parquet::thrift::ConvertedType::type> {
  using Parquetenum = ConvertedType;
};

template <>
struct ThriftenumTypeTraits<
    ::facebook::velox::parquet::thrift::FieldRepetitionType::type> {
  using Parquetenum = Repetition;
};

template <>
struct ThriftenumTypeTraits<
    ::facebook::velox::parquet::thrift::Encoding::type> {
  using Parquetenum = Encoding;
};

template <>
struct ThriftenumTypeTraits<
    ::facebook::velox::parquet::thrift::PageType::type> {
  using Parquetenum = PageType;
};

template <>
struct ThriftenumTypeTraits<
    ::facebook::velox::parquet::thrift::BoundaryOrder::type> {
  using Parquetenum = BoundaryOrder;
};

// If the parquet file is corrupted it is possible the enum value decoded.
// Will not be in the range of defined values, which is undefined behaviour.
// This facility prevents this by loading the value as the underlying type.
// And checking to make sure it is in range.

template <
    typename enumType,
    typename enumTypeRaw = typename std::underlying_type<enumType>::type>
inline static enumTypeRaw loadenumRaw(const enumType* in) {
  enumTypeRaw rawValue;
  // Use memcpy(), as a regular cast would be undefined behaviour on invalid.
  // Values.
  memcpy(&rawValue, in, sizeof(enumType));
  return rawValue;
}

template <typename ApiType>
struct SafeLoader {
  using ApiTypeenum = typename ApiType::type;
  using ApiTypeRawenum = typename std::underlying_type<ApiTypeenum>::type;

  template <typename ThriftType>
  inline static ApiTypeRawenum loadRaw(const ThriftType* in) {
    static_assert(
        sizeof(ApiTypeenum) == sizeof(ThriftType),
        "parquet type should always be the same size as thrift type");
    return static_cast<ApiTypeRawenum>(loadenumRaw(in));
  }

  template <typename ThriftType, bool IsUnsigned = true>
  inline static ApiTypeenum loadChecked(
      const typename std::enable_if<IsUnsigned, ThriftType>::type* in) {
    auto rawValue = loadRaw(in);
    if (ARROW_PREDICT_FALSE(
            rawValue >= static_cast<ApiTypeRawenum>(ApiType::kUndefined))) {
      return ApiType::kUndefined;
    }
    return fromThriftUnsafe(static_cast<ThriftType>(rawValue));
  }

  template <typename ThriftType, bool IsUnsigned = false>
  inline static ApiTypeenum loadChecked(
      const typename std::enable_if<!IsUnsigned, ThriftType>::type* in) {
    auto rawValue = loadRaw(in);
    if (ARROW_PREDICT_FALSE(
            rawValue >= static_cast<ApiTypeRawenum>(ApiType::kUndefined) ||
            rawValue < 0)) {
      return ApiType::kUndefined;
    }
    return fromThriftUnsafe(static_cast<ThriftType>(rawValue));
  }

  template <typename ThriftType>
  inline static ApiTypeenum load(const ThriftType* in) {
    return loadChecked<ThriftType, std::is_unsigned<ApiTypeRawenum>::value>(in);
  }
};

} // namespace internal

// Safe enum loader: will check for invalid enum value before converting.

template <
    typename ThriftType,
    typename Parquetenum =
        typename internal::ThriftenumTypeTraits<ThriftType>::Parquetenum>
inline typename Parquetenum::type loadenumSafe(const ThriftType* in) {
  return internal::SafeLoader<Parquetenum>::load(in);
}

inline typename Compression::type loadenumSafe(
    const facebook::velox::parquet::thrift::CompressionCodec::type* in) {
  const auto rawValue = internal::loadenumRaw(in);
  // Check bounds manually, as Compression::type doesn't have the same values.
  // As facebook::velox::parquet::thrift::CompressionCodec.
  const auto minValue = static_cast<decltype(rawValue)>(
      facebook::velox::parquet::thrift::CompressionCodec::UNCOMPRESSED);
  const auto maxValue = static_cast<decltype(rawValue)>(
      facebook::velox::parquet::thrift::CompressionCodec::LZ4_RAW);
  if (rawValue < minValue || rawValue > maxValue) {
    return Compression::UNCOMPRESSED;
  }
  return fromThriftUnsafe(*in);
}

// Safe non-enum converters.

static inline AadMetadata fromThrift(
    facebook::velox::parquet::thrift::AesGcmV1 aesGcmV1) {
  return AadMetadata{
      aesGcmV1.aad_prefix,
      aesGcmV1.aad_file_unique,
      aesGcmV1.supply_aad_prefix};
}

static inline AadMetadata fromThrift(
    facebook::velox::parquet::thrift::AesGcmCtrV1 aesGcmCtrV1) {
  return AadMetadata{
      aesGcmCtrV1.aad_prefix,
      aesGcmCtrV1.aad_file_unique,
      aesGcmCtrV1.supply_aad_prefix};
}

static inline EncryptionAlgorithm fromThrift(
    facebook::velox::parquet::thrift::EncryptionAlgorithm encryption) {
  EncryptionAlgorithm encryptionAlgorithm;

  if (encryption.__isset.AES_GCM_V1) {
    encryptionAlgorithm.algorithm = ParquetCipher::kAesGcmV1;
    encryptionAlgorithm.aad = fromThrift(encryption.AES_GCM_V1);
  } else if (encryption.__isset.AES_GCM_CTR_V1) {
    encryptionAlgorithm.algorithm = ParquetCipher::kAesGcmCtrV1;
    encryptionAlgorithm.aad = fromThrift(encryption.AES_GCM_CTR_V1);
  } else {
    throw ParquetException("Unsupported algorithm");
  }
  return encryptionAlgorithm;
}

static inline SortingColumn fromThrift(
    facebook::velox::parquet::thrift::SortingColumn thriftSortingColumn) {
  SortingColumn sortingColumn;
  sortingColumn.columnIdx = thriftSortingColumn.column_idx;
  sortingColumn.nullsFirst = thriftSortingColumn.nulls_first;
  sortingColumn.descending = thriftSortingColumn.descending;
  return sortingColumn;
}

// ----------------------------------------------------------------------.
// Convert Thrift enums from Parquet enums.

static inline facebook::velox::parquet::thrift::Type::type toThrift(
    Type::type type) {
  return static_cast<facebook::velox::parquet::thrift::Type::type>(type);
}

static fmt::underlying_t<ConvertedType::type> formatAs(
    ConvertedType::type type) {
  return fmt::underlying(type);
}

static inline facebook::velox::parquet::thrift::ConvertedType::type toThrift(
    ConvertedType::type type) {
  // Item 0 is NONE.
  const int typeValue = static_cast<int>(type);
  VELOX_DCHECK_NE(typeValue, static_cast<int>(ConvertedType::kNone));
  // it is forbidden to emit "NA" (PARQUET-1990)
  VELOX_DCHECK_NE(typeValue, static_cast<int>(ConvertedType::kNa));
  VELOX_DCHECK_NE(typeValue, static_cast<int>(ConvertedType::kUndefined));
  return static_cast<facebook::velox::parquet::thrift::ConvertedType::type>(
      typeValue - 1);
}

static inline facebook::velox::parquet::thrift::FieldRepetitionType::type
toThrift(Repetition::type type) {
  return static_cast<
      facebook::velox::parquet::thrift::FieldRepetitionType::type>(type);
}

static inline facebook::velox::parquet::thrift::Encoding::type toThrift(
    Encoding::type type) {
  return static_cast<facebook::velox::parquet::thrift::Encoding::type>(type);
}

static inline facebook::velox::parquet::thrift::CompressionCodec::type toThrift(
    Compression::type type) {
  switch (type) {
    case Compression::UNCOMPRESSED:
      return facebook::velox::parquet::thrift::CompressionCodec::UNCOMPRESSED;
    case Compression::SNAPPY:
      return facebook::velox::parquet::thrift::CompressionCodec::SNAPPY;
    case Compression::GZIP:
      return facebook::velox::parquet::thrift::CompressionCodec::GZIP;
    case Compression::LZO:
      return facebook::velox::parquet::thrift::CompressionCodec::LZO;
    case Compression::BROTLI:
      return facebook::velox::parquet::thrift::CompressionCodec::BROTLI;
    case Compression::LZ4:
      return facebook::velox::parquet::thrift::CompressionCodec::LZ4_RAW;
    case Compression::LZ4_HADOOP:
      // Deprecated "LZ4" Parquet compression has Hadoop-specific framing.
      return facebook::velox::parquet::thrift::CompressionCodec::LZ4;
    case Compression::ZSTD:
      return facebook::velox::parquet::thrift::CompressionCodec::ZSTD;
    default:
      VELOX_DCHECK(false, "Cannot reach here");
      return facebook::velox::parquet::thrift::CompressionCodec::UNCOMPRESSED;
  }
}

static inline facebook::velox::parquet::thrift::BoundaryOrder::type toThrift(
    BoundaryOrder::type type) {
  switch (type) {
    case BoundaryOrder::Unordered:
    case BoundaryOrder::Ascending:
    case BoundaryOrder::Descending:
      return static_cast<facebook::velox::parquet::thrift::BoundaryOrder::type>(
          type);
    default:
      VELOX_DCHECK(false, "Cannot reach here");
      return facebook::velox::parquet::thrift::BoundaryOrder::UNORDERED;
  }
}

static inline facebook::velox::parquet::thrift::SortingColumn toThrift(
    SortingColumn sortingColumn) {
  facebook::velox::parquet::thrift::SortingColumn thriftSortingColumn;
  thriftSortingColumn.column_idx = sortingColumn.columnIdx;
  thriftSortingColumn.descending = sortingColumn.descending;
  thriftSortingColumn.nulls_first = sortingColumn.nullsFirst;
  return thriftSortingColumn;
}

static inline facebook::velox::parquet::thrift::Statistics toThrift(
    const EncodedStatistics& stats) {
  facebook::velox::parquet::thrift::Statistics Statistics;
  if (stats.hasMin) {
    Statistics.__set_min_value(stats.min());
    // If the order is SIGNED, then the old min value must be set too.
    // This for backward compatibility.
    if (stats.isSigned()) {
      Statistics.__set_min(stats.min());
    }
  }
  if (stats.hasMax) {
    Statistics.__set_max_value(stats.max());
    // If the order is SIGNED, then the old max value must be set too.
    // This for backward compatibility.
    if (stats.isSigned()) {
      Statistics.__set_max(stats.max());
    }
  }
  if (stats.hasNullCount) {
    Statistics.__set_null_count(stats.nullCount);
  }
  if (stats.hasDistinctCount) {
    Statistics.__set_distinct_count(stats.distinctCount);
  }

  return Statistics;
}

static inline facebook::velox::parquet::thrift::AesGcmV1 toAesGcmV1Thrift(
    AadMetadata aad) {
  facebook::velox::parquet::thrift::AesGcmV1 aesGcmV1;
  // Aad_file_unique is always set.
  aesGcmV1.__set_aad_file_unique(aad.aadFileUnique);
  aesGcmV1.__set_supply_aad_prefix(aad.supplyAadPrefix);
  if (!aad.aadPrefix.empty()) {
    aesGcmV1.__set_aad_prefix(aad.aadPrefix);
  }
  return aesGcmV1;
}

static inline facebook::velox::parquet::thrift::AesGcmCtrV1 toAesGcmCtrV1Thrift(
    AadMetadata aad) {
  facebook::velox::parquet::thrift::AesGcmCtrV1 aesGcmCtrV1;
  // Aad_file_unique is always set.
  aesGcmCtrV1.__set_aad_file_unique(aad.aadFileUnique);
  aesGcmCtrV1.__set_supply_aad_prefix(aad.supplyAadPrefix);
  if (!aad.aadPrefix.empty()) {
    aesGcmCtrV1.__set_aad_prefix(aad.aadPrefix);
  }
  return aesGcmCtrV1;
}

static inline facebook::velox::parquet::thrift::EncryptionAlgorithm toThrift(
    EncryptionAlgorithm encryption) {
  facebook::velox::parquet::thrift::EncryptionAlgorithm encryptionAlgorithm;
  if (encryption.algorithm == ParquetCipher::kAesGcmV1) {
    encryptionAlgorithm.__set_AES_GCM_V1(toAesGcmV1Thrift(encryption.aad));
  } else {
    encryptionAlgorithm.__set_AES_GCM_CTR_V1(
        toAesGcmCtrV1Thrift(encryption.aad));
  }
  return encryptionAlgorithm;
}

// ----------------------------------------------------------------------.
// Thrift struct serialization / deserialization utilities.

using ThriftBuffer = apache::thrift::transport::TMemoryBuffer;

class ThriftDeserializer {
 public:
  explicit ThriftDeserializer(const ReaderProperties& properties)
      : ThriftDeserializer(
            properties.thriftStringSizeLimit(),
            properties.thriftContainerSizeLimit()) {}

  ThriftDeserializer(int32_t stringSizeLimit, int32_t containerSizeLimit)
      : stringSizeLimit_(stringSizeLimit),
        containerSizeLimit_(containerSizeLimit) {}

  // Deserialize a thrift message from buf/len.  buf/len must at least contain.
  // All the bytes needed to store the thrift message.  On return, len will be.
  // Set to the actual length of the header.
  template <class T>
  void deserializeMessage(
      const uint8_t* buf,
      uint32_t* len,
      T* deserializedMsg,
      const std::shared_ptr<Decryptor>& Decryptor = NULLPTR) {
    if (Decryptor == NULLPTR) {
      // Thrift message is not encrypted.
      deserializeUnencryptedMessage(buf, len, deserializedMsg);
    } else {
      // Thrift message is encrypted.
      uint32_t clen;
      clen = *len;
      // Decrypt.
      auto decryptedBuffer =
          std::static_pointer_cast<ResizableBuffer>(allocateBuffer(
              Decryptor->pool(),
              static_cast<int64_t>(clen - Decryptor->ciphertextSizeDelta())));
      const uint8_t* cipherBuf = buf;
      uint32_t decryptedBufferLen =
          Decryptor->decrypt(cipherBuf, 0, decryptedBuffer->mutable_data());
      if (decryptedBufferLen <= 0) {
        throw ParquetException("Couldn't decrypt buffer\n");
      }
      *len = decryptedBufferLen + Decryptor->ciphertextSizeDelta();
      deserializeUnencryptedMessage(
          decryptedBuffer->data(), &decryptedBufferLen, deserializedMsg);
    }
  }

 private:
  // On Thrift 0.14.0+, we want to use TConfiguration to raise the max message.
  // Size limit (ARROW-13655).  If we wanted to protect against huge messages,.
  // We could do it ourselves since we know the message size up front.
  std::shared_ptr<ThriftBuffer> createReadOnlyMemoryBuffer(
      uint8_t* buf,
      uint32_t len) {
#if PARQUET_THRIFT_VERSION_MAJOR > 0 || PARQUET_THRIFT_VERSION_MINOR >= 14
    auto conf = std::make_shared<apache::thrift::TConfiguration>();
    conf->setMaxMessageSize(std::numeric_limits<int>::max());
    return std::make_shared<ThriftBuffer>(
        buf, len, ThriftBuffer::OBSERVE, conf);
#else
    return std::make_shared<ThriftBuffer>(buf, len);
#endif
  }

  template <class T>
  void deserializeUnencryptedMessage(
      const uint8_t* buf,
      uint32_t* len,
      T* deserializedMsg) {
    // Deserialize msg bytes into c++ thrift msg using memory transport.
    auto tmemTransport =
        createReadOnlyMemoryBuffer(const_cast<uint8_t*>(buf), *len);
    apache::thrift::protocol::TCompactProtocolFactoryT<ThriftBuffer>
        tprotoFactory;
    // Protect against CPU and memory bombs.
    tprotoFactory.setStringSizeLimit(stringSizeLimit_);
    tprotoFactory.setContainerSizeLimit(containerSizeLimit_);
    auto tproto = tprotoFactory.getProtocol(tmemTransport);
    try {
      deserializedMsg->read(tproto.get());
    } catch (std::exception& e) {
      std::stringstream ss;
      ss << "Couldn't deserialize thrift: " << e.what() << "\n";
      throw ParquetException(ss.str());
    }
    uint32_t bytesLeft = tmemTransport->available_read();
    *len = *len - bytesLeft;
  }

  const int32_t stringSizeLimit_;
  const int32_t containerSizeLimit_;
};

/// Utility class to serialize thrift objects to a binary format.  This object.
/// Should be reused if possible to reuse the underlying memory.
/// Note: thrift will encode NULLs into the serialized buffer so it is not
/// valid. To treat it as a string.
class ThriftSerializer {
 public:
  explicit ThriftSerializer(int initialBufferSize = 1024)
      : memBuffer_(std::make_shared<ThriftBuffer>(initialBufferSize)) {
    apache::thrift::protocol::TCompactProtocolFactoryT<ThriftBuffer> factory;
    protocol_ = factory.getProtocol(memBuffer_);
  }

  /// Serialize obj into a memory buffer.  The result is returned in buffer/len.
  /// The memory returned is owned by this object and will be invalid when.
  /// Another object is serialized.
  template <class T>
  void serializeToBuffer(const T* obj, uint32_t* len, uint8_t** buffer) {
    serializeObject(obj);
    memBuffer_->getBuffer(buffer, len);
  }

  template <class T>
  void serializeToString(const T* obj, std::string* result) {
    serializeObject(obj);
    *result = memBuffer_->getBufferAsString();
  }

  template <class T>
  int64_t serialize(
      const T* obj,
      ArrowOutputStream* out,
      const std::shared_ptr<Encryptor>& Encryptor = NULLPTR) {
    uint8_t* outBuffer;
    uint32_t outLength;
    serializeToBuffer(obj, &outLength, &outBuffer);

    // Obj is not encrypted.
    if (Encryptor == NULLPTR) {
      PARQUET_THROW_NOT_OK(out->Write(outBuffer, outLength));
      return static_cast<int64_t>(outLength);
    } else { // obj is encrypted
      return serializeEncryptedObj(out, outBuffer, outLength, Encryptor);
    }
  }

 private:
  template <class T>
  void serializeObject(const T* obj) {
    try {
      memBuffer_->resetBuffer();
      obj->write(protocol_.get());
    } catch (std::exception& e) {
      std::stringstream ss;
      ss << "Couldn't serialize thrift: " << e.what() << "\n";
      throw ParquetException(ss.str());
    }
  }

  int64_t serializeEncryptedObj(
      ArrowOutputStream* out,
      uint8_t* outBuffer,
      uint32_t outLength,
      const std::shared_ptr<Encryptor>& Encryptor) {
    auto cipherBuffer =
        std::static_pointer_cast<ResizableBuffer>(allocateBuffer(
            Encryptor->pool(),
            static_cast<int64_t>(
                Encryptor->ciphertextSizeDelta() + outLength)));
    int cipherBufferLen =
        Encryptor->encrypt(outBuffer, outLength, cipherBuffer->mutable_data());

    PARQUET_THROW_NOT_OK(out->Write(cipherBuffer->data(), cipherBufferLen));
    return static_cast<int64_t>(cipherBufferLen);
  }

  std::shared_ptr<ThriftBuffer> memBuffer_;
  std::shared_ptr<apache::thrift::protocol::TProtocol> protocol_;
};

} // namespace facebook::velox::parquet::arrow
