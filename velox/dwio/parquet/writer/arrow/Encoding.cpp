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

#include "velox/dwio/parquet/writer/arrow/Encoding.h"

#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_dict.h"
#include "arrow/stl_allocator.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/bitmap_writer.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/ubsan.h"
#include "arrow/visit_data_inline.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/common/RleEncodingInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/util/ByteStreamSplitInternal.h"
#include "velox/dwio/parquet/writer/arrow/util/Hashing.h"
#include "velox/dwio/parquet/writer/arrow/util/OverflowUtilInternal.h"

using ::arrow::Buffer;
using ::arrow::MemoryPool;
using ::arrow::ResizableBuffer;
using arrow::Status;
using arrow::VisitNullBitmapInline;
using arrow::internal::addWithOverflow;
using arrow::internal::checked_cast;
using arrow::internal::multiplyWithOverflow;
using arrow::internal::SubtractWithOverflow;
using std::string_view;

template <typename T>
using ArrowPoolVector = std::vector<T, ::arrow::stl::allocator<T>>;

namespace facebook::velox::parquet::arrow {
namespace {

template <typename T>
inline std::enable_if_t<std::is_trivially_copyable_v<T>, T> SafeLoadAs(
    const uint8_t* unaligned) {
  std::remove_const_t<T> ret;
  std::memcpy(&ret, unaligned, sizeof(T));
  return ret;
}

template <typename T>
inline std::enable_if_t<std::is_trivially_copyable_v<T>, T> safeLoad(
    const T* unaligned) {
  std::remove_const_t<T> ret;
  std::memcpy(&ret, unaligned, sizeof(T));
  return ret;
}

std::shared_ptr<ResizableBuffer> allocateBuffer(
    MemoryPool* pool,
    int64_t size) {
  PARQUET_ASSIGN_OR_THROW(
      auto result, ::arrow::AllocateResizableBuffer(size, pool));
  return std::move(result);
}

// The Parquet spec isn't very clear whether ByteArray lengths are signed or.
// Unsigned, but the Java implementation uses signed ints.
constexpr size_t kMaxByteArraySize = std::numeric_limits<int32_t>::max();

class EncoderImpl : virtual public Encoder {
 public:
  EncoderImpl(
      const ColumnDescriptor* descr,
      Encoding::type encoding,
      MemoryPool* pool)
      : descr_(descr),
        encoding_(encoding),
        pool_(pool),
        typeLength_(descr ? descr->typeLength() : -1) {}

  Encoding::type encoding() const override {
    return encoding_;
  }

  MemoryPool* memoryPool() const override {
    return pool_;
  }

 protected:
  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY.
  const ColumnDescriptor* descr_;
  const Encoding::type encoding_;
  MemoryPool* pool_;

  /// Type length from descr.
  int typeLength_;
};

// ----------------------------------------------------------------------.
// Plain encoder implementation.

template <typename DType>
class PlainEncoder : public EncoderImpl, virtual public TypedEncoder<DType> {
 public:
  using T = typename DType::CType;

  explicit PlainEncoder(const ColumnDescriptor* descr, MemoryPool* pool)
      : EncoderImpl(descr, Encoding::kPlain, pool), sink_(pool) {}

  int64_t estimatedDataEncodedSize() override {
    return sink_.length();
  }

  std::shared_ptr<::arrow::Buffer> flushValues() override {
    std::shared_ptr<Buffer> buffer;
    PARQUET_THROW_NOT_OK(sink_.Finish(&buffer));
    return buffer;
  }

  using TypedEncoder<DType>::put;

  void put(const T* buffer, int numValues) override;

  void put(const ::arrow::Array& values) override;

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    if (validBits != NULLPTR) {
      auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
      T* data = reinterpret_cast<T*>(buffer->mutable_data());
      int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
          src, numValues, validBits, validBitsOffset, data);
      put(data, numValidValues);
    } else {
      put(src, numValues);
    }
  }

  void unsafePutByteArray(const void* data, uint32_t length) {
    VELOX_DCHECK(length == 0 || data != nullptr, "Value ptr cannot be NULL");
    sink_.UnsafeAppend(&length, sizeof(uint32_t));
    sink_.UnsafeAppend(data, static_cast<int64_t>(length));
  }

  void put(const ByteArray& val) {
    // Write the result to the output stream.
    const int64_t increment = static_cast<int64_t>(val.len + sizeof(uint32_t));
    if (ARROW_PREDICT_FALSE(sink_.length() + increment > sink_.capacity())) {
      PARQUET_THROW_NOT_OK(sink_.Reserve(increment));
    }
    unsafePutByteArray(val.ptr, val.len);
  }

 protected:
  template <typename ArrayType>
  void putBinaryArray(const ArrayType& array) {
    const int64_t totalBytes =
        array.value_offset(array.length()) - array.value_offset(0);
    PARQUET_THROW_NOT_OK(
        sink_.Reserve(totalBytes + array.length() * sizeof(uint32_t)));

    PARQUET_THROW_NOT_OK(
        ::arrow::VisitArraySpanInline<typename ArrayType::TypeClass>(
            *array.data(),
            [&](std::string_view view) {
              if (ARROW_PREDICT_FALSE(view.size() > kMaxByteArraySize)) {
                return Status::Invalid(
                    "Parquet cannot store strings with size 2GB or more");
              }
              unsafePutByteArray(
                  view.data(), static_cast<uint32_t>(view.size()));
              return Status::OK();
            },
            []() { return Status::OK(); }));
  }

  ::arrow::BufferBuilder sink_;
};

template <typename DType>
void PlainEncoder<DType>::put(const T* buffer, int numValues) {
  if (numValues > 0) {
    PARQUET_THROW_NOT_OK(sink_.Append(buffer, numValues * sizeof(T)));
  }
}

template <>
inline void PlainEncoder<ByteArrayType>::put(
    const ByteArray* src,
    int numValues) {
  for (int i = 0; i < numValues; ++i) {
    put(src[i]);
  }
}

template <typename ArrayType>
void directPutImpl(const ::arrow::Array& values, ::arrow::BufferBuilder* sink) {
  if (values.type_id() != ArrayType::TypeClass::type_id) {
    std::string typeName = ArrayType::TypeClass::type_name();
    throw ParquetException(
        "direct put to " + typeName + " from " + values.type()->ToString() +
        " not supported");
  }

  using ValueType = typename ArrayType::value_type;
  constexpr auto valueSize = sizeof(ValueType);
  auto rawValues = checked_cast<const ArrayType&>(values).raw_values();

  if (values.null_count() == 0) {
    // No nulls, just dump the data.
    PARQUET_THROW_NOT_OK(sink->Append(rawValues, values.length() * valueSize));
  } else {
    PARQUET_THROW_NOT_OK(
        sink->Reserve((values.length() - values.null_count()) * valueSize));

    for (int64_t i = 0; i < values.length(); i++) {
      if (values.IsValid(i)) {
        sink->UnsafeAppend(&rawValues[i], valueSize);
      }
    }
  }
}

template <>
void PlainEncoder<Int32Type>::put(const ::arrow::Array& values) {
  directPutImpl<::arrow::Int32Array>(values, &sink_);
}

template <>
void PlainEncoder<Int64Type>::put(const ::arrow::Array& values) {
  directPutImpl<::arrow::Int64Array>(values, &sink_);
}

template <>
void PlainEncoder<Int96Type>::put(const ::arrow::Array& values) {
  ParquetException::NYI("direct put to Int96");
}

template <>
void PlainEncoder<FloatType>::put(const ::arrow::Array& values) {
  directPutImpl<::arrow::FloatArray>(values, &sink_);
}

template <>
void PlainEncoder<DoubleType>::put(const ::arrow::Array& values) {
  directPutImpl<::arrow::DoubleArray>(values, &sink_);
}

template <typename DType>
void PlainEncoder<DType>::put(const ::arrow::Array& values) {
  ParquetException::NYI("direct put of " + values.type()->ToString());
}

void assertBaseBinary(const ::arrow::Array& values) {
  if (!::arrow::is_base_binary_like(values.type_id())) {
    throw ParquetException("Only BaseBinaryArray and subclasses supported");
  }
}

template <>
inline void PlainEncoder<ByteArrayType>::put(const ::arrow::Array& values) {
  assertBaseBinary(values);

  if (::arrow::is_binary_like(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::BinaryArray&>(values));
  } else {
    VELOX_DCHECK(::arrow::is_large_binary_like(values.type_id()));
    putBinaryArray(checked_cast<const ::arrow::LargeBinaryArray&>(values));
  }
}

void assertFixedSizeBinary(const ::arrow::Array& values, int typeLength) {
  if (values.type_id() != ::arrow::Type::FIXED_SIZE_BINARY &&
      values.type_id() != ::arrow::Type::DECIMAL128 &&
      values.type_id() != ::arrow::Type::DECIMAL256) {
    throw ParquetException(
        "Only FixedSizeBinaryArray and subclasses supported");
  }
  if (checked_cast<const ::arrow::FixedSizeBinaryType&>(*values.type())
          .byte_width() != typeLength) {
    throw ParquetException(
        "Size mismatch: " + values.type()->ToString() + " should have been " +
        std::to_string(typeLength) + " wide");
  }
}

template <>
inline void PlainEncoder<FLBAType>::put(const ::arrow::Array& values) {
  assertFixedSizeBinary(values, descr_->typeLength());
  const auto& data = checked_cast<const ::arrow::FixedSizeBinaryArray&>(values);

  if (data.null_count() == 0) {
    // No nulls, just dump the data.
    PARQUET_THROW_NOT_OK(
        sink_.Append(data.raw_values(), data.length() * data.byte_width()));
  } else {
    const int64_t totalBytes = data.length() * data.byte_width() -
        data.null_count() * data.byte_width();
    PARQUET_THROW_NOT_OK(sink_.Reserve(totalBytes));
    for (int64_t i = 0; i < data.length(); i++) {
      if (data.IsValid(i)) {
        sink_.UnsafeAppend(data.Value(i), data.byte_width());
      }
    }
  }
}

template <>
inline void PlainEncoder<FLBAType>::put(
    const FixedLenByteArray* src,
    int numValues) {
  if (descr_->typeLength() == 0) {
    return;
  }
  for (int i = 0; i < numValues; ++i) {
    // Write the result to the output stream.
    VELOX_DCHECK(src[i].ptr != nullptr, "Value ptr cannot be NULL");
    PARQUET_THROW_NOT_OK(sink_.Append(src[i].ptr, descr_->typeLength()));
  }
}

template <>
class PlainEncoder<BooleanType> : public EncoderImpl,
                                  virtual public BooleanEncoder {
 public:
  explicit PlainEncoder(const ColumnDescriptor* descr, MemoryPool* pool)
      : EncoderImpl(descr, Encoding::kPlain, pool), sink_(pool) {}

  int64_t estimatedDataEncodedSize() override;
  std::shared_ptr<::arrow::Buffer> flushValues() override;

  void put(const bool* src, int numValues) override;

  void put(const std::vector<bool>& src, int numValues) override;

  void putSpaced(
      const bool* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    if (validBits != NULLPTR) {
      auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
      T* data = reinterpret_cast<T*>(buffer->mutable_data());
      int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
          src, numValues, validBits, validBitsOffset, data);
      put(data, numValidValues);
    } else {
      put(src, numValues);
    }
  }

  void put(const ::arrow::Array& values) override {
    if (values.type_id() != ::arrow::Type::BOOL) {
      throw ParquetException(
          "direct put to boolean from " + values.type()->ToString() +
          " not supported");
    }
    const auto& data = checked_cast<const ::arrow::BooleanArray&>(values);

    if (data.null_count() == 0) {
      // No nulls, just dump the data.
      PARQUET_THROW_NOT_OK(sink_.Reserve(data.length()));
      sink_.UnsafeAppend(
          data.data()->GetValues<uint8_t>(1, 0), data.offset(), data.length());
    } else {
      PARQUET_THROW_NOT_OK(sink_.Reserve(data.length() - data.null_count()));
      for (int64_t i = 0; i < data.length(); i++) {
        if (data.IsValid(i)) {
          sink_.UnsafeAppend(data.Value(i));
        }
      }
    }
  }

 private:
  ::arrow::TypedBufferBuilder<bool> sink_;

  template <typename SequenceType>
  void putImpl(const SequenceType& src, int numValues);
};

template <typename SequenceType>
void PlainEncoder<BooleanType>::putImpl(
    const SequenceType& src,
    int numValues) {
  PARQUET_THROW_NOT_OK(sink_.Reserve(numValues));
  for (int i = 0; i < numValues; ++i) {
    sink_.UnsafeAppend(src[i]);
  }
}

int64_t PlainEncoder<BooleanType>::estimatedDataEncodedSize() {
  return ::arrow::bit_util::BytesForBits(sink_.length());
}

std::shared_ptr<::arrow::Buffer> PlainEncoder<BooleanType>::flushValues() {
  std::shared_ptr<Buffer> buffer;
  PARQUET_THROW_NOT_OK(sink_.Finish(&buffer));
  return buffer;
}

void PlainEncoder<BooleanType>::put(const bool* src, int numValues) {
  putImpl(src, numValues);
}

void PlainEncoder<BooleanType>::put(
    const std::vector<bool>& src,
    int numValues) {
  putImpl(src, numValues);
}

// ----------------------------------------------------------------------.
// DictEncoder<T> implementations.

template <typename DType>
struct DictEncoderTraits {
  using CType = typename DType::CType;
  using MemoTableType = arrow::internal::ScalarMemoTable<CType>;
};

template <>
struct DictEncoderTraits<ByteArrayType> {
  using MemoTableType =
      arrow::internal::BinaryMemoTable<::arrow::BinaryBuilder>;
};

template <>
struct DictEncoderTraits<FLBAType> {
  using MemoTableType =
      arrow::internal::BinaryMemoTable<::arrow::BinaryBuilder>;
};

// Initially 1024 elements.
static constexpr int32_t kInitialHashTableSize = 1 << 10;

int rlePreserveBufferSize(int numValues, int bitWidth) {
  // Note: because of the way RleEncoder::CheckBufferFull()
  // Is called, we have to Reserve an extra "RleEncoder::MinBufferSize".
  // Bytes. These extra bytes won't be used but not reserving them.
  // Would cause the encoder to fail.
  return RleEncoder::MaxBufferSize(bitWidth, numValues) +
      RleEncoder::MinBufferSize(bitWidth);
}

/// See the dictionary encoding section of.
/// https://github.com/Parquet/parquet-format.  The encoding supports
/// Streaming encoding. Values are encoded as they are added while the.
/// Dictionary is being constructed. At any time, the buffered values.
/// Can be written out with the current dictionary size. More values.
/// Can then be added to the encoder, including new dictionary.
/// Entries.
template <typename DType>
class DictEncoderImpl : public EncoderImpl, virtual public DictEncoder<DType> {
  using MemoTableType = typename DictEncoderTraits<DType>::MemoTableType;

 public:
  typedef typename DType::CType T;

  /// In data page, the bit width used to encode the entry.
  /// Ids stored as 1 byte (max bit width = 32).
  constexpr static int32_t kDataPageBitWidthBytes = 1;

  explicit DictEncoderImpl(const ColumnDescriptor* desc, MemoryPool* pool)
      : EncoderImpl(desc, Encoding::kPlainDictionary, pool),
        bufferedIndices_(::arrow::stl::allocator<int32_t>(pool)),
        dictEncodedSize_(0),
        memoTable_(pool, kInitialHashTableSize) {}

  ~DictEncoderImpl() = default;

  int dictEncodedSize() const override {
    return dictEncodedSize_;
  }

  int writeIndices(uint8_t* buffer, int bufferLen) override {
    // Write bit width in first byte.
    *buffer = static_cast<uint8_t>(bitWidth());
    ++buffer;
    --bufferLen;

    RleEncoder encoder(buffer, bufferLen, bitWidth());

    for (int32_t index : bufferedIndices_) {
      if (ARROW_PREDICT_FALSE(!encoder.Put(index)))
        return -1;
    }
    encoder.Flush();

    clearIndices();
    return kDataPageBitWidthBytes + encoder.len();
  }

  void setTypeLength(int typeLength) {
    this->typeLength_ = typeLength;
  }

  /// Returns a conservative estimate of the number of bytes needed to encode.
  /// The buffered indices. Used to size the buffer passed to WriteIndices().
  int64_t estimatedDataEncodedSize() override {
    return kDataPageBitWidthBytes +
        rlePreserveBufferSize(
               static_cast<int>(bufferedIndices_.size()), bitWidth());
  }

  /// The minimum bit width required to encode the currently buffered indices.
  int bitWidth() const override {
    if (ARROW_PREDICT_FALSE(numEntries() == 0))
      return 0;
    if (ARROW_PREDICT_FALSE(numEntries() == 1))
      return 1;
    return ::arrow::bit_util::Log2(numEntries());
  }

  /// Encode value. Note that this does not actually write any data, just.
  /// Buffers the value's index to be written later.
  inline void put(const T& value);

  // Not implemented for other data types.
  inline void putByteArray(const void* ptr, int32_t length);

  void put(const T* src, int numValues) override {
    for (int32_t i = 0; i < numValues; i++) {
      put(safeLoad(src + i));
    }
  }

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    ::arrow::internal::VisitSetBitRunsVoid(
        validBits,
        validBitsOffset,
        numValues,
        [&](int64_t position, int64_t length) {
          for (int64_t i = 0; i < length; i++) {
            put(safeLoad(src + i + position));
          }
        });
  }

  using TypedEncoder<DType>::put;

  void put(const ::arrow::Array& values) override;
  void putDictionary(const ::arrow::Array& values) override;

  template <typename ArrowType, typename T = typename ArrowType::c_type>
  void putIndicesTyped(const ::arrow::Array& data) {
    auto values = data.data()->GetValues<T>(1);
    size_t bufferPosition = bufferedIndices_.size();
    bufferedIndices_.resize(
        bufferPosition +
        static_cast<size_t>(data.length() - data.null_count()));
    ::arrow::internal::VisitSetBitRunsVoid(
        data.null_bitmap_data(),
        data.offset(),
        data.length(),
        [&](int64_t position, int64_t length) {
          for (int64_t i = 0; i < length; ++i) {
            bufferedIndices_[bufferPosition++] =
                static_cast<int32_t>(values[i + position]);
          }
        });
  }

  void putIndices(const ::arrow::Array& data) override {
    switch (data.type()->id()) {
      case ::arrow::Type::UINT8:
      case ::arrow::Type::INT8:
        return putIndicesTyped<::arrow::UInt8Type>(data);
      case ::arrow::Type::UINT16:
      case ::arrow::Type::INT16:
        return putIndicesTyped<::arrow::UInt16Type>(data);
      case ::arrow::Type::UINT32:
      case ::arrow::Type::INT32:
        return putIndicesTyped<::arrow::UInt32Type>(data);
      case ::arrow::Type::UINT64:
      case ::arrow::Type::INT64:
        return putIndicesTyped<::arrow::UInt64Type>(data);
      default:
        throw ParquetException("Passed non-integer array to PutIndices");
    }
  }

  std::shared_ptr<::arrow::Buffer> flushValues() override {
    std::shared_ptr<ResizableBuffer> buffer =
        allocateBuffer(this->pool_, estimatedDataEncodedSize());
    int resultSize = writeIndices(
        buffer->mutable_data(), static_cast<int>(estimatedDataEncodedSize()));
    PARQUET_THROW_NOT_OK(buffer->Resize(resultSize, false));
    return std::move(buffer);
  }

  /// Writes out the encoded dictionary to buffer. buffer must be preallocated.
  /// To dict_encoded_size() bytes.
  void writeDict(uint8_t* buffer) const override;

  /// The number of entries in the dictionary.
  int numEntries() const override {
    return memoTable_.size();
  }

 private:
  /// Clears all the indices (but leaves the dictionary).
  void clearIndices() {
    bufferedIndices_.clear();
  }

  /// Indices that have not yet be written out by WriteIndices().
  ArrowPoolVector<int32_t> bufferedIndices_;

  template <typename ArrayType>
  void putBinaryArray(const ArrayType& array) {
    PARQUET_THROW_NOT_OK(
        ::arrow::VisitArraySpanInline<typename ArrayType::TypeClass>(
            *array.data(),
            [&](std::string_view view) {
              if (ARROW_PREDICT_FALSE(view.size() > kMaxByteArraySize)) {
                return Status::Invalid(
                    "Parquet cannot store strings with size 2GB or more");
              }
              putByteArray(view.data(), static_cast<uint32_t>(view.size()));
              return Status::OK();
            },
            []() { return Status::OK(); }));
  }

  template <typename ArrayType>
  void putBinaryDictionaryArray(const ArrayType& array) {
    VELOX_DCHECK_EQ(array.null_count(), 0);
    for (int64_t i = 0; i < array.length(); i++) {
      auto v = array.GetView(i);
      if (ARROW_PREDICT_FALSE(v.size() > kMaxByteArraySize)) {
        throw ParquetException(
            "Parquet cannot store strings with size 2GB or more");
      }
      dictEncodedSize_ += static_cast<int>(v.size() + sizeof(uint32_t));
      int32_t unusedMemoIndex;
      PARQUET_THROW_NOT_OK(memoTable_.getOrInsert(
          v.data(), static_cast<int32_t>(v.size()), &unusedMemoIndex));
    }
  }

  /// The number of bytes needed to encode the dictionary.
  int dictEncodedSize_;

  MemoTableType memoTable_;
};

template <typename DType>
void DictEncoderImpl<DType>::writeDict(uint8_t* buffer) const {
  // For primitive types, only a memcpy.
  VELOX_DCHECK_EQ(
      static_cast<size_t>(dictEncodedSize_), sizeof(T) * memoTable_.size());
  memoTable_.copyValues(0 /* start_pos */, reinterpret_cast<T*>(buffer));
}

// ByteArray and FLBA already have the dictionary encoded in their data heaps.
template <>
void DictEncoderImpl<ByteArrayType>::writeDict(uint8_t* buffer) const {
  memoTable_.visitValues(0, [&buffer](std::string_view v) {
    uint32_t len = static_cast<uint32_t>(v.length());
    memcpy(buffer, &len, sizeof(len));
    buffer += sizeof(len);
    memcpy(buffer, v.data(), len);
    buffer += len;
  });
}

template <>
void DictEncoderImpl<FLBAType>::writeDict(uint8_t* buffer) const {
  memoTable_.visitValues(0, [&](std::string_view v) {
    VELOX_DCHECK_EQ(v.length(), static_cast<size_t>(typeLength_));
    memcpy(buffer, v.data(), typeLength_);
    buffer += typeLength_;
  });
}

template <typename DType>
inline void DictEncoderImpl<DType>::put(const T& v) {
  // Put() implementation for primitive types.
  auto onFound = [](int32_t memoIndex) {};
  auto onNotFound = [this](int32_t memoIndex) {
    dictEncodedSize_ += static_cast<int>(sizeof(T));
  };

  int32_t memoIndex;
  PARQUET_THROW_NOT_OK(
      memoTable_.getOrInsert(v, onFound, onNotFound, &memoIndex));
  bufferedIndices_.push_back(memoIndex);
}

template <typename DType>
inline void DictEncoderImpl<DType>::putByteArray(
    const void* ptr,
    int32_t length) {
  VELOX_DCHECK(false);
}

template <>
inline void DictEncoderImpl<ByteArrayType>::putByteArray(
    const void* ptr,
    int32_t length) {
  static const uint8_t empty[] = {0};

  auto onFound = [](int32_t memoIndex) {};
  auto onNotFound = [&](int32_t memoIndex) {
    dictEncodedSize_ += static_cast<int>(length + sizeof(uint32_t));
  };

  VELOX_DCHECK(ptr != nullptr || length == 0);
  ptr = (ptr != nullptr) ? ptr : empty;
  int32_t memoIndex;
  PARQUET_THROW_NOT_OK(
      memoTable_.getOrInsert(ptr, length, onFound, onNotFound, &memoIndex));
  bufferedIndices_.push_back(memoIndex);
}

template <>
inline void DictEncoderImpl<ByteArrayType>::put(const ByteArray& val) {
  return putByteArray(val.ptr, static_cast<int32_t>(val.len));
}

template <>
inline void DictEncoderImpl<FLBAType>::put(const FixedLenByteArray& v) {
  static const uint8_t empty[] = {0};

  auto onFound = [](int32_t memoIndex) {};
  auto onNotFound = [this](int32_t memoIndex) {
    dictEncodedSize_ += typeLength_;
  };

  VELOX_DCHECK(v.ptr != nullptr || typeLength_ == 0);
  const void* ptr = (v.ptr != nullptr) ? v.ptr : empty;
  int32_t memoIndex;
  PARQUET_THROW_NOT_OK(memoTable_.getOrInsert(
      ptr, typeLength_, onFound, onNotFound, &memoIndex));
  bufferedIndices_.push_back(memoIndex);
}

template <>
void DictEncoderImpl<Int96Type>::put(const ::arrow::Array& values) {
  ParquetException::NYI("Direct put to Int96");
}

template <>
void DictEncoderImpl<Int96Type>::putDictionary(const ::arrow::Array& values) {
  ParquetException::NYI("Direct put to Int96");
}

template <typename DType>
void DictEncoderImpl<DType>::put(const ::arrow::Array& values) {
  using ArrayType =
      typename ::arrow::CTypeTraits<typename DType::CType>::ArrayType;
  const auto& data = checked_cast<const ArrayType&>(values);
  if (data.null_count() == 0) {
    // No nulls, just dump the data.
    for (int64_t i = 0; i < data.length(); i++) {
      put(data.Value(i));
    }
  } else {
    for (int64_t i = 0; i < data.length(); i++) {
      if (data.IsValid(i)) {
        put(data.Value(i));
      }
    }
  }
}

template <>
void DictEncoderImpl<FLBAType>::put(const ::arrow::Array& values) {
  assertFixedSizeBinary(values, typeLength_);
  const auto& data = checked_cast<const ::arrow::FixedSizeBinaryArray&>(values);
  if (data.null_count() == 0) {
    // No nulls, just dump the data.
    for (int64_t i = 0; i < data.length(); i++) {
      put(FixedLenByteArray(data.Value(i)));
    }
  } else {
    std::vector<uint8_t> empty(typeLength_, 0);
    for (int64_t i = 0; i < data.length(); i++) {
      if (data.IsValid(i)) {
        put(FixedLenByteArray(data.Value(i)));
      }
    }
  }
}

template <>
void DictEncoderImpl<ByteArrayType>::put(const ::arrow::Array& values) {
  assertBaseBinary(values);
  if (::arrow::is_binary_like(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::BinaryArray&>(values));
  } else {
    VELOX_DCHECK(::arrow::is_large_binary_like(values.type_id()));
    putBinaryArray(checked_cast<const ::arrow::LargeBinaryArray&>(values));
  }
}

template <typename DType>
void assertCanPutDictionary(
    DictEncoderImpl<DType>* encoder,
    const ::arrow::Array& dict) {
  if (dict.null_count() > 0) {
    throw ParquetException("Inserted dictionary cannot cannot contain nulls");
  }

  if (encoder->numEntries() > 0) {
    throw ParquetException(
        "Can only call PutDictionary on an empty DictEncoder");
  }
}

template <typename DType>
void DictEncoderImpl<DType>::putDictionary(const ::arrow::Array& values) {
  assertCanPutDictionary(this, values);

  using ArrayType =
      typename ::arrow::CTypeTraits<typename DType::CType>::ArrayType;
  const auto& data = checked_cast<const ArrayType&>(values);

  dictEncodedSize_ +=
      static_cast<int>(sizeof(typename DType::CType) * data.length());
  for (int64_t i = 0; i < data.length(); i++) {
    int32_t unusedMemoIndex;
    PARQUET_THROW_NOT_OK(
        memoTable_.getOrInsert(data.Value(i), &unusedMemoIndex));
  }
}

template <>
void DictEncoderImpl<FLBAType>::putDictionary(const ::arrow::Array& values) {
  assertFixedSizeBinary(values, typeLength_);
  assertCanPutDictionary(this, values);

  const auto& data = checked_cast<const ::arrow::FixedSizeBinaryArray&>(values);

  dictEncodedSize_ += static_cast<int>(typeLength_ * data.length());
  for (int64_t i = 0; i < data.length(); i++) {
    int32_t unusedMemoIndex;
    PARQUET_THROW_NOT_OK(
        memoTable_.getOrInsert(data.Value(i), typeLength_, &unusedMemoIndex));
  }
}

template <>
void DictEncoderImpl<ByteArrayType>::putDictionary(
    const ::arrow::Array& values) {
  assertBaseBinary(values);
  assertCanPutDictionary(this, values);

  if (::arrow::is_binary_like(values.type_id())) {
    putBinaryDictionaryArray(checked_cast<const ::arrow::BinaryArray&>(values));
  } else {
    VELOX_DCHECK(::arrow::is_large_binary_like(values.type_id()));
    putBinaryDictionaryArray(
        checked_cast<const ::arrow::LargeBinaryArray&>(values));
  }
}

// ----------------------------------------------------------------------.
// ByteStreamSplitEncoder<T> implementations.

template <typename DType>
class ByteStreamSplitEncoder : public EncoderImpl,
                               virtual public TypedEncoder<DType> {
 public:
  using T = typename DType::CType;
  using TypedEncoder<DType>::put;

  explicit ByteStreamSplitEncoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  int64_t estimatedDataEncodedSize() override;
  std::shared_ptr<::arrow::Buffer> flushValues() override;

  void put(const T* buffer, int numValues) override;
  void put(const ::arrow::Array& values) override;
  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override;

 protected:
  template <typename ArrowType>
  void putImpl(const ::arrow::Array& values) {
    if (values.type_id() != ArrowType::type_id) {
      throw ParquetException(
          std::string() + "direct put to " + ArrowType::type_name() + " from " +
          values.type()->ToString() + " not supported");
    }
    const auto& data = *values.data();
    putSpaced(
        data.GetValues<typename ArrowType::c_type>(1),
        static_cast<int>(data.length),
        data.GetValues<uint8_t>(0, 0),
        data.offset);
  }

  ::arrow::BufferBuilder sink_;
  int64_t numValuesInBuffer_;
};

template <typename DType>
ByteStreamSplitEncoder<DType>::ByteStreamSplitEncoder(
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool)
    : EncoderImpl(descr, Encoding::kByteStreamSplit, pool),
      sink_{pool},
      numValuesInBuffer_{0} {}

template <typename DType>
int64_t ByteStreamSplitEncoder<DType>::estimatedDataEncodedSize() {
  return sink_.length();
}

template <typename DType>
std::shared_ptr<::arrow::Buffer> ByteStreamSplitEncoder<DType>::flushValues() {
  std::shared_ptr<ResizableBuffer> outputBuffer =
      allocateBuffer(this->memoryPool(), estimatedDataEncodedSize());
  uint8_t* outputBufferRaw = outputBuffer->mutable_data();
  const uint8_t* rawValues = sink_.data();
  byteStreamSplitEncode<T>(rawValues, numValuesInBuffer_, outputBufferRaw);
  sink_.Reset();
  numValuesInBuffer_ = 0;
  return std::move(outputBuffer);
}

template <typename DType>
void ByteStreamSplitEncoder<DType>::put(const T* buffer, int numValues) {
  if (numValues > 0) {
    PARQUET_THROW_NOT_OK(sink_.Append(buffer, numValues * sizeof(T)));
    numValuesInBuffer_ += numValues;
  }
}

template <>
void ByteStreamSplitEncoder<FloatType>::put(const ::arrow::Array& values) {
  putImpl<::arrow::FloatType>(values);
}

template <>
void ByteStreamSplitEncoder<DoubleType>::put(const ::arrow::Array& values) {
  putImpl<::arrow::DoubleType>(values);
}

template <typename DType>
void ByteStreamSplitEncoder<DType>::putSpaced(
    const T* src,
    int numValues,
    const uint8_t* validBits,
    int64_t validBitsOffset) {
  if (validBits != NULLPTR) {
    auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
        src, numValues, validBits, validBitsOffset, data);
    put(data, numValidValues);
  } else {
    put(src, numValues);
  }
}

class DecoderImpl : virtual public Decoder {
 public:
  void setData(int numValues, const uint8_t* data, int len) override {
    numValues_ = numValues;
    data_ = data;
    len_ = len;
  }

  int valuesLeft() const override {
    return numValues_;
  }
  Encoding::type encoding() const override {
    return encoding_;
  }

 protected:
  explicit DecoderImpl(const ColumnDescriptor* descr, Encoding::type encoding)
      : descr_(descr),
        encoding_(encoding),
        numValues_(0),
        data_(NULLPTR),
        len_(0) {}

  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY.
  const ColumnDescriptor* descr_;

  const Encoding::type encoding_;
  int numValues_;
  const uint8_t* data_;
  int len_;
  int typeLength_;
};

template <typename DType>
class PlainDecoder : public DecoderImpl, virtual public TypedDecoder<DType> {
 public:
  using T = typename DType::CType;
  explicit PlainDecoder(const ColumnDescriptor* descr);

  int decode(T* buffer, int maxValues) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::Accumulator* Builder) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::DictAccumulator* Builder) override;
};

template <>
inline int PlainDecoder<Int96Type>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<Int96Type>::Accumulator* Builder) {
  ParquetException::NYI("DecodeArrow not supported for Int96");
}

template <>
inline int PlainDecoder<Int96Type>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<Int96Type>::DictAccumulator* Builder) {
  ParquetException::NYI("DecodeArrow not supported for Int96");
}

template <>
inline int PlainDecoder<BooleanType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<BooleanType>::DictAccumulator* Builder) {
  ParquetException::NYI("dictionaries of BooleanType");
}

template <typename DType>
int PlainDecoder<DType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<DType>::Accumulator* Builder) {
  using ValueType = typename DType::CType;

  constexpr int valueSize = static_cast<int>(sizeof(ValueType));
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(len_ < valueSize * valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        Builder->UnsafeAppend(SafeLoadAs<ValueType>(data_));
        data_ += sizeof(ValueType);
      },
      [&]() { Builder->UnsafeAppendNull(); });

  numValues_ -= valuesDecoded;
  len_ -= sizeof(ValueType) * valuesDecoded;
  return valuesDecoded;
}

template <typename DType>
int PlainDecoder<DType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<DType>::DictAccumulator* Builder) {
  using ValueType = typename DType::CType;

  constexpr int valueSize = static_cast<int>(sizeof(ValueType));
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(len_ < valueSize * valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        PARQUET_THROW_NOT_OK(Builder->Append(SafeLoadAs<ValueType>(data_)));
        data_ += sizeof(ValueType);
      },
      [&]() { PARQUET_THROW_NOT_OK(Builder->AppendNull()); });

  numValues_ -= valuesDecoded;
  len_ -= sizeof(ValueType) * valuesDecoded;
  return valuesDecoded;
}

// Decode routine templated on C++ type rather than type enum.
template <typename T>
inline int decodePlain(
    const uint8_t* data,
    int64_t dataSize,
    int numValues,
    int typeLength,
    T* out) {
  int64_t bytesToDecode = numValues * static_cast<int64_t>(sizeof(T));
  if (bytesToDecode > dataSize || bytesToDecode > INT_MAX) {
    ParquetException::eofException();
  }
  // If bytes_to_decode == 0, data could be null.
  if (bytesToDecode > 0) {
    memcpy(out, data, bytesToDecode);
  }
  return static_cast<int>(bytesToDecode);
}

template <typename DType>
PlainDecoder<DType>::PlainDecoder(const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::kPlain) {
  if (descr_ && descr_->physicalType() == Type::kFixedLenByteArray) {
    typeLength_ = descr_->typeLength();
  } else {
    typeLength_ = -1;
  }
}

// Template specialization for BYTE_ARRAY. The written values do not own their.
// Own data.

static inline int64_t
readByteArray(const uint8_t* data, int64_t dataSize, ByteArray* out) {
  if (ARROW_PREDICT_FALSE(dataSize < 4)) {
    ParquetException::eofException();
  }
  const int32_t len = SafeLoadAs<int32_t>(data);
  if (len < 0) {
    throw ParquetException("Invalid BYTE_ARRAY value");
  }
  const int64_t consumedLength = static_cast<int64_t>(len) + 4;
  if (ARROW_PREDICT_FALSE(dataSize < consumedLength)) {
    ParquetException::eofException();
  }
  *out = ByteArray{static_cast<uint32_t>(len), data + 4};
  return consumedLength;
}

template <>
inline int decodePlain<ByteArray>(
    const uint8_t* data,
    int64_t dataSize,
    int numValues,
    int typeLength,
    ByteArray* out) {
  int bytesDecoded = 0;
  for (int i = 0; i < numValues; ++i) {
    const auto increment = readByteArray(data, dataSize, out + i);
    if (ARROW_PREDICT_FALSE(increment > INT_MAX - bytesDecoded)) {
      throw ParquetException("BYTE_ARRAY chunk too large");
    }
    data += increment;
    dataSize -= increment;
    bytesDecoded += static_cast<int>(increment);
  }
  return bytesDecoded;
}

// Template specialization for FIXED_LEN_BYTE_ARRAY. The written values do not.
// Own their own data.
template <>
inline int decodePlain<FixedLenByteArray>(
    const uint8_t* data,
    int64_t dataSize,
    int numValues,
    int typeLength,
    FixedLenByteArray* out) {
  int64_t bytesToDecode = static_cast<int64_t>(typeLength) * numValues;
  if (bytesToDecode > dataSize || bytesToDecode > INT_MAX) {
    ParquetException::eofException();
  }
  for (int i = 0; i < numValues; ++i) {
    out[i].ptr = data;
    data += typeLength;
    dataSize -= typeLength;
  }
  return static_cast<int>(bytesToDecode);
}

template <typename DType>
int PlainDecoder<DType>::decode(T* buffer, int maxValues) {
  maxValues = std::min(maxValues, numValues_);
  int bytesConsumed =
      decodePlain<T>(data_, len_, maxValues, typeLength_, buffer);
  data_ += bytesConsumed;
  len_ -= bytesConsumed;
  numValues_ -= maxValues;
  return maxValues;
}

class PlainBooleanDecoder : public DecoderImpl, virtual public BooleanDecoder {
 public:
  explicit PlainBooleanDecoder(const ColumnDescriptor* descr);
  void setData(int numValues, const uint8_t* data, int len) override;

  // Two flavors of bool decoding.
  int decode(uint8_t* buffer, int maxValues) override;
  int decode(bool* buffer, int maxValues) override;
  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<BooleanType>::Accumulator* out) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<BooleanType>::DictAccumulator* out) override;

 private:
  std::unique_ptr<BitReader> bitReader_;
};

PlainBooleanDecoder::PlainBooleanDecoder(const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::kPlain) {}

void PlainBooleanDecoder::setData(int numValues, const uint8_t* data, int len) {
  numValues_ = numValues;
  bitReader_ = std::make_unique<BitReader>(data, len);
}

int PlainBooleanDecoder::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<BooleanType>::Accumulator* Builder) {
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(numValues_ < valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        bool value;
        ((void)(bitReader_->GetValue(1, &value)));
        Builder->UnsafeAppend(value);
      },
      [&]() { Builder->UnsafeAppendNull(); });

  numValues_ -= valuesDecoded;
  return valuesDecoded;
}

inline int PlainBooleanDecoder::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<BooleanType>::DictAccumulator* Builder) {
  ParquetException::NYI("dictionaries of BooleanType");
}

int PlainBooleanDecoder::decode(uint8_t* buffer, int maxValues) {
  maxValues = std::min(maxValues, numValues_);
  bool val;
  ::arrow::internal::BitmapWriter bitWriter(buffer, 0, maxValues);
  for (int i = 0; i < maxValues; ++i) {
    if (!bitReader_->GetValue(1, &val)) {
      ParquetException::eofException();
    }
    if (val) {
      bitWriter.Set();
    }
    bitWriter.Next();
  }
  bitWriter.Finish();
  numValues_ -= maxValues;
  return maxValues;
}

int PlainBooleanDecoder::decode(bool* buffer, int maxValues) {
  maxValues = std::min(maxValues, numValues_);
  if (bitReader_->GetBatch(1, buffer, maxValues) != maxValues) {
    ParquetException::eofException();
  }
  numValues_ -= maxValues;
  return maxValues;
}

// A helper class to abstract away differences between.
// EncodingTraits<DType>::Accumulator for ByteArrayType and FLBAType.
template <typename DType>
struct ArrowBinaryHelper;

template <>
struct ArrowBinaryHelper<ByteArrayType> {
  using Accumulator = typename EncodingTraits<ByteArrayType>::Accumulator;

  ArrowBinaryHelper(Accumulator* acc, int64_t length)
      : acc_(acc),
        entriesRemaining_(length),
        chunkSpaceRemaining_(
            ::arrow::kBinaryMemoryLimit - acc_->Builder->value_data_length()) {}

  Status prepare(std::optional<int64_t> estimatedDataLength = {}) {
    RETURN_NOT_OK(acc_->Builder->Reserve(entriesRemaining_));
    if (estimatedDataLength.has_value()) {
      RETURN_NOT_OK(acc_->Builder->ReserveData(
          std::min<int64_t>(
              *estimatedDataLength, ::arrow::kBinaryMemoryLimit)));
    }
    return Status::OK();
  }

  Status prepareNextInput(
      int64_t nextValueLength,
      std::optional<int64_t> estimatedRemainingDataLength = {}) {
    if (ARROW_PREDICT_FALSE(!canFit(nextValueLength))) {
      // This element would exceed the capacity of a chunk.
      RETURN_NOT_OK(pushChunk());
      RETURN_NOT_OK(acc_->Builder->Reserve(entriesRemaining_));
      if (estimatedRemainingDataLength.has_value()) {
        RETURN_NOT_OK(acc_->Builder->ReserveData(
            std::min<int64_t>(
                *estimatedRemainingDataLength, chunkSpaceRemaining_)));
      }
    }
    return Status::OK();
  }

  void UnsafeAppend(const uint8_t* data, int32_t length) {
    VELOX_DCHECK(canFit(length));
    VELOX_DCHECK_GT(entriesRemaining_, 0);
    chunkSpaceRemaining_ -= length;
    --entriesRemaining_;
    acc_->Builder->UnsafeAppend(data, length);
  }

  Status Append(const uint8_t* data, int32_t length) {
    VELOX_DCHECK(canFit(length));
    VELOX_DCHECK_GT(entriesRemaining_, 0);
    chunkSpaceRemaining_ -= length;
    --entriesRemaining_;
    return acc_->Builder->Append(data, length);
  }

  void UnsafeAppendNull() {
    --entriesRemaining_;
    acc_->Builder->UnsafeAppendNull();
  }

  Status AppendNull() {
    --entriesRemaining_;
    return acc_->Builder->AppendNull();
  }

 private:
  Status pushChunk() {
    ARROW_ASSIGN_OR_RAISE(auto chunk, acc_->Builder->Finish());
    acc_->chunks.push_back(std::move(chunk));
    chunkSpaceRemaining_ = ::arrow::kBinaryMemoryLimit;
    return Status::OK();
  }

  bool canFit(int64_t length) const {
    return length <= chunkSpaceRemaining_;
  }

  Accumulator* acc_;
  int64_t entriesRemaining_;
  int64_t chunkSpaceRemaining_;
};

template <>
struct ArrowBinaryHelper<FLBAType> {
  using Accumulator = typename EncodingTraits<FLBAType>::Accumulator;

  ArrowBinaryHelper(Accumulator* acc, int64_t length)
      : acc_(acc), entriesRemaining_(length) {}

  Status prepare(std::optional<int64_t> estimatedDataLength = {}) {
    return acc_->Reserve(entriesRemaining_);
  }

  Status prepareNextInput(
      int64_t nextValueLength,
      std::optional<int64_t> estimatedRemainingDataLength = {}) {
    return Status::OK();
  }

  void UnsafeAppend(const uint8_t* data, int32_t length) {
    VELOX_DCHECK_GT(entriesRemaining_, 0);
    --entriesRemaining_;
    acc_->UnsafeAppend(data);
  }

  Status Append(const uint8_t* data, int32_t length) {
    VELOX_DCHECK_GT(entriesRemaining_, 0);
    --entriesRemaining_;
    return acc_->Append(data);
  }

  void UnsafeAppendNull() {
    --entriesRemaining_;
    acc_->UnsafeAppendNull();
  }

  Status AppendNull() {
    --entriesRemaining_;
    return acc_->AppendNull();
  }

 private:
  Accumulator* acc_;
  int64_t entriesRemaining_;
};

template <>
inline int PlainDecoder<ByteArrayType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<ByteArrayType>::Accumulator* Builder) {
  ParquetException::NYI();
}

template <>
inline int PlainDecoder<ByteArrayType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<ByteArrayType>::DictAccumulator* Builder) {
  ParquetException::NYI();
}

template <>
inline int PlainDecoder<FLBAType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<FLBAType>::Accumulator* Builder) {
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(len_ < descr_->typeLength() * valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        Builder->UnsafeAppend(data_);
        data_ += descr_->typeLength();
      },
      [&]() { Builder->UnsafeAppendNull(); });

  numValues_ -= valuesDecoded;
  len_ -= descr_->typeLength() * valuesDecoded;
  return valuesDecoded;
}

template <>
inline int PlainDecoder<FLBAType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<FLBAType>::DictAccumulator* Builder) {
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(len_ < descr_->typeLength() * valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        PARQUET_THROW_NOT_OK(Builder->Append(data_));
        data_ += descr_->typeLength();
      },
      [&]() { PARQUET_THROW_NOT_OK(Builder->AppendNull()); });

  numValues_ -= valuesDecoded;
  len_ -= descr_->typeLength() * valuesDecoded;
  return valuesDecoded;
}

class PlainByteArrayDecoder : public PlainDecoder<ByteArrayType>,
                              virtual public ByteArrayDecoder {
 public:
  using Base = PlainDecoder<ByteArrayType>;
  using Base::decodeSpaced;
  using Base::PlainDecoder;

  // ----------------------------------------------------------------------.
  // Dictionary read paths.

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      ::arrow::BinaryDictionary32Builder* Builder) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(decodeArrow(
        numValues, nullCount, validBits, validBitsOffset, Builder, &result));
    return result;
  }

  // ----------------------------------------------------------------------.
  // Optimized dense binary read paths.

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(decodeArrowDense(
        numValues, nullCount, validBits, validBitsOffset, out, &result));
    return result;
  }

 private:
  Status decodeArrowDense(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out,
      int* outValuesDecoded) {
    ArrowBinaryHelper<ByteArrayType> helper(out, numValues);
    int valuesDecoded = 0;

    RETURN_NOT_OK(helper.prepare(len_));

    int i = 0;
    RETURN_NOT_OK(VisitNullBitmapInline(
        validBits,
        validBitsOffset,
        numValues,
        nullCount,
        [&]() {
          if (ARROW_PREDICT_FALSE(len_ < 4)) {
            ParquetException::eofException();
          }
          auto valueLen = SafeLoadAs<int32_t>(data_);
          if (ARROW_PREDICT_FALSE(valueLen < 0 || valueLen > INT32_MAX - 4)) {
            return Status::Invalid(
                "Invalid or corrupted value_len '", valueLen, "'");
          }
          auto increment = valueLen + 4;
          if (ARROW_PREDICT_FALSE(len_ < increment)) {
            ParquetException::eofException();
          }
          RETURN_NOT_OK(helper.prepareNextInput(valueLen, len_));
          helper.UnsafeAppend(data_ + 4, valueLen);
          data_ += increment;
          len_ -= increment;
          ++valuesDecoded;
          ++i;
          return Status::OK();
        },
        [&]() {
          helper.UnsafeAppendNull();
          ++i;
          return Status::OK();
        }));

    numValues_ -= valuesDecoded;
    *outValuesDecoded = valuesDecoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      BuilderType* Builder,
      int* outValuesDecoded) {
    RETURN_NOT_OK(Builder->Reserve(numValues));
    int valuesDecoded = 0;

    RETURN_NOT_OK(VisitNullBitmapInline(
        validBits,
        validBitsOffset,
        numValues,
        nullCount,
        [&]() {
          if (ARROW_PREDICT_FALSE(len_ < 4)) {
            ParquetException::eofException();
          }
          auto valueLen = SafeLoadAs<int32_t>(data_);
          if (ARROW_PREDICT_FALSE(valueLen < 0 || valueLen > INT32_MAX - 4)) {
            return Status::Invalid(
                "Invalid or corrupted value_len '", valueLen, "'");
          }
          auto increment = valueLen + 4;
          if (ARROW_PREDICT_FALSE(len_ < increment)) {
            ParquetException::eofException();
          }
          RETURN_NOT_OK(Builder->Append(data_ + 4, valueLen));
          data_ += increment;
          len_ -= increment;
          ++valuesDecoded;
          return Status::OK();
        },
        [&]() { return Builder->AppendNull(); }));

    numValues_ -= valuesDecoded;
    *outValuesDecoded = valuesDecoded;
    return Status::OK();
  }
};

class PlainFLBADecoder : public PlainDecoder<FLBAType>,
                         virtual public FLBADecoder {
 public:
  using Base = PlainDecoder<FLBAType>;
  using Base::PlainDecoder;
};

// ----------------------------------------------------------------------.
// Dictionary encoding and decoding.

template <typename Type>
class DictDecoderImpl : public DecoderImpl, virtual public DictDecoder<Type> {
 public:
  typedef typename Type::CType T;

  // Initializes the dictionary with values from 'dictionary'. The data in.
  // Dictionary is not guaranteed to persist in memory after this call so the.
  // Dictionary decoder needs to copy the data out if necessary.
  explicit DictDecoderImpl(
      const ColumnDescriptor* descr,
      MemoryPool* pool = ::arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::kRleDictionary),
        dictionary_(allocateBuffer(pool, 0)),
        dictionaryLength_(0),
        byteArrayData_(allocateBuffer(pool, 0)),
        byteArrayOffsets_(allocateBuffer(pool, 0)),
        indicesScratchSpace_(allocateBuffer(pool, 0)) {}

  // Perform type-specific initialization.
  void setDict(TypedDecoder<Type>* dictionary) override;

  void setData(int numValues, const uint8_t* data, int len) override {
    numValues_ = numValues;
    if (len == 0) {
      // Initialize dummy decoder to avoid crashes later on.
      idxDecoder_ = RleDecoder(data, len, /*bitWidth=*/1);
      return;
    }
    uint8_t bitWidth = *data;
    if (ARROW_PREDICT_FALSE(bitWidth > 32)) {
      throw ParquetException(
          "Invalid or corrupted bit_width " + std::to_string(bitWidth) +
          ". Maximum allowed is 32.");
    }
    idxDecoder_ = RleDecoder(++data, --len, bitWidth);
  }

  int decode(T* buffer, int numValues) override {
    numValues = std::min(numValues, numValues_);
    int decodedValues = idxDecoder_.GetBatchWithDict(
        reinterpret_cast<const T*>(dictionary_->data()),
        dictionaryLength_,
        buffer,
        numValues);
    if (decodedValues != numValues) {
      ParquetException::eofException();
    }
    numValues_ -= numValues;
    return numValues;
  }

  int decodeSpaced(
      T* buffer,
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    numValues = std::min(numValues, numValues_);
    if (numValues !=
        idxDecoder_.GetBatchWithDictSpaced(
            reinterpret_cast<const T*>(dictionary_->data()),
            dictionaryLength_,
            buffer,
            numValues,
            nullCount,
            validBits,
            validBitsOffset)) {
      ParquetException::eofException();
    }
    numValues_ -= numValues;
    return numValues;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<Type>::Accumulator* out) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<Type>::DictAccumulator* out) override;

  void insertDictionary(::arrow::ArrayBuilder* Builder) override;

  int decodeIndicesSpaced(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      ::arrow::ArrayBuilder* Builder) override {
    if (numValues > 0) {
      // TODO(wesm): Refactor to batch reads for improved memory use. It is not.
      // Trivial because the null_count is relative to the entire bitmap.
      PARQUET_THROW_NOT_OK(indicesScratchSpace_->TypedResize<int32_t>(
          numValues, /*shrink_to_fit=*/false));
    }

    auto indicesBuffer =
        reinterpret_cast<int32_t*>(indicesScratchSpace_->mutable_data());

    if (numValues !=
        idxDecoder_.GetBatchSpaced(
            numValues, nullCount, validBits, validBitsOffset, indicesBuffer)) {
      ParquetException::eofException();
    }

    // XXX(wesm): Cannot Append "valid bits" directly to the builder.
    std::vector<uint8_t> validBytes(numValues, 0);
    int64_t i = 0;
    VisitNullBitmapInline(
        validBits,
        validBitsOffset,
        numValues,
        nullCount,
        [&]() { validBytes[i++] = 1; },
        [&]() { ++i; });

    auto binaryBuilder =
        checked_cast<::arrow::BinaryDictionary32Builder*>(Builder);
    PARQUET_THROW_NOT_OK(binaryBuilder->AppendIndices(
        indicesBuffer, numValues, validBytes.data()));
    numValues_ -= numValues - nullCount;
    return numValues - nullCount;
  }

  int decodeIndices(int numValues, ::arrow::ArrayBuilder* Builder) override {
    numValues = std::min(numValues, numValues_);
    if (numValues > 0) {
      // TODO(wesm): Refactor to batch reads for improved memory use. This is.
      // Relatively simple here because we don't have to do any bookkeeping of.
      // Nulls.
      PARQUET_THROW_NOT_OK(indicesScratchSpace_->TypedResize<int32_t>(
          numValues, /*shrink_to_fit=*/false));
    }
    auto indicesBuffer =
        reinterpret_cast<int32_t*>(indicesScratchSpace_->mutable_data());
    if (numValues != idxDecoder_.GetBatch(indicesBuffer, numValues)) {
      ParquetException::eofException();
    }
    auto binaryBuilder =
        checked_cast<::arrow::BinaryDictionary32Builder*>(Builder);
    PARQUET_THROW_NOT_OK(
        binaryBuilder->AppendIndices(indicesBuffer, numValues));
    numValues_ -= numValues;
    return numValues;
  }

  int decodeIndices(int numValues, int32_t* indices) override {
    if (numValues != idxDecoder_.GetBatch(indices, numValues)) {
      ParquetException::eofException();
    }
    numValues_ -= numValues;
    return numValues;
  }

  void getDictionary(const T** dictionary, int32_t* dictionaryLength) override {
    *dictionaryLength = dictionaryLength_;
    *dictionary = reinterpret_cast<T*>(dictionary_->mutable_data());
  }

 protected:
  Status indexInBounds(int32_t index) {
    if (ARROW_PREDICT_TRUE(0 <= index && index < dictionaryLength_)) {
      return Status::OK();
    }
    return Status::Invalid("Index not in dictionary bounds");
  }

  inline void decodeDict(TypedDecoder<Type>* dictionary) {
    dictionaryLength_ = static_cast<int32_t>(dictionary->valuesLeft());
    PARQUET_THROW_NOT_OK(dictionary_->Resize(
        dictionaryLength_ * sizeof(T),
        /*shrink_to_fit=*/false));
    dictionary->decode(
        reinterpret_cast<T*>(dictionary_->mutable_data()), dictionaryLength_);
  }

  // Only one is set.
  std::shared_ptr<ResizableBuffer> dictionary_;

  int32_t dictionaryLength_;

  // Data that contains the byte array data (byte_array_dictionary_ just has
  // the. Pointers).
  std::shared_ptr<ResizableBuffer> byteArrayData_;

  // Arrow-style byte offsets for each dictionary value. We maintain two.
  // Representations of the dictionary, one as ByteArray* for non-Arrow.
  // Consumers and this one for Arrow consumers. Since dictionaries are.
  // Generally pretty small to begin with this doesn't mean too much extra.
  // Memory use in most cases.
  std::shared_ptr<ResizableBuffer> byteArrayOffsets_;

  // Reusable buffer for decoding dictionary indices to be appended to a.
  // BinaryDictionary32Builder.
  std::shared_ptr<ResizableBuffer> indicesScratchSpace_;

  RleDecoder idxDecoder_;
};

template <typename Type>
void DictDecoderImpl<Type>::setDict(TypedDecoder<Type>* dictionary) {
  decodeDict(dictionary);
}

template <>
void DictDecoderImpl<BooleanType>::setDict(
    TypedDecoder<BooleanType>* dictionary) {
  ParquetException::NYI(
      "Dictionary encoding is not implemented for boolean values");
}

template <>
void DictDecoderImpl<ByteArrayType>::setDict(
    TypedDecoder<ByteArrayType>* dictionary) {
  decodeDict(dictionary);

  auto dictValues = reinterpret_cast<ByteArray*>(dictionary_->mutable_data());

  int totalSize = 0;
  for (int i = 0; i < dictionaryLength_; ++i) {
    totalSize += dictValues[i].len;
  }
  PARQUET_THROW_NOT_OK(byteArrayData_->Resize(
      totalSize,
      /*shrink_to_fit=*/false));
  PARQUET_THROW_NOT_OK(byteArrayOffsets_->Resize(
      (dictionaryLength_ + 1) * sizeof(int32_t),
      /*shrink_to_fit=*/false));

  int32_t offset = 0;
  uint8_t* bytesData = byteArrayData_->mutable_data();
  int32_t* bytesOffsets =
      reinterpret_cast<int32_t*>(byteArrayOffsets_->mutable_data());
  for (int i = 0; i < dictionaryLength_; ++i) {
    memcpy(bytesData + offset, dictValues[i].ptr, dictValues[i].len);
    bytesOffsets[i] = offset;
    dictValues[i].ptr = bytesData + offset;
    offset += dictValues[i].len;
  }
  bytesOffsets[dictionaryLength_] = offset;
}

template <>
inline void DictDecoderImpl<FLBAType>::setDict(
    TypedDecoder<FLBAType>* dictionary) {
  decodeDict(dictionary);

  auto dictValues = reinterpret_cast<FLBA*>(dictionary_->mutable_data());

  int fixedLen = descr_->typeLength();
  int totalSize = dictionaryLength_ * fixedLen;

  PARQUET_THROW_NOT_OK(byteArrayData_->Resize(
      totalSize,
      /*shrink_to_fit=*/false));
  uint8_t* bytesData = byteArrayData_->mutable_data();
  for (int32_t i = 0, offset = 0; i < dictionaryLength_;
       ++i, offset += fixedLen) {
    memcpy(bytesData + offset, dictValues[i].ptr, fixedLen);
    dictValues[i].ptr = bytesData + offset;
  }
}

template <>
inline int DictDecoderImpl<Int96Type>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<Int96Type>::Accumulator* Builder) {
  ParquetException::NYI("DecodeArrow to Int96Type");
}

template <>
inline int DictDecoderImpl<Int96Type>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<Int96Type>::DictAccumulator* Builder) {
  ParquetException::NYI("DecodeArrow to Int96Type");
}

template <>
inline int DictDecoderImpl<ByteArrayType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<ByteArrayType>::Accumulator* Builder) {
  ParquetException::NYI("DecodeArrow implemented elsewhere");
}

template <>
inline int DictDecoderImpl<ByteArrayType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<ByteArrayType>::DictAccumulator* Builder) {
  ParquetException::NYI("DecodeArrow implemented elsewhere");
}

template <typename DType>
int DictDecoderImpl<DType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<DType>::DictAccumulator* Builder) {
  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  auto dictValues =
      reinterpret_cast<const typename DType::CType*>(dictionary_->data());

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        int32_t index;
        if (ARROW_PREDICT_FALSE(!idxDecoder_.Get(&index))) {
          throw ParquetException("");
        }
        PARQUET_THROW_NOT_OK(indexInBounds(index));
        PARQUET_THROW_NOT_OK(Builder->Append(dictValues[index]));
      },
      [&]() { PARQUET_THROW_NOT_OK(Builder->AppendNull()); });

  return numValues - nullCount;
}

template <>
int DictDecoderImpl<BooleanType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<BooleanType>::DictAccumulator* Builder) {
  ParquetException::NYI("No dictionary encoding for BooleanType");
}

template <>
inline int DictDecoderImpl<FLBAType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<FLBAType>::Accumulator* Builder) {
  if (Builder->byte_width() != descr_->typeLength()) {
    throw ParquetException(
        "Byte width mismatch: builder was " +
        std::to_string(Builder->byte_width()) + " but decoder was " +
        std::to_string(descr_->typeLength()));
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  auto dictValues = reinterpret_cast<const FLBA*>(dictionary_->data());

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        int32_t index;
        if (ARROW_PREDICT_FALSE(!idxDecoder_.Get(&index))) {
          throw ParquetException("");
        }
        PARQUET_THROW_NOT_OK(indexInBounds(index));
        Builder->UnsafeAppend(dictValues[index].ptr);
      },
      [&]() { Builder->UnsafeAppendNull(); });

  return numValues - nullCount;
}

template <>
int DictDecoderImpl<FLBAType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<FLBAType>::DictAccumulator* Builder) {
  auto valueType =
      checked_cast<const ::arrow::DictionaryType&>(*Builder->type())
          .value_type();
  auto byteWidth = checked_cast<const ::arrow::FixedSizeBinaryType&>(*valueType)
                       .byte_width();
  if (byteWidth != descr_->typeLength()) {
    throw ParquetException(
        "Byte width mismatch: builder was " + std::to_string(byteWidth) +
        " but decoder was " + std::to_string(descr_->typeLength()));
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  auto dictValues = reinterpret_cast<const FLBA*>(dictionary_->data());

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        int32_t index;
        if (ARROW_PREDICT_FALSE(!idxDecoder_.Get(&index))) {
          throw ParquetException("");
        }
        PARQUET_THROW_NOT_OK(indexInBounds(index));
        PARQUET_THROW_NOT_OK(Builder->Append(dictValues[index].ptr));
      },
      [&]() { PARQUET_THROW_NOT_OK(Builder->AppendNull()); });

  return numValues - nullCount;
}

template <typename Type>
int DictDecoderImpl<Type>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<Type>::Accumulator* Builder) {
  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  using ValueType = typename Type::CType;
  auto dictValues = reinterpret_cast<const ValueType*>(dictionary_->data());

  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        int32_t index;
        if (ARROW_PREDICT_FALSE(!idxDecoder_.Get(&index))) {
          throw ParquetException("");
        }
        PARQUET_THROW_NOT_OK(indexInBounds(index));
        Builder->UnsafeAppend(dictValues[index]);
      },
      [&]() { Builder->UnsafeAppendNull(); });

  return numValues - nullCount;
}

template <typename Type>
void DictDecoderImpl<Type>::insertDictionary(::arrow::ArrayBuilder* Builder) {
  ParquetException::NYI(
      "InsertDictionary only implemented for BYTE_ARRAY types");
}

template <>
void DictDecoderImpl<ByteArrayType>::insertDictionary(
    ::arrow::ArrayBuilder* Builder) {
  auto binaryBuilder =
      checked_cast<::arrow::BinaryDictionary32Builder*>(Builder);

  // Make a BinaryArray referencing the internal dictionary data.
  auto arr = std::make_shared<::arrow::BinaryArray>(
      dictionaryLength_, byteArrayOffsets_, byteArrayData_);
  PARQUET_THROW_NOT_OK(binaryBuilder->InsertMemoValues(*arr));
}

class DictByteArrayDecoderImpl : public DictDecoderImpl<ByteArrayType>,
                                 virtual public ByteArrayDecoder {
 public:
  using BASE = DictDecoderImpl<ByteArrayType>;
  using BASE::DictDecoderImpl;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      ::arrow::BinaryDictionary32Builder* Builder) override {
    int result = 0;
    if (nullCount == 0) {
      PARQUET_THROW_NOT_OK(decodeArrowNonNull(numValues, Builder, &result));
    } else {
      PARQUET_THROW_NOT_OK(decodeArrow(
          numValues, nullCount, validBits, validBitsOffset, Builder, &result));
    }
    return result;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out) override {
    int result = 0;
    if (nullCount == 0) {
      PARQUET_THROW_NOT_OK(decodeArrowDenseNonNull(numValues, out, &result));
    } else {
      PARQUET_THROW_NOT_OK(decodeArrowDense(
          numValues, nullCount, validBits, validBitsOffset, out, &result));
    }
    return result;
  }

 private:
  Status decodeArrowDense(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out,
      int* outNumValues) {
    constexpr int32_t kBufferSize = 1024;
    int32_t indices[kBufferSize];

    ArrowBinaryHelper<ByteArrayType> helper(out, numValues);
    RETURN_NOT_OK(helper.prepare());

    auto dictValues = reinterpret_cast<const ByteArray*>(dictionary_->data());
    int valuesDecoded = 0;
    int numIndices = 0;
    int posIndices = 0;

    auto visitValid = [&](int64_t position) -> Status {
      if (numIndices == posIndices) {
        // Refill indices buffer.
        const auto batchSize = std::min<int32_t>(
            kBufferSize, numValues - nullCount - valuesDecoded);
        numIndices = idxDecoder_.GetBatch(indices, batchSize);
        if (ARROW_PREDICT_FALSE(numIndices < 1)) {
          return Status::Invalid("Invalid number of indices: ", numIndices);
        }
        posIndices = 0;
      }
      const auto index = indices[posIndices++];
      RETURN_NOT_OK(indexInBounds(index));
      const auto& val = dictValues[index];
      RETURN_NOT_OK(helper.prepareNextInput(val.len));
      RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
      ++valuesDecoded;
      return Status::OK();
    };

    auto visitNull = [&]() -> Status {
      RETURN_NOT_OK(helper.AppendNull());
      return Status::OK();
    };

    ::arrow::internal::BitBlockCounter bitBlocks(
        validBits, validBitsOffset, numValues);
    int64_t position = 0;
    while (position < numValues) {
      const auto block = bitBlocks.NextWord();
      if (block.AllSet()) {
        for (int64_t i = 0; i < block.length; ++i, ++position) {
          ARROW_RETURN_NOT_OK(visitValid(position));
        }
      } else if (block.NoneSet()) {
        for (int64_t i = 0; i < block.length; ++i, ++position) {
          ARROW_RETURN_NOT_OK(visitNull());
        }
      } else {
        for (int64_t i = 0; i < block.length; ++i, ++position) {
          if (::arrow::bit_util::GetBit(
                  validBits, validBitsOffset + position)) {
            ARROW_RETURN_NOT_OK(visitValid(position));
          } else {
            ARROW_RETURN_NOT_OK(visitNull());
          }
        }
      }
    }

    *outNumValues = valuesDecoded;
    return Status::OK();
  }

  Status decodeArrowDenseNonNull(
      int numValues,
      typename EncodingTraits<ByteArrayType>::Accumulator* out,
      int* outNumValues) {
    constexpr int32_t kBufferSize = 2048;
    int32_t indices[kBufferSize];
    int valuesDecoded = 0;

    ArrowBinaryHelper<ByteArrayType> helper(out, numValues);
    RETURN_NOT_OK(helper.prepare(len_));

    auto dictValues = reinterpret_cast<const ByteArray*>(dictionary_->data());

    while (valuesDecoded < numValues) {
      const int32_t batchSize =
          std::min<int32_t>(kBufferSize, numValues - valuesDecoded);
      const int numIndices = idxDecoder_.GetBatch(indices, batchSize);
      if (numIndices == 0)
        ParquetException::eofException();
      for (int i = 0; i < numIndices; ++i) {
        auto idx = indices[i];
        RETURN_NOT_OK(indexInBounds(idx));
        const auto& val = dictValues[idx];
        RETURN_NOT_OK(helper.prepareNextInput(val.len));
        RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
      }
      valuesDecoded += numIndices;
    }
    *outNumValues = valuesDecoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      BuilderType* Builder,
      int* outNumValues) {
    constexpr int32_t kBufferSize = 1024;
    int32_t indices[kBufferSize];

    RETURN_NOT_OK(Builder->Reserve(numValues));
    ::arrow::internal::BitmapReader bitReader(
        validBits, validBitsOffset, numValues);

    auto dictValues = reinterpret_cast<const ByteArray*>(dictionary_->data());

    int valuesDecoded = 0;
    int numAppended = 0;
    while (numAppended < numValues) {
      bool isValid = bitReader.IsSet();
      bitReader.Next();

      if (isValid) {
        int32_t batchSize =
            std::min<int32_t>(kBufferSize, numValues - numAppended - nullCount);
        int numIndices = idxDecoder_.GetBatch(indices, batchSize);

        int i = 0;
        while (true) {
          // Consume all indices.
          if (isValid) {
            auto idx = indices[i];
            RETURN_NOT_OK(indexInBounds(idx));
            const auto& val = dictValues[idx];
            RETURN_NOT_OK(Builder->Append(val.ptr, val.len));
            ++i;
            ++valuesDecoded;
          } else {
            RETURN_NOT_OK(Builder->AppendNull());
            --nullCount;
          }
          ++numAppended;
          if (i == numIndices) {
            // Do not advance the bit_reader if we have fulfilled the decode.
            // Request.
            break;
          }
          isValid = bitReader.IsSet();
          bitReader.Next();
        }
      } else {
        RETURN_NOT_OK(Builder->AppendNull());
        --nullCount;
        ++numAppended;
      }
    }
    *outNumValues = valuesDecoded;
    return Status::OK();
  }

  template <typename BuilderType>
  Status
  decodeArrowNonNull(int numValues, BuilderType* Builder, int* outNumValues) {
    constexpr int32_t kBufferSize = 2048;
    int32_t indices[kBufferSize];

    RETURN_NOT_OK(Builder->Reserve(numValues));

    auto dictValues = reinterpret_cast<const ByteArray*>(dictionary_->data());

    int valuesDecoded = 0;
    while (valuesDecoded < numValues) {
      int32_t batchSize =
          std::min<int32_t>(kBufferSize, numValues - valuesDecoded);
      int numIndices = idxDecoder_.GetBatch(indices, batchSize);
      if (numIndices == 0)
        ParquetException::eofException();
      for (int i = 0; i < numIndices; ++i) {
        auto idx = indices[i];
        RETURN_NOT_OK(indexInBounds(idx));
        const auto& val = dictValues[idx];
        RETURN_NOT_OK(Builder->Append(val.ptr, val.len));
      }
      valuesDecoded += numIndices;
    }
    *outNumValues = valuesDecoded;
    return Status::OK();
  }
};

// ----------------------------------------------------------------------.
// DeltaBitPackEncoder.

/// DeltaBitPackEncoder is an encoder for the DeltaBinary Packing format.
/// As per the parquet spec. See:
/// https://github.com/apache/parquet-format/blob/master/Encodings.md#delta-encoding-delta_binary_packed--5
///
/// Consists of a header followed by blocks of delta encoded values binary.
/// Packed.
///
///  Format.
///    [Header] [block 1] [block 2] ... [block N].
///
///  Header.
///    [Block size] [number of mini blocks per block] [total value count]
///    [first. Value].
///
///  Block.
///    [Min delta] [list of bitwidths of the mini blocks] [miniblocks].
///
/// Sets aside bytes at the start of the internal buffer where the header will.
/// Be written, and only writes the header when FlushValues is called before.
/// Returning it.
///
/// To encode a block, we will:
///
/// 1. Compute the differences between consecutive elements. For the first.
/// Element in the block, use the last element in the previous block or, in the.
/// Case of the first block, use the first value of the whole sequence, stored.
/// In the header.
///
/// 2. Compute the frame of reference (the minimum of the deltas in the block).
/// Subtract this min delta from all deltas in the block. This guarantees that.
/// All values are non-negative.
///
/// 3. Encode the frame of reference (min delta) as a zigzag ULEB128 int.
/// Followed by the bit widths of the mini blocks and the delta values (minus.
/// The min delta) bit packed per mini block.
///
/// Supports only INT32 and INT64.

template <typename DType>
class DeltaBitPackEncoder : public EncoderImpl,
                            virtual public TypedEncoder<DType> {
  // Maximum possible header size.
  static constexpr uint32_t kMaxPageHeaderWriterSize = 32;
  static constexpr uint32_t kValuesPerBlock =
      std::is_same_v<int32_t, typename DType::CType> ? 128 : 256;
  static constexpr uint32_t kMiniBlocksPerBlock = 4;

 public:
  using T = typename DType::CType;
  using UT = std::make_unsigned_t<T>;
  using TypedEncoder<DType>::put;

  explicit DeltaBitPackEncoder(
      const ColumnDescriptor* descr,
      MemoryPool* pool,
      const uint32_t valuesPerBlock = kValuesPerBlock,
      const uint32_t miniBlocksPerBlock = kMiniBlocksPerBlock)
      : EncoderImpl(descr, Encoding::kDeltaBinaryPacked, pool),
        valuesPerBlock_(valuesPerBlock),
        miniBlocksPerBlock_(miniBlocksPerBlock),
        valuesPerMiniBlock_(valuesPerBlock / miniBlocksPerBlock),
        deltas_(valuesPerBlock, ::arrow::stl::allocator<T>(pool)),
        bitsBuffer_(allocateBuffer(
            pool,
            (kMiniBlocksPerBlock + valuesPerBlock) * sizeof(T))),
        sink_(pool),
        bitWriter_(
            bitsBuffer_->mutable_data(),
            static_cast<int>(bitsBuffer_->size())) {
    if (valuesPerBlock_ % 128 != 0) {
      throw ParquetException(
          "the number of values in a block must be multiple of 128, but it's " +
          std::to_string(valuesPerBlock_));
    }
    if (valuesPerMiniBlock_ % 32 != 0) {
      throw ParquetException(
          "the number of values in a miniblock must be multiple of 32, but it's " +
          std::to_string(valuesPerMiniBlock_));
    }
    if (valuesPerBlock % miniBlocksPerBlock != 0) {
      throw ParquetException(
          "the number of values per block % number of miniblocks per block must be 0, "
          "but it's " +
          std::to_string(valuesPerBlock % miniBlocksPerBlock));
    }
    // Reserve enough space at the beginning of the buffer for largest possible.
    // Header.
    PARQUET_THROW_NOT_OK(sink_.Advance(kMaxPageHeaderWriterSize));
  }

  std::shared_ptr<::arrow::Buffer> flushValues() override;

  int64_t estimatedDataEncodedSize() override {
    return sink_.length();
  }

  void put(const ::arrow::Array& values) override;

  void put(const T* buffer, int numValues) override;

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override;

  void flushBlock();

 private:
  const uint32_t valuesPerBlock_;
  const uint32_t miniBlocksPerBlock_;
  const uint32_t valuesPerMiniBlock_;
  uint32_t valuesCurrentBlock_{0};
  uint32_t totalValueCount_{0};
  UT firstValue_{0};
  UT currentValue_{0};
  ArrowPoolVector<UT> deltas_;
  std::shared_ptr<ResizableBuffer> bitsBuffer_;
  ::arrow::BufferBuilder sink_;
  BitWriter bitWriter_;
};

template <typename DType>
void DeltaBitPackEncoder<DType>::put(const T* src, int numValues) {
  if (numValues == 0) {
    return;
  }

  int idx = 0;
  if (totalValueCount_ == 0) {
    currentValue_ = src[0];
    firstValue_ = currentValue_;
    idx = 1;
  }
  totalValueCount_ += numValues;

  while (idx < numValues) {
    UT value = static_cast<UT>(src[idx]);
    // Calculate deltas. The possible overflow is handled by use of unsigned.
    // Integers making subtraction operations well-defined and correct even in.
    // Case of overflow. Encoded integers will wrap back around on decoding.
    // See. http://en.wikipedia.org/wiki/Modular_arithmetic#Integers_modulo_n
    deltas_[valuesCurrentBlock_] = value - currentValue_;
    currentValue_ = value;
    idx++;
    valuesCurrentBlock_++;
    if (valuesCurrentBlock_ == valuesPerBlock_) {
      flushBlock();
    }
  }
}

template <typename DType>
void DeltaBitPackEncoder<DType>::flushBlock() {
  if (valuesCurrentBlock_ == 0) {
    return;
  }

  const UT minDelta =
      *std::min_element(deltas_.begin(), deltas_.begin() + valuesCurrentBlock_);
  bitWriter_.PutZigZagVlqInt(static_cast<T>(minDelta));

  // Call to GetNextBytePtr reserves mini_blocks_per_block_ bytes of space to.
  // Write bit widths of miniblocks as they become known during the encoding.
  uint8_t* bitWidthData = bitWriter_.GetNextBytePtr(miniBlocksPerBlock_);
  VELOX_DCHECK(bitWidthData != nullptr);

  const uint32_t numMiniblocks = static_cast<uint32_t>(std::ceil(
      static_cast<double>(valuesCurrentBlock_) /
      static_cast<double>(valuesPerMiniBlock_)));
  for (uint32_t i = 0; i < numMiniblocks; i++) {
    const uint32_t valuesCurrentMiniBlock =
        std::min(valuesPerMiniBlock_, valuesCurrentBlock_);

    const uint32_t start = i * valuesPerMiniBlock_;
    const UT maxDelta = *std::max_element(
        deltas_.begin() + start,
        deltas_.begin() + start + valuesCurrentMiniBlock);

    // The minimum number of bits required to write any of values in deltas_.
    // Vector. See overflow comment above.
    const auto bitWidth = bitWidthData[i] =
        ::arrow::bit_util::NumRequiredBits(maxDelta - minDelta);

    for (uint32_t j = start; j < start + valuesCurrentMiniBlock; j++) {
      // See overflow comment above.
      const UT value = deltas_[j] - minDelta;
      bitWriter_.PutValue(value, bitWidth);
    }
    // If there are not enough values to fill the last mini block, we pad the.
    // Mini block with zeroes so that its length is the number of values in a.
    // Full mini block multiplied by the bit width.
    for (uint32_t j = valuesCurrentMiniBlock; j < valuesPerMiniBlock_; j++) {
      bitWriter_.PutValue(0, bitWidth);
    }
    valuesCurrentBlock_ -= valuesCurrentMiniBlock;
  }

  // If, in the last block, less than <number of miniblocks in a block>.
  // Miniblocks are needed to store the values, the bytes storing the bit
  // widths. Of the unneeded miniblocks are still present, their value should be
  // zero,. But readers must accept arbitrary values as well.
  for (uint32_t i = numMiniblocks; i < miniBlocksPerBlock_; i++) {
    bitWidthData[i] = 0;
  }
  VELOX_DCHECK_EQ(valuesCurrentBlock_, 0);

  bitWriter_.Flush();
  PARQUET_THROW_NOT_OK(
      sink_.Append(bitWriter_.buffer(), bitWriter_.bytesWritten()));
  bitWriter_.Clear();
}

template <typename DType>
std::shared_ptr<::arrow::Buffer> DeltaBitPackEncoder<DType>::flushValues() {
  if (valuesCurrentBlock_ > 0) {
    flushBlock();
  }
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink_.Finish(/*shrink_to_fit=*/true));

  uint8_t headerBuffer_[kMaxPageHeaderWriterSize] = {};
  BitWriter headerWriter(headerBuffer_, sizeof(headerBuffer_));
  if (!headerWriter.PutVlqInt(valuesPerBlock_) ||
      !headerWriter.PutVlqInt(miniBlocksPerBlock_) ||
      !headerWriter.PutVlqInt(totalValueCount_) ||
      !headerWriter.PutZigZagVlqInt(static_cast<T>(firstValue_))) {
    throw ParquetException("header writing error");
  }
  headerWriter.Flush();

  // We reserved enough space at the beginning of the buffer for largest.
  // Possible header and data was written immediately after. We now write the.
  // Header data immediately before the end of reserved space.
  const size_t offsetBytes =
      kMaxPageHeaderWriterSize - headerWriter.bytesWritten();
  std::memcpy(
      buffer->mutable_data() + offsetBytes,
      headerBuffer_,
      headerWriter.bytesWritten());

  // Reset counter of cached values.
  totalValueCount_ = 0;
  // Reserve enough space at the beginning of the buffer for largest possible.
  // Header.
  PARQUET_THROW_NOT_OK(sink_.Advance(kMaxPageHeaderWriterSize));

  // Excess bytes at the beginning are sliced off and ignored.
  return ::arrow::SliceBuffer(buffer, offsetBytes);
}

template <>
void DeltaBitPackEncoder<Int32Type>::put(const ::arrow::Array& values) {
  const ::arrow::ArrayData& data = *values.data();
  if (values.type_id() != ::arrow::Type::INT32) {
    throw ParquetException(
        "Expected Int32TArray, got ", values.type()->ToString());
  }
  if (data.length > std::numeric_limits<int32_t>::max()) {
    throw ParquetException(
        "Array cannot be longer than ", std::numeric_limits<int32_t>::max());
  }

  if (values.null_count() == 0) {
    put(data.GetValues<int32_t>(1), static_cast<int>(data.length));
  } else {
    putSpaced(
        data.GetValues<int32_t>(1),
        static_cast<int>(data.length),
        data.GetValues<uint8_t>(0, 0),
        data.offset);
  }
}

template <>
void DeltaBitPackEncoder<Int64Type>::put(const ::arrow::Array& values) {
  const ::arrow::ArrayData& data = *values.data();
  if (values.type_id() != ::arrow::Type::INT64) {
    throw ParquetException(
        "Expected Int64TArray, got ", values.type()->ToString());
  }
  if (data.length > std::numeric_limits<int32_t>::max()) {
    throw ParquetException(
        "Array cannot be longer than ", std::numeric_limits<int32_t>::max());
  }
  if (values.null_count() == 0) {
    put(data.GetValues<int64_t>(1), static_cast<int>(data.length));
  } else {
    putSpaced(
        data.GetValues<int64_t>(1),
        static_cast<int>(data.length),
        data.GetValues<uint8_t>(0, 0),
        data.offset);
  }
}

template <typename DType>
void DeltaBitPackEncoder<DType>::putSpaced(
    const T* src,
    int numValues,
    const uint8_t* validBits,
    int64_t validBitsOffset) {
  if (validBits != NULLPTR) {
    auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
        src, numValues, validBits, validBitsOffset, data);
    put(data, numValidValues);
  } else {
    put(src, numValues);
  }
}

// ----------------------------------------------------------------------.
// DeltaBitPackDecoder.

template <typename DType>
class DeltaBitPackDecoder : public DecoderImpl,
                            virtual public TypedDecoder<DType> {
 public:
  typedef typename DType::CType T;
  using UT = std::make_unsigned_t<T>;

  explicit DeltaBitPackDecoder(
      const ColumnDescriptor* descr,
      MemoryPool* pool = ::arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::kDeltaBinaryPacked), pool_(pool) {
    if (DType::typeNum != Type::kInt32 && DType::typeNum != Type::kInt64) {
      throw ParquetException(
          "Delta bit pack encoding should only be for integer data.");
    }
  }

  void setData(int numValues, const uint8_t* data, int len) override {
    // Num_values is equal to page's num_values, including null values in this.
    // Page.
    this->numValues_ = numValues;
    decoder_ = std::make_shared<BitReader>(data, len);
    initHeader();
  }

  // Set BitReader which is already initialized by DeltaLengthByteArrayDecoder.
  // Or DeltaByteArrayDecoder.
  void setDecoder(int numValues, std::shared_ptr<BitReader> decoder) {
    this->numValues_ = numValues;
    decoder_ = std::move(decoder);
    initHeader();
  }

  int validValuesCount() {
    // Total_values_remaining_ in header ignores of null values.
    return static_cast<int>(totalValuesRemaining_);
  }

  int decode(T* buffer, int maxValues) override {
    return getInternal(buffer, maxValues);
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::Accumulator* out) override {
    if (nullCount != 0) {
      // TODO(ARROW-34660): implement DecodeArrow with null slots.
      ParquetException::NYI("Delta bit pack DecodeArrow with null slots");
    }
    std::vector<T> values(numValues);
    int decodedCount = getInternal(values.data(), numValues);
    PARQUET_THROW_NOT_OK(out->AppendValues(values.data(), decodedCount));
    return decodedCount;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::DictAccumulator* out) override {
    if (nullCount != 0) {
      // TODO(ARROW-34660): implement DecodeArrow with null slots.
      ParquetException::NYI("Delta bit pack DecodeArrow with null slots");
    }
    std::vector<T> values(numValues);
    int decodedCount = getInternal(values.data(), numValues);
    PARQUET_THROW_NOT_OK(out->Reserve(decodedCount));
    for (int i = 0; i < decodedCount; ++i) {
      PARQUET_THROW_NOT_OK(out->Append(values[i]));
    }
    return decodedCount;
  }

 private:
  static constexpr int kMaxDeltaBitWidth = static_cast<int>(sizeof(T) * 8);

  void initHeader() {
    if (!decoder_->GetVlqInt(&valuesPerBlock_) ||
        !decoder_->GetVlqInt(&miniBlocksPerBlock_) ||
        !decoder_->GetVlqInt(&totalValueCount_) ||
        !decoder_->GetZigZagVlqInt(&lastValue_)) {
      ParquetException::eofException("InitHeader EOF");
    }

    if (valuesPerBlock_ == 0) {
      throw ParquetException("cannot have zero value per block");
    }
    if (valuesPerBlock_ % 128 != 0) {
      throw ParquetException(
          "the number of values in a block must be multiple of 128, but it's " +
          std::to_string(valuesPerBlock_));
    }
    if (miniBlocksPerBlock_ == 0) {
      throw ParquetException("cannot have zero miniblock per block");
    }
    valuesPerMiniBlock_ = valuesPerBlock_ / miniBlocksPerBlock_;
    if (valuesPerMiniBlock_ == 0) {
      throw ParquetException("cannot have zero value per miniblock");
    }
    if (valuesPerMiniBlock_ % 32 != 0) {
      throw ParquetException(
          "the number of values in a miniblock must be multiple of 32, but it's " +
          std::to_string(valuesPerMiniBlock_));
    }

    totalValuesRemaining_ = totalValueCount_;
    if (deltaBitWidths_ == nullptr) {
      deltaBitWidths_ = allocateBuffer(pool_, miniBlocksPerBlock_);
    } else {
      PARQUET_THROW_NOT_OK(deltaBitWidths_->Resize(
          miniBlocksPerBlock_, /*shrink_to_fit*/ false));
    }
    firstBlockInitialized_ = false;
    valuesRemainingCurrentMiniBlock_ = 0;
  }

  void initBlock() {
    VELOX_DCHECK_GT(totalValuesRemaining_, 0, "InitBlock called at EOF");

    if (!decoder_->GetZigZagVlqInt(&minDelta_))
      ParquetException::eofException("InitBlock EOF");

    // Read the bitwidth of each miniblock.
    uint8_t* bitWidthData = deltaBitWidths_->mutable_data();
    for (uint32_t i = 0; i < miniBlocksPerBlock_; ++i) {
      if (!decoder_->GetAligned(1, bitWidthData + i)) {
        ParquetException::eofException("Decode bit-width EOF");
      }
      // Note that non-conformant bitwidth entries are allowed by the Parquet.
      // Spec for extraneous miniblocks in the last block (GH-14923), so we.
      // Check the bitwidths when actually using them (see InitMiniBlock()).
    }

    miniBlockIdx_ = 0;
    firstBlockInitialized_ = true;
    initMiniBlock(bitWidthData[0]);
  }

  void initMiniBlock(int bitWidth) {
    if (ARROW_PREDICT_FALSE(bitWidth > kMaxDeltaBitWidth)) {
      throw ParquetException("delta bit width larger than integer bit width");
    }
    deltaBitWidth_ = bitWidth;
    valuesRemainingCurrentMiniBlock_ = valuesPerMiniBlock_;
  }

  int getInternal(T* buffer, int maxValues) {
    maxValues =
        static_cast<int>(std::min<int64_t>(maxValues, totalValuesRemaining_));
    if (maxValues == 0) {
      return 0;
    }

    int i = 0;

    if (ARROW_PREDICT_FALSE(!firstBlockInitialized_)) {
      // This is the first time we decode this data page, first output the.
      // Last value and initialize the first block.
      buffer[i++] = lastValue_;
      if (ARROW_PREDICT_FALSE(i == maxValues)) {
        // When i reaches max_values here we have two different possibilities:
        // 1. Total_value_count_ == 1, which means that the page may have only.
        //    One value (encoded in the header), and we should not initialize.
        //    Any block, nor should we skip any padding bits below.
        // 2. Total_value_count_ != 1, which means we should initialize the.
        //    Incoming block for subsequent reads.
        if (totalValueCount_ != 1) {
          initBlock();
        }
        totalValuesRemaining_ -= maxValues;
        this->numValues_ -= maxValues;
        return maxValues;
      }
      initBlock();
    }

    VELOX_DCHECK(firstBlockInitialized_);
    while (i < maxValues) {
      // Ensure we have an initialized mini-block.
      if (ARROW_PREDICT_FALSE(valuesRemainingCurrentMiniBlock_ == 0)) {
        ++miniBlockIdx_;
        if (miniBlockIdx_ < miniBlocksPerBlock_) {
          initMiniBlock(deltaBitWidths_->data()[miniBlockIdx_]);
        } else {
          initBlock();
        }
      }

      int valuesDecode = std::min(
          valuesRemainingCurrentMiniBlock_,
          static_cast<uint32_t>(maxValues - i));
      if (decoder_->GetBatch(deltaBitWidth_, buffer + i, valuesDecode) !=
          valuesDecode) {
        ParquetException::eofException();
      }
      for (int j = 0; j < valuesDecode; ++j) {
        // Addition between min_delta, packed int and last_value should be.
        // Treated as unsigned addition. Overflow is as expected.
        buffer[i + j] = static_cast<UT>(minDelta_) +
            static_cast<UT>(buffer[i + j]) + static_cast<UT>(lastValue_);
        lastValue_ = buffer[i + j];
      }
      valuesRemainingCurrentMiniBlock_ -= valuesDecode;
      i += valuesDecode;
    }
    totalValuesRemaining_ -= maxValues;
    this->numValues_ -= maxValues;

    if (ARROW_PREDICT_FALSE(totalValuesRemaining_ == 0)) {
      uint32_t paddingBits = valuesRemainingCurrentMiniBlock_ * deltaBitWidth_;
      // Skip the padding bits.
      if (!decoder_->Advance(paddingBits)) {
        ParquetException::eofException();
      }
      valuesRemainingCurrentMiniBlock_ = 0;
    }
    return maxValues;
  }

  MemoryPool* pool_;
  std::shared_ptr<BitReader> decoder_;
  uint32_t valuesPerBlock_;
  uint32_t miniBlocksPerBlock_;
  uint32_t valuesPerMiniBlock_;
  uint32_t totalValueCount_;

  uint32_t totalValuesRemaining_;
  // Remaining values in current mini block. If the current block is the last.
  // Mini block, values_remaining_current_mini_block_ may greater than.
  // Total_values_remaining_.
  uint32_t valuesRemainingCurrentMiniBlock_;

  // If the page doesn't contain any block, `first_block_initialized_` will.
  // Always be false. Otherwise, it will be true when first block initialized.
  bool firstBlockInitialized_;
  T minDelta_;
  uint32_t miniBlockIdx_;
  std::shared_ptr<ResizableBuffer> deltaBitWidths_;
  int deltaBitWidth_;

  T lastValue_;
};

// ----------------------------------------------------------------------.
// DELTA_LENGTH_BYTE_ARRAY.

// ----------------------------------------------------------------------.
// DeltaLengthByteArrayEncoder.

template <typename DType>
class DeltaLengthByteArrayEncoder : public EncoderImpl,
                                    virtual public TypedEncoder<ByteArrayType> {
 public:
  explicit DeltaLengthByteArrayEncoder(
      const ColumnDescriptor* descr,
      MemoryPool* pool)
      : EncoderImpl(
            descr,
            Encoding::kDeltaLengthByteArray,
            pool = ::arrow::default_memory_pool()),
        sink_(pool),
        lengthEncoder_(nullptr, pool),
        encodedSize_{0} {}

  std::shared_ptr<::arrow::Buffer> flushValues() override;

  int64_t estimatedDataEncodedSize() override {
    return encodedSize_ + lengthEncoder_.estimatedDataEncodedSize();
  }

  using TypedEncoder<ByteArrayType>::put;

  void put(const ::arrow::Array& values) override;

  void put(const T* buffer, int numValues) override;

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override;

 protected:
  template <typename ArrayType>
  void putBinaryArray(const ArrayType& array) {
    PARQUET_THROW_NOT_OK(
        ::arrow::VisitArraySpanInline<typename ArrayType::TypeClass>(
            *array.data(),
            [&](std::string_view view) {
              if (ARROW_PREDICT_FALSE(view.size() > kMaxByteArraySize)) {
                return Status::Invalid(
                    "Parquet cannot store strings with size 2GB or more");
              }
              lengthEncoder_.put({static_cast<int32_t>(view.length())}, 1);
              PARQUET_THROW_NOT_OK(sink_.Append(view.data(), view.length()));
              return Status::OK();
            },
            []() { return Status::OK(); }));
  }

  ::arrow::BufferBuilder sink_;
  DeltaBitPackEncoder<Int32Type> lengthEncoder_;
  uint32_t encodedSize_;
};

template <typename DType>
void DeltaLengthByteArrayEncoder<DType>::put(const ::arrow::Array& values) {
  assertBaseBinary(values);
  if (::arrow::is_binary_like(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::BinaryArray&>(values));
  } else {
    putBinaryArray(checked_cast<const ::arrow::LargeBinaryArray&>(values));
  }
}

template <typename DType>
void DeltaLengthByteArrayEncoder<DType>::put(const T* src, int numValues) {
  if (numValues == 0) {
    return;
  }

  constexpr int kBatchSize = 256;
  std::array<int32_t, kBatchSize> lengths;
  uint32_t totalIncrementSize = 0;
  for (int idx = 0; idx < numValues; idx += kBatchSize) {
    const int batchSize = std::min(kBatchSize, numValues - idx);
    for (int j = 0; j < batchSize; ++j) {
      const int32_t len = src[idx + j].len;
      if (addWithOverflow(totalIncrementSize, len, &totalIncrementSize)) {
        throw ParquetException("excess expansion in DELTA_LENGTH_BYTE_ARRAY");
      }
      lengths[j] = len;
    }
    lengthEncoder_.put(lengths.data(), batchSize);
  }

  if (addWithOverflow(encodedSize_, totalIncrementSize, &encodedSize_)) {
    throw ParquetException("excess expansion in DELTA_LENGTH_BYTE_ARRAY");
  }
  PARQUET_THROW_NOT_OK(sink_.Reserve(totalIncrementSize));
  for (int idx = 0; idx < numValues; idx++) {
    sink_.UnsafeAppend(src[idx].ptr, src[idx].len);
  }
}

template <typename DType>
void DeltaLengthByteArrayEncoder<DType>::putSpaced(
    const T* src,
    int numValues,
    const uint8_t* validBits,
    int64_t validBitsOffset) {
  if (validBits != NULLPTR) {
    auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
        src, numValues, validBits, validBitsOffset, data);
    put(data, numValidValues);
  } else {
    put(src, numValues);
  }
}

template <typename DType>
std::shared_ptr<::arrow::Buffer>
DeltaLengthByteArrayEncoder<DType>::flushValues() {
  std::shared_ptr<Buffer> encodedLengths = lengthEncoder_.flushValues();

  std::shared_ptr<Buffer> data;
  PARQUET_THROW_NOT_OK(sink_.Finish(&data));
  sink_.Reset();

  PARQUET_THROW_NOT_OK(sink_.Resize(encodedLengths->size() + data->size()));
  PARQUET_THROW_NOT_OK(
      sink_.Append(encodedLengths->data(), encodedLengths->size()));
  PARQUET_THROW_NOT_OK(sink_.Append(data->data(), data->size()));

  std::shared_ptr<Buffer> buffer;
  PARQUET_THROW_NOT_OK(sink_.Finish(&buffer, true));
  encodedSize_ = 0;
  return buffer;
}

// ----------------------------------------------------------------------.
// DeltaLengthByteArrayDecoder.

class DeltaLengthByteArrayDecoder : public DecoderImpl,
                                    virtual public TypedDecoder<ByteArrayType> {
 public:
  explicit DeltaLengthByteArrayDecoder(
      const ColumnDescriptor* descr,
      MemoryPool* pool = ::arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::kDeltaLengthByteArray),
        lenDecoder_(nullptr, pool),
        bufferedLength_(allocateBuffer(pool, 0)) {}

  void setData(int numValues, const uint8_t* data, int len) override {
    DecoderImpl::setData(numValues, data, len);
    decoder_ = std::make_shared<BitReader>(data, len);
    decodeLengths();
  }

  int decode(ByteArray* buffer, int maxValues) override {
    // Decode up to `max_values` strings into an internal buffer.
    // And reference them into `buffer`.
    maxValues = std::min(maxValues, numValidValues_);
    VELOX_DCHECK_GE(maxValues, 0);
    if (maxValues == 0) {
      return 0;
    }

    int32_t dataSize = 0;
    const int32_t* lengthPtr =
        reinterpret_cast<const int32_t*>(bufferedLength_->data()) + lengthIdx_;
    int bytesOffset = len_ - decoder_->bytesLeft();
    for (int i = 0; i < maxValues; ++i) {
      int32_t len = lengthPtr[i];
      if (ARROW_PREDICT_FALSE(len < 0)) {
        throw ParquetException("negative string delta length");
      }
      buffer[i].len = len;
      if (addWithOverflow(dataSize, len, &dataSize)) {
        throw ParquetException("excess expansion in DELTA_(LENGTH_)BYTE_ARRAY");
      }
    }
    lengthIdx_ += maxValues;
    if (ARROW_PREDICT_FALSE(
            !decoder_->Advance(8 * static_cast<int64_t>(dataSize)))) {
      ParquetException::eofException();
    }
    const uint8_t* dataPtr = data_ + bytesOffset;
    for (int i = 0; i < maxValues; ++i) {
      buffer[i].ptr = dataPtr;
      dataPtr += buffer[i].len;
    }
    this->numValues_ -= maxValues;
    numValidValues_ -= maxValues;
    return maxValues;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(decodeArrowDense(
        numValues, nullCount, validBits, validBitsOffset, out, &result));
    return result;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::DictAccumulator* out) override {
    ParquetException::NYI(
        "DecodeArrow of DictAccumulator for DeltaLengthByteArrayDecoder");
  }

 private:
  // Decode all the encoded lengths. The decoder_ will be at the start of the.
  // Encoded data after that.
  void decodeLengths() {
    lenDecoder_.setDecoder(numValues_, decoder_);

    // Get the number of encoded lengths.
    int numLength = lenDecoder_.validValuesCount();
    PARQUET_THROW_NOT_OK(bufferedLength_->Resize(numLength * sizeof(int32_t)));

    // Call len_decoder_.Decode to decode all the lengths.
    // All the lengths are buffered in buffered_length_.
    VELOX_DEBUG_ONLY int ret = lenDecoder_.decode(
        reinterpret_cast<int32_t*>(bufferedLength_->mutable_data()), numLength);
    VELOX_DCHECK_EQ(ret, numLength);
    lengthIdx_ = 0;
    numValidValues_ = numLength;
  }

  Status decodeArrowDense(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<ByteArrayType>::Accumulator* out,
      int* outNumValues) {
    ArrowBinaryHelper<ByteArrayType> helper(out, numValues);
    RETURN_NOT_OK(helper.prepare());

    std::vector<ByteArray> values(numValues - nullCount);
    const int numValidValues = decode(values.data(), numValues - nullCount);
    if (ARROW_PREDICT_FALSE(numValues - nullCount != numValidValues)) {
      throw ParquetException(
          "Expected to decode ",
          numValues - nullCount,
          " values, but decoded ",
          numValidValues,
          " values.");
    }

    auto valuesPtr = values.data();
    int valueIdx = 0;

    RETURN_NOT_OK(VisitNullBitmapInline(
        validBits,
        validBitsOffset,
        numValues,
        nullCount,
        [&]() {
          const auto& val = valuesPtr[valueIdx];
          RETURN_NOT_OK(helper.prepareNextInput(val.len));
          RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
          ++valueIdx;
          return Status::OK();
        },
        [&]() {
          RETURN_NOT_OK(helper.AppendNull());
          --nullCount;
          return Status::OK();
        }));

    VELOX_DCHECK_EQ(nullCount, 0);
    *outNumValues = numValidValues;
    return Status::OK();
  }

  std::shared_ptr<BitReader> decoder_;
  DeltaBitPackDecoder<Int32Type> lenDecoder_;
  int numValidValues_{0};
  uint32_t lengthIdx_{0};
  std::shared_ptr<ResizableBuffer> bufferedLength_;
};

// ----------------------------------------------------------------------.
// RLE_BOOLEAN_ENCODER.

class RleBooleanEncoder final : public EncoderImpl,
                                virtual public BooleanEncoder {
 public:
  explicit RleBooleanEncoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool)
      : EncoderImpl(descr, Encoding::kRle, pool),
        bufferedAppendValues_(::arrow::stl::allocator<T>(pool)) {}

  int64_t estimatedDataEncodedSize() override {
    return kRleLengthInBytes + maxRleBufferSize();
  }

  std::shared_ptr<Buffer> flushValues() override;

  void put(const T* buffer, int numValues) override;
  void put(const ::arrow::Array& values) override {
    if (values.type_id() != ::arrow::Type::BOOL) {
      throw ParquetException(
          "RleBooleanEncoder expects BooleanArray, got ",
          values.type()->ToString());
    }
    const auto& booleanArray =
        checked_cast<const ::arrow::BooleanArray&>(values);
    if (values.null_count() == 0) {
      for (int i = 0; i < booleanArray.length(); ++i) {
        // Null_count == 0, so just call Value directly is ok.
        bufferedAppendValues_.push_back(booleanArray.Value(i));
      }
    } else {
      PARQUET_THROW_NOT_OK(
          ::arrow::VisitArraySpanInline<::arrow::BooleanType>(
              *booleanArray.data(),
              [&](bool value) {
                bufferedAppendValues_.push_back(value);
                return Status::OK();
              },
              []() { return Status::OK(); }));
    }
  }

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    if (validBits != NULLPTR) {
      auto buffer = allocateBuffer(this->memoryPool(), numValues * sizeof(T));
      T* data = reinterpret_cast<T*>(buffer->mutable_data());
      int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
          src, numValues, validBits, validBitsOffset, data);
      put(data, numValidValues);
    } else {
      put(src, numValues);
    }
  }

  void put(const std::vector<bool>& src, int numValues) override;

 protected:
  template <typename SequenceType>
  void putImpl(const SequenceType& src, int numValues);

  int maxRleBufferSize() const noexcept {
    return rlePreserveBufferSize(
        static_cast<int>(bufferedAppendValues_.size()), kBitWidth);
  }

  constexpr static int32_t kBitWidth = 1;
  /// 4 Bytes in little-endian, which indicates the length.
  constexpr static int32_t kRleLengthInBytes = 4;

  // Std::vector<bool> in C++ is tricky, because it's a bitmap.
  // Here RleBooleanEncoder will only Append values into it, and.
  // Dump values into Buffer, so using it here is ok.
  ArrowPoolVector<bool> bufferedAppendValues_;
};

void RleBooleanEncoder::put(const bool* src, int numValues) {
  putImpl(src, numValues);
}

void RleBooleanEncoder::put(const std::vector<bool>& src, int numValues) {
  putImpl(src, numValues);
}

template <typename SequenceType>
void RleBooleanEncoder::putImpl(const SequenceType& src, int numValues) {
  for (int i = 0; i < numValues; ++i) {
    bufferedAppendValues_.push_back(src[i]);
  }
}

std::shared_ptr<Buffer> RleBooleanEncoder::flushValues() {
  int rleBufferSizeMax = maxRleBufferSize();
  std::shared_ptr<ResizableBuffer> buffer =
      allocateBuffer(this->pool_, rleBufferSizeMax + kRleLengthInBytes);
  RleEncoder encoder(
      buffer->mutable_data() + kRleLengthInBytes,
      rleBufferSizeMax,
      /*bit_width*/ kBitWidth);

  for (bool value : bufferedAppendValues_) {
    encoder.Put(value ? 1 : 0);
  }
  encoder.Flush();
  ::arrow::util::SafeStore(
      buffer->mutable_data(), ::arrow::bit_util::ToLittleEndian(encoder.len()));
  PARQUET_THROW_NOT_OK(buffer->Resize(kRleLengthInBytes + encoder.len()));
  bufferedAppendValues_.clear();
  return buffer;
}

// ----------------------------------------------------------------------.
// RLE_BOOLEAN_DECODER.

// TODO - Commented out as arrow/util/endian.h needs to be updated first.
/*
class RleBooleanDecoder : public DecoderImpl, virtual public BooleanDecoder {
 public:
  explicit RleBooleanDecoder(const ColumnDescriptor* descr)
      : DecoderImpl(descr, Encoding::RLE) {}

  void SetData(int num_values, const uint8_t* data, int len) override {
    num_values_ = num_values;
    uint32_t num_bytes = 0;

    if (len < 4) {
      throw ParquetException("Received invalid length : " + std::to_string(len)
+ " (corrupt data page?)");
    }
    // Load the first 4 bytes in little-endian, which indicates the length.
    num_bytes = ::arrow::bit_util::FromLittleEndian(SafeLoadAs<uint32_t>(data));
    if (num_bytes < 0 || num_bytes > static_cast<uint32_t>(len - 4)) {
      throw ParquetException("Received invalid number of bytes : " +
                             std::to_string(num_bytes) + " (corrupt data
page?)");
    }

    auto decoder_data = data + 4;
    if (decoder_ == nullptr) {
      decoder_ = std::make_shared<::arrow::util::RleDecoder>(decoder_data,
num_bytes, / *bit_width=* /1); } else { decoder_->Reset(decoder_data, num_bytes,
/ *bit_width=* /1);
    }
  }

  int Decode(bool* buffer, int max_values) override {
    max_values = std::min(max_values, num_values_);

    if (decoder_->GetBatch(buffer, max_values) != max_values) {
      ParquetException::EofException();
    }
    num_values_ -= max_values;
    return max_values;
  }

  int Decode(uint8_t* buffer, int max_values) override {
    ParquetException::NYI("Decode(uint8_t*, int) for RleBooleanDecoder");
  }

  int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                  int64_t valid_bits_offset,
                  typename EncodingTraits<BooleanType>::Accumulator* out)
override { if (null_count != 0) {
      // TODO(ARROW-34660): implement DecodeArrow with null slots.
      ParquetException::NYI("RleBoolean DecodeArrow with null slots");
    }
    constexpr int kBatchSize = 1024;
    std::array<bool, kBatchSize> values;
    int sum_decode_count = 0;
    do {
      int current_batch = std::min(kBatchSize, num_values);
      int decoded_count = decoder_->GetBatch(values.data(), current_batch);
      if (decoded_count == 0) {
        break;
      }
      sum_decode_count += decoded_count;
      PARQUET_THROW_NOT_OK(out->Reserve(sum_decode_count));
      for (int i = 0; i < decoded_count; ++i) {
        PARQUET_THROW_NOT_OK(out->Append(values[i]));
      }
      num_values -= decoded_count;
    } while (num_values > 0);
    return sum_decode_count;
  }

  int DecodeArrow(
      int num_values, int null_count, const uint8_t* valid_bits,
      int64_t valid_bits_offset,
      typename EncodingTraits<BooleanType>::DictAccumulator* builder) override {
    ParquetException::NYI("DecodeArrow for RleBooleanDecoder");
  }

 private:
  std::shared_ptr<::arrow::util::RleDecoder> decoder_;
};
*/

// ----------------------------------------------------------------------.
// DELTA_BYTE_ARRAY.

/// Delta Byte Array encoding also known as incremental encoding or front.
/// Compression: for each element in a sequence of strings, store the prefix.
/// Length of the previous entry plus the suffix.
///
/// This is stored as a sequence of delta-encoded prefix lengths.
/// (DELTA_BINARY_PACKED), followed by the suffixes encoded as delta length
/// byte. Arrays (DELTA_LENGTH_BYTE_ARRAY).

// ----------------------------------------------------------------------.
// DeltaByteArrayEncoder.

template <typename DType>
class DeltaByteArrayEncoder : public EncoderImpl,
                              virtual public TypedEncoder<DType> {
  static constexpr std::string_view kEmpty = "";

 public:
  using T = typename DType::CType;

  explicit DeltaByteArrayEncoder(
      const ColumnDescriptor* descr,
      MemoryPool* pool = ::arrow::default_memory_pool())
      : EncoderImpl(descr, Encoding::kDeltaByteArray, pool),
        sink_(pool),
        prefixLengthEncoder_(/*descr=*/nullptr, pool),
        suffixEncoder_(descr, pool),
        lastValue_(""),
        empty_(
            static_cast<uint32_t>(kEmpty.size()),
            reinterpret_cast<const uint8_t*>(kEmpty.data())) {}

  std::shared_ptr<Buffer> flushValues() override;

  int64_t estimatedDataEncodedSize() override {
    return prefixLengthEncoder_.estimatedDataEncodedSize() +
        suffixEncoder_.estimatedDataEncodedSize();
  }

  using TypedEncoder<DType>::put;

  void put(const ::arrow::Array& values) override;

  void put(const T* buffer, int numValues) override;

  void putSpaced(
      const T* src,
      int numValues,
      const uint8_t* validBits,
      int64_t validBitsOffset) override {
    if (validBits != nullptr) {
      if (buffer_ == nullptr) {
        PARQUET_ASSIGN_OR_THROW(
            buffer_,
            ::arrow::AllocateResizableBuffer(
                numValues * sizeof(T), this->memoryPool()));
      } else {
        PARQUET_THROW_NOT_OK(buffer_->Resize(numValues * sizeof(T), false));
      }
      T* data = reinterpret_cast<T*>(buffer_->mutable_data());
      int numValidValues = ::arrow::util::internal::SpacedCompress<T>(
          src, numValues, validBits, validBitsOffset, data);
      put(data, numValidValues);
    } else {
      put(src, numValues);
    }
  }

 protected:
  template <typename VisitorType>
  void putInternal(const T* src, int numValues, const VisitorType visitor) {
    if (numValues == 0) {
      return;
    }

    std::string_view lastValueView = lastValue_;
    constexpr int kBatchSize = 256;
    std::array<int32_t, kBatchSize> prefixLengths;
    std::array<ByteArray, kBatchSize> suffixes;

    for (int i = 0; i < numValues; i += kBatchSize) {
      const int batchSize = std::min(kBatchSize, numValues - i);

      for (int j = 0; j < batchSize; ++j) {
        const int idx = i + j;
        const auto view = visitor[idx];
        const auto len = static_cast<const uint32_t>(view.length());

        uint32_t commonPrefixLength = 0;
        const uint32_t maximumCommonPrefixLength =
            std::min(len, static_cast<uint32_t>(lastValueView.length()));
        while (commonPrefixLength < maximumCommonPrefixLength) {
          if (lastValueView[commonPrefixLength] != view[commonPrefixLength]) {
            break;
          }
          commonPrefixLength++;
        }

        lastValueView = view;
        prefixLengths[j] = commonPrefixLength;
        const uint32_t suffixLength = len - commonPrefixLength;
        const uint8_t* suffixPtr = src[idx].ptr + commonPrefixLength;

        // Convert to ByteArray, so it can be passed to the suffix_encoder_.
        const ByteArray suffix(suffixLength, suffixPtr);
        suffixes[j] = suffix;
      }
      suffixEncoder_.put(suffixes.data(), batchSize);
      prefixLengthEncoder_.put(prefixLengths.data(), batchSize);
    }
    lastValue_ = lastValueView;
  }

  template <typename ArrayType>
  void putBinaryArray(const ArrayType& array) {
    auto previousLen = static_cast<uint32_t>(lastValue_.length());
    std::string_view lastValueView = lastValue_;

    PARQUET_THROW_NOT_OK(
        ::arrow::VisitArraySpanInline<typename ArrayType::TypeClass>(
            *array.data(),
            [&](std::string_view view) {
              if (ARROW_PREDICT_FALSE(view.size() >= kMaxByteArraySize)) {
                return Status::Invalid(
                    "Parquet cannot store strings with size 2GB or more");
              }
              const ByteArray src{std::string_view(view.data(), view.size())};

              uint32_t commonPrefixLength = 0;
              const uint32_t len = src.len;
              const uint32_t maximumCommonPrefixLength =
                  std::min(previousLen, len);
              while (commonPrefixLength < maximumCommonPrefixLength) {
                if (lastValueView[commonPrefixLength] !=
                    view[commonPrefixLength]) {
                  break;
                }
                commonPrefixLength++;
              }
              previousLen = len;
              prefixLengthEncoder_.put(
                  {static_cast<int32_t>(commonPrefixLength)}, 1);

              lastValueView = std::string_view(view.data(), view.size());
              const auto suffixLength =
                  static_cast<uint32_t>(len - commonPrefixLength);
              if (suffixLength == 0) {
                suffixEncoder_.put(&empty_, 1);
                return Status::OK();
              }
              const uint8_t* suffixPtr = src.ptr + commonPrefixLength;
              // Convert to ByteArray, so it can be passed to the.
              // Suffix_encoder_.
              const ByteArray suffix(suffixLength, suffixPtr);
              suffixEncoder_.put(&suffix, 1);

              return Status::OK();
            },
            []() { return Status::OK(); }));
    lastValue_ = lastValueView;
  }

  ::arrow::BufferBuilder sink_;
  DeltaBitPackEncoder<Int32Type> prefixLengthEncoder_;
  DeltaLengthByteArrayEncoder<ByteArrayType> suffixEncoder_;
  std::string lastValue_;
  const ByteArray empty_;
  std::unique_ptr<ResizableBuffer> buffer_;
};

struct ByteArrayVisitor {
  const ByteArray* src;

  std::string_view operator[](int i) const {
    if (ARROW_PREDICT_FALSE(src[i].len >= kMaxByteArraySize)) {
      throw ParquetException(
          "Parquet cannot store strings with size 2GB or more");
    }
    return std::string_view{src[i]};
  }

  uint32_t len(int i) const {
    return src[i].len;
  }
};

struct FLBAVisitor {
  const FLBA* src;
  const uint32_t typeLength;

  std::string_view operator[](int i) const {
    return std::string_view{
        reinterpret_cast<const char*>(src[i].ptr), typeLength};
  }

  uint32_t len(int i) const {
    return typeLength;
  }
};

template <>
void DeltaByteArrayEncoder<ByteArrayType>::put(
    const ByteArray* src,
    int numValues) {
  auto visitor = ByteArrayVisitor{src};
  putInternal<ByteArrayVisitor>(src, numValues, visitor);
}

template <>
void DeltaByteArrayEncoder<FLBAType>::put(const FLBA* src, int numValues) {
  auto visitor = FLBAVisitor{src, static_cast<uint32_t>(descr_->typeLength())};
  putInternal<FLBAVisitor>(src, numValues, visitor);
}

template <typename DType>
void DeltaByteArrayEncoder<DType>::put(const ::arrow::Array& values) {
  if (::arrow::is_binary_like(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::BinaryArray&>(values));
  } else if (::arrow::is_large_binary_like(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::LargeBinaryArray&>(values));
  } else if (::arrow::is_fixed_size_binary(values.type_id())) {
    putBinaryArray(checked_cast<const ::arrow::FixedSizeBinaryArray&>(values));
  } else {
    throw ParquetException("Only BaseBinaryArray and subclasses supported");
  }
}

template <typename DType>
std::shared_ptr<Buffer> DeltaByteArrayEncoder<DType>::flushValues() {
  PARQUET_THROW_NOT_OK(sink_.Resize(estimatedDataEncodedSize(), false));

  std::shared_ptr<Buffer> prefixLengths = prefixLengthEncoder_.flushValues();
  PARQUET_THROW_NOT_OK(
      sink_.Append(prefixLengths->data(), prefixLengths->size()));

  std::shared_ptr<Buffer> suffixes = suffixEncoder_.flushValues();
  PARQUET_THROW_NOT_OK(sink_.Append(suffixes->data(), suffixes->size()));

  std::shared_ptr<Buffer> buffer;
  PARQUET_THROW_NOT_OK(sink_.Finish(&buffer, true));
  lastValue_.clear();
  return buffer;
}

// ----------------------------------------------------------------------.
// DeltaByteArrayDecoder.

template <typename DType>
class DeltaByteArrayDecoderImpl : public DecoderImpl,
                                  virtual public TypedDecoder<DType> {
  using T = typename DType::CType;

 public:
  explicit DeltaByteArrayDecoderImpl(
      const ColumnDescriptor* descr,
      MemoryPool* pool = ::arrow::default_memory_pool())
      : DecoderImpl(descr, Encoding::kDeltaByteArray),
        pool_(pool),
        prefixLenDecoder_(nullptr, pool),
        suffixDecoder_(nullptr, pool),
        lastValueInPreviousPage_(""),
        bufferedPrefixLength_(allocateBuffer(pool, 0)),
        bufferedData_(allocateBuffer(pool, 0)) {}

  void setData(int numValues, const uint8_t* data, int len) override {
    numValues_ = numValues;
    decoder_ = std::make_shared<BitReader>(data, len);
    prefixLenDecoder_.setDecoder(numValues, decoder_);

    // Get the number of encoded prefix lengths.
    int numPrefix = prefixLenDecoder_.validValuesCount();
    // Call prefix_len_decoder_.Decode to decode all the prefix lengths.
    // All the prefix lengths are buffered in buffered_prefix_length_.
    PARQUET_THROW_NOT_OK(
        bufferedPrefixLength_->Resize(numPrefix * sizeof(int32_t)));
    VELOX_DEBUG_ONLY int ret = prefixLenDecoder_.decode(
        reinterpret_cast<int32_t*>(bufferedPrefixLength_->mutable_data()),
        numPrefix);
    VELOX_DCHECK_EQ(ret, numPrefix);
    prefixLenOffset_ = 0;
    numValidValues_ = numPrefix;

    int bytesLeft = decoder_->bytesLeft();
    // If len < bytes_left, prefix_len_decoder.Decode will throw exception.
    VELOX_DCHECK_GE(len, bytesLeft);
    int suffixBegins = len - bytesLeft;
    // At this time, the decoder_ will be at the start of the encoded suffix.
    // Data.
    suffixDecoder_.setData(numValues, data + suffixBegins, bytesLeft);

    // TODO: read corrupted files written with bug(PARQUET-246). last_value_.
    // Should be set to last_value_in_previous_page_ when decoding a new.
    // page(except the first page)
    lastValue_ = "";
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::Accumulator* out) override {
    int result = 0;
    PARQUET_THROW_NOT_OK(decodeArrowDense(
        numValues, nullCount, validBits, validBitsOffset, out, &result));
    return result;
  }

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::DictAccumulator* Builder) override {
    ParquetException::NYI(
        "DecodeArrow of DictAccumulator for DeltaByteArrayDecoder");
  }

 protected:
  int getInternal(ByteArray* buffer, int maxValues) {
    // Decode up to `max_values` strings into an internal buffer.
    // And reference them into `buffer`.
    maxValues = std::min(maxValues, numValidValues_);
    if (maxValues == 0) {
      return maxValues;
    }

    int suffixRead = suffixDecoder_.decode(buffer, maxValues);
    if (ARROW_PREDICT_FALSE(suffixRead != maxValues)) {
      ParquetException::eofException(
          "Read " + std::to_string(suffixRead) + ", expecting " +
          std::to_string(maxValues) + " from suffix decoder");
    }

    int64_t dataSize = 0;
    const int32_t* prefixLenPtr =
        reinterpret_cast<const int32_t*>(bufferedPrefixLength_->data()) +
        prefixLenOffset_;
    for (int i = 0; i < maxValues; ++i) {
      if (ARROW_PREDICT_FALSE(prefixLenPtr[i] < 0)) {
        throw ParquetException("negative prefix length in DELTA_BYTE_ARRAY");
      }
      if (ARROW_PREDICT_FALSE(
              addWithOverflow(dataSize, prefixLenPtr[i], &dataSize) ||
              addWithOverflow(dataSize, buffer[i].len, &dataSize))) {
        throw ParquetException("excess expansion in DELTA_BYTE_ARRAY");
      }
    }
    PARQUET_THROW_NOT_OK(bufferedData_->Resize(dataSize));

    string_view prefix{lastValue_};
    uint8_t* dataPtr = bufferedData_->mutable_data();
    for (int i = 0; i < maxValues; ++i) {
      if (ARROW_PREDICT_FALSE(
              static_cast<size_t>(prefixLenPtr[i]) > prefix.length())) {
        throw ParquetException("prefix length too large in DELTA_BYTE_ARRAY");
      }
      memcpy(dataPtr, prefix.data(), prefixLenPtr[i]);
      // Buffer[i] currently points to the string suffix.
      memcpy(dataPtr + prefixLenPtr[i], buffer[i].ptr, buffer[i].len);
      buffer[i].ptr = dataPtr;
      buffer[i].len += prefixLenPtr[i];
      dataPtr += buffer[i].len;
      prefix = std::string_view{buffer[i]};
    }
    prefixLenOffset_ += maxValues;
    this->numValues_ -= maxValues;
    numValidValues_ -= maxValues;
    lastValue_ = std::string{prefix};

    if (numValidValues_ == 0) {
      lastValueInPreviousPage_ = lastValue_;
    }
    return maxValues;
  }

  Status decodeArrowDense(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::Accumulator* out,
      int* outNumValues) {
    ArrowBinaryHelper<DType> helper(out, numValues);
    RETURN_NOT_OK(helper.prepare());

    std::vector<ByteArray> values(numValues);
    const int numValidValues =
        getInternal(values.data(), numValues - nullCount);
    VELOX_DCHECK_EQ(numValues - nullCount, numValidValues);

    auto valuesPtr = reinterpret_cast<const ByteArray*>(values.data());
    int valueIdx = 0;

    RETURN_NOT_OK(VisitNullBitmapInline(
        validBits,
        validBitsOffset,
        numValues,
        nullCount,
        [&]() {
          const auto& val = valuesPtr[valueIdx];
          RETURN_NOT_OK(helper.prepareNextInput(val.len));
          RETURN_NOT_OK(helper.Append(val.ptr, static_cast<int32_t>(val.len)));
          ++valueIdx;
          return Status::OK();
        },
        [&]() {
          RETURN_NOT_OK(helper.AppendNull());
          --nullCount;
          return Status::OK();
        }));

    VELOX_DCHECK_EQ(nullCount, 0);
    *outNumValues = numValidValues;
    return Status::OK();
  }

  MemoryPool* pool_;

 private:
  std::shared_ptr<BitReader> decoder_;
  DeltaBitPackDecoder<Int32Type> prefixLenDecoder_;
  DeltaLengthByteArrayDecoder suffixDecoder_;
  std::string lastValue_;
  // String buffer for last value in previous page.
  std::string lastValueInPreviousPage_;
  int numValidValues_{0};
  uint32_t prefixLenOffset_{0};
  std::shared_ptr<ResizableBuffer> bufferedPrefixLength_;
  std::shared_ptr<ResizableBuffer> bufferedData_;
};

class DeltaByteArrayDecoder : public DeltaByteArrayDecoderImpl<ByteArrayType> {
 public:
  using Base = DeltaByteArrayDecoderImpl<ByteArrayType>;
  using Base::DeltaByteArrayDecoderImpl;

  int decode(ByteArray* buffer, int maxValues) override {
    return getInternal(buffer, maxValues);
  }
};

class DeltaByteArrayFLBADecoder : public DeltaByteArrayDecoderImpl<FLBAType>,
                                  virtual public FLBADecoder {
 public:
  using Base = DeltaByteArrayDecoderImpl<FLBAType>;
  using Base::DeltaByteArrayDecoderImpl;
  using Base::pool_;

  int decode(FixedLenByteArray* buffer, int maxValues) override {
    // GetInternal currently only support ByteArray.
    std::vector<ByteArray> decodeByteArray(maxValues);
    const int decodedValuesSize =
        getInternal(decodeByteArray.data(), maxValues);
    const uint32_t typeLength = descr_->typeLength();

    for (int i = 0; i < decodedValuesSize; i++) {
      if (ARROW_PREDICT_FALSE(decodeByteArray[i].len != typeLength)) {
        throw ParquetException("Fixed length byte array length mismatch");
      }
      buffer[i].ptr = decodeByteArray[i].ptr;
    }
    return decodedValuesSize;
  }
};

// ----------------------------------------------------------------------.
// BYTE_STREAM_SPLIT.

template <typename DType>
class ByteStreamSplitDecoder : public DecoderImpl,
                               virtual public TypedDecoder<DType> {
 public:
  using T = typename DType::CType;
  explicit ByteStreamSplitDecoder(const ColumnDescriptor* descr);

  int decode(T* buffer, int maxValues) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::Accumulator* Builder) override;

  int decodeArrow(
      int numValues,
      int nullCount,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      typename EncodingTraits<DType>::DictAccumulator* Builder) override;

  void setData(int numValues, const uint8_t* data, int len) override;

  T* ensureDecodeBuffer(int64_t minValues) {
    const int64_t size = sizeof(T) * minValues;
    if (!decodeBuffer_ || decodeBuffer_->size() < size) {
      decodeBuffer_ = allocateBuffer(::arrow::default_memory_pool(), size);
    }
    return reinterpret_cast<T*>(decodeBuffer_->mutable_data());
  }

 private:
  int numValuesInBuffer_{0};
  std::shared_ptr<Buffer> decodeBuffer_;

  static constexpr size_t kNumStreams = sizeof(T);
};

template <typename DType>
ByteStreamSplitDecoder<DType>::ByteStreamSplitDecoder(
    const ColumnDescriptor* descr)
    : DecoderImpl(descr, Encoding::kByteStreamSplit) {}

template <typename DType>
void ByteStreamSplitDecoder<DType>::setData(
    int numValues,
    const uint8_t* data,
    int len) {
  if (numValues * static_cast<int64_t>(sizeof(T)) < len) {
    throw ParquetException(
        "Data size too large for number of values (padding in byte stream split data "
        "page?)");
  }
  if (len % sizeof(T) != 0) {
    throw ParquetException(
        "ByteStreamSplit data size " + std::to_string(len) +
        " not aligned with type " + typeToString(DType::typeNum));
  }
  numValues = len / sizeof(T);
  DecoderImpl::setData(numValues, data, len);
  numValuesInBuffer_ = numValues_;
}

template <typename DType>
int ByteStreamSplitDecoder<DType>::decode(T* buffer, int maxValues) {
  const int valuesToDecode = std::min(numValues_, maxValues);
  const int numDecodedPreviously = numValuesInBuffer_ - numValues_;
  const uint8_t* data = data_ + numDecodedPreviously;

  byteStreamSplitDecode<T>(data, valuesToDecode, numValuesInBuffer_, buffer);
  numValues_ -= valuesToDecode;
  len_ -= sizeof(T) * valuesToDecode;
  return valuesToDecode;
}

template <typename DType>
int ByteStreamSplitDecoder<DType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<DType>::Accumulator* Builder) {
  constexpr int valueSize = static_cast<int>(kNumStreams);
  int valuesDecoded = numValues - nullCount;
  if (ARROW_PREDICT_FALSE(len_ < valueSize * valuesDecoded)) {
    ParquetException::eofException();
  }

  PARQUET_THROW_NOT_OK(Builder->Reserve(numValues));

  const int numDecodedPreviously = numValuesInBuffer_ - numValues_;
  const uint8_t* data = data_ + numDecodedPreviously;
  int offset = 0;

#if defined(ARROW_HAVE_SIMD_SPLIT)
  // Use fast decoding into intermediate buffer.  This will also decode.
  // Some null values, but it's fast enough that we don't care.
  T* decodeOut = ensureDecodeBuffer(valuesDecoded);
  ::arrow::util::internal::byte_stream_split_decode<T>(
      data, valuesDecoded, numValuesInBuffer_, decodeOut);

  // XXX If null_count is 0, we could even Append in bulk or decode directly.
  // Into builder.
  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        Builder->UnsafeAppend(decodeOut[offset]);
        ++offset;
      },
      [&]() { Builder->UnsafeAppendNull(); });

#else
  VisitNullBitmapInline(
      validBits,
      validBitsOffset,
      numValues,
      nullCount,
      [&]() {
        uint8_t gatheredByteData[kNumStreams];
        for (size_t b = 0; b < kNumStreams; ++b) {
          const size_t byteIndex = b * numValuesInBuffer_ + offset;
          gatheredByteData[b] = data[byteIndex];
        }
        Builder->UnsafeAppend(SafeLoadAs<T>(&gatheredByteData[0]));
        ++offset;
      },
      [&]() { Builder->UnsafeAppendNull(); });
#endif

  numValues_ -= valuesDecoded;
  len_ -= sizeof(T) * valuesDecoded;
  return valuesDecoded;
}

template <typename DType>
int ByteStreamSplitDecoder<DType>::decodeArrow(
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    typename EncodingTraits<DType>::DictAccumulator* Builder) {
  ParquetException::NYI("DecodeArrow for ByteStreamSplitDecoder");
}

} // namespace

// ----------------------------------------------------------------------.
// Encoder and decoder factory functions.

std::unique_ptr<Encoder> makeEncoder(
    Type::type typeNum,
    Encoding::type encoding,
    bool useDictionary,
    const ColumnDescriptor* descr,
    MemoryPool* pool) {
  if (useDictionary) {
    switch (typeNum) {
      case Type::kInt32:
        return std::make_unique<DictEncoderImpl<Int32Type>>(descr, pool);
      case Type::kInt64:
        return std::make_unique<DictEncoderImpl<Int64Type>>(descr, pool);
      case Type::kInt96:
        return std::make_unique<DictEncoderImpl<Int96Type>>(descr, pool);
      case Type::kFloat:
        return std::make_unique<DictEncoderImpl<FloatType>>(descr, pool);
      case Type::kDouble:
        return std::make_unique<DictEncoderImpl<DoubleType>>(descr, pool);
      case Type::kByteArray:
        return std::make_unique<DictEncoderImpl<ByteArrayType>>(descr, pool);
      case Type::kFixedLenByteArray:
        return std::make_unique<DictEncoderImpl<FLBAType>>(descr, pool);
      default:
        VELOX_DCHECK(false, "Encoder not implemented");
        break;
    }
  } else if (encoding == Encoding::kPlain) {
    switch (typeNum) {
      case Type::kBoolean:
        return std::make_unique<PlainEncoder<BooleanType>>(descr, pool);
      case Type::kInt32:
        return std::make_unique<PlainEncoder<Int32Type>>(descr, pool);
      case Type::kInt64:
        return std::make_unique<PlainEncoder<Int64Type>>(descr, pool);
      case Type::kInt96:
        return std::make_unique<PlainEncoder<Int96Type>>(descr, pool);
      case Type::kFloat:
        return std::make_unique<PlainEncoder<FloatType>>(descr, pool);
      case Type::kDouble:
        return std::make_unique<PlainEncoder<DoubleType>>(descr, pool);
      case Type::kByteArray:
        return std::make_unique<PlainEncoder<ByteArrayType>>(descr, pool);
      case Type::kFixedLenByteArray:
        return std::make_unique<PlainEncoder<FLBAType>>(descr, pool);
      default:
        VELOX_DCHECK(false, "Encoder not implemented");
        break;
    }
  } else if (encoding == Encoding::kByteStreamSplit) {
    switch (typeNum) {
      case Type::kFloat:
        return std::make_unique<ByteStreamSplitEncoder<FloatType>>(descr, pool);
      case Type::kDouble:
        return std::make_unique<ByteStreamSplitEncoder<DoubleType>>(
            descr, pool);
      default:
        throw ParquetException(
            "BYTE_STREAM_SPLIT only supports FLOAT and DOUBLE");
    }
  } else if (encoding == Encoding::kDeltaBinaryPacked) {
    switch (typeNum) {
      case Type::kInt32:
        return std::make_unique<DeltaBitPackEncoder<Int32Type>>(descr, pool);
      case Type::kInt64:
        return std::make_unique<DeltaBitPackEncoder<Int64Type>>(descr, pool);
      default:
        throw ParquetException(
            "DELTA_BINARY_PACKED encoder only supports INT32 and INT64");
    }
  } else if (encoding == Encoding::kDeltaLengthByteArray) {
    switch (typeNum) {
      case Type::kByteArray:
        return std::make_unique<DeltaLengthByteArrayEncoder<ByteArrayType>>(
            descr, pool);
      default:
        throw ParquetException(
            "DELTA_LENGTH_BYTE_ARRAY only supports BYTE_ARRAY");
    }
  } else if (encoding == Encoding::kRle) {
    switch (typeNum) {
      case Type::kBoolean:
        return std::make_unique<RleBooleanEncoder>(descr, pool);
      default:
        throw ParquetException("RLE only supports BOOLEAN");
    }
  } else if (encoding == Encoding::kDeltaByteArray) {
    switch (typeNum) {
      case Type::kByteArray:
        return std::make_unique<DeltaByteArrayEncoder<ByteArrayType>>(
            descr, pool);
      case Type::kFixedLenByteArray:
        return std::make_unique<DeltaByteArrayEncoder<FLBAType>>(descr, pool);
      default:
        throw ParquetException(
            "DELTA_BYTE_ARRAY only supports BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY");
    }
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  VELOX_DCHECK(false, "Should not be able to reach this code");
  return nullptr;
}

std::unique_ptr<Decoder> makeDecoder(
    Type::type typeNum,
    Encoding::type encoding,
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool) {
  if (encoding == Encoding::kPlain) {
    switch (typeNum) {
      case Type::kBoolean:
        return std::make_unique<PlainBooleanDecoder>(descr);
      case Type::kInt32:
        return std::make_unique<PlainDecoder<Int32Type>>(descr);
      case Type::kInt64:
        return std::make_unique<PlainDecoder<Int64Type>>(descr);
      case Type::kInt96:
        return std::make_unique<PlainDecoder<Int96Type>>(descr);
      case Type::kFloat:
        return std::make_unique<PlainDecoder<FloatType>>(descr);
      case Type::kDouble:
        return std::make_unique<PlainDecoder<DoubleType>>(descr);
      case Type::kByteArray:
        return std::make_unique<PlainByteArrayDecoder>(descr);
      case Type::kFixedLenByteArray:
        return std::make_unique<PlainFLBADecoder>(descr);
      default:
        break;
    }
  } else if (encoding == Encoding::kByteStreamSplit) {
    switch (typeNum) {
      case Type::kFloat:
        return std::make_unique<ByteStreamSplitDecoder<FloatType>>(descr);
      case Type::kDouble:
        return std::make_unique<ByteStreamSplitDecoder<DoubleType>>(descr);
      default:
        throw ParquetException(
            "BYTE_STREAM_SPLIT only supports FLOAT and DOUBLE");
    }
  } else if (encoding == Encoding::kDeltaBinaryPacked) {
    switch (typeNum) {
      case Type::kInt32:
        return std::make_unique<DeltaBitPackDecoder<Int32Type>>(descr, pool);
      case Type::kInt64:
        return std::make_unique<DeltaBitPackDecoder<Int64Type>>(descr, pool);
      default:
        throw ParquetException(
            "DELTA_BINARY_PACKED decoder only supports INT32 and INT64");
    }
  } else if (encoding == Encoding::kDeltaByteArray) {
    switch (typeNum) {
      case Type::kByteArray:
        return std::make_unique<DeltaByteArrayDecoder>(descr, pool);
      case Type::kFixedLenByteArray:
        return std::make_unique<DeltaByteArrayFLBADecoder>(descr, pool);
      default:
        throw ParquetException(
            "DELTA_BYTE_ARRAY only supports BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY");
    }
  } else if (encoding == Encoding::kDeltaLengthByteArray) {
    if (typeNum == Type::kByteArray) {
      return std::make_unique<DeltaLengthByteArrayDecoder>(descr, pool);
    }
    throw ParquetException("DELTA_LENGTH_BYTE_ARRAY only supports BYTE_ARRAY");
  } else if (encoding == Encoding::kRle) {
    if (typeNum == Type::kBoolean) {
      throw ParquetException("RleBooleanDecoder has been disabled.");
      // return std::make_unique<RleBooleanDecoder>(descr);
    }
    throw ParquetException("RLE encoding only supports BOOLEAN");
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  VELOX_DCHECK(false, "Should not be able to reach this code");
  return nullptr;
}

namespace detail {
std::unique_ptr<Decoder> makeDictDecoder(
    Type::type typeNum,
    const ColumnDescriptor* descr,
    MemoryPool* pool) {
  switch (typeNum) {
    case Type::kBoolean:
      ParquetException::NYI(
          "Dictionary encoding not implemented for boolean type");
    case Type::kInt32:
      return std::make_unique<DictDecoderImpl<Int32Type>>(descr, pool);
    case Type::kInt64:
      return std::make_unique<DictDecoderImpl<Int64Type>>(descr, pool);
    case Type::kInt96:
      return std::make_unique<DictDecoderImpl<Int96Type>>(descr, pool);
    case Type::kFloat:
      return std::make_unique<DictDecoderImpl<FloatType>>(descr, pool);
    case Type::kDouble:
      return std::make_unique<DictDecoderImpl<DoubleType>>(descr, pool);
    case Type::kByteArray:
      return std::make_unique<DictByteArrayDecoderImpl>(descr, pool);
    case Type::kFixedLenByteArray:
      return std::make_unique<DictDecoderImpl<FLBAType>>(descr, pool);
    default:
      break;
  }
  VELOX_DCHECK(false, "Should not be able to reach this code");
  return nullptr;
}

} // namespace detail
} // namespace facebook::velox::parquet::arrow
