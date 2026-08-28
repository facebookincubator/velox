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
#include "velox/dwio/nimble/reader/FieldReader.h"

#include <velox/type/StringView.h>
#include <cstddef>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/TrivialEncoding.h"

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/FlatMapHelper.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble {

namespace {

constexpr uint32_t kSkipBatchSize = 1024;

uint32_t scatterCount(
    uint32_t count,
    const velox::bits::Bitmap* scatterBitmap) {
  return scatterBitmap ? scatterBitmap->size() : count;
}

// Bytes needed for a packed null bitvector in velox.
constexpr uint64_t nullBytes(uint32_t rowCount) {
  return velox::bits::nbytes(rowCount);
}

// Ensures nulls can hold |rowCount| values.
char* ensureNulls(velox::BaseVector* vector, uint32_t rowCount) {
  return vector->mutableNulls(rowCount)->template asMutable<char>();
}

// Zeroes vector's null vector (aka make it 'all null').
void zeroNulls(velox::BaseVector* vector, uint32_t rowCount) {
  memset(ensureNulls(vector, rowCount), 0, nullBytes(rowCount));
}

// TODO: consider to use prepareForReuse in velox.
template <typename T>
T* verifyVectorState(velox::VectorPtr& vector) {
  // we want vector AND all its nested children to not be referenced by anyone
  // else (e.g. ref count of 1 recursively). Use BaseVector::reusable
  // from Velox to check this.
  if (vector != nullptr) {
    auto* casted = vector->as<T>();
    if ((casted != nullptr) && velox::BaseVector::recursivelyReusable(vector)) {
      return casted;
    }
    vector.reset();
  }
  return nullptr;
}

// Ensure the internal buffer to vector are refCounted to one
template <typename... T>
inline void resetIfNotWritable(velox::VectorPtr& vector, T&... buffer) {
  // The result vector and the buffer both hold reference, so refCount is at
  // least 2
  auto resetIfShared = [](auto& buffer) {
    if (!buffer) {
      return false;
    }
    const bool reset = buffer->refCount() > 2;
    if (reset) {
      buffer.reset();
    }
    return reset;
  };

  if ((... || resetIfShared(buffer))) {
    vector.reset();
  }
}

template <typename T, bool ShouldAllocate = true>
void ensureBuffer(
    velox::BufferPtr* buffer,
    size_t elementCount,
    velox::memory::MemoryPool* pool) {
  if constexpr (std::is_same_v<T, bool>) {
    const auto newSize = velox::bits::nbytes(elementCount);
    if (*buffer) {
      if (newSize <= (*buffer)->capacity()) {
        (*buffer)->setSize(newSize);
        return;
      }
    }

    if constexpr (ShouldAllocate) {
      *buffer = velox::AlignedBuffer::allocate<char>(newSize, pool);
    } else {
      *buffer = nullptr;
    }
  } else {
    if (*buffer) {
      const auto newSize = velox::checkedMultiply(elementCount, sizeof(T));
      if (newSize <= (*buffer)->capacity()) {
        (*buffer)->setSize(newSize);
        return;
      }
    }

    if constexpr (ShouldAllocate) {
      *buffer = velox::AlignedBuffer::allocate<T>(elementCount, pool);
    } else {
      *buffer = nullptr;
    }
  }
}

template <typename T>
struct VectorInitializer {};

template <typename T>
struct VectorInitializer<velox::FlatVector<T>> {
  static velox::FlatVector<T>* initialize(
      const velox::TypePtr& veloxType,
      uint64_t rowCount,
      velox::memory::MemoryPool* pool,
      velox::VectorPtr& output,
      velox::BufferPtr values = nullptr) {
    auto vector = verifyVectorState<velox::FlatVector<T>>(output);
    velox::BufferPtr nulls;

    if (vector) {
      nulls = vector->nulls();
      values = vector->values();
      resetIfNotWritable(output, nulls, values);
    }

    ensureBuffer<bool, /* ShouldAllocate */ false>(&nulls, rowCount, pool);
    ensureBuffer<T>(&values, rowCount, pool);

    if (!output) {
      output = std::make_shared<velox::FlatVector<T>>(
          pool,
          veloxType,
          nulls,
          rowCount,
          values,
          std::vector<velox::BufferPtr>());
    }

    return static_cast<velox::FlatVector<T>*>(output.get());
  }
};

template <>
struct VectorInitializer<velox::ArrayVector> {
  static velox::ArrayVector* initialize(
      const velox::TypePtr& veloxType,
      uint64_t rowCount,
      velox::memory::MemoryPool* pool,
      velox::VectorPtr& output) {
    auto vector = verifyVectorState<velox::ArrayVector>(output);
    velox::BufferPtr nulls, sizes, offsets;
    velox::VectorPtr elements;

    if (vector) {
      nulls = vector->nulls();
      sizes = vector->sizes();
      offsets = vector->offsets();
      elements = vector->elements();
      resetIfNotWritable(output, nulls, sizes, offsets);
    }

    ensureBuffer<bool, /* ShouldAllocate */ false>(&nulls, rowCount, pool);
    ensureBuffer<velox::vector_size_t>(&offsets, rowCount, pool);
    ensureBuffer<velox::vector_size_t>(&sizes, rowCount, pool);

    if (!output) {
      output = std::make_shared<velox::ArrayVector>(
          pool,
          veloxType,
          nulls,
          rowCount,
          std::move(offsets),
          std::move(sizes),
          /* elements */ elements,
          0 /*nullCount*/);
    }

    return static_cast<velox::ArrayVector*>(output.get());
  }
};

template <>
struct VectorInitializer<velox::MapVector> {
  static velox::MapVector* initialize(
      const velox::TypePtr& veloxType,
      uint64_t rowCount,
      velox::memory::MemoryPool* pool,
      velox::VectorPtr& output) {
    auto vector = verifyVectorState<velox::MapVector>(output);
    velox::BufferPtr nulls, sizes, offsets;
    velox::VectorPtr mapKeys, mapValues;

    if (vector) {
      nulls = vector->nulls();
      sizes = vector->sizes();
      offsets = vector->offsets();
      mapKeys = vector->mapKeys();
      mapValues = vector->mapValues();
      resetIfNotWritable(output, nulls, sizes, offsets);
    }

    ensureBuffer<bool, /* ShouldAllocate */ false>(&nulls, rowCount, pool);
    ensureBuffer<velox::vector_size_t>(&offsets, rowCount, pool);
    ensureBuffer<velox::vector_size_t>(&sizes, rowCount, pool);

    if (!output) {
      output = std::make_shared<velox::MapVector>(
          pool,
          veloxType,
          nulls,
          rowCount,
          std::move(offsets),
          std::move(sizes),
          /* keys*/ mapKeys,
          /*values*/ mapValues,
          0 /*nullCount*/);
    }

    return static_cast<velox::MapVector*>(output.get());
  }
};

template <>
struct VectorInitializer<velox::RowVector> {
  static velox::RowVector* initialize(
      const velox::TypePtr& veloxType,
      uint64_t rowCount,
      velox::memory::MemoryPool* pool,
      velox::VectorPtr& output) {
    auto* vector = verifyVectorState<velox::RowVector>(output);
    velox::BufferPtr nulls;
    std::vector<velox::VectorPtr> childrenVectors;
    if (vector != nullptr) {
      nulls = vector->nulls();
      childrenVectors = vector->children();
      resetIfNotWritable(output, nulls);
    } else {
      childrenVectors.resize(veloxType->size());
    }

    ensureBuffer<bool, /*ShouldAllocate=*/false>(&nulls, rowCount, pool);

    if (output == nullptr) {
      output = std::make_shared<velox::RowVector>(
          pool,
          veloxType,
          nulls,
          rowCount,
          std::move(childrenVectors),
          /*nullCount=*/0);
    }
    return static_cast<velox::RowVector*>(output.get());
  }
};

class NullColumnReader final : public FieldReader {
 public:
  NullColumnReader(velox::memory::MemoryPool& pool, velox::TypePtr type)
      : FieldReader{pool, std::move(type), nullptr} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    return std::optional<std::pair<uint32_t, uint64_t>>({0, 0});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    ensureNullConstant(type_, scatterCount(count, scatterBitmap), output);
  }

  void skip(uint32_t /* count */) final {}
};

class NullFieldReaderFactory final : public FieldReaderFactory {
 public:
  NullFieldReaderFactory(
      velox::TypePtr veloxType,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), /*nimbleType=*/nullptr, pool} {
  }

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<
          offset_size,
          std::unique_ptr<Decoder>>& /* decoders */) final {
    return createNullColumnReader();
  }
};

template <typename T>
static auto wrap(T& t) {
  return [&]() -> T& { return t; };
}

template <typename TRequested, typename TData>
struct IsBool : std::false_type {};

template <>
struct IsBool<bool, bool> : std::true_type {};

template <typename TRequested, typename TData, typename = void>
struct ScalarFieldReaderBase;

template <typename TRequested, typename TData>
struct ScalarFieldReaderBase<
    TRequested,
    TData,
    std::enable_if_t<IsBool<TRequested, TData>::value>> {
  explicit ScalarFieldReaderBase(velox::memory::MemoryPool& pool)
      : buf_{&pool} {}

  bool* ensureBuffer(uint32_t rowCount) {
    buf_.reserve(rowCount);
    auto* data = buf_.data();
    std::fill(data, data + rowCount, false);
    return data;
  }

  Vector<bool> buf_;
};

template <typename TRequested, typename TData>
struct ScalarFieldReaderBase<
    TRequested,
    TData,
    std::enable_if_t<!IsBool<TRequested, TData>::value>> {
  explicit ScalarFieldReaderBase(velox::memory::MemoryPool& /* pool */) {}
};

// TRequested is the requested data type from the reader, TData is the
// data type as stored in the file's schema
template <typename TRequested, typename TData>
class ScalarFieldReader final
    : public FieldReader,
      private ScalarFieldReaderBase<TRequested, TData> {
 public:
  ScalarFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder)
      : FieldReader(pool, std::move(type), decoder),
        ScalarFieldReaderBase<TRequested, TData>{pool} {
    if constexpr (
        (isSignedIntegralType<TRequested>() && !isSignedIntegralType<TData>() &&
         !isBoolType<TData>()) ||
        (isUnsignedIntegralType<TRequested>() &&
         !isUnsignedIntegralType<TData>()) ||
        (isFloatingPointType<TRequested>() && !isFloatingPointType<TData>()) ||
        sizeof(TRequested) < sizeof(TData)) {
      NIMBLE_FAIL("Incompatabile data type and requested type");
    }
  }

  using FieldReader::FieldReader;

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");
    const auto rowCount = encoding->rowCount();

    if (encoding->isNullable()) {
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
    }

    NIMBLE_CHECK(
        type_->isPrimitiveType(),
        "Velox type must be primitive in ScalarFieldReader");
    NIMBLE_CHECK(
        type_->isFixedWidth(),
        "Velox type must be fixed width in ScalarFieldReader");
    const auto veloxType = type_->kind();

    switch (veloxType) {
      case velox::TypeKind::BOOLEAN:
        // Bit packed representation for bool type
        totalBytes += rowCount / 8;
        break;
      case velox::TypeKind::TINYINT:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::TINYINT>::NativeType);
        break;
      case velox::TypeKind::SMALLINT:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::SMALLINT>::NativeType);
        break;
      case velox::TypeKind::INTEGER:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::INTEGER>::NativeType);
        break;
      case velox::TypeKind::BIGINT:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::BIGINT>::NativeType);
        break;
      case velox::TypeKind::REAL:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::REAL>::NativeType);
        break;
      case velox::TypeKind::DOUBLE:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::DOUBLE>::NativeType);
        break;
      case velox::TypeKind::TIMESTAMP:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::TIMESTAMP>::NativeType);
        break;
      case velox::TypeKind::HUGEINT:
        totalBytes += rowCount *
            sizeof(velox::TypeTraits<velox::TypeKind::HUGEINT>::NativeType);
        break;
      default:
        return std::nullopt;
    }
    return std::optional<std::pair<uint32_t, uint64_t>>(
        {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    const auto rowCount = scatterCount(count, scatterBitmap);
    auto vector = VectorInitializer<velox::FlatVector<TRequested>>::initialize(
        type_, rowCount, pool_, output);
    vector->resize(rowCount);

    const auto upcastNoNulls = [&vector]() {
      const auto vecRowCount = vector->size();
      if (vecRowCount == 0) {
        return;
      }
      auto* to = vector->mutableRawValues();
      const auto* from = vector->template rawValues<TData>();
      // we can't use for (uint32_t i = vecRowCount - 1; i >= 0; --i)
      // for the loop control, because for unsigned int, i >= 0 is always true,
      // it becomes an infinite loop
      for (uint32_t i = 0; i < vecRowCount; ++i) {
        to[vecRowCount - i - 1] =
            static_cast<TRequested>(from[vecRowCount - i - 1]);
      }
    };

    const auto upcastWithNulls = [&vector]() {
      const auto vecRowCount = vector->size();
      if (vecRowCount == 0) {
        return;
      }
      auto* to = vector->mutableRawValues();
      const auto* from = vector->template rawValues<TData>();
      for (uint32_t i = 0; i < vecRowCount; ++i) {
        if (vector->isNullAt(vecRowCount - i - 1)) {
          to[vecRowCount - i - 1] = TRequested();
        } else {
          to[vecRowCount - i - 1] =
              static_cast<TRequested>(from[vecRowCount - i - 1]);
        }
      }
    };

    uint32_t nonNullCount{0};
    // Unused string buffer container for api
    std::vector<velox::BufferPtr> stringBuffers;
    if constexpr (IsBool<TRequested, TData>::value) {
      // TODO: implement method for bitpacked bool
      auto* buf = this->ensureBuffer(rowCount);
      nonNullCount = decoder_->next(
          count,
          buf,
          stringBuffers,
          [&]() { return ensureNulls(vector, rowCount); },
          scatterBitmap);

      NIMBLE_DCHECK_EQ(
          vector->values()->size(),
          velox::bits::nbytes(rowCount),
          "Unexpected values buffer size.");
      auto* target = vector->values()->template asMutable<char>();
      std::fill(target, target + velox::bits::nbytes(rowCount), 0);
      if (nonNullCount == rowCount) {
        velox::bits::packBitmap(std::span<const bool>{buf, rowCount}, target);
      } else {
        for (uint32_t i = 0; i < rowCount; ++i) {
          if (!vector->isNullAt(i)) {
            velox::bits::maybeSetBit(target, i, buf[i]);
          }
        }
      }
    } else {
      NIMBLE_DCHECK_EQ(
          vector->values()->size(),
          (rowCount * sizeof(TRequested)),
          "Unexpected values buffer size.");
      nonNullCount = decoder_->next(
          count,
          vector->values()->template asMutable<TRequested>(),
          stringBuffers,
          [&]() { return ensureNulls(vector, rowCount); },
          scatterBitmap);
    }

    if (nonNullCount == rowCount) {
      vector->resetNulls();
      if constexpr (sizeof(TRequested) > sizeof(TData)) {
        upcastNoNulls();
      }
    } else {
      vector->setNullCount(rowCount - nonNullCount);
      if constexpr (sizeof(TRequested) > sizeof(TData)) {
        upcastWithNulls();
      }
    }
  }

  void skip(uint32_t count) final {
    decoder_->skip(count);
  }
};

template <typename T>
class ScalarFieldReaderFactory final : public FieldReaderFactory {
 public:
  ScalarFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    const auto& descriptor = nimbleType_->asScalar().scalarDescriptor();
    switch (descriptor.scalarKind()) {
      case ScalarKind::Bool: {
        return createReaderImpl<ScalarFieldReader<T, bool>>(
            decoders, descriptor);
      }
      case ScalarKind::Int8: {
        return createReaderImpl<ScalarFieldReader<T, int8_t>>(
            decoders, descriptor);
      }
      case ScalarKind::Int16: {
        return createReaderImpl<ScalarFieldReader<T, int16_t>>(
            decoders, descriptor);
      }
      case ScalarKind::Int32: {
        return createReaderImpl<ScalarFieldReader<T, int32_t>>(
            decoders, descriptor);
      }
      case ScalarKind::Int64: {
        return createReaderImpl<ScalarFieldReader<T, int64_t>>(
            decoders, descriptor);
      }
      case ScalarKind::Float: {
        return createReaderImpl<ScalarFieldReader<T, float>>(
            decoders, descriptor);
      }
      case ScalarKind::Double: {
        return createReaderImpl<ScalarFieldReader<T, double>>(
            decoders, descriptor);
      }
      case ScalarKind::UInt8:
      case ScalarKind::UInt16:
      case ScalarKind::UInt32: {
        return createReaderImpl<ScalarFieldReader<T, uint32_t>>(
            decoders, descriptor);
      }
      case ScalarKind::UInt64:
      case ScalarKind::String:
      case ScalarKind::Binary:
      case ScalarKind::Undefined: {
        NIMBLE_UNSUPPORTED(
            "Unsupported nimble scalar type: {}.",
            toString(descriptor.scalarKind()));
      }
    }
    NIMBLE_UNREACHABLE(
        "Should not have nimble scalar type: {}.",
        toString(descriptor.scalarKind()))
  }
};

class StringFieldReader final : public FieldReader {
 public:
  StringFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::string_view>& buffer)
      : FieldReader{pool, std::move(type), decoder}, buffer_{buffer} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");
    const auto* innerEncoding = encoding;
    const auto rowCount = encoding->rowCount();

    if (encoding->isNullable()) {
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
      const auto* nullableEncoding =
          dynamic_cast<const NullableEncoding<std::string_view>*>(encoding);
      NIMBLE_CHECK_NOT_NULL(
          nullableEncoding,
          "NullableEncoding is not used for nullable string field.");
      innerEncoding = nullableEncoding->nonNulls();
    }

    // TODO: support more encodings (or do encoding traversal), DICT, RLE, etc.
    // We currently only estimate trivial encoded string field.
    if (const auto* trivialEncoding =
            dynamic_cast<const TrivialEncoding<std::string_view>*>(
                innerEncoding)) {
      // Adding overhead for velox::StringView. 4 bytes for inline, 16 bytes for
      // non-inline
      const auto nonNullCount = trivialEncoding->rowCount();
      const auto payloadBytes = trivialEncoding->uncompressedDataBytes();
      // Non-null entries overhead
      totalBytes +=
          ((payloadBytes / nonNullCount) > velox::StringView::kInlineSize ? 16
                                                                          : 4) *
          nonNullCount;
      // Null entries overhead
      totalBytes += (rowCount - nonNullCount) * 16;

      // Adding actual string content payload size
      totalBytes += payloadBytes;
    } else {
      return std::nullopt;
    }

    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    auto rowCount = scatterCount(count, scatterBitmap);
    auto vector =
        VectorInitializer<velox::FlatVector<velox::StringView>>::initialize(
            type_, rowCount, pool_, output);
    vector->resize(rowCount);
    buffer_.resize(rowCount);

    std::vector<velox::BufferPtr> stringBuffers;
    // NOTE: the next diff will branch and use an earlier version
    // of copying string values or simple setter in flat vector.
    auto nonNullCount = decoder_->next(
        count,
        buffer_.data(),
        stringBuffers,
        [&]() { return ensureNulls(vector, rowCount); },
        scatterBitmap);
    auto* valuesPtr = vector->mutableValues()->asMutable<velox::StringView>();
    if (nonNullCount == rowCount) {
      vector->resetNulls();
      for (uint32_t i = 0; i < rowCount; ++i) {
        valuesPtr[i] =
            velox::StringView(buffer_[i].data(), buffer_[i].length());
      }
    } else {
      vector->setNullCount(rowCount - nonNullCount);
      for (uint32_t i = 0; i < rowCount; ++i) {
        if (!vector->isNullAt(i)) {
          valuesPtr[i] =
              velox::StringView(buffer_[i].data(), buffer_[i].length());
        }
      }
    }
    // Use shared ownership of the buffers
    vector->setStringBuffers(std::move(stringBuffers));
  }

  void skip(uint32_t count) final {
    decoder_->skip(count);
  }

 private:
  std::vector<std::string_view>& buffer_;
};

// This is the legacy string reader that is used for the legacy encodings.
class LegacyStringFieldReader final : public FieldReader {
 public:
  LegacyStringFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::string_view>& buffer)
      : FieldReader{pool, std::move(type), decoder}, buffer_{buffer} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");
    const auto* innerEncoding = encoding;
    const auto rowCount = encoding->rowCount();

    if (encoding->isNullable()) {
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
      const auto* nullableEncoding =
          dynamic_cast<const legacy::NullableEncoding<std::string_view>*>(
              encoding);
      NIMBLE_CHECK_NOT_NULL(
          nullableEncoding,
          "NullableEncoding is not used for nullable string field.");
      innerEncoding = nullableEncoding->nonNulls();
    }

    // TODO: support more encodings (or do encoding traversal), DICT, RLE, etc.
    // We currently only estimate trivial encoded string field.
    if (const auto* trivialEncoding =
            dynamic_cast<const legacy::TrivialEncoding<std::string_view>*>(
                innerEncoding)) {
      // Adding overhead for velox::StringView. 4 bytes for inline, 16 bytes for
      // non-inline
      const auto nonNullCount = trivialEncoding->rowCount();
      const auto payloadBytes = trivialEncoding->uncompressedDataBytes();
      // Non-null entries overhead
      totalBytes +=
          ((payloadBytes / nonNullCount) > velox::StringView::kInlineSize ? 16
                                                                          : 4) *
          nonNullCount;
      // Null entries overhead
      totalBytes += (rowCount - nonNullCount) * 16;

      // Adding actual string content payload size
      totalBytes += payloadBytes;
    } else {
      return std::nullopt;
    }

    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    auto rowCount = scatterCount(count, scatterBitmap);
    auto vector =
        VectorInitializer<velox::FlatVector<velox::StringView>>::initialize(
            type_, rowCount, pool_, output);
    vector->resize(rowCount);
    buffer_.resize(rowCount);

    // Unused place holder for api.
    std::vector<velox::BufferPtr> stringBuffers;
    auto nonNullCount = decoder_->next(
        count,
        buffer_.data(),
        stringBuffers,
        [&]() { return ensureNulls(vector, rowCount); },
        scatterBitmap);
    size_t totalLength = 0;
    const bool hasNulls = (nonNullCount != rowCount);
    // A column that is entirely null over this batch would make both passes
    // over |rowCount| below run only to move nothing.
    const bool allNull = (nonNullCount == 0);
    if (!hasNulls) {
      vector->resetNulls();
      for (uint32_t i = 0; i < rowCount; ++i) {
        totalLength += buffer_[i].length();
      }
    } else {
      vector->setNullCount(rowCount - nonNullCount);
      if (!allNull) {
        for (uint32_t i = 0; i < rowCount; ++i) {
          if (!vector->isNullAt(i)) {
            totalLength += buffer_[i].length();
          }
        }
      }
    }
    // Copy the strings into a single string buffer.
    velox::BufferPtr data =
        velox::AlignedBuffer::allocate<char>(totalLength, pool_);
    char* dataPtr = data->asMutable<char>();
    auto* valuesPtr = vector->mutableValues()->asMutable<velox::StringView>();
    size_t currentOffset = 0;
    if (!hasNulls) {
      for (uint32_t i = 0; i < rowCount; ++i) {
        std::copy(buffer_[i].data(), buffer_[i].end(), dataPtr + currentOffset);
        valuesPtr[i] =
            velox::StringView(dataPtr + currentOffset, buffer_[i].length());
        currentOffset += buffer_[i].length();
      }
    } else if (!allNull) {
      for (uint32_t i = 0; i < rowCount; ++i) {
        if (!vector->isNullAt(i)) {
          std::copy(
              buffer_[i].data(), buffer_[i].end(), dataPtr + currentOffset);
          valuesPtr[i] =
              velox::StringView(dataPtr + currentOffset, buffer_[i].length());
          currentOffset += buffer_[i].length();
        }
      }
    }
    vector->setStringBuffers({data});
  }

  void skip(uint32_t count) final {
    decoder_->skip(count);
  }

 private:
  std::vector<std::string_view>& buffer_;
};

class StringFieldReaderFactory final : public FieldReaderFactory {
 public:
  StringFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      bool optimizeStringBufferHandling,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        optimizeStringBufferHandling_{optimizeStringBufferHandling} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return optimizeStringBufferHandling_
        ? createReaderImpl<StringFieldReader>(
              decoders,
              nimbleType_->asScalar().scalarDescriptor(),
              wrap(buffer_))
        : createReaderImpl<LegacyStringFieldReader>(
              decoders,
              nimbleType_->asScalar().scalarDescriptor(),
              wrap(buffer_));
  }

 private:
  std::vector<std::string_view> buffer_;
  bool optimizeStringBufferHandling_;
};

class TimestampMicroNanoFieldReader final : public FieldReader {
 public:
  TimestampMicroNanoFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* microsDecoder,
      Decoder* nanosDecoder)
      : FieldReader{pool, std::move(type), microsDecoder},
        nanosDecoder_{nanosDecoder},
        microsBuffer_{&pool},
        nanosBuffer_{&pool} {
    NIMBLE_DCHECK_NOT_NULL(
        nanosDecoder,
        "Nanoseconds decoder must exist when microseconds decoder exists");
  }

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");

    const auto rowCount = encoding->rowCount();

    if (rowCount == 0) {
      return std::optional<std::pair<uint32_t, uint64_t>>({0, 0});
    }

    uint64_t totalBytes = rowCount *
        sizeof(velox::TypeTraits<velox::TypeKind::TIMESTAMP>::NativeType);

    if (encoding->isNullable()) {
      totalBytes += rowCount / 8;
    }

    return std::optional<std::pair<uint32_t, uint64_t>>(
        {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    const auto rowCount = scatterCount(count, scatterBitmap);
    auto vector =
        VectorInitializer<velox::FlatVector<velox::Timestamp>>::initialize(
            type_, rowCount, pool_, output);
    vector->resize(rowCount);
    microsBuffer_.resize(rowCount);

    std::vector<velox::BufferPtr> stringBuffers;
    auto nonNullCount = decoder_->next(
        count,
        microsBuffer_.data(),
        stringBuffers,
        [&]() { return ensureNulls(vector, rowCount); },
        scatterBitmap);

    stringBuffers.clear();
    nanosBuffer_.resize(nonNullCount);
    nanosDecoder_->next(
        nonNullCount,
        nanosBuffer_.data(),
        stringBuffers,
        []() { return nullptr; },
        nullptr);

    auto* rawValues = vector->mutableRawValues();

    if (nonNullCount == rowCount) {
      vector->resetNulls();
      for (velox::vector_size_t i = 0; i < rowCount; ++i) {
        rawValues[i] =
            convertToVeloxTimestamp(microsBuffer_[i], nanosBuffer_[i]);
      }
    } else {
      vector->setNullCount(rowCount - nonNullCount);
      velox::vector_size_t nanosIndex = 0;
      for (velox::vector_size_t i = 0; i < rowCount; ++i) {
        if (!vector->isNullAt(i)) {
          rawValues[i] = convertToVeloxTimestamp(
              microsBuffer_[i], nanosBuffer_[nanosIndex++]);
        }
      }
    }
  }

  void skip(uint32_t count) final {
    std::array<int64_t, kSkipBatchSize> microsBuffer{};
    std::array<char, nullBytes(kSkipBatchSize)> nulls{};
    uint32_t nonNullCount = 0;

    std::vector<velox::BufferPtr> stringBuffers;
    while (count > 0) {
      auto readSize = std::min(count, kSkipBatchSize);
      nonNullCount += decoder_->next(
          readSize,
          microsBuffer.data(),
          stringBuffers,
          [&]() { return nulls.data(); },
          /* scatterBitmap */ nullptr);
      count -= readSize;
    }

    if (nonNullCount > 0) {
      nanosDecoder_->skip(nonNullCount);
    }
  }

  void reset() final {
    FieldReader::reset();
    nanosDecoder_->reset();
  }

 private:
  // - Nimble stores time as:
  //     micros         -> whole microseconds since epoch
  //     subMicrosNanos -> extra nanoseconds inside the microsecond [0, 999]
  // - Velox stores time as:
  //     seconds + nanos (0 <= nanos < 1_000_000_000)
  //
  // The math below splits 'micros' into whole seconds and the remainder, then
  // converts the remainder to nanoseconds and adds the sub-microsecond nanos.
  // For negative remainders, we use a branchless correction to ensure nanos
  // is always positive.
  static velox::Timestamp convertToVeloxTimestamp(
      int64_t micros,
      uint16_t subMicrosNanos) {
    int64_t seconds = micros / 1000000;
    int64_t remainder = micros % 1000000;
    // Branchless Sign Correction
    // If micros was negative (e.g., -100us), remainder will be -100.
    // We need to borrow 1 second to make the remainder positive.
    // mask will be -1 (0xFF...FF) if remainder < 0, else 0.
    int64_t mask = remainder >> 63;
    // If negative: seconds -= 1; remainder += 1000000;
    seconds += mask;
    remainder += (1000000 & mask);
    // remainder is now guaranteed [0, 999999].
    // Convert remainder micros to nanos (* 1000) and add the fractional nanos.
    uint64_t nanos = static_cast<uint64_t>(remainder) * 1000 + subMicrosNanos;
    return velox::Timestamp(seconds, nanos);
  }

  Decoder* nanosDecoder_;
  Vector<int64_t> microsBuffer_;
  Vector<uint16_t> nanosBuffer_;
};

class TimestampMicroNanoFieldReaderFactory final : public FieldReaderFactory {
 public:
  TimestampMicroNanoFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return createReaderImpl<TimestampMicroNanoFieldReader>(
        decoders,
        nimbleType_->asTimestampMicroNano().microsDescriptor(),
        [&]() {
          return getDecoder(
              decoders, nimbleType_->asTimestampMicroNano().nanosDescriptor());
        });
  }
};

class MultiValueFieldReader : public FieldReader {
 public:
  using FieldReader::FieldReader;

 protected:
  template <typename T, typename... Args>
  velox::vector_size_t loadOffsets(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap,
      velox::vector_size_t allocationSize = 0,
      Args&&... args) {
    auto rowCount = scatterCount(count, scatterBitmap);
    if (allocationSize == 0) {
      allocationSize = rowCount;
    }
    NIMBLE_CHECK_GE(
        allocationSize,
        rowCount,
        "readCount should be less than allocationSize");

    auto vector = VectorInitializer<T>::initialize(
        type_, allocationSize, pool_, output, std::forward<Args>(args)...);
    vector->resize(allocationSize);

    NIMBLE_DCHECK_EQ(
        vector->sizes()->size(),
        (allocationSize * sizeof(velox::vector_size_t)),
        "Unexpected 'sizes' buffer size.");
    NIMBLE_DCHECK_EQ(
        vector->offsets()->size(),
        (allocationSize * sizeof(velox::vector_size_t)),
        "Unexpected 'offsets' buffer size.");
    auto* sizes = vector->sizes()->template asMutable<velox::vector_size_t>();
    auto* offsets =
        vector->offsets()->template asMutable<velox::vector_size_t>();

    auto nonNullCount = decoder_->next(
        count,
        sizes,
        stringBuffers_,
        [&]() { return ensureNulls(vector, allocationSize); },
        scatterBitmap);

    size_t childrenRows = 0;
    if (nonNullCount == rowCount) {
      vector->resetNulls();
    } else {
      vector->setNullCount(rowCount - nonNullCount);

      // Zero out sizes for null rows so the offset loop below is branchless.
      const auto* rawNulls = vector->rawNulls();
      NIMBLE_CHECK_NOT_NULL(
          rawNulls,
          "rawNulls must be initialized when nonNullCount < rowCount");
      velox::bits::forEachUnsetBit(
          reinterpret_cast<const uint64_t*>(rawNulls),
          0,
          rowCount,
          [sizes](int32_t i) { sizes[i] = 0; });
    }
    for (uint32_t i = 0; i < rowCount; ++i) {
      offsets[i] = static_cast<velox::vector_size_t>(childrenRows);
      childrenRows += sizes[i];
    }

    NIMBLE_CHECK_LE(
        childrenRows,
        std::numeric_limits<velox::vector_size_t>::max(),
        "Unsupported children count");
    return static_cast<velox::vector_size_t>(childrenRows);
  }

  // Reusable empty container to satisfy decoder_->next() API. Avoids heap
  // allocation on every call for non-string types.
  std::vector<velox::BufferPtr> stringBuffers_;

  uint32_t skipLengths(uint32_t count) {
    size_t childrenCount = 0;
    std::array<int32_t, kSkipBatchSize> sizes;

    constexpr auto byteSize = nullBytes(kSkipBatchSize);
    std::array<char, byteSize> nulls;

    // Unused place holder for api.
    std::vector<velox::BufferPtr> stringBuffers;
    while (count > 0) {
      auto readSize = std::min(count, kSkipBatchSize);
      auto nonNullCount = decoder_->next(
          readSize,
          sizes.data(),
          stringBuffers,
          [&]() { return nulls.data(); },
          /* scatterBitmap */ nullptr);

      if (nonNullCount == readSize) {
        for (uint32_t i = 0; i < readSize; ++i) {
          childrenCount += sizes[i];
        }
      } else {
        for (uint32_t i = 0; i < readSize; ++i) {
          if (velox::bits::isBitSet(
                  reinterpret_cast<const uint8_t*>(nulls.data()), i)) {
            childrenCount += sizes[i];
          }
        }
      }
      count -= readSize;
    }

    NIMBLE_CHECK_LE(
        childrenCount,
        std::numeric_limits<uint32_t>::max(),
        "Unsupported children count");
    return static_cast<uint32_t>(childrenCount);
  }
};

class ArrayFieldReader final : public MultiValueFieldReader {
 public:
  ArrayFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      std::unique_ptr<FieldReader> elementsReader)
      : MultiValueFieldReader{pool, std::move(type), decoder},
        elementsReader_{std::move(elementsReader)} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");
    const auto rowCount = encoding->rowCount();

    // Adding memory for velox::BaseVector::nulls_.
    // NOTE: We are not traversing encoding to get the number of nulls as it is
    // expensive for an estimation. We try to be conservative and assume it is
    // nullable.
    totalBytes += rowCount / 8;

    // Adding memory for velox::ArrayVectorBase::sizes_ and
    // velox::ArrayVectorBase::offsets_
    totalBytes += rowCount * sizeof(int32_t) * 2;

    auto rowSize = elementsReader_->estimatedRowSize();
    if (!rowSize.has_value()) {
      return std::nullopt;
    } else {
      const auto elementCount = rowSize.value().first;
      const auto elementAvgSize = rowSize.value().second;
      totalBytes += elementCount * elementAvgSize;
      return rowCount == 0
          ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
          : std::optional<std::pair<uint32_t, uint64_t>>(
                {rowCount, totalBytes / rowCount});
    }
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    auto childrenRows = this->template loadOffsets<velox::ArrayVector>(
        count, output, scatterBitmap);

    // As the fields are aligned by lengths decoder so no need to pass scatter
    // to elements
    elementsReader_->next(
        childrenRows,
        static_cast<velox::ArrayVector&>(*output).elements(),
        /* scatterBitmap */ nullptr);
  }

  void skip(uint32_t count) final {
    auto childrenCount = this->skipLengths(count);
    if (childrenCount > 0) {
      elementsReader_->skip(childrenCount);
    }
  }

  void reset() final {
    FieldReader::reset();
    elementsReader_->reset();
  }

 private:
  std::unique_ptr<FieldReader> elementsReader_;
};

class ArrayFieldReaderFactory final : public FieldReaderFactory {
 public:
  // Here the index is the index of the array lengths.
  ArrayFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::unique_ptr<FieldReaderFactory> elements,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        elements_{std::move(elements)} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return createReaderImpl<ArrayFieldReader>(
        decoders, nimbleType_->asArray().lengthsDescriptor(), [&]() {
          return elements_->createReader(decoders);
        });
  }

 private:
  std::unique_ptr<FieldReaderFactory> elements_;
};

class ArrayWithOffsetsFieldReader final : public MultiValueFieldReader {
 public:
  using OffsetType = uint32_t;

  ArrayWithOffsetsFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      Decoder* offsetDecoder,
      std::unique_ptr<FieldReader> elementsReader)
      : MultiValueFieldReader{pool, std::move(type), decoder},
        offsetDecoder_{offsetDecoder},
        elementsReader_{std::move(elementsReader)},
        cached_{false},
        cachedValue_{nullptr},
        cachedIndex_{0},
        cachedSize_{0},
        cachedLazyLoad_{false},
        cachedLazyChildrenRows_{0} {
    VectorInitializer<velox::ArrayVector>::initialize(
        type_, 1, pool_, cachedValue_);
  }

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    // TODO: Implement estimatedTotalOutputSize for ArrayWithOffsetsFieldReader.
    return std::nullopt;
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    auto rowCount = scatterCount(count, scatterBitmap);
    // read the offsets/indices which is one value per rowCount
    // and filter out deduped arrays to be read
    uint32_t nonNullCount;

    auto dictionaryVector =
        verifyVectorState<velox::DictionaryVector<velox::ComplexType>>(output);

    if (dictionaryVector) {
      dictionaryVector->resize(rowCount);
      resetIfNotWritable(output, dictionaryVector->indices());
    } else {
      velox::VectorPtr child;
      VectorInitializer<velox::ArrayVector>::initialize(
          type_, rowCount, pool_, child);
      auto indices =
          velox::AlignedBuffer::allocate<OffsetType>(rowCount, pool_);

      // Note: when creating a dictionary vector, it validates the vector (in
      // debug builds) for correctness. Therefore, we allocate all the buffers
      // above with the right size, but we "resize" them to zero, before
      // creating the dictionary vector, to avoid failing this validation.
      // We will later resize the vector to the correct size.
      // These resize operations are "no cost" operations, as shrinking a
      // vector/buffer doesn't free its memory, and resizing to the original
      // size doesn't allocate, as capacity is guaranteed to be enough.
      child->resize(0);
      indices->setSize(0);

      output = velox::BaseVector::wrapInDictionary(
          /* nulls */ nullptr,
          /* indices */ std::move(indices),
          /* size */ 0,
          /* values */ std::move(child));
      dictionaryVector =
          output->as<velox::DictionaryVector<velox::ComplexType>>();
      dictionaryVector->resize(rowCount);
    }

    void* nullsPtr = nullptr;
    uint32_t dedupCount = getIndicesDeduplicated(
        dictionaryVector->indices()->asMutable<OffsetType>(),
        [&]() {
          // The pointer will be initialized ONLY if the data is nullable.
          // Otherwise, it will remain nullptr, and this is handled below.
          nullsPtr = ensureNulls(dictionaryVector, rowCount);
          return nullsPtr;
        },
        nonNullCount,
        count,
        scatterBitmap);

    bool hasNulls = nonNullCount != rowCount;
    auto indices = dictionaryVector->indices()->asMutable<OffsetType>();
    NIMBLE_DCHECK_NOT_NULL(indices, "Indices missing.");

    // Returns the first non-null index or -1 (if all are null).
    auto baseIndex = findFirstBit(rowCount, hasNulls, nullsPtr, indices);

    bool cachedLocally = rowCount > 0 && cached_ && (baseIndex == cachedIndex_);

    // Initializes sizes and offsets in the vector.
    auto& dictionaryValues =
        const_cast<velox::VectorPtr&>(dictionaryVector->valueVector());
    auto childrenRows = loadOffsets<velox::ArrayVector>(
        dedupCount - cachedLocally,
        dictionaryValues,
        /* scatterBitmap */ nullptr,
        dedupCount);

    if (cached_ && cachedLazyLoad_) {
      if (cachedLocally) {
        elementsReader_->next(
            cachedLazyChildrenRows_,
            static_cast<velox::ArrayVector&>(*cachedValue_).elements(),
            /* scatterBitmap */ nullptr);
      } else {
        elementsReader_->skip(cachedLazyChildrenRows_);
      }
      cachedLazyLoad_ = false;
    }

    elementsReader_->next(
        childrenRows,
        static_cast<velox::ArrayVector&>(*dictionaryValues).elements(),
        /* scatterBitmap */ nullptr);

    if (cachedLocally) {
      auto vector = static_cast<velox::ArrayVector*>(dictionaryValues.get());

      // Copy elements from cache
      const auto cacheIdx = static_cast<int64_t>(dedupCount) - 1;
      velox::BaseVector::CopyRange cacheRange{
          0, static_cast<velox::vector_size_t>(cacheIdx), 1};
      vector->copyRanges(cachedValue_.get(), folly::Range(&cacheRange, 1));

      // copyRanges overwrites offsets from the source array and must be reset
      NIMBLE_DCHECK_EQ(
          vector->sizes()->size(),
          (dedupCount * sizeof(OffsetType)),
          "Unexpected 'sizes' buffer size.");
      NIMBLE_DCHECK_EQ(
          vector->offsets()->size(),
          (dedupCount * sizeof(OffsetType)),
          "Unexpected 'offsets' buffer size.");
      OffsetType* sizes = vector->sizes()->template asMutable<OffsetType>();
      OffsetType* offsets = vector->offsets()->template asMutable<OffsetType>();

      size_t rows = 0;
      if (cacheIdx > 0) {
        rows = offsets[cacheIdx - 1] + sizes[cacheIdx - 1];
      }

      sizes[cacheIdx] = cachedSize_;
      offsets[cacheIdx] = static_cast<OffsetType>(rows);

      if (hasNulls) {
        vector->setNull(cacheIdx, false);
      }
    }

    // Cache last item
    if (dedupCount > 0) {
      const auto& values = dictionaryVector->valueVector();
      auto idxToCache = std::max(
          0, static_cast<velox::vector_size_t>(dedupCount - 1 - cachedLocally));
      velox::BaseVector::CopyRange cacheRange{
          static_cast<velox::vector_size_t>(idxToCache), 0, 1};

      cachedValue_->prepareForReuse();
      cachedValue_->copyRanges(values.get(), folly::Range(&cacheRange, 1));

      // Get the index for this last element which must be non-null
      cachedIndex_ = indices[findLastBit(rowCount, hasNulls, nullsPtr)];

      cachedSize_ =
          static_cast<velox::ArrayVector&>(*values).sizeAt(idxToCache);
      cached_ = true;
      cachedLazyLoad_ = false;
    }

    // normalize the indices if not all null
    if (nonNullCount > 0) {
      if (hasNulls) {
        NIMBLE_DCHECK_NOT_NULL(nullsPtr, "Nulls buffer missing.");
        for (OffsetType idx = 0; idx < rowCount; idx++) {
          if (velox::bits::isBitNull(
                  static_cast<const uint64_t*>(nullsPtr), idx)) {
            continue;
          }

          indices[idx] = indices[idx] - baseIndex;
        }
      } else {
        for (OffsetType idx = 0; idx < rowCount; idx++) {
          indices[idx] = indices[idx] - baseIndex;
        }
      }
    }

    // update the indices as per cached and null locations
    if (hasNulls) {
      dictionaryVector->setNullCount(nonNullCount != rowCount);
      NIMBLE_DCHECK_NOT_NULL(nullsPtr, "Nulls buffer missing.");
      for (OffsetType idx = 0; idx < rowCount; idx++) {
        if (velox::bits::isBitNull(
                static_cast<const uint64_t*>(nullsPtr), idx)) {
          indices[idx] = dedupCount - 1;
        } else {
          if (indices[idx] == 0 && cachedLocally) { // cached index
            indices[idx] = dedupCount - 1;
          } else {
            indices[idx] -= cachedLocally;
          }
        }
      }
    } else {
      dictionaryVector->resetNulls();
      for (OffsetType idx = 0; idx < rowCount; idx++) {
        if (indices[idx] == 0 && cachedLocally) { // cached index
          indices[idx] = dedupCount - 1;
        } else {
          indices[idx] -= cachedLocally;
        }
      }
    }
  }

  void skip(uint32_t count) final {
    // read the offsets/indices which is one value per rowCount
    // and filter out deduped arrays to be read
    std::array<OffsetType, kSkipBatchSize> indices;
    std::array<char, nullBytes(kSkipBatchSize)> nulls;
    void* nullsPtr = nulls.data();
    uint32_t nonNullCount;

    while (count > 0) {
      auto batchedRowCount = std::min(count, kSkipBatchSize);
      uint32_t dedupCount = getIndicesDeduplicated(
          indices.data(),
          [&]() { return nullsPtr; },
          nonNullCount,
          batchedRowCount);

      bool hasNulls = nonNullCount != batchedRowCount;

      // baseIndex is the first non-null index
      auto baseIndex =
          findFirstBit(batchedRowCount, hasNulls, nullsPtr, indices.data());

      bool cachedLocally = cached_ && (baseIndex == cachedIndex_);
      if (cachedLocally) {
        dedupCount--;
      }

      // skip all the children except the last one
      if (dedupCount > 0) {
        auto childrenRows =
            cached_ && cachedLazyLoad_ ? cachedLazyChildrenRows_ : 0;
        childrenRows += this->skipLengths(dedupCount - 1);
        if (childrenRows > 0) {
          elementsReader_->skip(childrenRows);
        }

        /// cache the last child

        // get the index for this last element which must be non-null
        cachedIndex_ =
            indices[findLastBit(batchedRowCount, hasNulls, nullsPtr)];
        cached_ = true;
        cachedLazyLoad_ = true;
        cachedLazyChildrenRows_ =
            loadOffsets<velox::ArrayVector>(1, cachedValue_, nullptr);

        cachedSize_ = static_cast<velox::ArrayVector&>(*cachedValue_).sizeAt(0);
      }

      count -= batchedRowCount;
    }
  }

  void reset() final {
    FieldReader::reset();
    offsetDecoder_->reset();
    cached_ = false;
    elementsReader_->reset();
  }

 private:
  Decoder* offsetDecoder_;
  std::unique_ptr<FieldReader> elementsReader_;
  bool cached_;
  velox::VectorPtr cachedValue_;
  OffsetType cachedIndex_;
  uint32_t cachedSize_;
  bool cachedLazyLoad_;
  uint32_t cachedLazyChildrenRows_;

  static inline OffsetType
  findLastBit(uint32_t rowCount, bool hasNulls, const void* nulls) {
    if (!hasNulls) {
      return rowCount - 1;
    }

    NIMBLE_DCHECK_NOT_NULL(nulls, "Nulls buffer missing.");
    auto index = velox::bits::findLastBit(
        static_cast<const uint64_t*>(nulls), 0, rowCount);
    if (index == -1) {
      return rowCount - 1;
    }

    return index;
  }

  static inline int32_t findFirstBit(
      uint32_t rowCount,
      bool hasNulls,
      const void* nulls,
      const OffsetType* indices) {
    if (!hasNulls) {
      return indices[0];
    }

    NIMBLE_DCHECK_NOT_NULL(nulls, "Nulls buffer missing.");
    auto index = velox::bits::findFirstBit(
        static_cast<const uint64_t*>(nulls), 0, rowCount);

    if (index == -1) {
      return -1;
    }

    return indices[index];
  }

  uint32_t getIndicesDeduplicated(
      OffsetType* indices,
      std::function<void*()> nulls,
      uint32_t& nonNullCount,
      uint32_t count,
      const velox::bits::Bitmap* scatterBitmap = nullptr) {
    auto rowCount = scatterCount(count, scatterBitmap);
    // OffsetType* indices = dictIndices->asMutable<OffsetType>();
    void* nullsPtr;

    std::vector<velox::BufferPtr> stringBuffers;
    nonNullCount = offsetDecoder_->next(
        count,
        indices,
        stringBuffers,
        [&]() {
          nullsPtr = nulls();
          return nullsPtr;
        },
        scatterBitmap);

    // remove duplicated indices and calculate unique count
    uint32_t uniqueCount = 0;
    uint32_t prevIdx = 0;
    bool hasNulls = nonNullCount != rowCount;

    if (hasNulls) {
      NIMBLE_DCHECK_NOT_NULL(
          nullsPtr, "Data contain nulls but nulls buffer is not initialized.");

      for (uint32_t idx = 0; idx < rowCount; idx++) {
        if (velox::bits::isBitNull(
                static_cast<const uint64_t*>(nullsPtr), idx)) {
          indices[idx] = 0;
          continue;
        }

        if (uniqueCount == 0 || indices[idx] != indices[prevIdx]) {
          uniqueCount++;
        }
        prevIdx = idx;
      }
    } else {
      for (uint32_t idx = 0; idx < rowCount; idx++) {
        if (uniqueCount == 0 || indices[idx] != indices[prevIdx]) {
          uniqueCount++;
        }
        prevIdx = idx;
      }
    }

    return uniqueCount;
  }
};

class ArrayWithOffsetsFieldReaderFactory final : public FieldReaderFactory {
 public:
  // Here the index is the index of the array lengths.
  ArrayWithOffsetsFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::unique_ptr<FieldReaderFactory> elements,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        elements_{std::move(elements)} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return createReaderImpl<ArrayWithOffsetsFieldReader>(
        decoders,
        nimbleType_->asArrayWithOffsets().lengthsDescriptor(),
        [&]() {
          return getDecoder(
              decoders, nimbleType_->asArrayWithOffsets().offsetsDescriptor());
        },
        [&]() { return elements_->createReader(decoders); });
  }

 private:
  std::unique_ptr<FieldReaderFactory> elements_;
};

class SlidingWindowMapFieldReader final : public FieldReader {
 public:
  SlidingWindowMapFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* offsetDecoder,
      Decoder* lengthsDecoder,
      std::unique_ptr<FieldReader> keysReader,
      std::unique_ptr<FieldReader> valuesReader)
      : FieldReader(pool, std::move(type), nullptr),
        offsetDecoder_{offsetDecoder},
        lengthsDecoder_{lengthsDecoder},
        keysReader_{std::move(keysReader)},
        valuesReader_{std::move(valuesReader)},
        currentOffset_{0},
        cacheOffset_{0} {
    VectorInitializer<velox::MapVector>::initialize(
        type_, 0, pool_, cachedMap_);
  }

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    // TODO: Implement estimatedTotalOutputSize for SlidingWindowMapFieldReader.
    return std::nullopt;
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) override {
    auto rowCount = scatterCount(count, scatterBitmap);

    auto dictionaryVector =
        verifyVectorState<velox::DictionaryVector<velox::ComplexType>>(output);

    // Initialize the output vector
    if (dictionaryVector) {
      dictionaryVector->resize(rowCount);
      dictionaryVector->resetNulls();
      resetIfNotWritable(output, dictionaryVector->indices());
      auto& dictionaryValues =
          const_cast<velox::VectorPtr&>(dictionaryVector->valueVector());
      auto child = verifyVectorState<velox::MapVector>(dictionaryValues);
      if (child) {
        child->resize(rowCount);
      } else {
        VectorInitializer<velox::MapVector>::initialize(
            type_, rowCount, pool_, dictionaryValues);
      }
    } else {
      velox::VectorPtr child;
      VectorInitializer<velox::MapVector>::initialize(
          type_, rowCount, pool_, child);
      auto indices = velox::AlignedBuffer::allocate<uint32_t>(rowCount, pool_);

      // Note: when creating a dictionary vector, it validates the vector (in
      // debug builds) for correctness. Therefore, we allocate all the buffers
      // above with the right size, but we "resize" them to zero, before
      // creating the dictionary vector, to avoid failing this validation.
      // We will later resize the vector to the correct size.
      // These resize operations are "no cost" operations, as shrinking a
      // vector/buffer doesn't free its memory, and resizing to the original
      // size doesn't allocate, as capacity is guaranteed to be enough.
      child->resize(0);
      indices->setSize(0);

      output = velox::BaseVector::wrapInDictionary(
          /* nulls */ nullptr,
          /* indices */ std::move(indices),
          /* size */ 0,
          /* values */ std::move(child));
      dictionaryVector =
          output->as<velox::DictionaryVector<velox::ComplexType>>();
      dictionaryVector->resize(rowCount);
    }

    // Read the offsets which can be nullable
    auto indices = dictionaryVector->indices()->asMutable<uint32_t>();
    void* nullsPtr = nullptr;
    std::vector<velox::BufferPtr> stringBuffers;
    const uint32_t nonNullCount = offsetDecoder_->next(
        count,
        indices,
        stringBuffers,
        [&]() {
          nullsPtr = ensureNulls(dictionaryVector, rowCount);
          return nullsPtr;
        },
        scatterBitmap);

    // Return early if everything is null
    if (nonNullCount == 0) {
      return;
    }

    bool hasNulls = nonNullCount != rowCount;
    // Read the lengths
    uint32_t lengthBuffer[nonNullCount];
    std::vector<velox::BufferPtr> lengthStringBuffers;
    lengthsDecoder_->next(nonNullCount, lengthBuffer, lengthStringBuffers);

    // Convert the offsets and lengths to a list of unique offsets and lengths
    // and update the indices to be 0-based indices
    std::vector<uint32_t> deduplicatedOffsets, deduplicatedLengths;
    deduplicatedOffsets.reserve(nonNullCount);
    deduplicatedLengths.reserve(nonNullCount);
    uint32_t uniqueCount = 0, startOffset = 0, endOffset = 0;
    if (hasNulls) {
      NIMBLE_DCHECK_NOT_NULL(
          nullsPtr, "Data contain nulls but nulls buffer is not initialized.");
      uint32_t nullCount = 0;
      for (uint32_t idx = 0; idx < rowCount; ++idx) {
        if (velox::bits::isBitNull(
                static_cast<const uint64_t*>(nullsPtr), idx)) {
          indices[idx] = 0;
          ++nullCount;
          continue;
        }
        // First non-null item
        if (deduplicatedOffsets.empty()) {
          deduplicatedOffsets.emplace_back(indices[idx]);
          deduplicatedLengths.emplace_back(lengthBuffer[idx - nullCount]);
          startOffset = deduplicatedOffsets.back();
          endOffset = deduplicatedOffsets.back() + deduplicatedLengths.back();
          uniqueCount = 1;
        } else if (
            // Check if the current item is the same as the last one
            // If not, update deduplicatedOffsets and deduplicatedLengths
            deduplicatedOffsets.back() != indices[idx] ||
            deduplicatedLengths.back() != lengthBuffer[idx - nullCount]) {
          deduplicatedOffsets.emplace_back(indices[idx]);
          deduplicatedLengths.emplace_back(lengthBuffer[idx - nullCount]);
          endOffset = std::max(
              deduplicatedOffsets.back() + deduplicatedLengths.back(),
              endOffset);
          ++uniqueCount;
        }
        indices[idx] = uniqueCount - 1;
      }
      NIMBLE_CHECK_EQ(
          nonNullCount + nullCount, rowCount, "Null Count is not matching");
    } else {
      deduplicatedOffsets.emplace_back(indices[0]);
      deduplicatedLengths.emplace_back(lengthBuffer[0]);
      startOffset = deduplicatedOffsets.back();
      endOffset = deduplicatedOffsets.back() + deduplicatedLengths.back();
      indices[0] = 0;
      ++uniqueCount;
      // Start from the second item, check if the current item is the same as
      // the last one If not, update deduplicatedOffsets and deduplicatedLengths
      for (uint32_t idx = 1; idx < rowCount; ++idx) {
        if (deduplicatedOffsets.back() != indices[idx] ||
            deduplicatedLengths.back() != lengthBuffer[idx]) {
          deduplicatedOffsets.emplace_back(indices[idx]);
          deduplicatedLengths.emplace_back(lengthBuffer[idx]);
          endOffset = std::max(
              deduplicatedOffsets.back() + deduplicatedLengths.back(),
              endOffset);
          ++uniqueCount;
        }
        indices[idx] = uniqueCount - 1;
      }
    }

    NIMBLE_DCHECK_GT(
        deduplicatedLengths.size(), 0, "Invalid deduplicatedLengths size.");
    NIMBLE_DCHECK_EQ(
        deduplicatedLengths.size(),
        uniqueCount,
        "deduplicatedLengths size mismatch.");

    // Fill the map vector
    auto map =
        static_cast<velox::MapVector*>(dictionaryVector->valueVector().get());
    map->resize(uniqueCount);
    map->mapKeys()->resize(0);
    map->mapValues()->resize(0);

    bool useCache = false;
    uint32_t childrenRows = endOffset - startOffset;
    if (childrenRows > 0) {
      if (isCached()) {
        // NOTE: We assume that the cache will either fully match the current
        // offset, or will fully not match it. There is another "possible" state
        // (in the future, but not now). When sliding window is actually
        // supported, it is possible that the cache will cover "part" of the
        // required map keys and values (but will not be an exact match). When
        // we add supprot for sliding windows, we need to add correct handling
        // for the cache, to handle hese partial mismatches.
        const uint32_t size = cacheSize();
        if (startOffset == cacheOffset_ && size == deduplicatedLengths[0]) {
          useCache = true;
          childrenRows -= size;
        } else {
          resetCache();
        }
      }
    }

    if (childrenRows > 0) {
      keysReader_->next(
          childrenRows,
          map->mapKeys(),
          /* scatterBitmap */ nullptr);
      valuesReader_->next(
          childrenRows,
          map->mapValues(),
          /* scatterBitmap */ nullptr);
    }

    currentOffset_ = endOffset;
    const uint32_t lastElement =
        static_cast<uint32_t>(deduplicatedOffsets.size()) - 1;

    map->sizes()->setSize(uniqueCount * sizeof(uint32_t));
    map->offsets()->setSize(uniqueCount * sizeof(uint32_t));
    auto sizes = map->sizes()->template asMutable<uint32_t>();
    auto offsets = map->offsets()->template asMutable<uint32_t>();

    if (useCache) {
      if (!map->mapKeys()->isWritable()) {
        velox::BaseVector::ensureWritable(
            velox::SelectivityVector::empty(),
            map->mapKeys()->type(),
            pool_,
            map->mapKeys());
      }
      if (!map->mapValues()->isWritable()) {
        velox::BaseVector::ensureWritable(
            velox::SelectivityVector::empty(),
            map->mapValues()->type(),
            pool_,
            map->mapValues());
      }
      velox::BaseVector::CopyRange cacheRange{/* sourceIndex */ 0,
                                              /* targetIndex */ 0,
                                              /* count */ 1};
      map->copyRanges(cachedMap_.get(), folly::Range(&cacheRange, 1));

      const uint32_t size = cacheSize();
      offsets[0] = childrenRows;
      sizes[0] = deduplicatedLengths[0];

      uint32_t mapOffset = deduplicatedLengths[0];
      for (uint32_t i = 1; i < uniqueCount; ++i) {
        sizes[i] = deduplicatedLengths[i];
        offsets[i] = mapOffset - size;

        mapOffset += deduplicatedLengths[i];
      }
    } else {
      uint32_t mapOffset = 0;
      for (uint32_t i = 0; i < uniqueCount; ++i) {
        sizes[i] = deduplicatedLengths[i];
        offsets[i] = mapOffset;
        mapOffset += sizes[i];
      }
    }

    // Populate cache
    if (deduplicatedLengths.back() == 0) {
      resetCache();
    } else if (
        !isCached() ||
        // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
        deduplicatedOffsets.back() != cacheOffset_) {
      if (!cachedMap_->isWritable()) {
        velox::BaseVector::ensureWritable(
            velox::SelectivityVector::empty(),
            cachedMap_->type(),
            pool_,
            cachedMap_);
      }
      velox::BaseVector::CopyRange cacheRange{
          static_cast<velox::vector_size_t>(lastElement), 0, 1};
      cachedMap_->resize(1);
      cachedMap_->copyRanges(map, folly::Range(&cacheRange, 1));
      // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
      cacheOffset_ = deduplicatedOffsets.back();
    }
  }

  void skip(uint32_t count) final {
    if (count == 0) {
      return;
    }

    // @lint-ignore CLANGTIDY cppcoreguidelines-pro-type-member-init
    std::array<uint32_t, kSkipBatchSize> offsets;
    // @lint-ignore CLANGTIDY cppcoreguidelines-pro-type-member-init
    std::array<uint32_t, kSkipBatchSize> lengths;
    std::array<char, nullBytes(kSkipBatchSize)> nulls;
    void* nullsPtr = nulls.data();

    uint32_t childrenSkip = 0;
    uint32_t lastOffset = 0;
    uint32_t lastLength = 0;

    // Unused string buffer container for api
    std::vector<velox::BufferPtr> stringBuffers;
    while (count > 0) {
      auto skipSize = std::min(count, kSkipBatchSize);
      auto nonNullCount = offsetDecoder_->next(
          skipSize,
          offsets.data(),
          stringBuffers,
          [&]() { return nullsPtr; },
          /* scatterBitmap */ nullptr);
      std::vector<velox::BufferPtr> lengthStringBuffers;
      lengthsDecoder_->next(nonNullCount, lengths.data(), lengthStringBuffers);

      const bool hasNulls = nonNullCount != skipSize;

      uint32_t offsetIndex = 0;
      uint32_t lengthIndex = 0;

      if (isCached()) {
        const uint32_t size = cacheSize();
        if (hasNulls) {
          for (; offsetIndex < skipSize; ++offsetIndex) {
            if (!velox::bits::isBitNull(
                    static_cast<const uint64_t*>(nullsPtr), offsetIndex)) {
              const uint32_t length = lengths[lengthIndex++];
              if (offsets[offsetIndex] == cacheOffset_ && length == size) {
                continue;
              } else {
                resetCache();
                currentOffset_ = offsets[offsetIndex];
                lastOffset = currentOffset_;
                lastLength = length;
                childrenSkip += length;
                break;
              }
            }
          }
        } else {
          for (; offsetIndex < skipSize; ++offsetIndex) {
            const uint32_t length = lengths[lengthIndex++];
            if (offsets[offsetIndex] == cacheOffset_ && length == size) {
              continue;
            } else {
              resetCache();
              currentOffset_ = offsets[offsetIndex];
              lastOffset = currentOffset_;
              lastLength = length;
              childrenSkip += length;
              break;
            }
          }
        }
      }

      count -= skipSize;

      if (isCached()) {
        // The entire skip was within the same cached "run"
        continue;
      }

      // Find how much to skip the children readers. This should not include the
      // last (non-null) item, as we are going to cache this item.
      if (hasNulls) {
        for (; offsetIndex < skipSize; ++offsetIndex) {
          if (velox::bits::isBitNull(
                  static_cast<const uint64_t*>(nullsPtr), offsetIndex)) {
            continue;
          }

          const uint32_t offset = offsets[offsetIndex];
          const uint32_t length = lengths[lengthIndex];
          if (lastOffset != offset || lastLength != length) {
            childrenSkip += length;
            currentOffset_ = offset;
          }
          lastOffset = offset;
          lastLength = length;
          ++lengthIndex;
        }
      } else {
        for (; offsetIndex < skipSize; ++offsetIndex) {
          const uint32_t offset = offsets[offsetIndex];
          const uint32_t length = lengths[lengthIndex];
          if (lastOffset != offset || lastLength != length) {
            childrenSkip += length;
            currentOffset_ = offset;
          }
          lastOffset = offset;
          lastLength = length;
          ++lengthIndex;
        }
      }
    }

    if (childrenSkip == 0) {
      return;
    }

    childrenSkip -= lastLength;

    if (childrenSkip > 0) {
      keysReader_->skip(childrenSkip);
      valuesReader_->skip(childrenSkip);
    }

    if (lastLength == 0) {
      return;
    }

    auto& cachedMap = static_cast<velox::MapVector&>(*cachedMap_);
    cachedMap_->resize(1);
    cacheOffset_ = lastOffset;
    currentOffset_ += lastLength;
    keysReader_->next(
        lastLength,
        cachedMap.mapKeys(),
        /* scatterBitmap */ nullptr);
    valuesReader_->next(
        lastLength,
        cachedMap.mapValues(),
        /* scatterBitmap */ nullptr);

    cachedMap.mutableOffsets(1)->template asMutable<uint32_t>()[0] = 0;
    cachedMap.mutableSizes(1)->template asMutable<uint32_t>()[0] = lastLength;
  }

  // Move the cursor of key and value readers to the given offset
  void seek(uint32_t offset) {
    if (offset == currentOffset_) {
      return;
    } else if (offset < currentOffset_) {
      keysReader_->reset();
      valuesReader_->reset();
      keysReader_->skip(offset);
      valuesReader_->skip(offset);
    } else {
      keysReader_->skip(offset - currentOffset_);
      valuesReader_->skip(offset - currentOffset_);
    }
    currentOffset_ = offset;
  }

  void reset() final {
    FieldReader::reset();
    offsetDecoder_->reset();
    lengthsDecoder_->reset();
    keysReader_->reset();
    valuesReader_->reset();
    resetCache();
  }

 private:
  inline bool isCached() const {
    return cachedMap_->size() > 0;
  }

  inline uint32_t cacheSize() {
    auto& cachedMap = static_cast<velox::MapVector&>(*cachedMap_);
    NIMBLE_DCHECK_GT(
        isCached() && cachedMap.sizes()->size(), 0, "Unexpected cache state.");
    return cachedMap.sizes()->as<int32_t>()[0];
  }

  inline void resetCache() {
    cachedMap_->resize(0);
  }

  Decoder* offsetDecoder_;
  Decoder* lengthsDecoder_;
  std::unique_ptr<FieldReader> keysReader_;
  std::unique_ptr<FieldReader> valuesReader_;
  uint32_t currentOffset_;

  // cache
  velox::VectorPtr cachedMap_;
  uint32_t cacheOffset_;
};

class SlidingWindowMapFieldReaderFactory final : public FieldReaderFactory {
 public:
  // Here the index is the index of the array lengths.
  SlidingWindowMapFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::unique_ptr<FieldReaderFactory> keys,
      std::unique_ptr<FieldReaderFactory> values,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        keys_{std::move(keys)},
        values_{std::move(values)} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return createReaderImpl<SlidingWindowMapFieldReader>(
        decoders,
        nimbleType_->asSlidingWindowMap().offsetsDescriptor(),
        [&]() {
          return getDecoder(
              decoders, nimbleType_->asSlidingWindowMap().lengthsDescriptor());
        },
        [&]() { return keys_->createReader(decoders); },
        [&]() { return values_->createReader(decoders); });
  }

 private:
  std::unique_ptr<FieldReaderFactory> keys_;
  std::unique_ptr<FieldReaderFactory> values_;
};

class MapFieldReader final : public MultiValueFieldReader {
 public:
  MapFieldReader(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      std::unique_ptr<FieldReader> keysReader,
      std::unique_ptr<FieldReader> valuesReader)
      : MultiValueFieldReader{pool, std::move(type), decoder},
        keysReader_{std::move(keysReader)},
        valuesReader_{std::move(valuesReader)} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    const auto* encoding = decoder_->encoding();
    NIMBLE_CHECK_NOT_NULL(
        encoding, "Decoder must be loaded for output size estimation.");
    const auto rowCount = encoding->rowCount();

    // Adding memory for velox::BaseVector::nulls_.
    // NOTE: We are not traversing encoding to get the number of nulls as it
    // is expensive for an estimation. We try to be conservative and assume it
    // is nullable.
    totalBytes += rowCount / 8;

    // Adding memory for velox::MapVector::sizes_ and
    // velox::MapVector::offsets_
    totalBytes += rowCount * sizeof(int32_t) * 2;

    auto keySize = keysReader_->estimatedRowSize();
    if (!keySize.has_value()) {
      return std::nullopt;
    }
    auto valueSize = valuesReader_->estimatedRowSize();
    if (!valueSize.has_value()) {
      return std::nullopt;
    }
    totalBytes += keySize.value().first * keySize.value().second +
        valueSize.value().first * valueSize.value().second;
    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    auto childrenRows = this->template loadOffsets<velox::MapVector>(
        count, output, scatterBitmap);

    // As the field is aligned by lengths decoder then no need to pass
    // scatterBitmap to keys and values
    auto& mapVector = static_cast<velox::MapVector&>(*output);
    keysReader_->next(
        childrenRows, mapVector.mapKeys(), /* scatterBitmap */ nullptr);
    valuesReader_->next(
        childrenRows, mapVector.mapValues(), /* scatterBitmap */ nullptr);
  }

  void skip(uint32_t count) final {
    auto childrenCount = this->skipLengths(count);
    if (childrenCount > 0) {
      keysReader_->skip(childrenCount);
      valuesReader_->skip(childrenCount);
    }
  }

  void reset() final {
    FieldReader::reset();
    keysReader_->reset();
    valuesReader_->reset();
  }

 private:
  std::unique_ptr<FieldReader> keysReader_;
  std::unique_ptr<FieldReader> valuesReader_;
};

class MapFieldReaderFactory final : public FieldReaderFactory {
 public:
  // Here the index is the index of the array lengths.
  MapFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::unique_ptr<FieldReaderFactory> keys,
      std::unique_ptr<FieldReaderFactory> values,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        keys_{std::move(keys)},
        values_{std::move(values)} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return createReaderImpl<MapFieldReader>(
        decoders,
        nimbleType_->asMap().lengthsDescriptor(),
        [&]() { return keys_->createReader(decoders); },
        [&]() { return values_->createReader(decoders); });
  }

 private:
  std::unique_ptr<FieldReaderFactory> keys_;
  std::unique_ptr<FieldReaderFactory> values_;
};

// Read values from boolean decoder and return number of true values.
template <typename TrueHandler>
uint32_t readBooleanValues(
    Decoder* decoder,
    bool* buffer,
    uint32_t count,
    TrueHandler handler) {
  std::vector<velox::BufferPtr> stringBuffers;
  decoder->next(count, buffer, stringBuffers);

  uint32_t trueCount = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (buffer[i]) {
      handler(i);
      ++trueCount;
    }
  }
  return trueCount;
}

uint32_t readBooleanValues(Decoder* decoder, bool* buffer, uint32_t count) {
  return readBooleanValues(decoder, buffer, count, [](auto /* ignored */) {});
}

namespace {
// Per row overhead on velox vector for null value. Returns overhead in bits.
uint64_t nullOverheadBits(const velox::TypePtr& type) {
  switch (type->kind()) {
    case velox::TypeKind::BOOLEAN:
      return 1;
    case velox::TypeKind::TINYINT:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::TINYINT>::NativeType);
    case velox::TypeKind::SMALLINT:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::SMALLINT>::NativeType);
    case velox::TypeKind::INTEGER:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::INTEGER>::NativeType);
    case velox::TypeKind::BIGINT:
      return 8 * sizeof(velox::TypeTraits<velox::TypeKind::BIGINT>::NativeType);
    case velox::TypeKind::REAL:
      return 8 * sizeof(velox::TypeTraits<velox::TypeKind::REAL>::NativeType);
    case velox::TypeKind::DOUBLE:
      return 8 * sizeof(velox::TypeTraits<velox::TypeKind::DOUBLE>::NativeType);
    case velox::TypeKind::HUGEINT:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::HUGEINT>::NativeType);
    case velox::TypeKind::TIMESTAMP:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::TIMESTAMP>::NativeType);
    case velox::TypeKind::UNKNOWN:
      return 8 *
          sizeof(velox::TypeTraits<velox::TypeKind::UNKNOWN>::NativeType);
    case velox::TypeKind::VARCHAR:
      [[fallthrough]];
    case velox::TypeKind::VARBINARY:
      return 8 * sizeof(velox::StringView);
    case velox::TypeKind::ARRAY:
      [[fallthrough]];
    case velox::TypeKind::MAP:
      // 4 bytes per row on sizes_ and 4 bytes per row on offsets_
      return 8 * 8;
    case velox::TypeKind::ROW:
      [[fallthrough]];
    default:
      // Not adding nulls overhead (reduced accuracy) for unknown types.
      return 0;
  }
}
} // namespace

template <bool hasNull>
class RowFieldReader final : public FieldReader {
 public:
  RowFieldReader(
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::unique_ptr<FieldReader>> childrenReaders,
      Vector<bool>& boolBuffer,
      velox::memory::MemoryPool* pool,
      const FieldReader::Options& options)
      : FieldReader{*pool, std::move(type), decoder, options},
        childrenReaders_{std::move(childrenReaders)},
        boolBuffer_{boolBuffer} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    uint64_t rowCount{0};
    if constexpr (hasNull) {
      const auto* encoding = decoder_->encoding();
      NIMBLE_CHECK_NOT_NULL(
          encoding, "Decoder must be loaded for output size estimation.");
      rowCount = encoding->rowCount();
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
    }

    for (auto& reader : childrenReaders_) {
      if (reader == nullptr) {
        continue;
      }
      auto childSize = reader->estimatedRowSize();
      if (!childSize.has_value()) {
        return std::nullopt;
      }

      // Add non-null size
      const auto childRowCount = childSize.value().first;
      totalBytes += childRowCount * childSize.value().second;

      // Add null size
      if constexpr (hasNull) {
        const auto nullCount = rowCount - childSize.value().first;
        totalBytes += nullCount * nullOverheadBits(reader->type()) / 8;
      } else if (rowCount == 0) {
        rowCount = childRowCount;
      } else if (childRowCount != 0) {
        NIMBLE_CHECK_EQ(
            rowCount,
            childRowCount,
            "rowCount should be equal to childRowCount under no null condition.");
      }
    }

    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    if (parallelDecodeEnabled(childrenReaders_.size())) {
      folly::coro::blockingWait(co_next(count, output, scatterBitmap));
      return;
    }
    const auto outputContext = prepareOutput(count, output, scatterBitmap);
    velox::bits::Bitmap bitmap{
        outputContext.scatterNullBits, outputContext.rowCount};
    auto* bitmapPtr =
        outputContext.scatterNullBits != nullptr ? &bitmap : nullptr;
    for (uint32_t i = 0; i < childrenReaders_.size(); ++i) {
      auto& reader = childrenReaders_[i];
      if (reader != nullptr) {
        reader->next(
            outputContext.selectedNonNullCount,
            outputContext.vector->childAt(i),
            bitmapPtr);
      }
    }
  }

  folly::coro::Task<void> co_next(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    const auto outputContext = prepareOutput(count, output, scatterBitmap);

    // Collect indices of non-null child readers.
    std::vector<uint32_t> nonNullChildren;
    nonNullChildren.reserve(childrenReaders_.size());
    for (uint32_t i = 0; i < childrenReaders_.size(); ++i) {
      if (childrenReaders_[i] != nullptr) {
        nonNullChildren.emplace_back(i);
      }
    }

    if (nonNullChildren.empty()) {
      co_return;
    }

    const uint32_t taskCount =
        computeParallelDecodeTaskCount(nonNullChildren.size());
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::RowFieldReader::co_next",
        const_cast<uint32_t*>(&taskCount));

    const uint32_t childrenPerTask = nonNullChildren.size() / taskCount;
    const uint32_t numRemainderChildren = nonNullChildren.size() % taskCount;

    // Decodes children in [startIdx, endIdx) synchronously.
    auto decodeRange = [this, &outputContext, &nonNullChildren](
                           uint32_t startIdx, uint32_t endIdx) {
      velox::bits::Bitmap bitmap{
          outputContext.scatterNullBits, outputContext.rowCount};
      auto* bitmapPtr =
          outputContext.scatterNullBits != nullptr ? &bitmap : nullptr;
      for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
        const auto i = nonNullChildren[idx];
        childrenReaders_[i]->next(
            outputContext.selectedNonNullCount,
            outputContext.vector->childAt(i),
            bitmapPtr);
      }
    };

    // First 'numRemainderChildren' tasks get one extra child each.
    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    tasks.reserve(taskCount);
    uint32_t nextChildIdx = 0;
    for (uint32_t task = 0; task < taskCount; ++task) {
      const uint32_t endChildIdx = nextChildIdx + childrenPerTask +
          (task < numRemainderChildren ? 1 : 0);
      tasks.emplace_back(
          folly::coro::co_withExecutor(
              decodeExecutor_,
              folly::coro::co_invoke(
                  [&decodeRange, nextChildIdx, endChildIdx]()
                      -> folly::coro::Task<void> {
                    decodeRange(nextChildIdx, endChildIdx);
                    co_return;
                  })));
      nextChildIdx = endChildIdx;
    }

    co_await folly::coro::collectAllRange(std::move(tasks));
  }

  void skip(uint32_t count) final {
    uint32_t childRowCount = count;
    if constexpr (hasNull) {
      std::array<bool, kSkipBatchSize> buffer{};
      childRowCount = 0;
      while (count > 0) {
        auto readSize = std::min(count, kSkipBatchSize);
        childRowCount += readBooleanValues(decoder_, buffer.data(), readSize);
        count -= readSize;
      }
    }

    if (childRowCount > 0) {
      for (auto& reader : childrenReaders_) {
        if (reader) {
          reader->skip(childRowCount);
        }
      }
    }
  }

  void reset() final {
    FieldReader::reset();
    for (auto& reader : childrenReaders_) {
      if (reader) {
        reader->reset();
      }
    }
  }

 private:
  struct OutputContext {
    // Initialized output row vector.
    velox::RowVector* vector;
    // Number of non-null rows after scatter filtering.
    uint32_t selectedNonNullCount;
    // Scatter bitmap bits for child null propagation, or nullptr if all
    // non-null.
    const void* scatterNullBits;
    // Total number of rows including nulls and scattered positions.
    uint32_t rowCount;
  };

  OutputContext prepareOutput(
      uint32_t count,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) {
    const auto rowCount = scatterCount(count, scatterBitmap);
    auto* vector = VectorInitializer<velox::RowVector>::initialize(
        type_, rowCount, pool_, output);
    vector->children().resize(childrenReaders_.size());
    vector->unsafeResize(rowCount);
    const void* scatterNullBits = nullptr;
    uint32_t selectedNonNullCount = 0;

    if constexpr (hasNull) {
      zeroNulls(vector, rowCount);
      // if it is a scattered read case then we can't read the rowCount
      // values from the nulls, we count the set value in scatterBitmap and
      // read only those values, if there is no scatter then we can read
      // rowCount values in its original place without removal
      boolBuffer_.resize(count);
      std::vector<velox::BufferPtr> stringBuffers;
      decoder_->next(count, boolBuffer_.data(), stringBuffers);

      auto* nullBuffer = ensureNulls(vector, rowCount);
      velox::bits::BitmapBuilder nullBits{nullBuffer, rowCount};
      if (scatterBitmap != nullptr) {
        uint32_t boolBufferOffset = 0;
        for (uint32_t i = 0; i < rowCount; ++i) {
          if (scatterBitmap->test(i) && boolBuffer_[boolBufferOffset++]) {
            nullBits.set(i);
            ++selectedNonNullCount;
          }
        }
      } else {
        for (uint32_t i = 0; i < rowCount; ++i) {
          if (boolBuffer_[i]) {
            nullBits.set(i);
            ++selectedNonNullCount;
          }
        }
      }
      if (UNLIKELY(selectedNonNullCount == rowCount)) {
        vector->resetNulls();
      } else {
        vector->setNullCount(rowCount - selectedNonNullCount);
        scatterNullBits = nullBuffer;
      }
    } else {
      selectedNonNullCount = count;
      if (scatterBitmap != nullptr) {
        const auto requiredBytes = velox::bits::nbytes(rowCount);
        auto* nullBuffer = ensureNulls(vector, rowCount);
        // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
        std::memcpy(
            nullBuffer,
            static_cast<const char*>(scatterBitmap->bits()),
            requiredBytes);
        vector->setNullCount(rowCount - count);
        scatterNullBits = scatterBitmap->bits();
      } else {
        vector->resetNulls();
      }
    }
    return {vector, selectedNonNullCount, scatterNullBits, rowCount};
  }

  std::vector<std::unique_ptr<FieldReader>> childrenReaders_;
  Vector<bool>& boolBuffer_;
};

class RowFieldReaderFactory final : public FieldReaderFactory {
 public:
  // Here the index is the index of the null decoder.
  RowFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::vector<std::unique_ptr<FieldReaderFactory>> children,
      velox::memory::MemoryPool* pool,
      const FieldReaderParams& params)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        decodeExecutor_{params.decodeExecutor},
        maxDecodeParallelism_{params.maxDecodeParallelism},
        minStreamsPerDecodeUnit_{params.minStreamsPerDecodeUnit},
        children_{std::move(children)},
        boolBuffer_{pool_} {}

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    auto nulls = getDecoder(decoders, nimbleType_->asRow().nullsDescriptor());

    std::vector<std::unique_ptr<FieldReader>> childrenReaders(children_.size());
    for (uint32_t i = 0; i < children_.size(); ++i) {
      auto& child = children_[i];
      if (child) {
        // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
        childrenReaders[i] = child->createReader(decoders);
      }
    }

    const FieldReader::Options options{
        decodeExecutor_, maxDecodeParallelism_, minStreamsPerDecodeUnit_};
    if (!nulls) {
      return std::make_unique<RowFieldReader<false>>(
          veloxType_,
          nulls,
          std::move(childrenReaders),
          boolBuffer_,
          pool_,
          options);
    }

    return std::make_unique<RowFieldReader<true>>(
        veloxType_,
        nulls,
        std::move(childrenReaders),
        boolBuffer_,
        pool_,
        options);
  }

 private:
  folly::Executor* const decodeExecutor_;
  const uint32_t maxDecodeParallelism_;
  const uint32_t minStreamsPerDecodeUnit_;
  std::vector<std::unique_ptr<FieldReaderFactory>> children_;
  Vector<bool> boolBuffer_;
};

// Represent a keyed value node for flat map
// Before reading the value, InMap vectors we need to call load()
template <typename T>
class FlatMapKeyNode {
 public:
  FlatMapKeyNode(
      velox::memory::MemoryPool& memoryPool,
      std::unique_ptr<FieldReader> valueReader,
      Decoder* inMapDecoder,
      const velox::dwio::common::flatmap::KeyValue<T>& key)
      : key_{key},
        valueReader_{std::move(valueReader)},
        inMapDecoder_{inMapDecoder},
        inMapData_{&memoryPool},
        mergedNulls_{&memoryPool} {}

  ~FlatMapKeyNode() = default;

  void readAsChild(
      velox::VectorPtr& vector,
      uint32_t numValues,
      uint32_t nonNullValues,
      const Vector<bool>& mapNulls,
      Vector<char>* mergedNulls = nullptr) {
    if (mergedNulls == nullptr) {
      mergedNulls = &mergedNulls_;
    }
    const auto nonNullCount =
        mergeNulls(numValues, nonNullValues, mapNulls, *mergedNulls);
    velox::bits::Bitmap bitmap{mergedNulls->data(), numValues};
    valueReader_->next(nonNullCount, vector, &bitmap);
    NIMBLE_DCHECK_EQ(numValues, vector->size(), "Items not loaded");
  }

  uint32_t readInMapData(uint32_t numValues) {
    inMapData_.resize(numValues);
    if (inMapDecoder_ == nullptr) {
      // Missing in-map stream means all rows are in-map.
      std::fill(inMapData_.data(), inMapData_.data() + numValues, true);
      numValues_ = numValues;
    } else {
      numValues_ =
          readBooleanValues(inMapDecoder_, inMapData_.data(), numValues);
    }
    return numValues_;
  }

  void loadValues(velox::VectorPtr& values) {
    valueReader_->next(numValues_, values, /*scatterBitmap=*/nullptr);
    NIMBLE_DCHECK_EQ(numValues_, values->size(), "Items not loaded");
  }

  void skip(uint32_t numValues) {
    auto numItems = readInMapData(numValues);
    if (numItems > 0) {
      valueReader_->skip(numItems);
    }
  }

  const velox::dwio::common::flatmap::KeyValue<T>& key() const {
    return key_;
  }

  const FieldReader* valueReader() const {
    return valueReader_.get();
  }

  const Decoder* inMapDecoder() const {
    return inMapDecoder_;
  }

  bool inMap(uint32_t index) const {
    return inMapData_[index];
  }

  void reset() {
    if (inMapDecoder_ != nullptr) {
      inMapDecoder_->reset();
    }
    valueReader_->reset();
  }

 private:
  // Merge the mapNulls and inMapData into mergedNulls
  uint32_t mergeNulls(
      uint32_t numValues,
      uint32_t nonNullMaps,
      const Vector<bool>& mapNulls,
      Vector<char>& mergedNulls) {
    const auto numItems = readInMapData(nonNullMaps);
    const auto requiredBytes = velox::bits::nbytes(numValues);
    mergedNulls.resize(requiredBytes);
    ::memset(mergedNulls.data(), 0, requiredBytes);
    if (numItems == 0) {
      return 0;
    }
    if (nonNullMaps == numValues) {
      // All values are nonNull
      velox::bits::packBitmap(inMapData_, mergedNulls.data());
      return numItems;
    }
    uint32_t inMapOffset{0};
    for (uint32_t i = 0; i < numValues; ++i) {
      if (mapNulls[i] && inMapData_[inMapOffset++]) {
        velox::bits::setBit(reinterpret_cast<uint8_t*>(mergedNulls.data()), i);
      }
    }
    return numItems;
  }

  const velox::dwio::common::flatmap::KeyValue<T>& key_;
  const std::unique_ptr<FieldReader> valueReader_;
  Decoder* const inMapDecoder_;
  Vector<bool> inMapData_;
  uint32_t numValues_;
  // nulls buffer used in parallel read cases.
  Vector<char> mergedNulls_;
};

template <typename T, bool hasNull>
class FlatMapFieldReaderBase : public FieldReader {
 public:
  FlatMapFieldReaderBase(
      velox::memory::MemoryPool& pool,
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::unique_ptr<FlatMapKeyNode<T>>> keyNodes,
      Vector<bool>& boolBuffer,
      const FieldReader::Options& options = {})
      : FieldReader{pool, std::move(type), decoder, options},
        keyNodes_{std::move(keyNodes)},
        boolBuffer_{boolBuffer} {}

  uint32_t loadNulls(uint32_t rowCount, velox::BaseVector* vector) {
    if constexpr (hasNull) {
      zeroNulls(vector, rowCount);
      auto* nullBuffer = ensureNulls(vector, rowCount);
      velox::bits::BitmapBuilder bitmap{nullBuffer, rowCount};

      boolBuffer_.resize(rowCount);
      auto nonNullCount = readBooleanValues(
          decoder_, boolBuffer_.data(), rowCount, [&](auto i) {
            bitmap.set(i);
          });

      if (UNLIKELY(nonNullCount == rowCount)) {
        vector->resetNulls();
      } else {
        vector->setNullCount(rowCount - nonNullCount);
      }
      return nonNullCount;
    } else {
      vector->resetNulls();
      return rowCount;
    }
  }

  void skip(uint32_t count) final {
    uint32_t nonNullCount = count;

    if constexpr (hasNull) {
      std::array<bool, kSkipBatchSize> buffer;
      nonNullCount = 0;
      while (count > 0) {
        auto readSize = std::min(count, kSkipBatchSize);
        nonNullCount += readBooleanValues(decoder_, buffer.data(), readSize);
        count -= readSize;
      }
    }

    if (nonNullCount > 0) {
      for (auto& node : keyNodes_) {
        if (node) {
          node->skip(nonNullCount);
        }
      }
    }
  }

  void reset() final {
    FieldReader::reset();
    for (auto& node : keyNodes_) {
      if (node) {
        node->reset();
      }
    }
  }

 protected:
  std::vector<std::unique_ptr<FlatMapKeyNode<T>>> keyNodes_;
  Vector<bool>& boolBuffer_;
};

// The decoders map may contain entries with null unique_ptr for streams that
// don't have data in the current stripe, so we check the pointer itself.
inline bool hasDecoderStream(
    offset_size offset,
    const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders) {
  auto it = decoders.find(offset);
  return it != decoders.end() && it->second != nullptr;
}

template <typename T>
class FlatMapFieldReaderFactoryBase : public FieldReaderFactory {
 public:
  FlatMapFieldReaderFactoryBase(
      velox::TypePtr veloxType,
      const Type* type,
      std::vector<const StreamDescriptor*> inMapDescriptors,
      std::vector<std::unique_ptr<FieldReaderFactory>> valueReaders,
      const std::vector<size_t>& selectedChildren,
      velox::memory::MemoryPool* pool)
      : FieldReaderFactory{std::move(veloxType), type, pool},
        inMapDescriptors_{std::move(inMapDescriptors)},
        valueReaders_{std::move(valueReaders)},
        boolBuffer_{pool_} {
    // inMapTypes contains all projected children, including those that don't
    // exist in the schema. selectedChildren and valuesReaders only contain
    // those that also exist in the schema.
    NIMBLE_CHECK_GE(
        inMapDescriptors_.size(),
        valueReaders_.size(),
        "Value and inMaps size mismatch!");
    NIMBLE_CHECK_EQ(
        selectedChildren.size(),
        valueReaders_.size(),
        "Selected children and value readers size mismatch!");

    const auto& flatMap = type->asFlatMap();
    keyValues_.reserve(selectedChildren.size());
    valueTypes_.reserve(selectedChildren.size());
    for (auto childIdx : selectedChildren) {
      keyValues_.emplace_back(
          velox::dwio::common::flatmap::parseKeyValue<T>(
              flatMap.nameAt(childIdx)));
      valueTypes_.emplace_back(flatMap.childAt(childIdx).get());
    }
  }

  template <
      template <bool> typename ReaderT,
      bool includeMissing,
      typename... Args>
  std::unique_ptr<FieldReader> createFlatMapReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders,
      Args&&... args) {
    auto nulls =
        getDecoder(decoders, nimbleType_->asFlatMap().nullsDescriptor());

    std::vector<std::unique_ptr<FlatMapKeyNode<T>>> keyNodes;
    keyNodes.reserve(valueReaders_.size());
    uint32_t childIdx{0};
    for (auto inMapDescriptor : inMapDescriptors_) {
      if (inMapDescriptor != nullptr) {
        const auto currentIdx = childIdx++;
        auto* decoder = getDecoder(decoders, *inMapDescriptor);
        NIMBLE_CHECK_LT(
            currentIdx,
            valueTypes_.size(),
            "currentIdx out of range for valueTypes_");
        if (decoder != nullptr ||
            visitValueStreamLeaves(
                *valueTypes_[currentIdx], [&decoders](offset_size offset) {
                  return hasDecoderStream(offset, decoders);
                })) {
          keyNodes.emplace_back(
              std::make_unique<FlatMapKeyNode<T>>(
                  *pool_,
                  // @lint-ignore CLANGTIDY
                  // facebook-hte-MemberUncheckedArrayBounds
                  valueReaders_[currentIdx]->createReader(decoders),
                  decoder,
                  // @lint-ignore CLANGTIDY
                  // facebook-hte-MemberUncheckedArrayBounds
                  keyValues_[currentIdx]));
          continue;
        }
      }

      if constexpr (includeMissing) {
        keyNodes.emplace_back(nullptr);
      }
    }

    if (nulls == nullptr) {
      return std::make_unique<ReaderT<false>>(
          this->veloxType_,
          nulls,
          std::move(keyNodes),
          boolBuffer_,
          pool_,
          std::forward<Args>(args)...);
    }

    return std::make_unique<ReaderT<true>>(
        this->veloxType_,
        nulls,
        std::move(keyNodes),
        boolBuffer_,
        pool_,
        std::forward<Args>(args)...);
  }

 protected:
  std::vector<const StreamDescriptor*> inMapDescriptors_;
  std::vector<std::unique_ptr<FieldReaderFactory>> valueReaders_;
  std::vector<velox::dwio::common::flatmap::KeyValue<T>> keyValues_;
  // Value types for each selected child, used to check if value streams
  // exist when the in-map decoder is missing.
  std::vector<const Type*> valueTypes_;
  Vector<bool> boolBuffer_;
};

template <typename T, bool hasNull>
class StructFlatMapFieldReader : public FlatMapFieldReaderBase<T, hasNull> {
 public:
  StructFlatMapFieldReader(
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::unique_ptr<FlatMapKeyNode<T>>> keyNodes,
      Vector<bool>& boolBuffer,
      velox::memory::MemoryPool* pool,
      Vector<char>& mergedNulls,
      const FieldReader::Options& options)
      : FlatMapFieldReaderBase<T, hasNull>(
            *pool,
            std::move(type),
            decoder,
            std::move(keyNodes),
            boolBuffer,
            options),
        mergedNulls_{mergedNulls} {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    uint64_t rowCount{0};
    if constexpr (hasNull) {
      NIMBLE_CHECK_NOT_NULL(
          FieldReader::decoder_, "decoder_ should be set when hasNull is true");
      const auto* encoding = FieldReader::decoder_->encoding();
      NIMBLE_CHECK_NOT_NULL(
          encoding, "Decoder must be loaded for output size estimation.");
      rowCount = encoding->rowCount();
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
    }

    for (const auto& node : this->keyNodes_) {
      if (node == nullptr) {
        // This could happen when selected feature does not exist.
        continue;
      }
      const auto keyNodeSizeOpt = node->valueReader()->estimatedRowSize();
      if (!keyNodeSizeOpt.has_value()) {
        return std::nullopt;
      }
      const auto nonNullCount = keyNodeSizeOpt.value().first;
      const auto keyNodeBytesPerRow = keyNodeSizeOpt.value().second;
      totalBytes += keyNodeBytesPerRow * nonNullCount;
      // Adding memory for additional null overhead in outer layer
      if constexpr (hasNull) {
        NIMBLE_CHECK_GE(
            rowCount, nonNullCount, "rowCount should be >= nonNullCount");
        totalBytes += (rowCount - nonNullCount) *
            nullOverheadBits(node->valueReader()->type()) / 8;
      } else if (rowCount == 0) {
        rowCount = nonNullCount;
      } else if (nonNullCount != 0) {
        NIMBLE_CHECK_EQ(
            rowCount,
            nonNullCount,
            "rowCount should be equal to nonNullCount under no null condition");
      }
    }
    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t rowCount,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    NIMBLE_CHECK_NULL(scatterBitmap, "unexpected scatterBitmap");
    if (this->parallelDecodeEnabled(this->keyNodes_.size())) {
      folly::coro::blockingWait(co_next(rowCount, output, scatterBitmap));
      return;
    }
    const auto outputContext = prepareOutput(rowCount, output);
    for (uint32_t i = 0; i < this->keyNodes_.size(); ++i) {
      if (this->keyNodes_[i] == nullptr) {
        this->ensureNullConstant(
            this->type_->childAt(i),
            rowCount,
            outputContext.vector->childAt(i));
      } else {
        this->keyNodes_[i]->readAsChild(
            outputContext.vector->childAt(i),
            rowCount,
            outputContext.nonNullCount,
            this->boolBuffer_,
            &mergedNulls_);
      }
    }
  }

  folly::coro::Task<void> co_next(
      uint32_t rowCount,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    NIMBLE_CHECK_NULL(scatterBitmap, "unexpected scatterBitmap");
    const auto outputContext = prepareOutput(rowCount, output);

    // Handle null constants inline (cheap) before dispatching parallel tasks.
    // Collect indices of non-null key nodes.
    std::vector<uint32_t> nonNullChildren;
    nonNullChildren.reserve(this->keyNodes_.size());
    for (uint32_t i = 0; i < this->keyNodes_.size(); ++i) {
      if (this->keyNodes_[i] == nullptr) {
        this->ensureNullConstant(
            this->type_->childAt(i),
            rowCount,
            outputContext.vector->childAt(i));
      } else {
        nonNullChildren.emplace_back(i);
      }
    }

    if (nonNullChildren.empty()) {
      co_return;
    }

    const uint32_t taskCount =
        this->computeParallelDecodeTaskCount(nonNullChildren.size());
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::StructFlatMapFieldReader::co_next",
        const_cast<uint32_t*>(&taskCount));

    const uint32_t childrenPerTask = nonNullChildren.size() / taskCount;
    const uint32_t numRemainderChildren = nonNullChildren.size() % taskCount;

    // Decodes children in [startIdx, endIdx).
    auto decodeRange = [this, rowCount, &outputContext, &nonNullChildren](
                           uint32_t startIdx, uint32_t endIdx) {
      for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
        const auto i = nonNullChildren[idx];
        this->keyNodes_[i]->readAsChild(
            outputContext.vector->childAt(i),
            rowCount,
            outputContext.nonNullCount,
            this->boolBuffer_);
      }
    };

    // First 'numRemainderChildren' tasks get one extra child each.
    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    tasks.reserve(taskCount);
    uint32_t nextChildIdx = 0;
    for (uint32_t task = 0; task < taskCount; ++task) {
      const uint32_t endChildIdx = nextChildIdx + childrenPerTask +
          (task < numRemainderChildren ? 1 : 0);
      tasks.emplace_back(
          folly::coro::co_withExecutor(
              this->decodeExecutor_,
              folly::coro::co_invoke(
                  [&decodeRange, nextChildIdx, endChildIdx]()
                      -> folly::coro::Task<void> {
                    decodeRange(nextChildIdx, endChildIdx);
                    co_return;
                  })));
      nextChildIdx = endChildIdx;
    }

    co_await folly::coro::collectAllRange(std::move(tasks));
  }

 private:
  struct OutputContext {
    // Initialized output row vector.
    velox::RowVector* vector;
    // Number of non-null rows.
    uint32_t nonNullCount;
  };

  OutputContext prepareOutput(uint32_t rowCount, velox::VectorPtr& output) {
    auto* vector = VectorInitializer<velox::RowVector>::initialize(
        this->type_, rowCount, this->pool_, output);
    vector->unsafeResize(rowCount);
    const uint32_t nonNullCount = this->loadNulls(rowCount, vector);
    return {vector, nonNullCount};
  }

  Vector<char>& mergedNulls_;
};

template <typename T>
class StructFlatMapFieldReaderFactory final
    : public FlatMapFieldReaderFactoryBase<T> {
  template <bool hasNull>
  using ReaderType = StructFlatMapFieldReader<T, hasNull>;

 public:
  StructFlatMapFieldReaderFactory(
      velox::TypePtr veloxType,
      const Type* type,
      std::vector<const StreamDescriptor*> inMapDescriptors,
      std::vector<std::unique_ptr<FieldReaderFactory>> valueReaders,
      const std::vector<size_t>& selectedChildren,
      velox::memory::MemoryPool* pool,
      const FieldReaderParams& params)
      : FlatMapFieldReaderFactoryBase<T>(
            std::move(veloxType),
            type,
            std::move(inMapDescriptors),
            std::move(valueReaders),
            selectedChildren,
            pool),
        decodeExecutor_{params.decodeExecutor},
        maxDecodeParallelism_{params.maxDecodeParallelism},
        minStreamsPerDecodeUnit_{params.minStreamsPerDecodeUnit},
        mergedNulls_{this->pool_} {
    NIMBLE_CHECK(this->nimbleType_->isFlatMap(), "Type should be a flat map.");
  }

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    const FieldReader::Options options{
        decodeExecutor_, maxDecodeParallelism_, minStreamsPerDecodeUnit_};
    return this->template createFlatMapReader<ReaderType, true>(
        decoders, mergedNulls_, options);
  }

 private:
  folly::Executor* const decodeExecutor_;
  const uint32_t maxDecodeParallelism_;
  const uint32_t minStreamsPerDecodeUnit_;
  Vector<char> mergedNulls_;
};

// Reads a flat map and produces a Velox MapVector. Each flat map key becomes a
// FlatMapKeyNode with its own in-map boolean stream and value stream. The
// reader transposes from column-wise (per-key) layout to row-wise map entries:
//   1. Reads in-map bitmaps for all keys and builds a row-wise in-map mask.
//   2. Computes per-row offsets and lengths from the mask.
//   3. Iterates key-by-key, copying value data into the merged MapVector.
//
// Contrast with StructFlatMapFieldReader, which reads a flat map as a Velox
// RowVector (one child per key) without transposing to map layout.
template <typename T, bool hasNull>
class MergedFlatMapFieldReader final
    : public FlatMapFieldReaderBase<T, hasNull> {
 public:
  MergedFlatMapFieldReader(
      velox::TypePtr type,
      Decoder* decoder,
      std::vector<std::unique_ptr<FlatMapKeyNode<T>>> keyNodes,
      Vector<bool>& boolBuffer,
      velox::memory::MemoryPool* pool)
      : FlatMapFieldReaderBase<T, hasNull>(
            *pool,
            std::move(type),
            decoder,
            std::move(keyNodes),
            boolBuffer) {}

  std::optional<std::pair<uint32_t, uint64_t>> estimatedRowSize() const final {
    uint64_t totalBytes{0};
    uint32_t rowCount{0};
    if constexpr (hasNull) {
      NIMBLE_CHECK_NOT_NULL(
          FieldReader::decoder_, "decoder_ should be set when hasNull is true");
      const auto* encoding = FieldReader::decoder_->encoding();
      NIMBLE_CHECK_NOT_NULL(
          encoding, "Decoder must be loaded for output size estimation.");
      rowCount = encoding->rowCount();
      // Adding memory for velox::BaseVector::nulls_
      totalBytes += rowCount / 8;
    } else {
      if (this->keyNodes_.empty()) {
        // This happens when selected feature does not exist in the flatmap.
        // As we cannot acquire row count in this case, nullopt will be
        // returned to indicate unsupported.
        return std::nullopt;
      }
      // Find the row count from an in-map decoder, or fall back to a value
      // reader's estimate. The in-map decoder may be null when the writer
      // omitted the all-true in-map stream.
      bool found = false;
      for (const auto& keyNode : this->keyNodes_) {
        NIMBLE_CHECK_NOT_NULL(
            keyNode,
            "MergedFlatMapFieldReader is created with includeMissing=false");
        if (keyNode->inMapDecoder() != nullptr) {
          rowCount = keyNode->inMapDecoder()->encoding()->rowCount();
          found = true;
          break;
        }
        if (const auto estimatedSize =
                keyNode->valueReader()->estimatedRowSize()) {
          rowCount = estimatedSize->first;
          found = true;
          break;
        }
      }
      if (!found) {
        return std::nullopt;
      }
    }

    // Adding memory for velox::ArrayVectorBase::offsets_ and
    // velox::ArrayVectorBase::sizes_
    totalBytes += rowCount * sizeof(int32_t) * 2;

    // Estimation of map key vector size in velox::MapVector.
    // Adding memory for key vector's BaseVector::nulls_
    totalBytes += rowCount * this->keyNodes_.size() / 8;
    // MergedFlatMap key field is either velox::StringView or primitive type
    uint64_t totalKeyBytesPerRow{0};
    if constexpr (std::is_same<T, velox::StringView>::value) {
      for (const auto& node : this->keyNodes_) {
        const auto keyBytes = node->key().get().size();
        // Adding memory for key vector's velox::FlatVector::stringBuffers_
        totalKeyBytesPerRow += keyBytes;
        // Adding overheads for StringView in velox::FlatVector::values_
        totalKeyBytesPerRow +=
            keyBytes > velox::StringView::kInlineSize ? 16 : 4;
      }
    } else {
      // Adding memory for key vector's velox::FlatVector::values_
      totalKeyBytesPerRow += this->keyNodes_.size() * sizeof(T);
    }
    // Null row count in this map cannot be easily obtained, we over-estimate
    // by multiplying total row count.
    totalBytes += rowCount * totalKeyBytesPerRow;

    // Estimation of map value vector size in velox::MapVector. As
    // MergedFlatMapReader transforms the dimension of the keys and values
    // from a flat map representation to velox map representation, there is no
    // easy way of doing a direct estimation of velox values column. So we
    // adopt the ways from StructFlatMapReader for estimating values size.
    for (const auto& node : this->keyNodes_) {
      auto valueSize = node->valueReader()->estimatedRowSize();
      if (!valueSize.has_value()) {
        return std::nullopt;
      }
      const auto nonNullCount = valueSize.value().first;
      const auto valueBytesPerRow = valueSize.value().second;
      totalBytes += nonNullCount * valueBytesPerRow;
      if constexpr (hasNull) {
        NIMBLE_CHECK_GE(
            rowCount, nonNullCount, "rowCount should be >= than nonNullCount");
        // Adding null overhead on outer layer
        totalBytes += (rowCount - nonNullCount) *
            nullOverheadBits(node->valueReader()->type()) / 8;
      }
    }
    return rowCount == 0 ? std::optional<std::pair<uint32_t, uint64_t>>({0, 0})
                         : std::optional<std::pair<uint32_t, uint64_t>>(
                               {rowCount, totalBytes / rowCount});
  }

  void next(
      uint32_t rowCount,
      velox::VectorPtr& output,
      const velox::bits::Bitmap* scatterBitmap) final {
    NIMBLE_CHECK_NULL(scatterBitmap, "unexpected scatterBitmap");
    auto* vector = VectorInitializer<velox::MapVector>::initialize(
        this->type_, rowCount, this->pool_, output);
    vector->resize(rowCount);
    velox::VectorPtr& keysVector = vector->mapKeys();
    // Check the refCount for key vector
    auto flatKeysVector = VectorInitializer<velox::FlatVector<T>>::initialize(
        std::static_pointer_cast<const velox::MapType>(this->type_)->keyType(),
        rowCount,
        this->pool_,
        keysVector);

    NIMBLE_DCHECK_EQ(
        vector->sizes()->size(),
        (rowCount * sizeof(velox::vector_size_t)),
        "Unexpected 'sizes' buffer size.");
    NIMBLE_DCHECK_EQ(
        vector->offsets()->size(),
        (rowCount * sizeof(velox::vector_size_t)),
        "Unexpected 'offsets' buffer size.");

    const velox::BufferPtr& lengths = vector->sizes();
    const velox::BufferPtr& offsets = vector->offsets();
    const uint32_t nonNullCount = this->loadNulls(rowCount, vector);
    nodes_.clear();
    size_t totalMapEntries{0};
    for (auto& node : this->keyNodes_) {
      const auto numValues = node->readInMapData(nonNullCount);
      if (numValues > 0) {
        nodes_.emplace_back(node.get());
        totalMapEntries += numValues;
      }
    }

    velox::VectorPtr& valuesVector = vector->mapValues();
    // The key vector holds one entry per map entry, but VectorInitializer above
    // sized it from rowCount, which shrinks its values buffer to
    // rowCount * sizeof(T). The resize below is what puts that back -- except
    // FlatVector::resize returns early when handed the size the vector already
    // has, which happens whenever two consecutive batches carry the same entry
    // count. The vector is then left claiming totalMapEntries entries over a
    // rowCount-sized buffer. Nothing notices on a plain read, because only
    // wrapping the vector in a dictionary -- what a filtered read does when it
    // drops rows -- ever validates it, and only in a debug build.
    //
    // Dropping to zero first makes the growth path always run. Capacity
    // survives the shrink, so this costs no allocation.
    //
    // The zero-entry case is deliberately left alone: a batch with no entries
    // keeps the previous batch's key length, which is the same inconsistency in
    // miniature. Clearing it here regresses flat map feature projection
    // (koski's NimbleTableTest.FeatureProjection*), so it needs its own
    // investigation rather than a drive-by fix.
    if (totalMapEntries > 0) {
      keysVector->resize(0, false);
      keysVector->resize(totalMapEntries, false);
      velox::BaseVector::prepareForReuse(valuesVector, totalMapEntries);
    }

    // Pre-load node values to enable direct element copy for ArrayVector
    // values, bypassing copyRangesImpl's incremental resize.
    nodeValues_.resize(nodes_.size());
    for (size_t j = 0; j < nodes_.size(); ++j) {
      nodes_[j]->loadValues(nodeValues_[j]);
    }

    auto* offsetsPtr = offsets->asMutable<velox::vector_size_t>();
    auto* lengthsPtr = lengths->asMutable<velox::vector_size_t>();
    initRowWiseInMap(rowCount);
    initOffsets(rowCount, offsetsPtr, lengthsPtr);

    // Always access inMap and value streams node-wise to avoid large striding
    // through the memory and destroying CPU cache performance.
    //
    // Index symbology:
    // i : Node index
    // j : Row index
    // For ArrayVector values (e.g., Map<K, Array<V>>), bypass copyRangesImpl
    // and directly copy inner elements into a pre-sized buffer to avoid
    // incremental realloc+memcpy. For other value types (FlatVector, etc.),
    // fall through to the copyRanges path which has no resize overhead.
    auto* arrayValues =
        totalMapEntries > 0 ? valuesVector->as<velox::ArrayVector>() : nullptr;

    if (arrayValues != nullptr) {
      copyArrayValues(
          arrayValues, flatKeysVector, rowCount, totalMapEntries, offsetsPtr);
    } else {
      // Non-array values (FlatVector, etc): use copyRanges.
      for (size_t i = 0; i < nodes_.size(); ++i) {
        copyRanges_.clear();
        for (velox::vector_size_t j = 0; j < rowCount; ++j) {
          if (!velox::bits::isBitSet(
                  rowWiseInMap_.data(), i + j * nodes_.size())) {
            continue;
          }
          const velox::vector_size_t sourceIndex = copyRanges_.size();
          copyRanges_.push_back({sourceIndex, offsetsPtr[j], 1});
          flatKeysVector->set(offsetsPtr[j], nodes_[i]->key().get());
          ++offsetsPtr[j];
        }
        valuesVector->copyRanges(nodeValues_[i].get(), copyRanges_);
      }
    }
    if (rowCount > 0) {
      NIMBLE_CHECK_EQ(
          offsetsPtr[rowCount - 1],
          totalMapEntries,
          "Total map entry size mismatch");
      // We updated `offsetsPtr' during the copy process, so that now it was
      // shifted to the left by 1 element (i.e. offsetsPtr[i] is really
      // offsetsPtr[i+1]).  Need to restore the values back to their correct
      // positions.
      std::copy_backward(
          offsetsPtr, offsetsPtr + rowCount - 1, offsetsPtr + rowCount);
      offsetsPtr[0] = 0;
    }

    // Reset the updated value vector to result
    vector->setKeysAndValues(std::move(keysVector), std::move(valuesVector));
  }

 private:
  // Pre-allocates inner elements to the exact final size and directly copies,
  // bypassing copyRangesImpl's incremental resize which causes O(N) reallocs.
  void copyArrayValues(
      velox::ArrayVector* arrayValues,
      velox::FlatVector<T>* flatKeysVector,
      velox::vector_size_t rowCount,
      size_t totalMapEntries,
      velox::vector_size_t* offsetsPtr) {
    velox::vector_size_t totalElements = 0;
    for (auto& nodeValue : nodeValues_) {
      auto* sourceArray =
          nodeValue->wrappedVector()->asUnchecked<velox::ArrayVector>();
      for (velox::vector_size_t i = 0; i < nodeValue->size(); ++i) {
        if (!nodeValue->isNullAt(i)) {
          totalElements += sourceArray->sizeAt(nodeValue->wrappedIndex(i));
        }
      }
    }

    auto& elements = arrayValues->elements();
    elements->resize(totalElements);
    // For arrays-of-arrays values, the per-node copyRanges() below grows the
    // innermost element buffer incrementally, causing quadratic
    // reallocate+memcpy. Pre-size it to an upper bound then shrink to 0, which
    // retains the capacity so each copy stays within it.
    if (auto* nestedArray = elements->as<velox::ArrayVector>()) {
      velox::vector_size_t nestedTotal = 0;
      for (auto& nodeValue : nodeValues_) {
        auto* sourceArray =
            nodeValue->wrappedVector()->asUnchecked<velox::ArrayVector>();
        // This is only an upper-bound estimate for the common unencoded case;
        // encoded inner elements are skipped and simply fall back to the
        // incremental growth path.
        if (auto* sourceNested =
                sourceArray->elements()->as<velox::ArrayVector>()) {
          nestedTotal += sourceNested->elements()->size();
        }
      }
      if (nestedTotal > 0) {
        auto& nestedElements = nestedArray->elements();
        nestedElements->resize(nestedTotal);
        nestedElements->resize(0);
      }
    }
    auto* valuesOffsets = arrayValues->mutableOffsets(totalMapEntries)
                              ->asMutable<velox::vector_size_t>();
    auto* valuesSizes = arrayValues->mutableSizes(totalMapEntries)
                            ->asMutable<velox::vector_size_t>();

    velox::vector_size_t elementOffset = 0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto* sourceArray =
          nodeValues_[i]->wrappedVector()->asUnchecked<velox::ArrayVector>();
      velox::vector_size_t sourceIndex = 0;
      copyRanges_.clear();

      auto copyValue = [&](velox::vector_size_t j) {
        const auto targetIndex = offsetsPtr[j];
        flatKeysVector->set(targetIndex, nodes_[i]->key().get());
        arrayValues->setNull(targetIndex, false);
        const auto wrappedIndex = nodeValues_[i]->wrappedIndex(sourceIndex);
        const auto copySize = sourceArray->sizeAt(wrappedIndex);
        valuesOffsets[targetIndex] = elementOffset;
        valuesSizes[targetIndex] = copySize;
        if (copySize > 0) {
          copyRanges_.push_back(
              {sourceArray->offsetAt(wrappedIndex), elementOffset, copySize});
          elementOffset += copySize;
        }
        ++sourceIndex;
        ++offsetsPtr[j];
      };

      auto copyNull = [&](velox::vector_size_t j) {
        const auto targetIndex = offsetsPtr[j];
        flatKeysVector->set(targetIndex, nodes_[i]->key().get());
        arrayValues->setNull(targetIndex, true);
        valuesOffsets[targetIndex] = elementOffset;
        valuesSizes[targetIndex] = 0;
        ++sourceIndex;
        ++offsetsPtr[j];
      };

      if (nodeValues_[i]->mayHaveNulls()) {
        for (velox::vector_size_t j = 0; j < rowCount; ++j) {
          if (!velox::bits::isBitSet(
                  rowWiseInMap_.data(), i + j * nodes_.size())) {
            continue;
          }
          if (nodeValues_[i]->isNullAt(sourceIndex)) {
            copyNull(j);
          } else {
            copyValue(j);
          }
        }
      } else {
        for (velox::vector_size_t j = 0; j < rowCount; ++j) {
          if (!velox::bits::isBitSet(
                  rowWiseInMap_.data(), i + j * nodes_.size())) {
            continue;
          }
          copyValue(j);
        }
      }
      elements->copyRanges(sourceArray->elements().get(), copyRanges_);
    }
    NIMBLE_CHECK_EQ(elementOffset, totalElements, "Element count mismatch");
  }

  void initRowWiseInMap(velox::vector_size_t rowCount) {
    rowWiseInMap_.resize(velox::bits::nwords(nodes_.size() * rowCount));
    std::fill(rowWiseInMap_.begin(), rowWiseInMap_.end(), 0);
    for (size_t j = 0; j < nodes_.size(); ++j) {
      uint32_t inMapIndex = 0;
      for (velox::vector_size_t i = 0; i < rowCount; ++i) {
        const bool isNull = hasNull && !this->boolBuffer_[i];
        if (!isNull && nodes_[j]->inMap(inMapIndex)) {
          velox::bits::setBit(rowWiseInMap_.data(), j + i * nodes_.size());
        }
        inMapIndex += !isNull;
      }
    }
  }

  void initOffsets(
      velox::vector_size_t rowCount,
      velox::vector_size_t* offsets,
      velox::vector_size_t* lengths) {
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < rowCount; ++i) {
      offsets[i] = offset;
      lengths[i] = velox::bits::countBits(
          rowWiseInMap_.data(), i * nodes_.size(), (i + 1) * nodes_.size());
      offset += lengths[i];
    }
  }

  // All the nodes that is selected to be read.
  std::vector<FlatMapKeyNode<T>*> nodes_;

  // Pre-loaded node values, used to compute total inner elements for
  // capacity pre-allocation before copying.
  std::vector<velox::VectorPtr> nodeValues_;

  // In-map mask (1 bit per value), organized in row first layout.
  std::vector<uint64_t> rowWiseInMap_;

  // Copy ranges from one node values into the merged values. Memory buffer
  // purpose only, no values stored between calls.
  std::vector<velox::BaseVector::CopyRange> copyRanges_;
};

template <typename T>
class MergedFlatMapFieldReaderFactory final
    : public FlatMapFieldReaderFactoryBase<T> {
  template <bool hasNull>
  using ReaderType = MergedFlatMapFieldReader<T, hasNull>;

 public:
  using FlatMapFieldReaderFactoryBase<T>::FlatMapFieldReaderFactoryBase;

  std::unique_ptr<FieldReader> createReader(
      const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders)
      final {
    return this->template createFlatMapReader<ReaderType, false>(decoders);
  }
};

std::unique_ptr<FieldReaderFactory> createFlatMapReaderFactory(
    velox::memory::MemoryPool* pool,
    velox::TypeKind keyKind,
    velox::TypePtr veloxType,
    const Type* type,
    std::vector<const StreamDescriptor*> inMapDescriptors,
    std::vector<std::unique_ptr<FieldReaderFactory>> valueReaders,
    const std::vector<size_t>& selectedChildren,
    bool flatMapAsStruct,
    const FieldReaderParams& params) {
  switch (keyKind) {
#define SCALAR_CASE(veloxKind, fieldType)                                  \
  case velox::TypeKind::veloxKind: {                                       \
    if (flatMapAsStruct) {                                                 \
      return std::make_unique<StructFlatMapFieldReaderFactory<fieldType>>( \
          std::move(veloxType),                                            \
          type,                                                            \
          std::move(inMapDescriptors),                                     \
          std::move(valueReaders),                                         \
          selectedChildren,                                                \
          pool,                                                            \
          params);                                                         \
    } else {                                                               \
      return std::make_unique<MergedFlatMapFieldReaderFactory<fieldType>>( \
          std::move(veloxType),                                            \
          type,                                                            \
          std::move(inMapDescriptors),                                     \
          std::move(valueReaders),                                         \
          selectedChildren,                                                \
          pool);                                                           \
    }                                                                      \
  }

    SCALAR_CASE(TINYINT, int8_t);
    SCALAR_CASE(SMALLINT, int16_t);
    SCALAR_CASE(INTEGER, int32_t);
    SCALAR_CASE(BIGINT, int64_t);
    SCALAR_CASE(VARCHAR, velox::StringView);
    SCALAR_CASE(VARBINARY, velox::StringView);
#undef SCALAR_CASE

    default:
      NIMBLE_UNSUPPORTED("Not supported flatmap key type: {} ", keyKind);
  }
}

std::shared_ptr<const velox::Type> createFlatType(
    const std::vector<std::string>& selectedFeatures,
    const velox::TypePtr& veloxType) {
  NIMBLE_CHECK(
      !selectedFeatures.empty(),
      "Empty feature selection not allowed for struct encoding.");

  auto& valueType = veloxType->asMap().valueType();
  return velox::ROW(
      std::vector<std::string>(selectedFeatures),
      std::vector<std::shared_ptr<const velox::Type>>(
          selectedFeatures.size(), valueType));
}

velox::TypePtr inferType(
    const FieldReaderParams& params,
    const std::string& name,
    const velox::TypePtr& type,
    size_t level) {
  // Special case for flatmaps. If the flatmap field is missing, still need to
  // honor the "as struct" intent by returning row instead of map.
  if (level == 1 && params.readFlatMapFieldAsStruct.contains(name)) {
    NIMBLE_CHECK(
        type->kind() == velox::TypeKind::MAP,
        "Unexpected type kind of flat maps.");
    auto it = params.flatMapFeatureSelector.find(name);
    NIMBLE_CHECK(
        it != params.flatMapFeatureSelector.end() &&
            !it->second.features.empty(),
        "Flat map feature selection for map '{}' has empty feature set.",
        name);
    NIMBLE_CHECK(
        it->second.mode == SelectionMode::Include,
        "Flat map exclusion list is not supported when flat map field is missing.");

    return createFlatType(it->second.features, type);
  }
  return type;
}

// TODO: use field reader params or another flag to control creating legacy
// string field reader.
std::unique_ptr<FieldReaderFactory> createFieldReaderFactory(
    const FieldReaderParams& parameters,
    const std::shared_ptr<const Type>& nimbleType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& veloxType,
    std::vector<uint32_t>& offsets,
    const std::function<bool(uint32_t)>& isSelected,
    size_t level,
    const std::string* name,
    velox::memory::MemoryPool* pool) {
  const auto veloxKind = veloxType->type()->kind();
  // compatibleKinds are the types that can be upcasted to nimbleType
  auto checkType = [&nimbleType](
                       const std::vector<ScalarKind>& compatibleKinds) {
    return std::any_of(
        compatibleKinds.begin(),
        compatibleKinds.end(),
        [&nimbleType](ScalarKind kind) {
          return nimbleType->asScalar().scalarDescriptor().scalarKind() == kind;
        });
  };

// Assuming no-upcasting is the most common case, putting the largest type size
// at the beginning so the compatibility check can finish quicker.
#define BOOLEAN_COMPATIBLE {ScalarKind::Bool}
#define TINYINT_COMPATIBLE {ScalarKind::Int8, ScalarKind::Bool}
#define SMALLINT_COMPATIBLE \
  {ScalarKind::Int16, ScalarKind::Int8, ScalarKind::Bool}
#define INTEGER_COMPATIBLE \
  {ScalarKind::Int32, ScalarKind::Int16, ScalarKind::Int8, ScalarKind::Bool}
#define BIGINT_COMPATIBLE \
  {ScalarKind::Int64,     \
   ScalarKind::Int32,     \
   ScalarKind::Int16,     \
   ScalarKind::Int8,      \
   ScalarKind::Bool}
#define FLOAT_COMPATIBLE {ScalarKind::Float}
#define DOUBLE_COMPATIBLE                 \
  {                                       \
    ScalarKind::Double, ScalarKind::Float \
  }

  switch (veloxKind) {
#define SCALAR_CASE(veloxKind, cppType, compatibleKinds)                      \
  case velox::TypeKind::veloxKind: {                                          \
    NIMBLE_CHECK(                                                             \
        nimbleType->isScalar() && checkType(compatibleKinds),                 \
        "Provided schema doesn't match file schema.");                        \
    offsets.emplace_back(nimbleType->asScalar().scalarDescriptor().offset()); \
    return std::make_unique<ScalarFieldReaderFactory<cppType>>(               \
        veloxType->type(), nimbleType.get(), pool);                           \
  }

    SCALAR_CASE(BOOLEAN, bool, BOOLEAN_COMPATIBLE);
    SCALAR_CASE(TINYINT, int8_t, TINYINT_COMPATIBLE);
    SCALAR_CASE(SMALLINT, int16_t, SMALLINT_COMPATIBLE);
    SCALAR_CASE(INTEGER, int32_t, INTEGER_COMPATIBLE);
    SCALAR_CASE(BIGINT, int64_t, BIGINT_COMPATIBLE);
    SCALAR_CASE(REAL, float, FLOAT_COMPATIBLE);
    SCALAR_CASE(DOUBLE, double, DOUBLE_COMPATIBLE);
#undef SCALAR_CASE

    case velox::TypeKind::VARCHAR:
    case velox::TypeKind::VARBINARY: {
      NIMBLE_CHECK(
          nimbleType->isScalar() &&
              ((veloxKind == velox::TypeKind::VARCHAR &&
                nimbleType->asScalar().scalarDescriptor().scalarKind() ==
                    ScalarKind::String) ||
               (veloxKind == velox::TypeKind::VARBINARY &&
                nimbleType->asScalar().scalarDescriptor().scalarKind() ==
                    ScalarKind::Binary)),
          "Provided schema doesn't match file schema.");
      offsets.emplace_back(nimbleType->asScalar().scalarDescriptor().offset());
      return std::make_unique<StringFieldReaderFactory>(
          veloxType->type(),
          nimbleType.get(),
          parameters.optimizeStringBufferHandling,
          pool);
    }
    case velox::TypeKind::TIMESTAMP: {
      NIMBLE_CHECK(
          nimbleType->isTimestampMicroNano(),
          "Provided schema doesn't match file schema.");
      offsets.emplace_back(
          nimbleType->asTimestampMicroNano().microsDescriptor().offset());
      offsets.emplace_back(
          nimbleType->asTimestampMicroNano().nanosDescriptor().offset());
      return std::make_unique<TimestampMicroNanoFieldReaderFactory>(
          veloxType->type(), nimbleType.get(), pool);
    }
    case velox::TypeKind::ARRAY: {
      NIMBLE_CHECK(
          nimbleType->isArray() || nimbleType->isArrayWithOffsets(),
          "Provided schema doesn't match file schema.");
      NIMBLE_CHECK_EQ(
          veloxType->size(),
          1,
          "Velox array type should have exactly one child.");
      if (nimbleType->isArray()) {
        auto& nimbleArray = nimbleType->asArray();
        auto& elementType = veloxType->childAt(0);
        offsets.emplace_back(nimbleArray.lengthsDescriptor().offset());
        auto elements = isSelected(elementType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleArray.elements(),
                  elementType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(
                  elementType->type(), pool);
        return std::make_unique<ArrayFieldReaderFactory>(
            veloxType->type(), nimbleType.get(), std::move(elements), pool);
      } else {
        auto& nimbleArrayWithOffsets = nimbleType->asArrayWithOffsets();
        offsets.emplace_back(
            nimbleArrayWithOffsets.lengthsDescriptor().offset());
        offsets.emplace_back(
            nimbleArrayWithOffsets.offsetsDescriptor().offset());

        auto& elementType = veloxType->childAt(0);
        auto elements = isSelected(elementType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleArrayWithOffsets.elements(),
                  elementType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(
                  elementType->type(), pool);
        return std::make_unique<ArrayWithOffsetsFieldReaderFactory>(
            veloxType->type(), nimbleType.get(), std::move(elements), pool);
      }
    }
    case velox::TypeKind::ROW: {
      NIMBLE_CHECK(
          nimbleType->isRow(), "Provided schema doesn't match file schema.");

      auto& nimbleRow = nimbleType->asRow();
      auto& veloxRow = veloxType->type()->as<velox::TypeKind::ROW>();
      std::vector<std::unique_ptr<FieldReaderFactory>> children;
      std::vector<velox::TypePtr> childTypes;
      children.reserve(veloxType->size());
      childTypes.reserve(veloxType->size());
      offsets.emplace_back(nimbleRow.nullsDescriptor().offset());

      for (auto i = 0; i < veloxType->size(); ++i) {
        auto& child = veloxType->childAt(i);
        std::unique_ptr<FieldReaderFactory> factory;
        if (isSelected(child->id())) {
          if (i < nimbleRow.childrenCount()) {
            factory = createFieldReaderFactory(
                parameters,
                nimbleRow.childAt(i),
                child,
                offsets,
                isSelected,
                level + 1,
                &veloxRow.nameOf(i),
                pool);
          } else {
            factory = std::make_unique<NullFieldReaderFactory>(
                inferType(
                    parameters,
                    veloxRow.nameOf(i),
                    veloxRow.childAt(i),
                    level + 1),
                pool);
          }
        }
        childTypes.emplace_back(factory ? factory->veloxType() : child->type());
        children.emplace_back(std::move(factory));
      }

      // Underlying reader may return a different vector type than what's
      // specified (eg. flat map read as struct). So create new ROW type based
      // on child types. Note this special logic is only for Row type based
      // on the constraint that flatmap can only be top level fields.
      return std::make_unique<RowFieldReaderFactory>(
          velox::ROW(
              std::vector<std::string>(veloxRow.names()),
              std::move(childTypes)),
          nimbleType.get(),
          std::move(children),
          pool,
          parameters);
    }
    case velox::TypeKind::MAP: {
      NIMBLE_CHECK(
          nimbleType->isMap() || nimbleType->isFlatMap() ||
              nimbleType->isSlidingWindowMap(),
          "Provided schema doesn't match file schema.");
      NIMBLE_CHECK_EQ(
          veloxType->size(),
          2,
          "Velox map type should have exactly two children.");

      if (nimbleType->isMap()) {
        const auto& nimbleMap = nimbleType->asMap();
        auto& keyType = veloxType->childAt(0);
        offsets.emplace_back(nimbleMap.lengthsDescriptor().offset());
        auto keys = isSelected(keyType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleMap.keys(),
                  keyType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(keyType->type(), pool);
        auto& valueType = veloxType->childAt(1);
        auto values = isSelected(valueType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleMap.values(),
                  valueType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(valueType->type(), pool);
        return std::make_unique<MapFieldReaderFactory>(
            veloxType->type(),
            nimbleType.get(),
            std::move(keys),
            std::move(values),
            pool);
      } else if (nimbleType->isSlidingWindowMap()) {
        const auto& nimbleMap = nimbleType->asSlidingWindowMap();
        offsets.emplace_back(nimbleMap.offsetsDescriptor().offset());
        offsets.emplace_back(nimbleMap.lengthsDescriptor().offset());
        auto& keyType = veloxType->childAt(0);
        auto keys = isSelected(keyType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleMap.keys(),
                  keyType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(keyType->type(), pool);
        auto& valueType = veloxType->childAt(1);
        auto values = isSelected(valueType->id())
            ? createFieldReaderFactory(
                  parameters,
                  nimbleMap.values(),
                  valueType,
                  offsets,
                  isSelected,
                  level + 1,
                  /*name=*/nullptr,
                  pool)
            : std::make_unique<NullFieldReaderFactory>(valueType->type(), pool);
        return std::make_unique<SlidingWindowMapFieldReaderFactory>(
            veloxType->type(),
            nimbleType.get(),
            std::move(keys),
            std::move(values),
            pool);
      } else {
        auto& nimbleFlatMap = nimbleType->asFlatMap();
        offsets.emplace_back(nimbleFlatMap.nullsDescriptor().offset());
        NIMBLE_CHECK(
            level == 1 && name != nullptr,
            "Flat map is only supported as top level fields");
        auto flatMapAsStruct =
            parameters.readFlatMapFieldAsStruct.contains(*name);

        // Extract features only when flat map is not empty. When flatmap is
        // empty, writer creates dummy child with empty name to carry schema
        // information. We need to capture actual children count here.
        auto childrenCount = nimbleFlatMap.childrenCount();
        if (childrenCount == 1 && nimbleFlatMap.nameAt(0).empty()) {
          childrenCount = 0;
        }

        folly::F14FastMap<std::string_view, size_t> namesToIndices;

        auto featuresIt = parameters.flatMapFeatureSelector.find(*name);
        auto hasFeatureSelection =
            featuresIt != parameters.flatMapFeatureSelector.end();
        if (hasFeatureSelection) {
          NIMBLE_CHECK(
              !featuresIt->second.features.empty(),
              "Flat map feature selection for map '{}' has empty feature set.",
              *name);

          if (featuresIt->second.mode == SelectionMode::Include) {
            // We have valid feature projection. Build name -> index lookup
            // table.
            namesToIndices.reserve(childrenCount);
            for (auto i = 0; i < childrenCount; ++i) {
              namesToIndices.emplace(nimbleFlatMap.nameAt(i), i);
            }
          } else {
            NIMBLE_CHECK(
                !flatMapAsStruct,
                "Exclusion can only be applied when flat map is returned as a regular map.");
          }
        } else {
          // Not specifying features for a flat map is only allowed when
          // reconstructing a map column. For struct encoding, we require the
          // caller to provide feature selection, as it dictates the order of
          // the returned features.
          NIMBLE_CHECK(
              !flatMapAsStruct,
              "Flat map '{}' is configured to be returned as a struct, but feature selection is missing. "
              "Feature selection is used to define the order of the features in the returned struct.",
              *name);
        }

        auto actualType = veloxType->type();
        const auto& valueType = veloxType->childAt(1);
        std::vector<size_t> selectedChildren;
        std::vector<const StreamDescriptor*> inMapDescriptors;

        if (flatMapAsStruct) {
          // When reading as struct, all children appear in the feature
          // selection will need to be in the result even if they don't exist
          // in the schema.
          auto& features = featuresIt->second.features;
          selectedChildren.reserve(features.size());
          inMapDescriptors.reserve(features.size());
          actualType = createFlatType(features, veloxType->type());

          for (const auto& feature : features) {
            auto it = namesToIndices.find(feature);
            if (it != namesToIndices.end()) {
              const auto childIdx = it->second;
              selectedChildren.emplace_back(childIdx);
              auto* inMapDescriptor =
                  &nimbleFlatMap.inMapDescriptorAt(childIdx);
              inMapDescriptors.emplace_back(inMapDescriptor);
              offsets.emplace_back(inMapDescriptor->offset());
            } else {
              inMapDescriptors.emplace_back(nullptr);
            }
          }
        } else if (childrenCount > 0) {
          // When reading as regular map, projection only matters if the map
          // is not empty.
          if (!hasFeatureSelection) {
            selectedChildren.reserve(childrenCount);
            for (auto i = 0; i < childrenCount; ++i) {
              selectedChildren.emplace_back(i);
            }
          } else {
            auto& features = featuresIt->second.features;
            if (featuresIt->second.mode == SelectionMode::Include) {
              // Note this path is slightly different from "as struct" path as
              // it doesn't need to add the missing children to the selection.
              selectedChildren.reserve(features.size());
              for (auto& feature : features) {
                auto it = namesToIndices.find(feature);
                if (it != namesToIndices.end()) {
                  selectedChildren.emplace_back(it->second);
                }
              }
            } else {
              folly::F14FastSet<std::string_view> exclusions(
                  features.begin(), features.end());
              selectedChildren.reserve(childrenCount);
              for (auto i = 0; i < childrenCount; ++i) {
                if (!exclusions.contains(nimbleFlatMap.nameAt(i))) {
                  selectedChildren.emplace_back(i);
                }
              }
            }
          }

          inMapDescriptors.reserve(selectedChildren.size());
          for (auto childIdx : selectedChildren) {
            auto* inMapDescriptor = &nimbleFlatMap.inMapDescriptorAt(childIdx);
            inMapDescriptors.emplace_back(inMapDescriptor);
            offsets.emplace_back(inMapDescriptor->offset());
          }
        }

        std::vector<std::unique_ptr<FieldReaderFactory>> valueReaders;
        valueReaders.reserve(selectedChildren.size());
        for (auto childIdx : selectedChildren) {
          valueReaders.emplace_back(createFieldReaderFactory(
              parameters,
              nimbleFlatMap.childAt(childIdx),
              valueType,
              offsets,
              isSelected,
              level + 1,
              nullptr,
              pool));
        }

        const auto& keySelectionCallback = parameters.keySelectionCallback;
        if (keySelectionCallback != nullptr) {
          keySelectionCallback(
              {.totalKeys = childrenCount,
               .selectedKeys = selectedChildren.size()});
        }

        return createFlatMapReaderFactory(
            pool,
            veloxType->childAt(0)->type()->kind(),
            std::move(actualType),
            nimbleType.get(),
            std::move(inMapDescriptors),
            std::move(valueReaders),
            selectedChildren,
            flatMapAsStruct,
            parameters);
      }
    }
    default:
      NIMBLE_UNSUPPORTED("Unsupported type: {}", veloxType->type()->kindName());
  }
}

} // namespace

FieldReader::FieldReader(
    velox::memory::MemoryPool& pool,
    velox::TypePtr type,
    Decoder* decoder)
    : FieldReader(pool, std::move(type), decoder, Options{}) {}

FieldReader::FieldReader(
    velox::memory::MemoryPool& pool,
    velox::TypePtr type,
    Decoder* decoder,
    const Options& options)
    : pool_{&pool},
      type_{std::move(type)},
      decoder_{decoder},
      decodeExecutor_{options.decodeExecutor},
      maxDecodeParallelism_{options.maxDecodeParallelism},
      minStreamsPerDecodeUnit_{options.minStreamsPerDecodeUnit} {}

uint32_t FieldReader::computeParallelDecodeTaskCount(
    uint32_t numStreamChildren) const {
  if (numStreamChildren == 0) {
    return 0;
  }
  const uint32_t maxByStreams =
      numStreamChildren / std::max(1u, minStreamsPerDecodeUnit_);
  return std::clamp(
      std::min(maxDecodeParallelism_, maxByStreams), 1u, numStreamChildren);
}

void FieldReader::ensureNullConstant(
    const std::shared_ptr<const velox::Type>& type,
    uint32_t rowCount,
    velox::VectorPtr& output) const {
  // If output is already single referenced null constant, resize. Otherwise,
  // allocate new one.
  if (output && output.use_count() == 1 &&
      output->encoding() == velox::VectorEncoding::Simple::CONSTANT &&
      output->isNullAt(0)) {
    output->resize(rowCount);
  } else {
    output = velox::BaseVector::createNullConstant(type, rowCount, pool_);
  }
}

void FieldReader::reset() {
  if (decoder_ != nullptr) {
    decoder_->reset();
  }
}

std::unique_ptr<FieldReader> FieldReaderFactory::createNullColumnReader()
    const {
  return std::make_unique<NullColumnReader>(*pool_, veloxType_);
}

Decoder* FieldReaderFactory::getDecoder(
    const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders,
    const StreamDescriptor& streamDescriptor) const {
  auto it = decoders.find(streamDescriptor.offset());
  if (it == decoders.end()) {
    // It is possible that for a given offset, we don't have a matching
    // decoder. Each stripe might see different number of streams, so for all
    // unknown streams, there won't be a matching decoder.
    return nullptr;
  }
  return it->second.get();
}

template <typename T, typename... Args>
std::unique_ptr<FieldReader> FieldReaderFactory::createReaderImpl(
    const folly::F14FastMap<offset_size, std::unique_ptr<Decoder>>& decoders,
    const StreamDescriptor& nullsDescriptor,
    Args&&... args) const {
  auto decoder = getDecoder(decoders, nullsDescriptor);
  if (!decoder) {
    return createNullColumnReader();
  }

  return std::make_unique<T>(*pool_, veloxType_, decoder, args()...);
}

std::unique_ptr<FieldReaderFactory> FieldReaderFactory::create(
    const FieldReaderParams& parameters,
    const std::shared_ptr<const Type>& nimbleType,
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& veloxType,
    std::vector<uint32_t>& offsets,
    const std::function<bool(uint32_t)>& isSelected,
    velox::memory::MemoryPool* pool) {
  return createFieldReaderFactory(
      parameters,
      nimbleType,
      veloxType,
      offsets,
      isSelected,
      /*level=*/0,
      /*name=*/nullptr,
      pool);
}

} // namespace facebook::nimble
