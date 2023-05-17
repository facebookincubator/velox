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
#include "velox/serializers/SingleSerializer.h"
#include "velox/common/base/Crc.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::serializer {
namespace {
constexpr int8_t kCompressedBitMask = 1;
constexpr int8_t kEncryptedBitMask = 2;
constexpr int8_t kCheckSumBitMask = 4;

int64_t computeChecksum(
    SingleOutputStreamListener* listener,
    int codecMarker,
    int numRows,
    int uncompressedSize) {
  auto result = listener->crc();
  result.process_bytes(&codecMarker, 1);
  result.process_bytes(&numRows, 4);
  result.process_bytes(&uncompressedSize, 4);
  return result.checksum();
}

int64_t computeChecksum(
    ByteStream* source,
    int codecMarker,
    int numRows,
    int uncompressedSize) {
  auto offset = source->tellp();
  bits::Crc32 crc32;

  auto remainingBytes = uncompressedSize;
  while (remainingBytes > 0) {
    auto data = source->nextView(remainingBytes);
    crc32.process_bytes(data.data(), data.size());
    remainingBytes -= data.size();
  }

  crc32.process_bytes(&codecMarker, 1);
  crc32.process_bytes(&numRows, 4);
  crc32.process_bytes(&uncompressedSize, 4);
  auto checksum = crc32.checksum();

  source->seekp(offset);

  return checksum;
}

char getCodecMarker() {
  char marker = 0;
  marker |= kCheckSumBitMask;
  return marker;
}

bool isCompressedBitSet(int8_t codec) {
  return (codec & kCompressedBitMask) == kCompressedBitMask;
}

bool isEncryptedBit(int8_t codec) {
  return (codec & kEncryptedBitMask) == kEncryptedBitMask;
}

bool isChecksumBitSet(int8_t codec) {
  return (codec & kCheckSumBitMask) == kCheckSumBitMask;
}

void readColumns(
    ByteStream* source,
    velox::memory::MemoryPool* pool,
    int32_t length,
    const std::vector<TypePtr>& types,
    std::vector<VectorPtr>& result);

template <TypeKind kind>
VectorPtr readFlatVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool);

VectorPtr readArrayVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  ArrayVector* arrayVector = nullptr;
  std::vector<TypePtr> childTypes = {type->childAt(0)};
  std::vector<VectorPtr> children(1);
  if (arrayVector) {
    children[0] = arrayVector->elements();
  }
  VELOX_UNSUPPORTED("Not support deserialize Array");
}

VectorPtr readMapVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  MapVector* mapVector = nullptr;
  std::vector<TypePtr> childTypes = {type->childAt(0), type->childAt(1)};
  std::vector<VectorPtr> children(2);
  if (mapVector) {
    children[0] = mapVector->mapKeys();
    children[1] = mapVector->mapValues();
  }
  // readColumns(source, pool, childTypes, children);
  VELOX_UNSUPPORTED("Not support deserialize Map");
}

VectorPtr readRowVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  MapVector* mapVector = nullptr;
  std::vector<TypePtr> childTypes = {type->childAt(0), type->childAt(1)};
  std::vector<VectorPtr> children(2);
  if (mapVector) {
    children[0] = mapVector->mapKeys();
    children[1] = mapVector->mapValues();
  }
  // readColumns(source, pool, childTypes, children);
  VELOX_UNSUPPORTED("Not support deserialize Row");
}

BufferPtr readBuffer(ByteStream* source, velox::memory::MemoryPool* pool) {
  auto size = source->read<int32_t>();
  if (size == 0) {
    return BufferPtr(nullptr);
  }
  auto buffer = AlignedBuffer::allocate<char>(size, pool);
  source->readBytes(buffer->asMutable<uint8_t>(), size);
  return buffer;
}

template <TypeKind kind>
VectorPtr readFlatVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  auto bodyBufferSize = source->read<int32_t>();
  VELOX_CHECK_EQ(bodyBufferSize, 2);
  auto nulls = readBuffer(source, pool);
  auto values = readBuffer(source, pool);
  auto stringBuffersSize = source->read<int32_t>();
  std::vector<BufferPtr> stringBuffers;
  stringBuffers.reserve(stringBuffersSize);
  for (int32_t i = 0; i < stringBuffersSize; i++) {
    stringBuffers.emplace_back(std::move(readBuffer(source, pool)));
  }
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_shared<FlatVector<T>>(
      pool,
      type,
      std::move(nulls),
      length,
      std::move(values),
      std::move(stringBuffers));
}

int128_t readInt128(ByteStream* source) {
  // ByteStream does not support reading int128_t values.
  auto low = source->read<int64_t>();
  auto high = source->read<int64_t>();
  return buildInt128(high, low);
}

template <typename T>
T deserializeVariable(ByteStream* source, velox::memory::MemoryPool* pool) {
  if constexpr (std::is_same_v<T, Timestamp>) {
    return Timestamp::fromMicros(source->read<int64_t>());
  } else if constexpr (std::is_same_v<T, Date>) {
    return Date(source->read<int32_t>());
  } else if constexpr (std::is_same_v<T, StringView>) {
    int32_t size = source->read<int32_t>();
    auto values = AlignedBuffer::allocate<char>(size, pool);
    source->readBytes(values->asMutable<uint8_t>(), size);
    return StringView(values->asMutable<char>(), size);
  } else {
    return source->read<T>();
  }
}

template <TypeKind kind>
VectorPtr deserializeVariableToVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  T value = deserializeVariable<T>(source, pool);
  return std::make_shared<ConstantVector<T>>(
      pool, length, false, type, std::move(value));
}

template <TypeKind kind>
VectorPtr readConstantVector(
    ByteStream* source,
    int32_t length,
    std::shared_ptr<const Type> type,
    velox::memory::MemoryPool* pool) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  auto constantIsVector = source->read<bool>();
  auto constantIsNull = source->read<bool>();
  if (constantIsVector) {
    std::vector<TypePtr> childTypes = {type};
    std::vector<VectorPtr> children(1);
    readColumns(source, pool, length, childTypes, children);
    VELOX_CHECK_EQ(1, children[0]->size());
    return BaseVector::wrapInConstant(length, 0, children[0]);
  }

  if (constantIsNull) {
    return BaseVector::createNullConstant(type, length, pool);
  }

  if constexpr (std::is_same_v<T, UnscaledShortDecimal>) {
    int64_t unscaledValue = source->read<int64_t>();
    return std::make_shared<ConstantVector<T>>(
        pool, length, false, type, T(unscaledValue));
  } else if constexpr (std::is_same_v<T, UnscaledLongDecimal>) {
    return std::make_shared<ConstantVector<T>>(
        pool, length, false, type, T(readInt128(source)));
  } else {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        deserializeVariableToVector, type->kind(), source, length, type, pool);
  }
}

void readColumns(
    ByteStream* source,
    velox::memory::MemoryPool* pool,
    int32_t numRows,
    const std::vector<TypePtr>& types,
    std::vector<VectorPtr>& result) {
  if (source->atEnd()) { // empty page
    return;
  }
  for (int32_t i = 0; i < types.size(); ++i) {
    auto vectorEncoding =
        static_cast<VectorEncoding::Simple>(source->read<uint8_t>());
    switch (vectorEncoding) {
      case VectorEncoding::Simple::FLAT: {
        auto res = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
            readFlatVector, types[i]->kind(), source, numRows, types[i], pool);
        result.emplace_back(std::move(res));
      } break;
      case VectorEncoding::Simple::CONSTANT: {
        auto res = VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
            readConstantVector,
            types[i]->kind(),
            source,
            numRows,
            types[i],
            pool);
        result.emplace_back(std::move(res));
      } break;
      case VectorEncoding::Simple::ROW:
        result.emplace_back(readRowVector(source, numRows, types[i], pool));
        break;
      case VectorEncoding::Simple::ARRAY:
        result.emplace_back(readArrayVector(source, numRows, types[i], pool));
        break;
      case VectorEncoding::Simple::MAP:
        result.emplace_back(readMapVector(source, numRows, types[i], pool));
        break;
      default:
        VELOX_NYI("{} unsupported", __FUNCTION__);
    }
  }
}

void writeInt32(OutputStream* out, int32_t value) {
  out->write(reinterpret_cast<char*>(&value), sizeof(value));
}

void writeInt64(OutputStream* out, int64_t value) {
  out->write(reinterpret_cast<char*>(&value), sizeof(value));
}

void writeBool(OutputStream* out, bool value) {
  out->write(reinterpret_cast<char*>(&value), sizeof(value));
}

// Appendable container for serialized values. To append a value at a
// time, call appendNull or appendNonNull first. Then call
// appendLength if the type has a length. A null value has a length of
// 0. Then call appendValue if the value was not null.
class SingleVectorStream {
 public:
  SingleVectorStream(
      const TypePtr type,
      StreamArena* streamArena,
      int32_t initialNumRows)
      : type_(type), streamArena_(streamArena), constValue_(streamArena) {
    if (initialNumRows > 0) {
      switch (type_->kind()) {
        case TypeKind::ROW:
        case TypeKind::ARRAY:
        case TypeKind::MAP:
          children_.resize(type_->size());
          for (int32_t i = 0; i < type_->size(); ++i) {
            children_[i] = std::make_unique<SingleVectorStream>(
                type_->childAt(i), streamArena, initialNumRows);
          }
          break;
        default:
          break;
      }
    }
  }

  void appendBuffers(BufferPtr buffers) {
    bodyBuffers_.emplace_back(buffers);
  }

  void appendStringBuffers(std::vector<BufferPtr> buffers) {
    stringBuffers_ = buffers;
  }

  void setEncoding(VectorEncoding::Simple encoding) {
    encoding_ = encoding;
  }

  void setConstantIsVector(bool constantIsVector) {
    constantIsVector_ = constantIsVector;
  }

  void appendConstantNull(bool isNull) {
    constantIsNull_ = isNull;
  }

  template <typename T>
  void appendOneConst(const T& value) {
    if (constValue_.size() == 0) {
      constValue_.startWrite(sizeof(T));
    };
    append(folly::Range(&value, 1));
  }

  SingleVectorStream* childAt(int32_t index) {
    return children_[index].get();
  }

  void newChild(const TypePtr type, int32_t initialNumRows) {
    children_.emplace_back(std::make_unique<SingleVectorStream>(
        type, streamArena_, initialNumRows));
  }

  template <typename E>
  constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
  }

  // similiar as flush
  vector_size_t maxSerializedSize() {
    vector_size_t size = 0;
    size += sizeof(uint8_t); /* encoding */
    if (constantIsVector_.has_value()) {
      size += sizeof(bool);
      size += sizeof(bool);
      if (constValue_.size() > 0) {
        size += constValue_.size();
      }
    } else {
      size += sizeof(int32_t);
      for (auto& buffer : bodyBuffers_) {
        size += sizeof(int32_t);
        if (buffer != nullptr) {
          size += buffer->size();
        }
      }
      size += sizeof(int32_t);
      for (const auto& buffer : stringBuffers_) {
        size += sizeof(int32_t);
        if (buffer != nullptr) {
          size += buffer->size();
        }
      }
    }

    for (const auto& child : children_) {
      size += child->maxSerializedSize();
    }
    return size;
  }

  // Writes out the accumulated contents. Does not change the state.
  void flush(OutputStream* out) {
    uint8_t encoding = to_underlying(encoding_);
    out->write(reinterpret_cast<char*>(&encoding), sizeof(encoding));

    if (constantIsVector_.has_value()) {
      writeBool(out, constantIsVector_.value());
      writeBool(out, constantIsNull_);
      if (constValue_.size() > 0) {
        constValue_.flush(out);
      }
    } else {
      writeInt32(out, bodyBuffers_.size());
      for (auto& buffer : bodyBuffers_) {
        if (buffer == nullptr) {
          writeInt32(out, 0);
        } else {
          writeInt32(out, buffer->size());
          out->write(buffer->asMutable<char>(), buffer->size());
        }
      }
      writeInt32(out, stringBuffers_.size());
      for (const auto& buffer : stringBuffers_) {
        if (buffer == nullptr) {
          writeInt32(out, 0);
        } else {
          writeInt32(out, buffer->size());
          out->write(buffer->asMutable<char>(), buffer->size());
        }
      }
    }

    for (const auto& child : children_) {
      child->flush(out);
    }
  }

 private:
  template <typename T>
  void append(folly::Range<const T*> values) {
    constValue_.append(values);
  }

  const TypePtr type_;
  StreamArena* streamArena_;
  VectorEncoding::Simple encoding_;
  std::optional<bool> constantIsVector_;
  bool constantIsNull_;
  ByteStream constValue_;
  // nulls, values or others
  std::vector<BufferPtr> bodyBuffers_;
  std::vector<BufferPtr> stringBuffers_;
  std::vector<std::unique_ptr<SingleVectorStream>> children_;
};

template <>
inline void SingleVectorStream::append(folly::Range<const StringView*> values) {
  for (auto& value : values) {
    auto size = value.size();
    constValue_.appendOne<int32_t>(size);
    constValue_.appendStringPiece(folly::StringPiece(value.data(), size));
  }
}

void serializeColumn(const BaseVector* vector, SingleVectorStream* stream);

template <TypeKind kind>
void serializeFlatVector(const BaseVector* vector, SingleVectorStream* stream) {
  using T = typename TypeTraits<kind>::NativeType;
  auto flatVector = dynamic_cast<const FlatVector<T>*>(vector);
  stream->appendBuffers(flatVector->nulls());
  stream->appendBuffers(flatVector->values());
  stream->appendStringBuffers(flatVector->stringBuffers());
}

void serializeRowVector(const BaseVector* vector, SingleVectorStream* stream) {
  auto rowVector = dynamic_cast<const RowVector*>(vector);
  for (int32_t i = 0; i < rowVector->childrenSize(); ++i) {
    serializeColumn(rowVector->childAt(i).get(), stream->childAt(i));
  }
}

void serializeArrayVector(
    const BaseVector* vector,
    SingleVectorStream* stream) {
  auto arrayVector = dynamic_cast<const ArrayVector*>(vector);
  stream->appendBuffers(arrayVector->nulls());
  stream->appendBuffers(arrayVector->offsets());
  stream->appendBuffers(arrayVector->sizes());
  serializeColumn(arrayVector->elements().get(), stream->childAt(0));
}

void serializeMapVector(const BaseVector* vector, SingleVectorStream* stream) {
  auto mapVector = dynamic_cast<const MapVector*>(vector);
  // Wait to serialize nullCount and sortedKeys
  stream->appendBuffers(mapVector->nulls());
  stream->appendBuffers(mapVector->offsets());
  stream->appendBuffers(mapVector->sizes());

  serializeColumn(mapVector->mapKeys().get(), stream->childAt(0));
  serializeColumn(mapVector->mapValues().get(), stream->childAt(1));
}

template <TypeKind kind>
void serializeVariable(const BaseVector* vector, SingleVectorStream* stream) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  auto constVector = dynamic_cast<const ConstantVector<T>*>(vector);
  T value = constVector->valueAtFast(0);
  if constexpr (std::is_same_v<T, Timestamp>) {
    // It may lost some nanos, same with UnsafeRowSerializer
    stream->appendOneConst<int64_t>(value.toMicros());
  } else if constexpr (std::is_same_v<T, Date>) {
    stream->appendOneConst<int32_t>(value.days());
  } else {
    stream->appendOneConst<T>(value);
  }
}

template <TypeKind kind>
void serializeConstantVector(
    const BaseVector* vector,
    SingleVectorStream* stream) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  auto constVector = dynamic_cast<const ConstantVector<T>*>(vector);

  stream->setConstantIsVector(constVector->valueVector() != nullptr);
  stream->appendConstantNull(vector->isNullAt(0));
  if (constVector->valueVector()) {
    const BaseVector* wrapped = vector->wrappedVector();
    stream->newChild(wrapped->type(), wrapped->size());
    serializeColumn(wrapped, stream->childAt(0));
    return;
  }

  if (!vector->isNullAt(0)) {
    if constexpr (std::is_same_v<T, UnscaledShortDecimal>) {
      T value = constVector->valueAtFast(0);
      stream->appendOneConst<int64_t>(value.unscaledValue());
    } else if constexpr (std::is_same_v<T, UnscaledLongDecimal>) {
      T value = constVector->valueAtFast(0);
      stream->appendOneConst<int128_t>(value.unscaledValue());
    } else {
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          serializeVariable, vector->typeKind(), vector, stream);
    }
  }
}

void serializeColumn(const BaseVector* vector, SingleVectorStream* stream) {
  stream->setEncoding(vector->encoding());
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          serializeFlatVector, vector->typeKind(), vector, stream);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          serializeConstantVector, vector->typeKind(), vector, stream);
      break;
    case VectorEncoding::Simple::ROW:
      serializeRowVector(vector, stream);
      break;
    case VectorEncoding::Simple::ARRAY:
      serializeArrayVector(vector, stream);
      break;
    case VectorEncoding::Simple::MAP:
      serializeMapVector(vector, stream);
      break;
    case VectorEncoding::Simple::LAZY:
      serializeColumn(vector->loadedVector(), stream);
      break;
    default:
      VELOX_NYI("{} unsupported", __FUNCTION__);
  }
}

class SingleVectorSerializer : public VectorSerializer {
 public:
  SingleVectorSerializer(
      std::shared_ptr<const RowType> rowType,
      int32_t numRows,
      StreamArena* streamArena) {
    auto types = rowType->children();
    auto numTypes = types.size();
    streams_.resize(numTypes);
    for (int i = 0; i < numTypes; i++) {
      streams_[i] =
          std::make_unique<SingleVectorStream>(types[i], streamArena, numRows);
    }
  }

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& /* ranges */) override {
    VELOX_CHECK(
        numRows_ == 0,
        "SingleVectorSerializer can only append RowVector only once");
    if (vector->size() > 0) {
      numRows_ += vector->size();
      for (int32_t i = 0; i < vector->childrenSize(); ++i) {
        serializeColumn(vector->childAt(i).get(), streams_[i].get());
      }
    }
  }

  vector_size_t maxSerializedSize() override {
    vector_size_t size = 0;
    if (numRows_ != 0) {
      for (auto& stream : streams_) {
        size += stream->maxSerializedSize();
      }
    }
    size += 25; /* flush header layout size */
    return size;
  }

  void flush(OutputStream* out) override {
    flushInternal(numRows_, out);
  }

  // Writes the contents to 'stream' in wire format
  void flushInternal(int32_t numRows, OutputStream* out) {
    auto listener = dynamic_cast<SingleOutputStreamListener*>(out->listener());
    // Reset CRC computation
    if (listener) {
      listener->reset();
    }

    char codec = 0;
    if (listener) {
      codec = getCodecMarker();
    }

    int32_t offset = out->tellp();

    // Pause CRC computation
    if (listener) {
      listener->pause();
    }

    writeInt32(out, numRows);
    out->write(&codec, 1);

    // Make space for uncompressedSizeInBytes & sizeInBytes
    writeInt32(out, 0);
    writeInt32(out, 0);
    writeInt64(out, 0); // Write zero checksum

    // Number of columns and stream content. Unpause CRC.
    if (listener) {
      listener->resume();
    }
    writeInt32(out, streams_.size());
    if (numRows != 0) {
      for (auto& stream : streams_) {
        stream->flush(out);
      }
    }

    // Pause CRC computation
    if (listener) {
      listener->pause();
    }

    // Fill in uncompressedSizeInBytes & sizeInBytes
    int32_t size = (int32_t)out->tellp() - offset;
    int32_t uncompressedSize = size - kHeaderSize;
    int64_t crc = 0;
    if (listener) {
      crc = computeChecksum(listener, codec, numRows, uncompressedSize);
    }

    out->seekp(offset + kSizeInBytesOffset);
    writeInt32(out, uncompressedSize);
    writeInt32(out, uncompressedSize);
    writeInt64(out, crc);
    out->seekp(offset + size);
  }

 private:
  static const int32_t kSizeInBytesOffset{4 + 1};
  static const int32_t kHeaderSize{kSizeInBytesOffset + 4 + 4 + 8};

  int32_t numRows_{0};
  std::vector<std::unique_ptr<SingleVectorStream>> streams_;
};
} // namespace

void SingleVectorSerde::estimateSerializedSize(
    VectorPtr vector,
    const folly::Range<const IndexRange*>& /* ranges */,
    vector_size_t** sizes) {
  if (sizes == nullptr) {
    return;
  }
  *(sizes[0]) += vector->estimateFlatSize();
}

std::unique_ptr<VectorSerializer> SingleVectorSerde::createSerializer(
    std::shared_ptr<const RowType> type,
    int32_t numRows,
    StreamArena* streamArena,
    const Options* options) {
  return std::make_unique<SingleVectorSerializer>(type, numRows, streamArena);
}

void SingleVectorSerde::deserialize(
    ByteStream* source,
    velox::memory::MemoryPool* pool,
    std::shared_ptr<const RowType> type,
    std::shared_ptr<RowVector>* result,
    const Options* options) {
  auto numRows = source->read<int32_t>();

  auto pageCodecMarker = source->read<int8_t>();
  auto uncompressedSize = source->read<int32_t>();
  // skip size in bytes
  source->skip(4);
  auto checksum = source->read<int64_t>();

  int64_t actualCheckSum = 0;
  if (isChecksumBitSet(pageCodecMarker)) {
    actualCheckSum =
        computeChecksum(source, pageCodecMarker, numRows, uncompressedSize);
  }

  VELOX_CHECK_EQ(
      checksum, actualCheckSum, "Received corrupted serialized page.");

  // skip number of columns
  source->skip(4);

  std::vector<VectorPtr> children;
  auto childTypes = type->as<TypeKind::ROW>().children();

  readColumns(source, pool, numRows, childTypes, children);
  *result = std::make_shared<RowVector>(
      pool, type, BufferPtr(nullptr), numRows, children);
}

// static
void SingleVectorSerde::registerVectorSerde() {
  velox::registerVectorSerde(std::make_unique<SingleVectorSerde>());
}

} // namespace facebook::velox::serializer
