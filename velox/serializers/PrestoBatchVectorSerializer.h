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
#pragma once

#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/vector/VectorStream.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::serializer::presto::detail {
class PrestoBatchVectorSerializer : public BatchVectorSerializer {
 public:
  PrestoBatchVectorSerializer(
      memory::MemoryPool* pool,
      const PrestoVectorSerde::PrestoOptions& opts)
      : arena_(pool),
        codec_(common::compressionKindToCodec(opts.compressionKind)),
        opts_(opts),
        nulls_(&arena_, true, true, true) {}

  void serialize(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& scratch,
      OutputStream* stream) override;

  void estimateSerializedSize(
      VectorPtr vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch) override {
    estimateSerializedSizeImpl(vector, ranges, sizes, scratch);
  }

 private:
  static inline constexpr char kZero = 0;
  static inline constexpr char kOne = 1;
  // The isNull flags to use when there is just a single null at position 0.
  static inline constexpr char kSingleNull = -128;

  void estimateSerializedSizeImpl(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch);

  template <TypeKind Kind>
  void estimateConstantSerializedSize(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch) {
    VELOX_CHECK(vector->encoding() == VectorEncoding::Simple::CONSTANT);
    using T = typename KindToFlatVector<Kind>::WrapperType;
    auto constantVector = vector->as<ConstantVector<T>>();
    vector_size_t elementSize = 0;
    if (constantVector->isNullAt(0)) {
      // There's just a bit mask for the one null.
      elementSize = 1;
    } else if (constantVector->valueVector()) {
      std::vector<IndexRange> newRanges;
      newRanges.push_back({constantVector->index(), 1});
      auto* elementSizePtr = &elementSize;
      estimateSerializedSizeImpl(
          constantVector->valueVector(), newRanges, &elementSizePtr, scratch);
    } else if (std::is_same_v<T, StringView>) {
      auto value = constantVector->valueAt(0);
      auto string = reinterpret_cast<const StringView*>(&value);
      elementSize = string->size();
    } else {
      elementSize = sizeof(T);
    }

    for (size_t i = 0; i < ranges.size(); ++i) {
      *sizes[i] += elementSize;
    }
  }

  void estimateFlattenedDictionarySize(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch);

  vector_size_t computeSelectedIndices(
      const VectorPtr& vector,
      const VectorPtr& wrappedVector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& scratch,
      vector_size_t* selectedIndices);

  template <TypeKind Kind>
  void estimateDictionarySerializedSize(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch) {
    VELOX_CHECK(vector->encoding() == VectorEncoding::Simple::DICTIONARY);
    using T = typename KindToFlatVector<Kind>::WrapperType;
    const auto& wrappedVector = BaseVector::wrappedVectorShared(vector);

    // We don't currently support serializing DictionaryVectors with nulls, so
    // use the flattened size.
    if (vector->nulls()) {
      estimateFlattenedDictionarySize(vector, ranges, sizes, scratch);
      return;
    }

    // This will ultimately get passed to simd::transpose, so it needs to be a
    // raw_vector.
    raw_vector<IndexRange> childRanges;
    std::vector<vector_size_t*> childSizes;
    for (int rangeIndex = 0; rangeIndex < ranges.size(); rangeIndex++) {
      ScratchPtr<vector_size_t, 64> selectedIndicesHolder(scratch);
      auto* mutableSelectedIndices =
          selectedIndicesHolder.get(wrappedVector->size());
      auto numUsed = computeSelectedIndices(
          vector,
          wrappedVector,
          ranges.subpiece(rangeIndex, 1),
          scratch,
          mutableSelectedIndices);
      for (int i = 0; i < numUsed; i++) {
        childRanges.push_back({mutableSelectedIndices[i], 1});
        childSizes.push_back(sizes[rangeIndex]);
      }

      // Add the size of the indices.
      *sizes[rangeIndex] += sizeof(int32_t) * ranges[rangeIndex].size;
    }

    // In PrestoBatchVectorSerializer we don't preserve the encodings for the
    // valueVector for a DictionaryVector.
    estimateSerializedSize(
        wrappedVector, childRanges, childSizes.data(), scratch);
  }

  void writeHeader(const TypePtr& type, BufferedOutputStream* stream);

  /// Are there any nulls in the Vector or introduced artificially in the
  /// ranges. Does not look recursively at values Vectors or children.
  template <typename RangeType>
  bool hasNulls(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges);

  /// Write out the null flags to the streams.
  template <typename RangeType>
  void writeNulls(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      vector_size_t numRows,
      BufferedOutputStream* stream);

  /// Write out all the null information needed by the PrestoPage, both the
  /// hasNulls and isNull flags.
  template <typename RangeType>
  inline void writeNullsSegment(
      bool hasNulls,
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      vector_size_t numRows,
      BufferedOutputStream* stream) {
    if (hasNulls) {
      // Has-nulls flag.
      stream->write(&kOne, 1);

      // Nulls flags.
      writeNulls(vector, ranges, numRows, stream);
    } else {
      // Has-nulls flag.
      stream->write(&kZero, 1);
    }
  }

  template <typename RangeType>
  void serializeColumn(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream);

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<
          kind != TypeKind::TIMESTAMP && kind != TypeKind::BOOLEAN &&
              kind != TypeKind::OPAQUE && kind != TypeKind::UNKNOWN &&
              !std::
                  is_same_v<typename TypeTraits<kind>::NativeType, StringView>,
          bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    using T = typename TypeTraits<kind>::NativeType;
    const auto* flatVector = vector->as<FlatVector<T>>();
    const auto* rawValues = flatVector->rawValues();
    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    if (this->hasNulls(vector, ranges)) {
      // Write out the has-nulls flag.
      stream->write(&kOne, 1);

      // Write out the nulls flags.
      writeNulls(vector, ranges, numRows, stream);

      // Write out the values.
      // This logic merges consecutive ranges of non-null values so we can make
      // long consecutive writes to the stream. A range ends when we detect a
      // discontinuity between ranges, a null, or the end of the ranges. When
      // this happens we write out the range.

      // Tracks the beginning of the current range.
      int firstNonNull = -1;
      // Tracks the end of the current range.
      int lastNonNull = -1;
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (static_cast<const IndexRangeWithNulls&>(range).isNull) {
            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!flatVector->isNullAt(i)) {
            if (firstNonNull == -1) {
              // We're at the beginning of a new range.
              firstNonNull = i;
              lastNonNull = i;
            } else if (i == lastNonNull + 1) {
              // We're continuing the current range.
              lastNonNull = i;
            } else {
              // We've reached a discontinuity (either because the previous
              // value was null or because the ranges are discontinuous).
              // Write out the current range and start a new one.
              const size_t rangeSize = (1 + lastNonNull - firstNonNull);
              stream->write(
                  reinterpret_cast<const char*>(&rawValues[firstNonNull]),
                  rangeSize * sizeof(T));
              firstNonNull = i;
              lastNonNull = i;
            }
          }
        }
      }
      // There's no more data, if we had a range waiting to be written out, do
      // so.
      if (firstNonNull != -1) {
        const size_t rangeSize = (1 + lastNonNull - firstNonNull);
        stream->write(
            reinterpret_cast<const char*>(&rawValues[firstNonNull]),
            rangeSize * sizeof(T));
      }
    } else {
      // Write out the has-nulls flag.
      stream->write(&kZero, 1);

      // Write out the values. Since there are no nulls, we optimistically
      // assume the ranges are long enough that the overhead of merging
      // consecutive ranges is not worth it.
      for (auto& range : ranges) {
        stream->write(
            reinterpret_cast<const char*>(&rawValues[range.begin]),
            range.size * sizeof(T));
      }
    }
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<kind == TypeKind::TIMESTAMP, bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    const auto* flatVector = vector->as<FlatVector<Timestamp>>();
    const auto* rawValues = flatVector->rawValues();
    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    if (this->hasNulls(vector, ranges)) {
      // Write out the has-nulls flag.
      stream->write(&kOne, 1);

      // Write out the nulls flags.
      writeNulls(vector, ranges, numRows, stream);

      // Write out the values.
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (static_cast<const IndexRangeWithNulls&>(range).isNull) {
            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!flatVector->isNullAt(i)) {
            if (opts_.useLosslessTimestamp) {
              writeInt64(stream, rawValues[i].getSeconds());
              writeInt64(stream, rawValues[i].getNanos());
            } else {
              writeInt64(stream, rawValues[i].toMillis());
            }
          }
        }
      }
    } else {
      // Write out the has-nulls flag.
      stream->write(&kZero, 1);

      // Write out the values.
      for (auto& range : ranges) {
        if (opts_.useLosslessTimestamp) {
          for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
            writeInt64(stream, rawValues[i].getSeconds());
            writeInt64(stream, rawValues[i].getNanos());
          }
        } else {
          for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
            writeInt64(stream, rawValues[i].toMillis());
          }
        }
      }
    }
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<
          std::is_same_v<typename TypeTraits<kind>::NativeType, StringView>,
          bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    const auto* flatVector = vector->as<FlatVector<StringView>>();
    const auto* rawValues = flatVector->rawValues();
    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    if (this->hasNulls(vector, ranges)) {
      // The total number of bytes we'll write out for the strings.
      int32_t numBytes = 0;

      // Write out the offsets.
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (range.isNull) {
            // If it's a range of nulls, we just write the last offset out n
            // times.
            for (int i = 0; i < range.size; i++) {
              writeInt32(stream, numBytes);
            }

            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!flatVector->isNullAt(i)) {
            numBytes += rawValues[i].size();
          }
          writeInt32(stream, numBytes);
        }
      }

      // Write out the has-nulls flag.
      stream->write(&kOne, 1);

      // Write out the nulls flags.
      writeNulls(vector, ranges, numRows, stream);

      // Write out the total number of bytes.
      writeInt32(stream, numBytes);

      // Write out the values.
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (static_cast<const IndexRangeWithNulls&>(range).isNull) {
            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!flatVector->isNullAt(i)) {
            stream->write(rawValues[i].data(), rawValues[i].size());
          }
        }
      }
    } else {
      // Write out the offsets.
      int32_t numBytes = 0;
      for (const auto& range : ranges) {
        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          numBytes += rawValues[i].size();
          writeInt32(stream, numBytes);
        }
      }

      // Write out the has-nulls flag.
      stream->write(&kZero, 1);

      // Write out the total number of bytes.
      writeInt32(stream, numBytes);

      // Write out the values.
      for (auto& range : ranges) {
        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          stream->write(rawValues[i].data(), rawValues[i].size());
        }
      }
    }
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<kind == TypeKind::BOOLEAN, bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    const auto* flatVector = vector->as<FlatVector<bool>>();
    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    if (this->hasNulls(vector, ranges)) {
      // Write out the has-nulls flag.
      stream->write(&kOne, 1);

      // Write out the nulls flags.
      writeNulls(vector, ranges, numRows, stream);

      // Write out the values.
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (static_cast<const IndexRangeWithNulls&>(range).isNull) {
            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!vector->isNullAt(i)) {
            stream->write(flatVector->valueAtFast(i) ? &kOne : &kZero, 1);
          }
        }
      }
    } else {
      // Write out the has-nulls flag.
      stream->write(&kZero, 1);

      // Write out the values.
      for (const auto& range : ranges) {
        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          stream->write(flatVector->valueAtFast(i) ? &kOne : &kZero, 1);
        }
      }
    }
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<kind == TypeKind::OPAQUE, bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    const auto* flatVector = vector->as<FlatVector<std::shared_ptr<void>>>();
    const auto* rawValues = flatVector->rawValues();
    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    int32_t numBytes = 0;

    // To avoid serializng the values twice, we hold the serialized data here
    // until we reach the point in the stream where we can write it out.
    ScratchPtr<std::string, 64> valuesHolder(scratch_);
    std::string* mutableValues = valuesHolder.get(numRows);
    size_t valuesIndex = 0;

    auto serializer = vector->type()->asOpaque().getSerializeFunc();

    const bool hasNulls = flatVector->rawValues();

    // Write out the offsets and serialize the values.
    if (hasNulls) {
      for (const auto& range : ranges) {
        if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
          if (range.isNull) {
            for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
              writeInt32(stream, numBytes);
            }
            continue;
          }
        }

        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          if (!flatVector->isNullAt(i)) {
            mutableValues[valuesIndex] = serializer(rawValues[i]);
            numBytes += mutableValues[valuesIndex].size();
            valuesIndex++;
          }

          writeInt32(stream, numBytes);
        }
      }
    } else {
      for (const auto& range : ranges) {
        for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
          mutableValues[valuesIndex] = serializer(rawValues[i]);
          numBytes += mutableValues[valuesIndex].size();
          valuesIndex++;

          writeInt32(stream, numBytes);
        }
      }
    }

    // Write out the nulls flag and nulls.
    writeNullsSegment(hasNulls, vector, ranges, numRows, stream);

    // Write out the total number of bytes.
    writeInt32(stream, numBytes);

    // Write out the serialized values.
    for (size_t i = 0; i < valuesIndex; ++i) {
      stream->write(mutableValues[i].data(), mutableValues[i].size());
    }
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<kind == TypeKind::UNKNOWN, bool> = true>
  void serializeFlatVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    VELOX_CHECK_NOT_NULL(vector->rawNulls());

    const auto numRows = rangesTotalSize(ranges);

    // Write out the header.
    writeHeader(vector->type(), stream);

    // Write out the number of rows.
    writeInt32(stream, numRows);

    // Write out the has-nulls flag.
    stream->write(&kOne, 1);

    // Write out the nulls.
    nulls_.startWrite(bits::nbytes(numRows));
    nulls_.appendBool(bits::kNull, numRows);
    nulls_.flush(stream);
  }

  template <typename RangeType>
  void serializeRowVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream);

  template <typename RangeType>
  void serializeArrayVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream);

  template <typename RangeType>
  void serializeMapVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream);

  /// A helper function for getting a value from a ConstantVector that can be
  /// written to a Presto Stream (mostly just abstracts the complexity needed
  /// for booleans).
  template <TypeKind kind>
  std::conditional_t<
      kind == TypeKind::BOOLEAN,
      char,
      typename KindToFlatVector<kind>::WrapperType>
  extractConstant(
      const ConstantVector<typename KindToFlatVector<kind>::WrapperType>*
          constVector) {
    if constexpr (kind == TypeKind::BOOLEAN) {
      return constVector->valueAtFast(0) ? kOne : kZero;
    } else {
      return constVector->valueAtFast(0);
    }
  }

  /// Helper function for writing a flat stream of any type with a single null
  /// value.
  void writeSingleNull(const TypePtr& type, BufferedOutputStream* stream);

  /// Helper function for writing an empty flat stream of any type.
  void writeEmptyVector(const TypePtr& type, BufferedOutputStream* stream);

  /// Helper function for writing a flat stream with a single primitive value
  /// `value`. If `withNull` is true a null is written as the first value,
  /// followed by `value`.
  template <typename T>
  void writeSingleValue(
      const T& value,
      const TypePtr& type,
      BufferedOutputStream* stream,
      bool withNull = false) {
    VELOX_CHECK(type->isPrimitiveType());

    // Write out the header.
    writeHeader(type, stream);
    // Write out the number of rows.
    writeInt32(stream, withNull ? 2 : 1);

    // Write out the lengths if necessary.
    if constexpr (std::is_same_v<T, StringView>) {
      if (withNull) {
        writeInt32(stream, 0);
      }
      writeInt32(stream, value.size());
    }

    // Write out the hasNull and isNull flags.
    if (withNull) {
      stream->write(&kOne, 1);
      stream->write(&kSingleNull, 1);
    } else {
      stream->write(&kZero, 1);
    }

    // Write out the single non-null value.
    if constexpr (std::is_same_v<T, StringView>) {
      // Write out the total length of the values, i.e. the length of the only
      // value.
      writeInt32(stream, value.size());
      stream->write(value.data(), value.size());
    } else {
      if constexpr (std::is_same_v<T, Timestamp>) {
        if (opts_.useLosslessTimestamp) {
          writeInt64(stream, value.getSeconds());
          writeInt64(stream, value.getNanos());
        } else {
          writeInt64(stream, value.toMillis());
        }
      } else {
        stream->write(reinterpret_cast<const char*>(&value), sizeof(T));
      }
    }
  }

  /// This is specifically for the case where we're flattening a
  /// DictionaryVector which has a ConstantVector as its values Vector, the
  /// DictionaryVector introduced nulls, and the type is primitive and small.
  /// In this case the value is no longer constant and the most efficient way to
  /// represent it is to flatten it.
  template <TypeKind kind>
  void serializeConstantVectorAsFlat(
      const ConstantVector<typename KindToFlatVector<kind>::WrapperType>*
          constVector,
      const folly::Range<const IndexRangeWithNulls*>& ranges,
      int32_t numRows,
      BufferedOutputStream* stream) {
    using T = typename KindToFlatVector<kind>::WrapperType;
    // If either of these is true, it's more efficient to create a dictionary.
    static_assert(TypeTraits<kind>::isFixedWidth && sizeof(T) <= 4);

    // Write out the header, in this case we're writing a flat stream according
    // to the type.
    const auto encoding = typeToEncodingName(constVector->type());
    writeInt32(stream, encoding.size());
    stream->write(encoding.data(), encoding.size());

    // Write out the number of rows.
    writeInt32(stream, numRows);

    // Write out the hasNulls flag (if we're calling this there must be a null).
    stream->write(&kOne, 1);

    using ValueType = std::conditional_t<kind == TypeKind::BOOLEAN, char, T>;
    // We use this to build up the values so we only have to iterate over the
    // ranges once.
    ScratchPtr<ValueType, 64> valuesHolder(scratch_);
    ValueType* mutableValues = valuesHolder.get(numRows);
    // This tracks where to write the next value in mutableValues.
    size_t valuesIndex = 0;

    // Extract the constant value from the Vector.
    const ValueType constValue = extractConstant<kind>(constVector);

    // Write out the isNull flags and build up mutableValues.
    nulls_.startWrite(bits::nbytes(numRows));
    for (auto& range : ranges) {
      if (range.isNull) {
        nulls_.appendBool(bits::kNull, range.size);
      } else {
        nulls_.appendBool(bits::kNotNull, range.size);
        std::fill_n(mutableValues + valuesIndex, range.size, constValue);
        valuesIndex += range.size;
      }
    }

    nulls_.flush(stream);

    // Write out the non-null constant values.
    stream->write(
        reinterpret_cast<char*>(mutableValues), sizeof(T) * valuesIndex);
  }

  /// This is specifically for the case where we're flattening a
  /// DictionaryVector which has a ConstantVector as its values Vector, the
  /// DictionaryVector introduced nulls, and the type is non-primitive or large.
  /// In this case the value is no longer constant and the most efficient way to
  /// represent it is as a DictionaryVector with 2 values, null and the constant
  /// value.
  template <TypeKind kind>
  void serializeConstantVectorAsDictionary(
      const ConstantVector<typename KindToFlatVector<kind>::WrapperType>*
          constVector,
      const folly::Range<const IndexRangeWithNulls*>& ranges,
      int32_t numRows,
      BufferedOutputStream* stream) {
    using T = typename KindToFlatVector<kind>::WrapperType;
    // If both of these are false, it's more efficient just to flatten the data.
    static_assert(!TypeTraits<kind>::isFixedWidth || sizeof(T) > 4);

    // Write out the header, in this case we're writing a Dictionary encoded
    // stream.
    const auto& encoding = kDictionary;
    writeInt32(stream, encoding.size());
    stream->write(encoding.data(), encoding.size());

    // Write out the number of rows.
    writeInt32(stream, numRows);

    // Write out the dictionary values, this is a null at index 0 and the
    // constant value at index 1.
    if (constVector->valueVector() != nullptr) {
      std::vector<IndexRangeWithNulls> selectedRanges;
      selectedRanges.reserve(2);
      // Create a range with a single artificial null value.
      selectedRanges.push_back({0, 1, true});
      // Create a range with the single constant value.
      selectedRanges.push_back({constVector->index(), 1, false});

      serializeColumn<IndexRangeWithNulls>(
          constVector->valueVector(), selectedRanges, stream);
    } else {
      const T value = constVector->valueAtFast(0);
      // Write out a single value with a null preceding it.
      writeSingleValue(
          value,
          constVector->type(),
          stream,
          // Inject a null before the value.
          true);
    }

    // Used to hold the dictionary indices, these are 0 for nulls inherited from
    // the dictionary and 1 for non-null values (the constant value).
    ScratchPtr<int32_t, 64> indicesHolder(scratch_);
    int32_t* mutableIndices = indicesHolder.get(numRows);
    // Where to write the next index in mutableIndices.
    size_t indicesIndex = 0;
    for (auto& range : ranges) {
      if (range.isNull) {
        std::fill_n(mutableIndices + indicesIndex, range.size, 0);
      } else {
        std::fill_n(mutableIndices + indicesIndex, range.size, 1);
      }

      indicesIndex += range.size;
    }
    // Write out the indices.
    stream->write(
        reinterpret_cast<char*>(mutableIndices),
        indicesIndex * sizeof(int32_t));

    // Write out the dictionary ID (we don't use this, so just write 0).
    static const int64_t unused{0};
    writeInt64(stream, unused);
    writeInt64(stream, unused);
    writeInt64(stream, unused);
  }

  template <TypeKind kind, typename RangeType>
  void serializeConstantVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    using T = typename KindToFlatVector<kind>::WrapperType;
    const auto* constVector = vector->as<ConstantVector<T>>();
    const auto numRows = rangesTotalSize(ranges);

    if (constVector->isNullAt(0)) {
      // Write a null constant regardless of RangeType.
      const auto& encoding = kRLE;
      writeInt32(stream, encoding.size());
      stream->write(encoding.data(), encoding.size());

      // Write the number of rows.
      writeInt32(stream, numRows);

      // Write out the constant null value.
      writeSingleNull(constVector->type(), stream);
      return;
    }

    if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
      // If this was wrapped by a dictionary that injected nulls, we either
      // need to flatten the vector or write it out as a dictionary.
      if constexpr (TypeTraits<kind>::isFixedWidth && sizeof(T) <= 4) {
        serializeConstantVectorAsFlat<kind>(
            constVector, ranges, numRows, stream);
      } else {
        serializeConstantVectorAsDictionary<kind>(
            constVector, ranges, numRows, stream);
      }
      return;
    }

    // Write the header.
    const auto& encoding = kRLE;
    writeInt32(stream, encoding.size());
    stream->write(encoding.data(), encoding.size());

    // Write the number of rows.
    writeInt32(stream, numRows);

    // Write the single constant value.
    if (constVector->valueVector() != nullptr) {
      const IndexRange range{constVector->index(), 1};
      serializeColumn<IndexRange>(
          constVector->valueVector(), {&range, 1}, stream);
    } else {
      writeSingleValue(
          extractConstant<kind>(constVector), constVector->type(), stream);
    }
  }

  template <bool WithNulls>
  void serializeFlattenedDictionary(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      const int32_t numRows,
      BufferedOutputStream* stream) {
    using RangeType =
        std::conditional_t<WithNulls, IndexRangeWithNulls, IndexRange>;
    // Holds the ranges to write out of the values Vector.
    ScratchPtr<RangeType, 64> selectedRangesHolder(scratch_);
    RangeType* mutableSelectedRanges = selectedRangesHolder.get(numRows);
    // Where to write the next range in mutableSelectedRanges;
    size_t selectedRangesIndex = 0;

    // Compute the ranges to write out from the base Vector.
    const VectorPtr& wrapped = BaseVector::wrappedVectorShared(vector);
    for (const auto& range : ranges) {
      for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
        if constexpr (WithNulls) {
          if (vector->isNullAt(i)) {
            mutableSelectedRanges[selectedRangesIndex++] =
                IndexRangeWithNulls{0, 1, true};
            continue;
          }
        }

        const auto innerIndex = vector->wrappedIndex(i);
        mutableSelectedRanges[selectedRangesIndex++] = {innerIndex, 1};
      }
    }

    // Write out the flattened data.
    serializeColumn(
        wrapped,
        folly::Range<const RangeType*>(mutableSelectedRanges, numRows),
        stream);
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<std::is_same_v<RangeType, IndexRange>, bool> =
          true>
  void serializeDictionaryVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    auto numRows = rangesTotalSize(ranges);

    // Cannot serialize dictionary as PrestoPage dictionary if it has nulls.
    if (vector->mayHaveNulls()) {
      serializeFlattenedDictionary<true>(vector, ranges, numRows, stream);
      return;
    }

    const auto& wrappedVector = BaseVector::wrappedVectorShared(vector);

    // This is used to track a mapping from output row to the index in the
    // dictionary's alphabet.
    ScratchPtr<vector_size_t, 64> selectedIndicesHolder(scratch_);
    auto* mutableSelectedIndices =
        selectedIndicesHolder.get(wrappedVector->size());
    auto numUsed = computeSelectedIndices(
        vector, wrappedVector, ranges, scratch_, mutableSelectedIndices);

    // If the values are fixed width and we aren't getting enough reuse to
    // justify the dictionary, flatten it. For variable width types, rather
    // than iterate over them computing their size, we simply assume we'll get
    // a benefit.
    if constexpr (TypeTraits<kind>::isFixedWidth) {
      // This calculation admittdely ignores some constants, but if they
      // really make a difference, they're small so there's not much
      // difference either way.
      if (!opts_.preserveEncodings &&
          numUsed * vector->type()->cppSizeInBytes() +
                  numRows * sizeof(int32_t) >=
              numRows * vector->type()->cppSizeInBytes()) {
        if (vector->mayHaveNulls()) {
          serializeFlattenedDictionary<true>(vector, ranges, numRows, stream);
        } else {
          serializeFlattenedDictionary<false>(vector, ranges, numRows, stream);
        }
        return;
      }
    }

    // If every element is unique the dictionary isn't giving us any benefit,
    // flatten it.
    if (!opts_.preserveEncodings && numUsed == numRows) {
      if (vector->mayHaveNulls()) {
        serializeFlattenedDictionary<true>(vector, ranges, numRows, stream);
      } else {
        serializeFlattenedDictionary<false>(vector, ranges, numRows, stream);
      }
      return;
    }

    // Write out the header.
    const auto& encoding = kDictionary;
    writeInt32(stream, encoding.size());
    stream->write(encoding.data(), encoding.size());

    // Write out the number of rows.
    writeInt32(stream, numRows);

    // This is used to track which ranges in the base Vector we need to
    // serialize as the dictionary's alphabet.
    ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);
    IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numUsed);

    for (vector_size_t i = 0; i < numUsed; ++i) {
      mutableSelectedRanges[i] = {mutableSelectedIndices[i], 1};
    }

    // Serialize the used elements from the Dictionary.
    serializeColumn(
        wrappedVector,
        folly::Range<const IndexRange*>(mutableSelectedRanges, numUsed),
        stream);

    // Create a mapping from the original indices to the indices in the shrunk
    // Dictionary of just used values.
    ScratchPtr<vector_size_t, 64> updatedIndicesHolder(scratch_);
    auto* updatedIndices = updatedIndicesHolder.get(wrappedVector->size());
    vector_size_t curIndex = 0;
    for (vector_size_t i = 0; i < numUsed; ++i) {
      updatedIndices[mutableSelectedIndices[i]] = curIndex++;
    }

    // Write out the indices, translating them using the above mapping.
    for (const auto& range : ranges) {
      for (auto i = 0; i < range.size; ++i) {
        writeInt32(
            stream, updatedIndices[vector->wrappedIndex(range.begin + i)]);
      }
    }

    // Write out the dictionary ID (we don't use it, so just write 0).
    static const int64_t unused{0};
    writeInt64(stream, unused);
    writeInt64(stream, unused);
    writeInt64(stream, unused);
  }

  template <
      TypeKind kind,
      typename RangeType,
      typename std::enable_if_t<!std::is_same_v<RangeType, IndexRange>, bool> =
          true>
  void serializeDictionaryVector(
      const VectorPtr& vector,
      const folly::Range<const RangeType*>& ranges,
      BufferedOutputStream* stream) {
    // This would mean this DictionaryVector is the base values of another
    // DictionaryVector that injected nulls.  The base values can only be a
    // ConstantVector or a flat-like Vector.
    VELOX_UNSUPPORTED();
  }

  StreamArena arena_;
  const std::unique_ptr<folly::io::Codec> codec_;
  const PrestoVectorSerde::PrestoOptions opts_;

  // A scratch space for writing null bits, this is a frequent operation that
  // the OutputStream interface is not well suited for.
  //
  // Since this is shared/reused, it is important that the usage of nulls_
  // once started when serializing a Vector is finished before serializing any
  // children. This can be guaranteed by using the writeNullsSegment or
  // writeNulls functions.
  ByteOutputStream nulls_;
  Scratch scratch_;
};
} // namespace facebook::velox::serializer::presto::detail
