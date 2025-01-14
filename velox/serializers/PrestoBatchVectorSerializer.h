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

  void estimateSerializedSizeImpl(
      const VectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch);

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
