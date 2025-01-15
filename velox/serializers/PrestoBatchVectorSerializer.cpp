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

#include "velox/serializers/PrestoBatchVectorSerializer.h"

#include "velox/serializers/PrestoSerializerEstimationUtils.h"
#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/serializers/VectorStream.h"

namespace facebook::velox::serializer::presto::detail {
namespace {
// Populates mutableOffsets with the starting offset of each collection and the
// total size of all collections.
//
// Populates mutableSelectedRanges with the ranges to write from the children.
//
// Populates rangeIndex with the number of ranges to write from the children.
template <typename VectorType, typename RangeType>
void computeCollectionRangesAndOffsets(
    const VectorType* vector,
    const folly::Range<const RangeType*>& ranges,
    bool hasNulls,
    int32_t* mutableOffsets,
    IndexRange* mutableSelectedRanges,
    size_t& rangeIndex) {
  auto* rawSizes = vector->rawSizes();
  auto* rawOffsets = vector->rawOffsets();

  // The first offset is always 0.
  mutableOffsets[0] = 0;
  // The index of the next offset to write in mutableOffsets.
  size_t offsetsIndex = 1;
  // The length all the collections in ranges seen so far. This is the offset to
  // write for the next collection.
  int32_t totalLength = 0;
  if (hasNulls) {
    for (const auto& range : ranges) {
      if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
        if (range.isNull) {
          std::fill_n(&mutableOffsets[offsetsIndex], range.size, totalLength);
          offsetsIndex += range.size;

          continue;
        }
      }

      for (int32_t i = range.begin; i < range.begin + range.size; ++i) {
        if (vector->isNullAt(i)) {
          mutableOffsets[offsetsIndex++] = totalLength;
        } else {
          auto length = rawSizes[i];
          totalLength += length;
          mutableOffsets[offsetsIndex++] = totalLength;

          // We only have to write anything from the children if the collection
          // is non-empty.
          if (length > 0) {
            mutableSelectedRanges[rangeIndex++] = {rawOffsets[i], rawSizes[i]};
          }
        }
      }
    }
  } else {
    for (const auto& range : ranges) {
      for (auto i = range.begin; i < range.begin + range.size; ++i) {
        auto length = rawSizes[i];
        totalLength += length;
        mutableOffsets[offsetsIndex++] = totalLength;

        // We only have to write anything from the children if the collection is
        // non-empty.
        if (length > 0) {
          mutableSelectedRanges[rangeIndex++] = {rawOffsets[i], length};
        }
      }
    }
  }
}

void flushSerialization(
    int32_t numRows,
    int32_t uncompressedSize,
    int32_t serializationSize,
    char codecMask,
    const std::unique_ptr<folly::IOBuf>& iobuf,
    OutputStream* output) {
  auto listener = dynamic_cast<PrestoOutputStreamListener*>(output->listener());

  // Pause CRC computation.
  if (listener) {
    listener->reset();
    listener->pause();
    codecMask |= getCodecMarker();
  }
  writeInt32(output, numRows);
  output->write(&codecMask, 1);
  writeInt32(output, uncompressedSize);
  writeInt32(output, serializationSize);
  auto crcOffset = output->tellp();
  // Write zero checksum.
  writeInt64(output, 0);
  // Number of columns and stream content. Unpause CRC.
  if (listener) {
    listener->resume();
  }
  for (auto range : *iobuf) {
    output->write(reinterpret_cast<const char*>(range.data()), range.size());
  }
  // Pause CRC computation.
  if (listener) {
    listener->pause();
  }
  const int32_t endSize = output->tellp();
  // Fill in crc.
  int64_t crc = 0;
  if (listener) {
    crc = computeChecksum(listener, codecMask, numRows, uncompressedSize);
  }
  output->seekp(crcOffset);
  writeInt64(output, crc);
  output->seekp(endSize);
}
} // namespace

int32_t PrestoBatchVectorSerializer::serializeUncompressed(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    OutputStream* stream) {
  const auto numRows = rangesTotalSize(ranges);
  const auto rowType = vector->type();
  const auto numChildren = vector->childrenSize();

  BufferedOutputStream out(stream, &arena_);

  auto listener = dynamic_cast<PrestoOutputStreamListener*>(stream->listener());
  // Reset CRC computation.
  if (listener) {
    listener->reset();
  }

  int32_t offset = out.tellp();

  char codecMask = 0;
  if (listener) {
    codecMask = getCodecMarker();
  }
  // Pause CRC computation.
  if (listener) {
    out.flush();
    listener->pause();
  }

  writeInt32(&out, numRows);
  out.write(&codecMask, 1);

  // Make space for uncompressedSizeInBytes & sizeInBytes.
  writeInt32(&out, 0);
  writeInt32(&out, 0);
  // Write zero checksum.
  writeInt64(&out, 0);

  // Number of columns and stream content. Unpause CRC.
  if (listener) {
    out.flush();
    listener->resume();
  }
  writeInt32(&out, numChildren);

  for (int i = 0; i < numChildren; i++) {
    serializeColumn(vector->childAt(i), ranges, &out);
  }

  out.flush();

  // Pause CRC computation.
  if (listener) {
    listener->pause();
  }

  // Fill in uncompressedSizeInBytes & sizeInBytes.
  int32_t size = (int32_t)out.tellp() - offset;
  const int32_t uncompressedSize = size - kHeaderSize;
  int64_t crc = 0;
  if (listener) {
    crc = computeChecksum(listener, codecMask, numRows, uncompressedSize);
  }

  out.seekp(offset + kSizeInBytesOffset);
  writeInt32(&out, uncompressedSize);
  writeInt32(&out, uncompressedSize);
  writeInt64(&out, crc);
  out.flush();
  out.seekp(offset + size);

  return uncompressedSize;
}

FlushSizes PrestoBatchVectorSerializer::serializeCompressed(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    OutputStream* stream) {
  const auto numRows = rangesTotalSize(ranges);
  const auto rowType = vector->type();
  const auto numChildren = vector->childrenSize();

  IOBufOutputStream ioBufStream(*(arena_.pool()), nullptr, arena_.size());
  BufferedOutputStream out(&ioBufStream, &arena_);

  writeInt32(&out, numChildren);

  for (int i = 0; i < numChildren; i++) {
    serializeColumn(vector->childAt(i), ranges, &out);
  }

  out.flush();

  const int32_t uncompressedSize = ioBufStream.tellp();
  VELOX_CHECK_LE(
      uncompressedSize,
      codec_->maxUncompressedLength(),
      "UncompressedSize exceeds limit");
  auto iobuf = ioBufStream.getIOBuf();
  const auto compressedBuffer = codec_->compress(iobuf.get());
  const int32_t compressedSize = compressedBuffer->length();

  if (compressedSize > uncompressedSize * opts_.minCompressionRatio) {
    flushSerialization(
        numRows, uncompressedSize, uncompressedSize, 0, iobuf, stream);

    return {uncompressedSize, uncompressedSize};
  } else {
    flushSerialization(
        numRows,
        uncompressedSize,
        compressedSize,
        kCompressedBitMask,
        compressedBuffer,
        stream);

    return {uncompressedSize, compressedSize};
  }
}

void PrestoBatchVectorSerializer::serialize(
    const RowVectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    OutputStream* stream) {
  if (!needCompression(*codec_)) {
    serializeUncompressed(vector, ranges, scratch, stream);
  } else {
    if (numCompressionsToSkip_ > 0) {
      const auto noCompressionCodec = common::compressionKindToCodec(
          common::CompressionKind::CompressionKind_NONE);
      const auto size = serializeUncompressed(vector, ranges, scratch, stream);
      stats_.compressionSkippedBytes += size;
      --numCompressionsToSkip_;
      ++stats_.numCompressionSkipped;
    } else {
      const auto [size, compressedSize] =
          serializeCompressed(vector, ranges, scratch, stream);
      stats_.compressionInputBytes += size;
      stats_.compressedBytes += compressedSize;
      if (compressedSize > size * opts_.minCompressionRatio) {
        numCompressionsToSkip_ = std::min<int64_t>(
            kMaxCompressionAttemptsToSkip, 1 + stats_.numCompressionSkipped);
      }
    }
  }

  arena_.clear();
}

void PrestoBatchVectorSerializer::estimateSerializedSizeImpl(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          estimateFlatSerializedSize,
          vector->typeKind(),
          vector.get(),
          ranges,
          sizes);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateConstantSerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          estimateDictionarySerializedSize,
          vector->typeKind(),
          vector,
          ranges,
          sizes,
          scratch);
      break;
    case VectorEncoding::Simple::ROW: {
      if (!vector->mayHaveNulls()) {
        // Add the size of the offsets in the Row encoding.
        for (int32_t i = 0; i < ranges.size(); ++i) {
          *sizes[i] += ranges[i].size * sizeof(int32_t);
        }

        auto rowVector = vector->as<RowVector>();
        auto& children = rowVector->children();
        for (auto& child : children) {
          if (child) {
            estimateSerializedSizeImpl(child, ranges, sizes, scratch);
          }
        }

        break;
      }

      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      for (int32_t i = 0; i < ranges.size(); ++i) {
        // Add the size of the nulls bit mask.
        *sizes[i] += bits::nbytes(ranges[i].size);

        auto begin = ranges[i].begin;
        auto end = begin + ranges[i].size;
        for (auto offset = begin; offset < end; ++offset) {
          // Add the size of the offset.
          *sizes[i] += sizeof(int32_t);
          if (!vector->isNullAt(offset)) {
            childRanges.push_back(IndexRange{offset, 1});
            childSizes.push_back(sizes[i]);
          }
        }
      }

      if (!childRanges.empty()) {
        auto rowVector = vector->as<RowVector>();
        auto& children = rowVector->children();
        for (auto& child : children) {
          if (child) {
            estimateSerializedSizeImpl(
                child,
                folly::Range(childRanges.data(), childRanges.size()),
                childSizes.data(),
                scratch);
          }
        }
      }

      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto mapVector = vector->as<MapVector>();
      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      expandRepeatedRanges(
          mapVector,
          mapVector->rawOffsets(),
          mapVector->rawSizes(),
          ranges,
          sizes,
          &childRanges,
          &childSizes);
      if (!childRanges.empty()) {
        estimateSerializedSizeImpl(
            mapVector->mapKeys(), childRanges, childSizes.data(), scratch);
        estimateSerializedSizeImpl(
            mapVector->mapValues(), childRanges, childSizes.data(), scratch);
      }
      break;
    }
    case VectorEncoding::Simple::ARRAY: {
      auto arrayVector = vector->as<ArrayVector>();
      std::vector<IndexRange> childRanges;
      std::vector<vector_size_t*> childSizes;
      expandRepeatedRanges(
          arrayVector,
          arrayVector->rawOffsets(),
          arrayVector->rawSizes(),
          ranges,
          sizes,
          &childRanges,
          &childSizes);
      if (!childRanges.empty()) {
        estimateSerializedSizeImpl(
            arrayVector->elements(), childRanges, childSizes.data(), scratch);
      }
      break;
    }
    case VectorEncoding::Simple::LAZY:
      estimateSerializedSizeImpl(
          vector->as<LazyVector>()->loadedVectorShared(),
          ranges,
          sizes,
          scratch);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported vector encoding {}", vector->encoding());
  }
}

void PrestoBatchVectorSerializer::writeHeader(
    const TypePtr& type,
    BufferedOutputStream* stream) {
  auto encoding = typeToEncodingName(type);
  writeInt32(stream, encoding.size());
  stream->write(encoding.data(), encoding.size());
}

template <>
bool PrestoBatchVectorSerializer::hasNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges) {
  if (vector->nulls()) {
    for (auto& range : ranges) {
      if (!bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        return true;
      }
    }
  }

  return false;
}

template <>
bool PrestoBatchVectorSerializer::hasNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRangeWithNulls*>& ranges) {
  if (vector->nulls()) {
    for (auto& range : ranges) {
      if (range.isNull ||
          !bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        return true;
      }
    }
  } else {
    for (auto& range : ranges) {
      if (range.isNull) {
        return true;
      }
    }
  }

  return false;
}

template <>
void PrestoBatchVectorSerializer::writeNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t numRows,
    BufferedOutputStream* stream) {
  nulls_.startWrite(bits::nbytes(numRows));
  for (auto& range : ranges) {
    nulls_.appendBits(
        vector->rawNulls(), range.begin, range.begin + range.size);
  }
  nulls_.flush(stream);
}

template <>
void PrestoBatchVectorSerializer::writeNulls(
    const VectorPtr& vector,
    const folly::Range<const IndexRangeWithNulls*>& ranges,
    vector_size_t numRows,
    BufferedOutputStream* stream) {
  nulls_.startWrite(bits::nbytes(numRows));
  for (auto& range : ranges) {
    if (range.isNull) {
      nulls_.appendBool(bits::kNull, range.size);
    } else if (vector->mayHaveNulls()) {
      nulls_.appendBits(
          vector->rawNulls(), range.begin, range.begin + range.size);
    } else {
      nulls_.appendBool(bits::kNotNull, range.size);
    }
  }
  nulls_.flush(stream);
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeRowVector(
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges,
    BufferedOutputStream* stream) {
  const auto* rowVector = vector->as<RowVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(vector->type(), stream);

  const bool hasNulls = this->hasNulls(vector, ranges);

  // The ranges to write of the child Vectors, this is the same as the ranges
  // of this RowVector to write except positions where the row is null.
  folly::Range<const IndexRange*> childRanges;
  // PrestoPage requires us to write out for each row 0 if the row is null or
  // i if the row is the i'th non-null row. We track these values here.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The first offset is always 0, this in addition to the offset per row.
  mutableOffsets[0] = 0;
  // The index at which we should write the next value in mutableOffsets.
  size_t offsetsIndex = 1;
  // The value of "offset" to write for the next non-null row.
  int32_t rowOffset = 1;

  // We use this to construct contiguous ranges to write for the children,
  // excluding any null rows.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);

  if (hasNulls) {
    IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
    // The index in mutableSelectedRanges to write the next range.
    size_t rangeIndex = 0;

    for (const auto& range : ranges) {
      if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
        if (range.isNull) {
          std::fill_n(&mutableOffsets[offsetsIndex], range.size, 0);
          offsetsIndex += range.size;

          continue;
        }
      }

      if (vector->mayHaveNulls() &&
          !bits::isAllSet(
              vector->rawNulls(), range.begin, range.begin + range.size)) {
        // The start of the current contiguous range.
        int rangeStart = -1;
        // The length of the current contiguous range.
        int rangeSize = 0;
        for (auto i = range.begin; i < range.begin + range.size; ++i) {
          if (!vector->isNullAt(i)) {
            mutableOffsets[offsetsIndex++] = rowOffset++;

            // If we aren't already in a contiguous range, mark the beginning.
            if (rangeStart == -1) {
              rangeStart = i;
            }
            // Continue the contiguous range.
            rangeSize++;
          } else {
            mutableOffsets[offsetsIndex++] = 0;

            // If we were in a contiguous range, write it out to the scratch
            // buffer and indicate we are no longer in one.
            if (rangeStart != -1) {
              mutableSelectedRanges[rangeIndex++] =
                  IndexRange{rangeStart, rangeSize};
              rangeStart = -1;
              rangeSize = 0;
            }
          }
        }

        // If we were in a contigous range, write out the last one.
        if (rangeStart != -1) {
          mutableSelectedRanges[rangeIndex++] =
              IndexRange{rangeStart, rangeSize};
        }
      } else {
        // There are now nulls in this range, write out the offsets and copy
        // the range to the scratch buffer.
        std::iota(
            &mutableOffsets[offsetsIndex],
            &mutableOffsets[offsetsIndex + range.size],
            rowOffset);
        rowOffset += range.size;
        offsetsIndex += range.size;

        mutableSelectedRanges[rangeIndex++] =
            IndexRange{range.begin, range.size};
      }
    }

    // Lastly update child ranges to exclude any null rows.
    childRanges =
        folly::Range<const IndexRange*>(mutableSelectedRanges, rangeIndex);
  } else {
    // There are no null rows, so offsets is just an incrementing series and
    // we can reuse ranges for the children.
    std::iota(&mutableOffsets[1], &mutableOffsets[numRows + 1], rowOffset);

    if constexpr (std::is_same_v<RangeType, IndexRangeWithNulls>) {
      IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
      // The index in mutableSelectedRanges to write the next range.
      size_t rangeIndex = 0;
      for (const auto& range : ranges) {
        mutableSelectedRanges[rangeIndex++] = {range.begin, range.size};
      }

      childRanges =
          folly::Range<const IndexRange*>(mutableSelectedRanges, ranges.size());
    } else {
      childRanges = ranges;
    }
  }

  if (opts_.nullsFirst) {
    // Write out the number of rows.
    writeInt32(stream, numRows);
    // Write out the hasNull and isNull flags.
    writeNullsSegment(hasNulls, vector, ranges, numRows, stream);
  }

  // Write out the number of children.
  writeInt32(stream, vector->type()->size());
  // Write out the children.
  for (int32_t i = 0; i < rowVector->childrenSize(); ++i) {
    serializeColumn(rowVector->childAt(i), childRanges, stream);
  }

  if (!opts_.nullsFirst) {
    // Write out the number of rows.
    writeInt32(stream, numRows);
    // Write out the offsets.
    stream->write(
        reinterpret_cast<char*>(mutableOffsets),
        (numRows + 1) * sizeof(int32_t));
    // Write out the hasNull and isNull flags.
    writeNullsSegment(hasNulls, vector, ranges, numRows, stream);
  }
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeColumn(
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges,
    BufferedOutputStream* stream) {
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          serializeFlatVector, vector->typeKind(), vector, ranges, stream);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          serializeConstantVector, vector->typeKind(), vector, ranges, stream);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          serializeDictionaryVector,
          vector->typeKind(),
          vector,
          ranges,
          stream);
      break;
    case VectorEncoding::Simple::ROW:
      serializeRowVector(vector, ranges, stream);
      break;
    case VectorEncoding::Simple::ARRAY:
      serializeArrayVector(vector, ranges, stream);
      break;
    case VectorEncoding::Simple::MAP:
      serializeMapVector(vector, ranges, stream);
      break;
    case VectorEncoding::Simple::LAZY:
      serializeColumn(BaseVector::loadedVectorShared(vector), ranges, stream);
      break;
    default:
      VELOX_UNSUPPORTED();
  }
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeArrayVector(
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges,
    BufferedOutputStream* stream) {
  const auto* arrayVector = vector->as<ArrayVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(vector->type(), stream);

  const bool hasNulls = this->hasNulls(vector, ranges);

  // This is used to hold the ranges of the elements Vector to write out.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);
  IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
  // This is used to hold the offsets of the arrays which we write out towards
  // the end.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The number of ranges to write out from the elements Vector. This is equal
  // to the number of non-empty, non-null arrays in ranges.
  size_t rangesSize = 0;
  computeCollectionRangesAndOffsets(
      arrayVector,
      ranges,
      hasNulls,
      mutableOffsets,
      mutableSelectedRanges,
      rangesSize);

  // Write out the elements.
  serializeColumn(
      arrayVector->elements(),
      folly::Range<const IndexRange*>(mutableSelectedRanges, rangesSize),
      stream);

  // Write out the number of rows.
  writeInt32(stream, numRows);
  // Write out the offsets.
  stream->write(
      reinterpret_cast<char*>(mutableOffsets), (numRows + 1) * sizeof(int32_t));

  // Write out the hasNull and isNUll flags.
  writeNullsSegment(hasNulls, vector, ranges, numRows, stream);
}

template <typename RangeType>
void PrestoBatchVectorSerializer::serializeMapVector(
    const VectorPtr& vector,
    const folly::Range<const RangeType*>& ranges,
    BufferedOutputStream* stream) {
  const auto* mapVector = vector->as<MapVector>();
  const auto numRows = rangesTotalSize(ranges);

  // Write out the header.
  writeHeader(vector->type(), stream);

  const bool hasNulls = this->hasNulls(vector, ranges);

  // This is used to hold the ranges of the keys/values Vectors to write out.
  ScratchPtr<IndexRange, 64> selectedRangesHolder(scratch_);
  IndexRange* mutableSelectedRanges = selectedRangesHolder.get(numRows);
  // This is used to hold the offsets of the maps which we write out towards the
  // end.
  ScratchPtr<int32_t, 64> offsetsHolder(scratch_);
  int32_t* mutableOffsets = offsetsHolder.get(numRows + 1);
  // The number of ranges to write out from the keys/values Vectors. This is
  // equal to the number of non-empty, non-null mpas in ranges.
  size_t rangesSize = 0;
  computeCollectionRangesAndOffsets(
      mapVector,
      ranges,
      hasNulls,
      mutableOffsets,
      mutableSelectedRanges,
      rangesSize);

  // Write out the keys.
  serializeColumn(
      mapVector->mapKeys(),
      folly::Range<const IndexRange*>(mutableSelectedRanges, rangesSize),
      stream);
  // Write out the values.
  serializeColumn(
      mapVector->mapValues(),
      folly::Range<const IndexRange*>(mutableSelectedRanges, rangesSize),
      stream);

  // Write out the hash table size (we don't write out the optional hash table,
  // so use -1 as the size).
  writeInt32(stream, -1);
  // Write out the number of rows.
  writeInt32(stream, numRows);
  // Write out the offsets.
  stream->write(
      reinterpret_cast<char*>(mutableOffsets), (numRows + 1) * sizeof(int32_t));

  // Wirte out the hasNull and isNull flags.
  writeNullsSegment(hasNulls, vector, ranges, numRows, stream);
}

/// Specialization for opaque types.
template <>
void PrestoBatchVectorSerializer::writeSingleValue(
    const std::shared_ptr<void>& value,
    const TypePtr& type,
    BufferedOutputStream* stream,
    bool withNull) {
  // Write out the header.
  writeHeader(type, stream);
  // Write out the number of nulls.
  writeInt32(stream, withNull ? 2 : 1);

  const std::string serializedValue =
      type->asOpaque().getSerializeFunc()(value);

  // Write out the lengths.
  if (withNull) {
    writeInt32(stream, 0);
  }
  writeInt32(stream, serializedValue.size());

  // Write out the hasNull and isNull flags.
  if (withNull) {
    stream->write(&kOne, 1);
    stream->write(&kSingleNull, 1);
  } else {
    stream->write(&kZero, 1);
  }

  // Write out the total length of the values, i.e. the length of the only
  // value.
  writeInt32(stream, serializedValue.size());

  // Write out the single non-null value.
  stream->write(serializedValue.data(), serializedValue.size());
}

void PrestoBatchVectorSerializer::writeSingleNull(
    const TypePtr& type,
    BufferedOutputStream* stream) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
    case TypeKind::UNKNOWN:
      // Write the header.
      writeHeader(type, stream);
      // Write the number of rows.
      writeInt32(stream, 1);
      // Write the hasNull flag.
      stream->write(&kOne, 1);
      // Write the isNull flags.
      stream->write(&kSingleNull, 1);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      // Write the header.
      writeHeader(type, stream);
      // Write the number of rows.
      writeInt32(stream, 1);
      // Write the offsets of the (non-existent) values.
      writeInt32(stream, 0);
      // Write the hasNull flag.
      stream->write(&kOne, 1);
      // Write the isNull flags.
      stream->write(&kSingleNull, 1);
      // Write the total size of the (non-existent) values.
      writeInt32(stream, 0);
      break;
    case TypeKind::ARRAY:
      // Write the header.
      writeHeader(type, stream);
      // Write the non-existent elements.
      writeEmptyVector(type->childAt(0), stream);
      // Write the number of rows.
      writeInt32(stream, 1);
      // Write the offsets of the (non-existent) values.
      writeInt32(stream, 0);
      writeInt32(stream, 0);
      // Write the hasNull flag.
      stream->write(&kOne, 1);
      // Write the isNull flags.
      stream->write(&kSingleNull, 1);
      break;
    case TypeKind::MAP:
      // Write the header.
      writeHeader(type, stream);
      // Write the non-existent keys.
      writeEmptyVector(type->childAt(0), stream);
      // Write the non-existent values.
      writeEmptyVector(type->childAt(1), stream);
      // Write the size of the hash map (which we don't use, so it's -1).
      writeInt32(stream, -1);
      // Write the number of rows.
      writeInt32(stream, 1);
      // Write the offsets of the (non-existent) values.
      writeInt32(stream, 0);
      writeInt32(stream, 0);
      // Write the hasNull flag.
      stream->write(&kOne, 1);
      // Write the isNull flags.
      stream->write(&kSingleNull, 1);
      break;
    case TypeKind::ROW:
      // Write the header.
      writeHeader(type, stream);
      if (opts_.nullsFirst) {
        // Write the number of rows.
        writeInt32(stream, 1);
        // Write the hasNull flag.
        stream->write(&kOne, 1);
        // Write the isNull flags.
        stream->write(&kSingleNull, 1);
      }
      // Write the number of children.
      writeInt32(stream, type->size());
      // Write the non-existent children.
      for (int i = 0; i < type->size(); ++i) {
        writeEmptyVector(type->childAt(i), stream);
      }
      if (!opts_.nullsFirst) {
        // Write the number of rows.
        writeInt32(stream, 1);
        // Write the offsets of the (non-existent) values.
        writeInt32(stream, 0);
        writeInt32(stream, 0);
        // Write the hasNull flag.
        stream->write(&kOne, 1);
        // Write the isNull flags.
        stream->write(&kSingleNull, 1);
      }
      break;
    default:
      VELOX_UNSUPPORTED();
  }
}

void PrestoBatchVectorSerializer::writeEmptyVector(
    const TypePtr& type,
    BufferedOutputStream* stream) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
    case TypeKind::UNKNOWN:
      // Write the header.
      writeHeader(type, stream);
      // Write the number of rows.
      writeInt32(stream, 0);
      // Write the hasNulls flag.
      stream->write(&kZero, 1);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      // Write the header.
      writeHeader(type, stream);
      // Write the number of rows.
      writeInt32(stream, 0);
      // Write the hasNulls flag.
      stream->write(&kZero, 1);
      // Write the total size of the non-existent values.
      writeInt32(stream, 0);
      break;
    case TypeKind::ARRAY:
      // Write the header.
      writeHeader(type, stream);
      // Write the non-existent elements.
      writeEmptyVector(type->childAt(0), stream);
      // Write the number of rows.
      writeInt32(stream, 0);
      // Write the offsets of the non-existent values.
      writeInt32(stream, 0);
      // Write the hasNulls flag.
      stream->write(&kZero, 1);
      break;
    case TypeKind::MAP:
      // Write the header.
      writeHeader(type, stream);
      // Write the non-existent keys.
      writeEmptyVector(type->childAt(0), stream);
      // Write the non-existent values.
      writeEmptyVector(type->childAt(1), stream);
      // Write the size of the hash map (which we don't use, so it's -1).
      writeInt32(stream, -1);
      // Write the number of rows.
      writeInt32(stream, 0);
      // Write the offsets of the non-existent values.
      writeInt32(stream, 0);
      // Write the hasNulls flag.
      stream->write(&kZero, 1);
      break;
    case TypeKind::ROW:
      // Write the header.
      writeHeader(type, stream);
      if (opts_.nullsFirst) {
        // Write the number of rows.
        writeInt32(stream, 0);
        // Write the hasNulls flag.
        stream->write(&kZero, 1);
      }
      // Write the number of children.
      writeInt32(stream, type->size());
      // Write the non-existent children.
      for (int i = 0; i < type->size(); ++i) {
        writeEmptyVector(type->childAt(i), stream);
      }
      if (!opts_.nullsFirst) {
        // Write the number of rows.
        writeInt32(stream, 0);
        // Write the offsets of the non-existent values.
        writeInt32(stream, 0);
        // Write the hasNulls flag.
        stream->write(&kZero, 1);
      }
      break;
    default:
      VELOX_UNSUPPORTED();
  }
}

void PrestoBatchVectorSerializer::estimateFlattenedDictionarySize(
    const VectorPtr& vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch) {
  const auto numRows = rangesTotalSize(ranges);

  // Used to hold the ranges of the values Vector that will get written out.
  ScratchPtr<IndexRange, 64> newRangesHolder(scratch_);
  auto* mutableNewRanges = newRangesHolder.get(numRows);
  // Used to track the pointer to the size for the given range to update.
  ScratchPtr<vector_size_t*, 64> childSizesHolder(scratch_);
  auto* mutableChildSizes = childSizesHolder.get(numRows);
  // The index in mutableNewRanges/mutableChildSizes to write the next value.
  int32_t offset = 0;

  const VectorPtr& wrapped = BaseVector::wrappedVectorShared(vector);
  for (int i = 0; i < ranges.size(); i++) {
    int nullCount = 0;
    const auto& range = ranges[i];
    for (int32_t rangeOffset = range.begin;
         rangeOffset < range.begin + range.size;
         ++rangeOffset) {
      if (vector->mayHaveNulls() && vector->isNullAt(rangeOffset)) {
        nullCount++;
        continue;
      }

      const auto innerIndex = vector->wrappedIndex(rangeOffset);
      mutableNewRanges[offset] = IndexRange{innerIndex, 1};
      mutableChildSizes[offset] = sizes[i];
      offset++;
    }

    *sizes[i] += bits::nbytes(nullCount);
  }

  if (offset > 0) {
    // If there were any non-null ranges, compute their size.
    estimateSerializedSize(
        wrapped,
        folly::Range<const IndexRange*>(mutableNewRanges, offset),
        mutableChildSizes,
        scratch);
  }
}

vector_size_t PrestoBatchVectorSerializer::computeSelectedIndices(
    const VectorPtr& vector,
    const VectorPtr& wrappedVector,
    const folly::Range<const IndexRange*>& ranges,
    Scratch& scratch,
    vector_size_t* selectedIndices) {
  // Create a bit set to track which values in the Dictionary are used.
  ScratchPtr<uint64_t, 64> usedIndicesHolder(scratch);
  auto* usedIndices =
      usedIndicesHolder.get(bits::nwords(wrappedVector->size()));
  simd::memset(usedIndices, 0, usedIndicesHolder.size() * sizeof(uint64_t));

  for (const auto& range : ranges) {
    for (auto i = 0; i < range.size; ++i) {
      bits::setBit(usedIndices, vector->wrappedIndex(range.begin + i));
    }
  }

  // Convert the bitset to a list of the used indices.
  return simd::indicesOfSetBits(
      usedIndices, 0, wrappedVector->size(), selectedIndices);
}
} // namespace facebook::velox::serializer::presto::detail
