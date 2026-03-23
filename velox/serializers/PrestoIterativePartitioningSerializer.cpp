/*
 * Copyright (c) International Business Machines Corporation
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
#include "velox/serializers/PrestoIterativePartitioningSerializer.h"

#include "velox/common/base/BitUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::serializer::presto {

namespace {

constexpr int8_t kCheckSumBitMask = 4;
constexpr int64_t kVectorSizeTypeSize{sizeof(vector_size_t)};
// [numRows:4][codec:1]
constexpr int64_t kUncompressedSizeOffset{kVectorSizeTypeSize + 1};
// [numRows:4][codec:1][uncompressedSize:4][compressedSize:4][checksum:8]
constexpr int64_t kHeaderSize{kUncompressedSizeOffset + 4 + 4 + 8};

static inline const std::string_view kByteArray{"BYTE_ARRAY"};
static inline const std::string_view kShortArray{"SHORT_ARRAY"};
static inline const std::string_view kIntArray{"INT_ARRAY"};
static inline const std::string_view kLongArray{"LONG_ARRAY"};
static inline const std::string_view kInt128Array{"INT128_ARRAY"};
static inline const std::string_view kVariableWidth{"VARIABLE_WIDTH"};
static inline const std::string_view kRow{"ROW"};

inline void writeInt32(OutputStream* out, int32_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void writeInt64(OutputStream* out, int64_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

char getCodecMarker() {
  char marker = 0;
  marker |= kCheckSumBitMask;
  return marker;
}

std::string_view typeToEncodingName(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return kByteArray;
    case TypeKind::SMALLINT:
      return kShortArray;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return kIntArray;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
      return kLongArray;
    case TypeKind::HUGEINT:
      return kInt128Array;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return kVariableWidth;
    case TypeKind::ROW:
      return kRow;
    default:
      VELOX_FAIL("Unsupported type kind: {}", static_cast<int>(type->kind()));
  }
}

/// Finalizes the Presto page CRC by mixing in the codec marker, row count,
/// and uncompressed size on top of the listener's accumulated data checksum.
int64_t computeChecksum(
    PrestoOutputStreamListener& listener,
    int8_t codecMarker,
    int32_t numRows,
    int32_t uncompressedSize) {
  auto crc = listener.crc();
  crc.process_bytes(&codecMarker, 1);
  crc.process_bytes(&numRows, 4);
  crc.process_bytes(&uncompressedSize, 4);
  return static_cast<int64_t>(crc.checksum());
}

/// Returns the serialized byte width of a fixed-width type, matching the
/// sizeof(T) used in flushFlatValues.
int32_t fixedTypeWidth(TypeKind kind) {
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return 1;
    case TypeKind::SMALLINT:
      return 2;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return 4;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
      return 8;
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      return 16;
    default:
      return 0;
  }
}

/// Returns the exact bytes for one fixed-width column in one partition.
int64_t
simpleColumnBytes(const TypePtr& colType, int64_t numRows, int64_t numNulls) {
  const auto encodingName = typeToEncodingName(colType);
  return 4 + static_cast<int64_t>(encodingName.size()) + // header
      4 + // rowCount
      1 + // nullFlag
      (numNulls > 0 ? bits::nbytes(numRows) : 0) + // null bitmap
      (numRows - numNulls) * fixedTypeWidth(colType->kind()); // values
}

/// Returns per-partition exact byte counts for one column (all partitions).
/// Recurses into nested ROW columns.
///
/// Byte layout per column type:
///   Fixed-width: simpleColumnBytes(colType, numRows, numNulls)
///   ROW:         7 (header) + 4 (numFields)
///                + sum(child sizes)
///                + 4 (numRows) + 4*(numRows+1) (offsets) + 1 (hasNulls)
///                + (rowNulls>0 ? bits::nbytes(numRows) : 0)
std::vector<int64_t> computeColumnFlushSizes(
    const std::vector<PartitionedVectorPtr>& columnVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<vector_size_t>& rowsPerPartition,
    uint32_t numPartitions) {
  std::vector<int64_t> sizes(numPartitions, 0);

  // Compute per-partition null counts by summing across batches.
  std::vector<int64_t> nullCounts(numPartitions, 0);
  for (uint32_t p : nonEmptyPartitions) {
    for (const auto& pv : columnVectors) {
      nullCounts[p] += pv->numNullsAt(p);
    }
  }

  switch (colType->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::HUGEINT:
      for (uint32_t p : nonEmptyPartitions) {
        sizes[p] =
            simpleColumnBytes(colType, rowsPerPartition[p], nullCounts[p]);
      }
      break;

    case TypeKind::TIMESTAMP:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
      VELOX_NYI(
          "computeColumnFlushSizes: unsupported type kind {}",
          TypeKindName::toName(colType->kind()));

    case TypeKind::ROW: {
      const auto& rowSchema = colType->asRow();
      const int32_t numFields = static_cast<int32_t>(rowSchema.size());

      // Fixed per-partition overhead: header(7) + numFields(4) + footer:
      // numRows(4)
      // + sequential offsets 4*(numRows+1) + hasNulls(1)
      // + null bitmap for the ROW vector itself if any rows in this partition
      // are null.
      for (uint32_t p : nonEmptyPartitions) {
        const int64_t numRows = rowsPerPartition[p];
        const int64_t rowNullBitmapBytes =
            nullCounts[p] > 0 ? bits::nbytes(numRows) : 0;
        sizes[p] = 7 + 4 + // "ROW" header + numFields
            4 + 4 * (numRows + 1) + 1 + // footer: numRows + offsets + hasNulls
            rowNullBitmapBytes;
      }
      // Add child column sizes recursively.
      for (uint32_t col = 0; col < static_cast<uint32_t>(numFields); ++col) {
        std::vector<PartitionedVectorPtr> childVectors;
        childVectors.reserve(columnVectors.size());
        for (const auto& pv : columnVectors) {
          childVectors.push_back(
              std::dynamic_pointer_cast<PartitionedRowVector>(pv)->childAt(
                  col));
        }
        const auto childSizes = computeColumnFlushSizes(
            childVectors,
            rowSchema.childAt(col),
            nonEmptyPartitions,
            rowsPerPartition,
            numPartitions);
        for (uint32_t p : nonEmptyPartitions) {
          sizes[p] += childSizes[p];
        }
      }
      break;
    }

    default:
      VELOX_UNSUPPORTED(
          "computeColumnFlushSizes: unsupported type kind {}",
          TypeKindName::toName(colType->kind()));
  }
  return sizes;
}

} // namespace

PrestoIterativePartitioningSerializer::PrestoIterativePartitioningSerializer(
    RowTypePtr inputType,
    uint32_t numPartitions,
    const SerdeOpts& opts,
    memory::MemoryPool* pool)
    : type_(std::move(inputType)),
      numPartitions_(numPartitions),
      opts_(opts),
      pool_(pool),
      rowsPerPartition_(numPartitions, 0) {
  VELOX_CHECK_GT(numPartitions_, 0);
  VELOX_CHECK_NOT_NULL(pool_);

  numColumns_ = type_->size();
}

void PrestoIterativePartitioningSerializer::append(
    const RowVectorPtr& input,
    const std::vector<uint32_t>& partitions) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_EQ(
      input->size(),
      partitions.size(),
      "partitions.size() must equal input->size()");

  if (input->size() == 0) {
    return;
  }

  PartitionBuildContext ctx;
  auto partitionedRowVector = PartitionedVector::create(
      std::static_pointer_cast<BaseVector>(input),
      partitions,
      numPartitions_,
      ctx,
      pool_);

  const vector_size_t* partitionOffsets =
      partitionedRowVector->rawPartitionOffsets();
  vector_size_t prevOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    rowsPerPartition_[p] += partitionOffsets[p] - prevOffset;
    prevOffset = partitionOffsets[p];
  }

  partitionedRowVectors_.push_back(std::move(partitionedRowVector));

  bytesBuffered_ += input->retainedSize();
  rowsBuffered_ += static_cast<int64_t>(input->size());
}

// ---------------------------------------------------------------------------
// Top-level flush
// ---------------------------------------------------------------------------

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flush() {
  auto pages =
      (opts_.compressionKind == common::CompressionKind::CompressionKind_NONE)
      ? flushUncompressed()
      : flushCompressed();

  partitionedRowVectors_.clear();
  flushSizes_.clear();
  std::fill(rowsPerPartition_.begin(), rowsPerPartition_.end(), 0);
  bytesBuffered_ = 0;
  rowsBuffered_ = 0;

  return pages;
}

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flushUncompressed() {
  if (partitionedRowVectors_.empty()) {
    return {};
  }

  const char codecMask = getCodecMarker();

  // 1. Determine non-empty partitions.
  std::vector<uint32_t> nonEmptyPartitions;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    if (rowsPerPartition_[p] > 0) {
      nonEmptyPartitions.push_back(p);
    }
  }

  // 2. Pre-compute exact byte sizes per top-level column and partition.
  const auto& rowSchema = type_->asRow();
  flushSizes_.assign(rowSchema.size(), std::vector<int64_t>(numPartitions_, 0));
  for (uint32_t col = 0; col < rowSchema.size(); ++col) {
    std::vector<PartitionedVectorPtr> columnVectors;
    columnVectors.reserve(partitionedRowVectors_.size());
    for (const auto& pRowVector : partitionedRowVectors_) {
      columnVectors.push_back(
          std::dynamic_pointer_cast<PartitionedRowVector>(pRowVector)
              ->childAt(col));
    }
    flushSizes_[col] = computeColumnFlushSizes(
        columnVectors,
        rowSchema.childAt(col),
        nonEmptyPartitions,
        rowsPerPartition_,
        numPartitions_);
  }

  // 3. Create output streams sized to the exact bytes each partition will need,
  // so that the entire payload fits. This avoids multiple resizing and copying.
  std::vector<std::unique_ptr<PrestoOutputStreamListener>> listeners(
      numPartitions_);
  std::vector<std::unique_ptr<IOBufOutputStream>> outputStreams(numPartitions_);
  std::vector<IOBufOutputStream*> rawOutputStreams(numPartitions_);
  std::vector<std::streampos> beginStreamPositions(numPartitions_);

  for (uint32_t p : nonEmptyPartitions) {
    int64_t initialSize = kHeaderSize + 4; // page header + numCols
    for (uint32_t col = 0; col < rowSchema.size(); ++col) {
      initialSize += flushSizes_[col][p];
    }
    listeners[p] = std::make_unique<PrestoOutputStreamListener>();
    outputStreams[p] = std::make_unique<IOBufOutputStream>(
        *pool_, listeners[p].get(), initialSize);
    rawOutputStreams[p] = outputStreams[p].get();
    beginStreamPositions[p] = outputStreams[p]->tellp();

    flushStart(*outputStreams[p], p, codecMask);
  }

  // 4. Flush column data.
  flushRowChildren(
      partitionedRowVectors_, rowSchema, nonEmptyPartitions, rawOutputStreams);

  // 5. Finalize the page by seeking back to fill in sizes and CRC, and get the
  // IOBuf and numOfRows from each stream.
  std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
      result;
  for (uint32_t p : nonEmptyPartitions) {
    flushFinish(
        *outputStreams[p],
        p,
        beginStreamPositions[p],
        codecMask,
        *listeners[p]);
    result[p] =
        std::make_pair(outputStreams[p]->getIOBuf(), rowsPerPartition_[p]);
  }

  return result;
}

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flushCompressed() {
  VELOX_NYI();
}

// ---------------------------------------------------------------------------
// Second level functions: start, columns and finish
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushStart(
    IOBufOutputStream& out,
    uint32_t partition,
    char codecMask) const {
  auto* listener = dynamic_cast<PrestoOutputStreamListener*>(out.listener());
  if (listener) {
    listener->pause();
  }

  // Write 21-byte Presto page header; sizes and CRC are filled in later.
  const int32_t numRows = static_cast<int32_t>(rowsPerPartition_[partition]);
  char header[kHeaderSize] = {};
  std::memcpy(&header[0], &numRows, 4);
  std::memcpy(&header[4], &codecMask, 1);
  out.write(header, kHeaderSize);

  if (listener) {
    listener->resume();
  }

  // Number of columns is included in the CRC.
  const int32_t numCols = static_cast<int32_t>(numColumns_);
  out.write(reinterpret_cast<const char*>(&numCols), 4);
}

void PrestoIterativePartitioningSerializer::flushRowChildren(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const RowType& rowSchema,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  for (uint32_t col = 0; col < rowSchema.size(); ++col) {
    std::vector<PartitionedVectorPtr> column;
    column.reserve(partitionedVectors.size());
    for (const auto& partitionedVector : partitionedVectors) {
      const auto& partitionedRowVector =
          std::dynamic_pointer_cast<PartitionedRowVector>(partitionedVector);
      VELOX_DCHECK_NOT_NULL(partitionedRowVector.get());
      column.push_back(partitionedRowVector->childAt(col));
    }

    flushColumn(
        column, rowSchema.childAt(col), nonEmptyPartitions, outputStreams);
  }
}

void PrestoIterativePartitioningSerializer::flushFinish(
    IOBufOutputStream& out,
    uint32_t partition,
    std::streampos beginOffset,
    char codecMask,
    PrestoOutputStreamListener& listener) const {
  listener.pause();

  const std::streampos totalSize =
      static_cast<int32_t>(out.tellp() - beginOffset);
  const std::streampos uncompressedSize = totalSize - kHeaderSize;
  const int64_t crc = computeChecksum(
      listener,
      static_cast<int8_t>(codecMask),
      static_cast<int32_t>(rowsPerPartition_[partition]),
      uncompressedSize);

  out.seekp(beginOffset + kUncompressedSizeOffset);
  writeInt32(&out, uncompressedSize);
  writeInt32(&out, uncompressedSize); // TODO: compressedSize
  writeInt64(&out, crc);
  out.seekp(beginOffset + totalSize);
}

// ---------------------------------------------------------------------------
// Column-level dispatch
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  VELOX_CHECK_GT(partitionedVectors.size(), 0);

  auto typeKind = partitionedVectors[0]->baseVector()->typeKind();
  switch (typeKind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::HUGEINT:
      flushSimpleColumn(
          partitionedVectors, colType, nonEmptyPartitions, outputStreams);
      break;

    case TypeKind::TIMESTAMP:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::ROW:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
      VELOX_NYI();

    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for PrestoIterativePartitioningSerializer: ",
          typeKind);
  }
}

void PrestoIterativePartitioningSerializer::flushSimpleColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  flushHeader(typeToEncodingName(colType), nonEmptyPartitions, outputStreams);
  flushRowCounts(nonEmptyPartitions, outputStreams);
  flushNulls(partitionedVectors, nonEmptyPartitions, outputStreams);

  for (size_t i = 0; i < partitionedVectors.size(); i++) {
    flushSingleSimpleVector(partitionedVectors[i], outputStreams);
  }
}

template <TypeKind kind>
void PrestoIterativePartitioningSerializer::flushSingleFlatVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  using T = typename TypeTraits<kind>::NativeType;
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<T>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawValues =
      flatVector->baseVector()->template as<FlatVector<T>>()->rawValues();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = flatVector->rawPartitionOffsets();

  flushFlatValues<T>(rawValues, rawNulls, partitionOffsets, outputStreams);
}

// BOOLEAN columns use kByteArray encoding: FlatVector<bool> stores bits
// packed, so rawValues() is unsupported. Each non-null value is written as
// one byte (0x00 or 0x01).
template <>
void PrestoIterativePartitioningSerializer::flushSingleFlatVector<
    TypeKind::BOOLEAN>(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<bool>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawBoolValues =
      flatVector->baseVector()->as<FlatVector<bool>>()->rawValues<uint64_t>();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = flatVector->rawPartitionOffsets();

  // TODO: Improve performance
  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    const auto numValues = offset - lastOffset;
    const auto numNulls = partitionedVector->numNullsAt(p);
    if (outputStreams[p] != nullptr && numValues > 0) {
      if (numNulls == 0) {
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          const int8_t val = bits::isBitSet(rawBoolValues, i) ? 1 : 0;
          outputStreams[p]->write(reinterpret_cast<const char*>(&val), 1);
        }
      } else {
        VELOX_DCHECK_NOT_NULL(rawNulls);
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          if (!bits::isBitNull(rawNulls, i)) {
            const int8_t val = bits::isBitSet(rawBoolValues, i) ? 1 : 0;
            outputStreams[p]->write(reinterpret_cast<const char*>(&val), 1);
          }
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::flushSingleSimpleVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  auto encoding = partitionedVector->baseVector()->encoding();
  auto typeKind = partitionedVector->baseVector()->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          flushSingleFlatVector, typeKind, partitionedVector, outputStreams);
      break;
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::CONSTANT:
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_NYI(
          "Unsupported vector encoding for PrestoIterativePartitioningSerializer: ",
          encoding);
    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for PrestoIterativePartitioningSerializer:flushSingleSimpleVector ",
          encoding);
  }
}

// ---------------------------------------------------------------------------
// Column building blocks
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushHeader(
    std::string_view name,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const int32_t nameLen = static_cast<int32_t>(name.size());
  for (uint32_t p : nonEmptyPartitions) {
    writeInt32(outputStreams[p], nameLen);
    outputStreams[p]->write(name.data(), nameLen);
  }
}

void PrestoIterativePartitioningSerializer::flushRowCounts(
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  for (uint32_t p : nonEmptyPartitions) {
    writeInt32(outputStreams[p], static_cast<int32_t>(rowsPerPartition_[p]));
  }
}

void PrestoIterativePartitioningSerializer::flushNulls(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  std::vector<vector_size_t> nullCounts(numPartitions_, 0);
  for (uint32_t p : nonEmptyPartitions) {
    for (const auto& pv : partitionedVectors) {
      nullCounts[p] += pv->numNullsAt(p);
    }
    const char flagByte = nullCounts[p] > 0 ? 1 : 0;
    outputStreams[p]->write(&flagByte, 1);
  }

  const bool hasAnyNulls = std::any_of(
      nonEmptyPartitions.begin(), nonEmptyPartitions.end(), [&](uint32_t p) {
        return nullCounts[p] > 0;
      });
  if (!hasAnyNulls) {
    return;
  }

  // Build each partition's null bitmap in a temporary buffer, accumulating
  // bits across all batches. Writing via write() correctly handles range
  // boundaries in the output stream without requiring seekp().
  // TODO: Avoid this extra memory allocation and copy
  std::vector<std::vector<uint8_t>> bitmaps(numPartitions_);
  for (uint32_t p : nonEmptyPartitions) {
    if (nullCounts[p] > 0) {
      bitmaps[p].assign(bits::nbytes(rowsPerPartition_[p]), bits::kNotNullByte);
    }
  }

  std::vector<vector_size_t> destBitOffsets(numPartitions_, 0);
  for (const auto& pv : partitionedVectors) {
    const uint64_t* rawNulls = pv->baseVector()->rawNulls();
    const auto* partitionOffsets = pv->rawPartitionOffsets();

    vector_size_t startBit = 0;
    for (uint32_t p : nonEmptyPartitions) {
      const vector_size_t numBits = partitionOffsets[p] - startBit;
      if (rawNulls && numBits > 0 && !bitmaps[p].empty()) {
        bits::copyBits(
            rawNulls,
            startBit,
            reinterpret_cast<uint64_t*>(bitmaps[p].data()),
            destBitOffsets[p],
            numBits);
      }
      if (!bitmaps[p].empty()) {
        destBitOffsets[p] += numBits;
      }
      startBit = partitionOffsets[p];
    }
  }

  for (uint32_t p : nonEmptyPartitions) {
    if (nullCounts[p] == 0) {
      continue;
    }

    // Convert Velox format (LSB-first, 1=not-null) to Presto wire format
    // (MSB-first, 1=null) in-place.
    const int32_t numBytes = bits::nbytes(rowsPerPartition_[p]);
    for (int32_t i = 0; i < numBytes; ++i) {
      bitmaps[p][i] = ~bitmaps[p][i];
      bits::reverseBits(&bitmaps[p][i], 1);
    }

    outputStreams[p]->write(
        reinterpret_cast<const char*>(bitmaps[p].data()), numBytes);
  }
}

template <typename T>
void PrestoIterativePartitioningSerializer::flushFlatValues(
    const T* partitionedValues,
    const uint64_t* rawNulls,
    const vector_size_t* partitionOffsets,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const auto typeWidth = sizeof(T);
  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    const auto numValues = offset - lastOffset;
    if (outputStreams[p] != nullptr && numValues > 0) {
      if (!rawNulls) {
        outputStreams[p]->write(
            reinterpret_cast<const char*>(&partitionedValues[lastOffset]),
            numValues * typeWidth);
      } else {
        // Presto writes only non-null values; null slots are omitted.
        // TODO: Improve performance
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          if (!bits::isBitNull(rawNulls, i)) {
            outputStreams[p]->write(
                reinterpret_cast<const char*>(&partitionedValues[i]),
                typeWidth);
          }
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::flushSequentialOffsets(
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  for (uint32_t p : nonEmptyPartitions) {
    const int32_t numRows = static_cast<int32_t>(rowsPerPartition_[p]);
    for (int32_t i = 0; i <= numRows; ++i) {
      writeInt32(outputStreams[p], i);
    }
  }
}

} // namespace facebook::velox::serializer::presto
