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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

namespace facebook::velox::cudf_velox {

cudf::type_id veloxToCudfTypeId(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::type_id::BOOL8;
    case TypeKind::TINYINT:
      return cudf::type_id::INT8;
    case TypeKind::SMALLINT:
      return cudf::type_id::INT16;
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        return cudf::type_id::TIMESTAMP_DAYS;
      }
      return cudf::type_id::INT32;
    case TypeKind::BIGINT:
      return cudf::type_id::INT64;
    case TypeKind::REAL:
      return cudf::type_id::FLOAT32;
    case TypeKind::DOUBLE:
      return cudf::type_id::FLOAT64;
    case TypeKind::VARCHAR:
      return cudf::type_id::STRING;
    case TypeKind::VARBINARY:
      return cudf::type_id::STRING;
    case TypeKind::TIMESTAMP:
      return cudf::type_id::TIMESTAMP_NANOSECONDS;
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    // TODO: Decimals are now logical types:
    // https://github.com/facebookincubator/velox/commit/73d2f935b55f084d30557c7be94b9768efb8e56f
    // case TypeKind::SHORT_DECIMAL: return cudf::type_id::DECIMAL64;
    // case TypeKind::LONG_DECIMAL: return cudf::type_id::DECIMAL128;
    case TypeKind::ARRAY:
      return cudf::type_id::LIST;
    case TypeKind::MAP:
      // MAP is represented as LIST<STRUCT<key, value>> in cudf.
      return cudf::type_id::LIST;
    case TypeKind::ROW:
      return cudf::type_id::STRUCT;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      CUDF_FAIL(
          "Unsupported Velox type: " +
          std::string(TypeKindName::toName(type->kind())));
      return cudf::type_id::EMPTY;
  }
}

namespace with_arrow {

namespace {

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto arrowDeviceArray = cudf::to_arrow_host(table, stream, mr);
  auto& arrowArray = arrowDeviceArray->array;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  auto veloxTable = importFromArrowAsOwner(*arrowSchema, arrowArray, pool);
  // BaseVector to RowVector
  auto castedPtr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(castedPtr);
  return castedPtr;
}

template <typename Iterator>
std::vector<cudf::column_metadata>
getMetadata(Iterator begin, Iterator end, const std::string& namePrefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(namePrefix + std::to_string(i)));
    metadata.back().children_meta = getMetadata(
        c->child_begin(), c->child_end(), namePrefix + std::to_string(i));
    std::cout <<"The child metadata size is " << metadata.back().children_meta.size() << std::endl;
    i++;
  }
  return metadata;
}

// Recursively generate metadata using exact names from Velox RowType.
cudf::column_metadata getMetadataWithName(
    const facebook::velox::TypePtr& type,
    const std::string& name) {
  cudf::column_metadata meta(name);
  if (type->kind() == facebook::velox::TypeKind::ROW) {
    auto rowType = std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
    for (size_t i = 0; i < rowType->size(); ++i) {
      meta.children_meta.push_back(getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
    }
  } else if (type->kind() == facebook::velox::TypeKind::MAP) {
    // MAP is represented as LIST<STRUCT<key, value>> in cudf.
    // STRUCT child inside LIST
    cudf::column_metadata structMeta;
    // key / value names
    structMeta.children_meta.push_back(getMetadataWithName(type->childAt(0), "key"));
    structMeta.children_meta.push_back(getMetadataWithName(type->childAt(1), "value"));
    meta.children_meta.emplace_back(structMeta);
  } else if (type->kind() == facebook::velox::TypeKind::ARRAY) {
      // cudf::lists_column_view::child_column_index is 1, the first metadata is offsets
    meta.children_meta.emplace_back(cudf::column_metadata(name + "_offsets"));
    meta.children_meta.push_back(getMetadataWithName(type->childAt(0), "element"));
  }
  return meta;
}

std::vector<cudf::column_metadata> getMetadataWithName(
    const RowTypePtr& rowType) {
  std::vector<cudf::column_metadata> metadata;
  for (size_t i = 0; i < rowType->size(); ++i) {
    metadata.push_back(getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  std::cout <<"The input table metadata size " << metadata.size() << std::endl;
  return toVeloxColumn(table, pool, metadata, stream, mr);
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return toVeloxColumn(table, pool, metadata, stream, mr);
}

// New overload: Accepts a Velox TypePtr for recursive metadata construction.
RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Recursively generate metadata using Velox type names for all columns.
  // This assumes 'type' is a RowType and its children match the cudf table columns.
  auto rowType = std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
  VELOX_CHECK_NOT_NULL(rowType);
  auto metadata = getMetadataWithName(rowType);
  return toVeloxColumn(table, pool, metadata, stream, mr);
}

} // namespace with_arrow

namespace {
std::unique_ptr<cudf::table> toCudfTableArrow(
    const VectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask.
  ArrowOptions arrowOptions{true, true};
  ArrowArray arrowArray;
  exportToArrow(
      veloxTable,
      arrowArray,
      pool,
      arrowOptions);
  ArrowSchema arrowSchema;
  exportToArrow(
      veloxTable,
      arrowSchema,
      arrowOptions);
  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream, mr);

  // Release Arrow resources
  if (arrowArray.release) {
    arrowArray.release(&arrowArray);
  }
  if (arrowSchema.release) {
    arrowSchema.release(&arrowSchema);
  }
  return tbl;
}

/// Holds the CUDA stream and memory resource used throughout the Velox-to-cudf
/// direct conversion, avoiding repetitive parameter passing.
struct DispatchColumn {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  /// Copies the Velox null bitmask to a GPU device buffer. Velox and cudf both
  /// use a validity bitmask where bit=1 means valid (not null), so no
  /// conversion is needed. Returns an empty buffer if there are no nulls.
  std::unique_ptr<rmm::device_buffer> copyNullMask(
      const BaseVector& vector) const {
    const auto* rawNulls = vector.rawNulls();
    if (rawNulls == nullptr) {
      return std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    // Velox stores null bits as uint64_t words; compute byte size needed.
    auto numBytes = bits::nbytes(vector.size());
    auto mask = std::make_unique<rmm::device_buffer>(numBytes, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        mask->data(),
        rawNulls,
        numBytes,
        cudaMemcpyHostToDevice,
        stream.value()));
    return mask;
  }

  /// Returns the null count for a Velox vector. Uses Velox's cached value if
  /// available, otherwise counts from the raw bitmap.
  static cudf::size_type getNullCount(const BaseVector& vector) {
    auto nullCount = vector.getNullCount();
    if (nullCount.has_value()) {
      return static_cast<cudf::size_type>(nullCount.value());
    }
    const auto* rawNulls = vector.rawNulls();
    if (rawNulls == nullptr) {
      return 0;
    }
    return static_cast<cudf::size_type>(
        bits::countNulls(rawNulls, 0, vector.size()));
  }

  /// Copies a host vector of int32_t offsets to a GPU column.
  /// This is the common building block for strings, lists, and maps which all
  /// need an offsets column of shape [numRows + 1] on the device.
  std::unique_ptr<cudf::column> makeOffsetsColumn(
      const std::vector<int32_t>& offsets) const {
    rmm::device_buffer offsetBuf(
        offsets.size() * sizeof(int32_t), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsetBuf.data(),
        offsets.data(),
        offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.value()));
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets.size()),
        std::move(offsetBuf),
        rmm::device_buffer{0, stream, mr},
        0);
  }

  /// Builds a cudf INT32 offsets column from Velox's (offset, size) pairs.
  /// Velox stores separate rawOffsets and rawSizes arrays while cudf uses
  /// Arrow-style cumulative offsets [0, size0, size0+size1, ...]. The returned
  /// pair contains the offsets column and the total element count.
  std::pair<std::unique_ptr<cudf::column>, int32_t> makeOffsetsColumnFromSizes(
      const vector_size_t* rawSizes,
      vector_size_t numRows) const {
    std::vector<int32_t> cudfOffsets(numRows + 1);
    int32_t runningOffset = 0;
    for (int32_t i = 0; i < numRows; ++i) {
      cudfOffsets[i] = runningOffset;
      runningOffset += rawSizes[i];
    }
    cudfOffsets[numRows] = runningOffset;
    return {makeOffsetsColumn(cudfOffsets), runningOffset};
  }

  /// Reads a flat scalar column from a Velox FlatVector and creates a cudf
  /// column by copying the data buffers to GPU. This mirrors the pattern from
  /// GpuBufferBatchResizer::DispatchColumn::readFlatColumn.
  template <TypeKind Kind>
  std::unique_ptr<cudf::column> readColumn(
      const BaseVector& vector,
      const TypePtr& type) const {
    using T = typename TypeTraits<Kind>::NativeType;
    const auto numRows = vector.size();

    auto* flatVector = vector.as<FlatVector<T>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector for direct conversion");

    // Copy values buffer to GPU.
    const auto* rawValues = flatVector->rawValues();
    const size_t valueBytes = numRows * sizeof(T);
    rmm::device_buffer dataBuf(valueBytes, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dataBuf.data(),
        rawValues,
        valueBytes,
        cudaMemcpyHostToDevice,
        stream.value()));

    // Copy null mask to GPU.
    auto nullBuf = copyNullMask(vector);
    auto nullCount = getNullCount(vector);

    cudf::data_type cudfType{veloxToCudfTypeId(type)};
    return std::make_unique<cudf::column>(
        cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
  }

  /// Converts a Velox FlatVector<StringView> to a cudf STRING column. Builds
  /// offsets and chars buffers on CPU from the StringView array, then copies
  /// them to GPU. This mirrors GpuBufferBatchResizer's string handling.
  std::unique_ptr<cudf::column> readStringColumn(
      const BaseVector& vector) const {
    const auto numRows = vector.size();

    auto* flatVector = vector.as<FlatVector<StringView>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<StringView> for direct conversion");

    // Build offsets (int32_t[numRows+1]) and concatenated chars on CPU.
    std::vector<int32_t> offsets(numRows + 1);
    int32_t totalChars = 0;
    for (int32_t i = 0; i < numRows; ++i) {
      offsets[i] = totalChars;
      if (!vector.isNullAt(i)) {
        totalChars += static_cast<int32_t>(flatVector->valueAt(i).size());
      }
    }
    offsets[numRows] = totalChars;

    // Build contiguous chars buffer.
    std::vector<char> chars(totalChars);
    for (int32_t i = 0; i < numRows; ++i) {
      if (!vector.isNullAt(i)) {
        auto sv = flatVector->valueAt(i);
        std::memcpy(chars.data() + offsets[i], sv.data(), sv.size());
      }
    }

    auto offsetsColumn = makeOffsetsColumn(offsets);

    // Copy chars to GPU.
    rmm::device_buffer charsBuf(totalChars, stream, mr);
    if (totalChars > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          charsBuf.data(),
          chars.data(),
          totalChars,
          cudaMemcpyHostToDevice,
          stream.value()));
    }

    // Copy null mask and get null count.
    auto nullBuf = copyNullMask(vector);
    auto nullCount = getNullCount(vector);

    return cudf::make_strings_column(
        numRows,
        std::move(offsetsColumn),
        std::move(charsBuf),
        nullCount,
        std::move(*nullBuf));
  }

  template <typename Vector>
  bool isCompact(const Vector& vec) const {
    for (vector_size_t i = 1; i < vec.size(); ++i) {
      if (vec.offsetAt(i - 1) + vec.sizeAt(i - 1) != vec.offsetAt(i)) {
        return false;
      }
    }
    return true;
  }

  /// Converts a Velox ArrayVector to a cudf LIST column.
  /// Velox stores lists with separate offsets and sizes arrays, while cudf uses
  /// Arrow-style cumulative offsets (length = num_rows + 1). We compute the
  /// cumulative offsets from Velox's (offset, size) pairs and recursively
  /// convert the elements child vector.
  std::unique_ptr<cudf::column> readArrayColumn(
      const VectorPtr& vector,
      const TypePtr& type,
      memory::MemoryPool* pool) const {
    const auto numRows = vector->size();
    auto* arrayVector = vector->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(arrayVector, "Expected ArrayVector for direct conversion");
    if (!isCompact(*arrayVector) && arrayVector->elements()->type()->isPrimitiveType()) {
      std::cout <<"array is not compact and primitive type"<< std::endl;
      auto* rawOffsets = arrayVector->rawOffsets();
      auto* rawSizes = arrayVector->rawSizes();
      int32_t totalElements = 0;
      std::vector<int32_t> offsets(numRows + 1);
      offsets[0] = 0;
      // Step 1: build compact offsets
      for (int i = 0; i < numRows; ++i) {
        std::cout <<"offsets at index " << i << " " <<rawOffsets[i] << std::endl;
        std::cout <<"sizes at index " << i << " " <<rawSizes[i] << std::endl;
        totalElements += rawSizes[i];
        offsets[i + 1] = totalElements; 
      }
      // Step 2: gather elements into contiguous buffer
      auto elements = BaseVector::loadedVectorShared(arrayVector->elements());
      // Create indices for gather
      std::vector<vector_size_t> gatherIndices;
      gatherIndices.reserve(totalElements);
      for (int i = 0; i < numRows; ++i) {
        auto offset = rawOffsets[i];
        auto size = rawSizes[i];
        std::cout <<"offsets at index " << i << " " <<offset << std::endl;
        std::cout <<"sizes at index " << i << " " <<size << std::endl;
        for (int j = 0; j < size; ++j) {
           gatherIndices.push_back(offset + j);
        }
      }
      // Wrap indices into a Velox buffer/vector
      auto indexBuffer = AlignedBuffer::allocate<vector_size_t>( gatherIndices.size(), pool);
      std::memcpy( indexBuffer->asMutable<vector_size_t>(), gatherIndices.data(), gatherIndices.size() * sizeof(vector_size_t));
      // Use Velox dictionary to compact
      auto compactElements = BaseVector::wrapInDictionary( BufferPtr(nullptr), indexBuffer, gatherIndices.size(), elements);
      std::cout <<"compactElements is " << compactElements->toString()<< std::endl;
      // Now convert compacted child
      BaseVector::flattenVector(compactElements);
      auto childColumn = convertColumn(compactElements, type->childAt(0), pool);
      auto nullBuf = copyNullMask(*vector);
      auto nullCount = getNullCount(*vector);
      auto offsetsColumn = makeOffsetsColumn(offsets);
      return cudf::make_lists_column(
        numRows,
        std::move(offsetsColumn),
        std::move(childColumn),
        nullCount,
        std::move(*nullBuf),
        stream,
        mr);   
    }

    if (!isCompact(*arrayVector)) {
      std::cout <<"array is not compact"<< std::endl;
      // Non-compact: accumulate only for non-null rows, always set offset for each row.
      // Need to reorder all the row, fallback to with arrow conversion
      // Wrap the vector in a RowVector for Arrow conversion
      auto rowType = ROW({type});
      std::vector<VectorPtr> children = {vector};
      auto rowVector = std::make_shared<RowVector>(
          pool,
          rowType,
          nullptr,
          vector->size(),
          children);
      auto columns = toCudfTableArrow(rowVector, pool, stream, mr)->release();
      return std::move(columns[0]);
    }
    std::vector<int32_t> offsets(numRows + 1);
    offsets[0] = 0;
    // If compact, copy offsets directly and compute the last offset.
    std::memcpy(offsets.data(), arrayVector->rawOffsets(), sizeof(int32_t) * numRows);
    offsets[numRows] = arrayVector->offsetAt(numRows - 1) + arrayVector->sizeAt(numRows - 1);

    auto offsetsColumn = makeOffsetsColumn(offsets);
    auto elements = arrayVector->elements();
    auto childColumn = convertColumn(elements, type->childAt(0), pool);

    auto nullBuf = copyNullMask(*vector);
    auto nullCount = getNullCount(*vector);

    return cudf::make_lists_column(
        numRows,
        std::move(offsetsColumn),
        std::move(childColumn),
        nullCount,
        std::move(*nullBuf),
        stream,
        mr);
  }

  /// Converts a Velox RowVector to a cudf STRUCT column.
  /// Recursively converts each child column and assembles them into a struct.
  std::unique_ptr<cudf::column> readStructColumn(
      const BaseVector& vector,
      const TypePtr& type,
      memory::MemoryPool* pool) const {
    const auto numRows = vector.size();

    auto* rowVector = vector.as<RowVector>();
    VELOX_CHECK_NOT_NULL(
        rowVector, "Expected RowVector for direct struct conversion");

    const auto numChildren = rowVector->childrenSize();
    std::vector<std::unique_ptr<cudf::column>> childColumns;
    childColumns.reserve(numChildren);

    for (auto i = 0; i < numChildren; ++i) {
      const auto& child = rowVector->childAt(i);
      auto childType = type->childAt(i);
      childColumns.push_back(convertColumn(child, childType, pool));
    }

    std::cout << "childColumns size " << childColumns.size() << std::endl;

    auto nullBuf = copyNullMask(vector);
    auto nullCount = getNullCount(vector);

    return cudf::make_structs_column(
      numRows,
      std::move(childColumns),
      nullCount,
      std::move(*nullBuf),
      stream,
      mr);
  }

  /// Converts any Velox vector to a cudf column by dispatching on TypeKind.
  std::unique_ptr<cudf::column> convertColumn(
      const VectorPtr& vector,
      const TypePtr& type,
      memory::MemoryPool* pool) const {
    switch (type->kind()) {
      case TypeKind::BOOLEAN:
        return readBooleanColumn(*vector, type);
      case TypeKind::TINYINT:
        return readColumn<TypeKind::TINYINT>(*vector, type);
      case TypeKind::SMALLINT:
        return readColumn<TypeKind::SMALLINT>(*vector, type);
      case TypeKind::INTEGER:
        return readColumn<TypeKind::INTEGER>(*vector, type);
      case TypeKind::BIGINT:
        return readColumn<TypeKind::BIGINT>(*vector, type);
      case TypeKind::REAL:
        return readColumn<TypeKind::REAL>(*vector, type);
      case TypeKind::DOUBLE:
        return readColumn<TypeKind::DOUBLE>(*vector, type);
      case TypeKind::TIMESTAMP:
        return readTimestampColumn(*vector, type);
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        return readStringColumn(*vector);
      case TypeKind::ARRAY:
        return readArrayColumn(vector, type, pool);
      case TypeKind::ROW:
        return readStructColumn(*vector, type, pool);
      default:
        VELOX_FAIL(
            "Unsupported Velox type for direct cudf conversion: {}",
            type->toString());
    }
  }

 private:
  /// Specialization for BOOLEAN: Velox stores booleans as bit-packed uint64_t
  /// arrays, and cudf BOOL8 stores them as one byte per value (int8_t). We
  /// need to expand the bit-packed representation.
  std::unique_ptr<cudf::column> readBooleanColumn(
      const BaseVector& vector,
      const TypePtr& type) const {
    const auto numRows = vector.size();
    auto* flatVector = vector.as<FlatVector<bool>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<bool> for direct conversion");

    // Velox stores booleans bit-packed. cudf BOOL8 uses int8_t per value.
    // Expand bits to bytes on CPU, then copy to GPU.
    const auto* rawBits = flatVector->template rawValues<uint64_t>();
    std::vector<int8_t> expanded(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      expanded[i] = bits::isBitSet(rawBits, i) ? 1 : 0;
    }

    rmm::device_buffer dataBuf(numRows * sizeof(int8_t), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dataBuf.data(),
        expanded.data(),
        numRows * sizeof(int8_t),
        cudaMemcpyHostToDevice,
        stream.value()));

    auto nullBuf = copyNullMask(vector);
    auto nullCount = getNullCount(vector);

    cudf::data_type cudfType{veloxToCudfTypeId(type)};
    return std::make_unique<cudf::column>(
        cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
  }

  /// Specialization for TIMESTAMP: Velox uses a Timestamp struct (seconds +
  /// nanos, 16 bytes) while cudf TIMESTAMP_NANOSECONDS uses a single int64_t.
  /// We convert each Timestamp to nanoseconds since epoch via toNanos().
  std::unique_ptr<cudf::column> readTimestampColumn(
      const BaseVector& vector,
      const TypePtr& type) const {
    const auto numRows = vector.size();
    auto* flatVector = vector.as<FlatVector<Timestamp>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<Timestamp> for direct conversion");

    // Convert Velox Timestamp (seconds + nanos) to int64_t nanoseconds.
    const auto* rawTimestamps = flatVector->rawValues();
    std::vector<int64_t> nanosValues(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      if (!vector.isNullAt(i)) {
        nanosValues[i] = rawTimestamps[i].toNanos();
      } else {
        nanosValues[i] = 0;
      }
    }

    rmm::device_buffer dataBuf(numRows * sizeof(int64_t), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dataBuf.data(),
        nanosValues.data(),
        numRows * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream.value()));

    auto nullBuf = copyNullMask(vector);
    auto nullCount = getNullCount(vector);

    cudf::data_type cudfType{veloxToCudfTypeId(type)};
    return std::make_unique<cudf::column>(
        cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
  }
};

} // namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numColumns = veloxTable->childrenSize();

  // Flatten dictionary and constant encodings to get flat vectors.
  // This matches what the Arrow path does with ArrowOptions{true, true}.
  VectorPtr flattenTarget = veloxTable;
  BaseVector::flattenVector(flattenTarget);
  auto flatRow = std::dynamic_pointer_cast<RowVector>(flattenTarget);

  const auto& rowType = flatRow->type()->as<TypeKind::ROW>();

  DispatchColumn dispatcher{stream, mr};

  std::vector<std::unique_ptr<cudf::column>> cudfColumns;
  cudfColumns.reserve(numColumns);

  for (auto i = 0; i < numColumns; ++i) {
    auto child = flatRow->childAt(i);
    const auto& colType = rowType.childAt(i);
    auto column = dispatcher.convertColumn(child, colType, pool);
    cudfColumns.push_back(std::move(column));
  }

  return std::make_unique<cudf::table>(std::move(cudfColumns));
}

} // namespace facebook::velox::cudf_velox
