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
#include "velox/type/StringView.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
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

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask.
  ArrowOptions arrowOptions{true, true};
  ArrowArray arrowArray;
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
      arrowArray,
      pool,
      arrowOptions);
  ArrowSchema arrowSchema;
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
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
    i++;
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

} // namespace with_arrow

namespace {

/// Copies the Velox null bitmask to a GPU device buffer. Velox and cudf both
/// use a validity bitmask where bit=1 means valid (not null), so no conversion
/// is needed. Returns an empty buffer if there are no nulls.
std::unique_ptr<rmm::device_buffer> copyNullMask(
    const BaseVector& vector,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto* rawNulls = vector.rawNulls();
  if (rawNulls == nullptr) {
    return std::make_unique<rmm::device_buffer>(0, stream, mr);
  }
  // Velox stores null bits as uint64_t words; compute byte size needed.
  auto numBytes = bits::nbytes(vector.size());
  auto mask = std::make_unique<rmm::device_buffer>(numBytes, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      mask->data(), rawNulls, numBytes, cudaMemcpyHostToDevice, stream.value()));
  return mask;
}

/// Returns the null count for a Velox vector. Uses Velox's cached value if
/// available, otherwise counts from the raw bitmap.
cudf::size_type getNullCount(const BaseVector& vector) {
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

/// Reads a flat scalar column from a Velox FlatVector and creates a cudf
/// column by copying the data buffers to GPU. This mirrors the pattern from
/// GpuBufferBatchResizer::DispatchColumn::readFlatColumn.
template <TypeKind Kind>
std::unique_ptr<cudf::column> readFlatColumn(
    const BaseVector& vector,
    cudf::type_id typeId,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  using T = typename TypeTraits<Kind>::NativeType;
  const auto numRows = vector.size();

  auto* flatVector = vector.as<FlatVector<T>>();
  VELOX_CHECK_NOT_NULL(flatVector, "Expected FlatVector for direct conversion");

  // Copy values buffer to GPU.
  const auto* rawValues = flatVector->rawValues();
  const size_t valueBytes = numRows * sizeof(T);
  rmm::device_buffer dataBuf(valueBytes, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dataBuf.data(), rawValues, valueBytes, cudaMemcpyHostToDevice, stream.value()));

  // Copy null mask to GPU.
  auto nullBuf = copyNullMask(vector, stream, mr);
  auto nullCount = getNullCount(vector);

  cudf::data_type cudfType{typeId};
  return std::make_unique<cudf::column>(
      cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
}

/// Specialization for BOOLEAN: Velox stores booleans as bit-packed uint64_t
/// arrays, and cudf BOOL8 stores them as one byte per value (int8_t). We need
/// to expand the bit-packed representation.
template <>
std::unique_ptr<cudf::column> readFlatColumn<TypeKind::BOOLEAN>(
    const BaseVector& vector,
    cudf::type_id typeId,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = vector.size();
  auto* flatVector = vector.as<FlatVector<bool>>();
  VELOX_CHECK_NOT_NULL(flatVector, "Expected FlatVector<bool> for direct conversion");

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

  auto nullBuf = copyNullMask(vector, stream, mr);
  auto nullCount = getNullCount(vector);

  cudf::data_type cudfType{typeId};
  return std::make_unique<cudf::column>(
      cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
}

/// Specialization for TIMESTAMP: Velox uses a Timestamp struct (seconds +
/// nanos, 16 bytes) while cudf TIMESTAMP_NANOSECONDS uses a single int64_t.
/// We convert each Timestamp to nanoseconds since epoch via toNanos().
template <>
std::unique_ptr<cudf::column> readFlatColumn<TypeKind::TIMESTAMP>(
    const BaseVector& vector,
    cudf::type_id typeId,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
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

  auto nullBuf = copyNullMask(vector, stream, mr);
  auto nullCount = getNullCount(vector);

  cudf::data_type cudfType{typeId};
  return std::make_unique<cudf::column>(
      cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
}

/// Converts a Velox FlatVector<StringView> to a cudf STRING column. Builds
/// offsets and chars buffers on CPU from the StringView array, then copies
/// them to GPU. This mirrors GpuBufferBatchResizer's string handling.
std::unique_ptr<cudf::column> readStringColumn(
    const BaseVector& vector,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = vector.size();

  if (numRows == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

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

  // Copy offsets to GPU and create offsets column.
  rmm::device_buffer offsetBuf(
      offsets.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetBuf.data(),
      offsets.data(),
      offsets.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(numRows + 1),
      std::move(offsetBuf),
      rmm::device_buffer{0, stream, mr},
      0);

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
  auto nullBuf = copyNullMask(vector, stream, mr);
  auto nullCount = getNullCount(vector);

  return cudf::make_strings_column(
      numRows,
      std::move(offsetsColumn),
      std::move(charsBuf),
      nullCount,
      std::move(*nullBuf));
}

/// Specialization for VARCHAR.
template <>
std::unique_ptr<cudf::column> readFlatColumn<TypeKind::VARCHAR>(
    const BaseVector& vector,
    cudf::type_id /*typeId*/,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return readStringColumn(vector, stream, mr);
}

/// Specialization for VARBINARY.
template <>
std::unique_ptr<cudf::column> readFlatColumn<TypeKind::VARBINARY>(
    const BaseVector& vector,
    cudf::type_id /*typeId*/,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return readStringColumn(vector, stream, mr);
}

} // namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto numRows = veloxTable->size();
  const auto numColumns = veloxTable->childrenSize();

  // Flatten dictionary and constant encodings to get flat vectors.
  // This matches what the Arrow path does with ArrowOptions{true, true}.
  std::vector<VectorPtr> flatChildren(numColumns);
  for (auto i = 0; i < numColumns; ++i) {
    flatChildren[i] = BaseVector::loadedVectorShared(veloxTable->childAt(i));
    BaseVector::flattenVector(flatChildren[i]);
  }

  const auto& rowType = veloxTable->type()->as<TypeKind::ROW>();

  std::vector<std::unique_ptr<cudf::column>> cudfColumns;
  cudfColumns.reserve(numColumns);

  for (auto i = 0; i < numColumns; ++i) {
    const auto& child = flatChildren[i];
    const auto& colType = rowType.childAt(i);
    auto cudfTypeId = veloxToCudfTypeId(colType);

    auto column = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        readFlatColumn, colType->kind(), *child, cudfTypeId, stream, mr);
    cudfColumns.push_back(std::move(column));
  }

  return std::make_unique<cudf::table>(std::move(cudfColumns));
}

} // namespace facebook::velox::cudf_velox
