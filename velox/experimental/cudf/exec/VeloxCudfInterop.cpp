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
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/strings/strings_column_view.hpp>

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

namespace {

void setVeloxNulls(
    const cudf::column_view& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    BufferPtr& nulls) {

  auto size = col.size();

  auto* dst = nulls->asMutable<uint64_t>(); // Velox uses bitmask (uint64_t aligned)

  const cudf::bitmask_type* srcMask = nullptr;
  rmm::device_buffer tempMask;

  // 🔥 Handle offset (same as Arrow code)
  if (col.offset() > 0) {
    tempMask = cudf::copy_bitmask(col, stream, mr);
    srcMask = reinterpret_cast<const cudf::bitmask_type*>(tempMask.data());
  } else {
    srcMask = col.null_mask();
  }

  size_t bytes = ((size + 7) / 8);

  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst,
      reinterpret_cast<const uint8_t*>(srcMask),
      bytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
}

void setVeloxNulls(
    const cudf::column_view& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    VectorPtr& flat) {
  if (!col.has_nulls()) {
    return;
  }
  setVeloxNulls(col, stream, mr, flat->mutableNulls(col.size()));
}

template <typename T>
VectorPtr makeFlatVectorFromDevice(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = col.size();
  auto vec = BaseVector::create(type, size, pool);
  auto flat = vec->asFlatVector<T>();

  setVeloxNulls(col, stream, mr, vec);
  T* rawValues = flat->template mutableRawValues<T>();

  CUDF_CUDA_TRY(cudaMemcpyAsync(
    rawValues,
    col.data<T>(),
    size * sizeof(T),
    cudaMemcpyDeviceToHost,
    stream.value()));
  return vec;
}

VectorPtr makeStringFlatVectorFromDevice(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    const TypePtr& type) {
    cudf::strings_column_view scv(col);
    auto size = scv.size();

    auto vec = BaseVector::create(VARCHAR(), size, pool);
    auto flat = vec->template asFlatVector<StringView>();

    setVeloxNulls(col, stream, mr, vec);

    // Copy offsets and chars from device to host
    std::vector<int32_t> offsets(size + 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsets.data(),
      scv.offsets().data<int32_t>(),
      (size + 1) * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    auto bufferSize = offsets[size];
    auto rawBuffer = flat->getRawStringBufferWithSpace(bufferSize, true);
    auto charsCount = scv.chars_size(stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      rawBuffer,
      scv.chars_begin(stream),
      bufferSize,
      cudaMemcpyDeviceToHost,
      stream.value()));


    // Fill StringView values
    for (auto i = 0; i < size; ++i) {
      if (!flat->isNullAt(i)) {
        flat->setNoCopy(i, StringView(rawBuffer + offsets[i], offsets[i + 1] - offsets[i]));
      }
    }

    return vec;
}

template <>
VectorPtr makeFlatVectorFromDevice<Timestamp>(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = col.size();
  auto vec = BaseVector::create(type, size, pool);
  auto flat = vec->asFlatVector<Timestamp>();

  setVeloxNulls(col, stream, mr, vec);
  Timestamp* rawValues = flat->template mutableRawValues<Timestamp>();
  // Get device data (int64 nanoseconds)
  auto dData = col.data<int64_t>();
    // Copy to host (simplest version; optimize to GPU execution later if needed)
  std::vector<int64_t> hData(size);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hData.data(),
      dData,
      sizeof(int64_t) * size,
      cudaMemcpyDeviceToHost,
      stream.value()));

  // Convert ns → (seconds, nanos)
  for (size_t i = 0; i < size; ++i) {
    int64_t ns = hData[i];

    int64_t seconds = ns / 1'000'000'000;
    int64_t nanos = ns % 1'000'000'000;

    // Handle negative timestamps correctly
    if (nanos < 0) {
      nanos += 1'000'000'000;
      --seconds;
    }

    rawValues[i] = Timestamp(seconds, nanos);
  }
  
  return vec;
}

template<>
VectorPtr makeFlatVectorFromDevice<StringView>(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return makeStringFlatVectorFromDevice(col, pool, stream, mr, type);
}

template<>
VectorPtr makeFlatVectorFromDevice<bool>(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto size = col.size();
  auto vec = BaseVector::create(BOOLEAN(), size, pool);
  auto flat = vec->asFlatVector<bool>();

  // 🔥 GPU: pack bools → bitmask
  auto [mask, _] =
      cudf::detail::bools_to_mask(col, stream, mr);
  // allocate Velox buffer
  auto raw = flat->mutableRawValues<uint64_t>();
  size_t bytes = ((size + 7) / 8);
  // copy packed bits (device → host)
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      raw,
      mask->data(),
      bytes,
      cudaMemcpyDeviceToHost,
      stream.value()));

  // nulls
  setVeloxNulls(col, stream, mr, vec);
  return vec;
}

VectorPtr toVeloxColumn(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const cudf::column_metadata& meta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

VectorPtr makeRowVectorFromDevice(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const cudf::column_metadata& meta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<VectorPtr> children;
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (size_t i = 0; i < col.num_children(); ++i) {
    auto child = toVeloxColumn(
        col.child(i),
        pool,
        meta.children_meta.size() > i ? meta.children_meta[i]
                                  : cudf::column_metadata{},
        stream,
        mr);

    children.push_back(child);

    std::string name =
        (meta.children_meta.size() > i && !meta.children_meta[i].name.empty())
        ? meta.children_meta[i].name
        : "c" + std::to_string(i);

    names.push_back(name);
    types.push_back(child->type());
  }
  BufferPtr nulls;
  if (col.has_nulls()) {
    nulls = AlignedBuffer::allocate<uint64_t>(
          bits::nwords(col.size()),
          pool);;
    setVeloxNulls(col, stream, mr, nulls);
  }

  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  return std::make_shared<RowVector>(
    pool,
    rowType,
    nulls,
    col.size(),
    std::move(children));
}

VectorPtr makeArrayVectorFromDevice(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const cudf::column_metadata& meta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto offsetsCol = col.child(0);
  auto childCol = col.child(1);

  int size = col.size();
  BufferPtr nulls;
  if (col.has_nulls()) {
    nulls = AlignedBuffer::allocate<uint64_t>(
          bits::nwords(col.size()),
          pool);;
    setVeloxNulls(col, stream, mr, nulls);  
  }

  std::vector<int32_t> offsets(size + 1);

  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsets.data(),
      offsetsCol.data<int32_t>(),
      (size + 1) * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  auto offsetsBuf = AlignedBuffer::allocate<vector_size_t>(size, pool);
  auto sizesBuf = AlignedBuffer::allocate<vector_size_t>(size, pool);

  auto* rawOffsets = offsetsBuf->asMutable<vector_size_t>();
  auto* rawSizes = sizesBuf->asMutable<vector_size_t>();

  for (int i = 0; i < size; ++i) {
    rawOffsets[i] = offsets[i];
    rawSizes[i] = offsets[i + 1] - offsets[i];
  }

  auto childVec = toVeloxColumn(
      childCol,
      pool,
      meta.children_meta.empty() ? cudf::column_metadata{} : meta.children_meta[0],
      stream,
      mr);

  return std::make_shared<ArrayVector>(
      pool,
      ARRAY(childVec->type()),
      nulls,
      size,
      offsetsBuf,
      sizesBuf,
      childVec);
}

VectorPtr toVeloxColumn(
    const cudf::column_view& col,
    memory::MemoryPool* pool,
    const cudf::column_metadata& meta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (col.type().id()) {
    case cudf::type_id::INT8:
      return makeFlatVectorFromDevice<int8_t>(
          col, pool, TINYINT(), stream, mr);
    case cudf::type_id::INT16:
      return makeFlatVectorFromDevice<int16_t>(
          col, pool, SMALLINT(), stream, mr);

    case cudf::type_id::INT32:
      return makeFlatVectorFromDevice<int32_t>(
          col, pool, INTEGER(), stream, mr);

    case cudf::type_id::INT64:
      return makeFlatVectorFromDevice<int64_t>(
          col, pool, BIGINT(), stream, mr);

    case cudf::type_id::FLOAT32:
      return makeFlatVectorFromDevice<float>(
          col, pool, REAL(), stream, mr);

    case cudf::type_id::FLOAT64:
      return makeFlatVectorFromDevice<double>(
          col, pool, DOUBLE(), stream, mr);

    case cudf::type_id::BOOL8:
      return makeFlatVectorFromDevice<bool>(
          col, pool, BOOLEAN(), stream, mr);
    case cudf::type_id::STRING:
      return makeFlatVectorFromDevice<StringView>(
          col, pool, VARCHAR(), stream, mr);
    case cudf::type_id::TIMESTAMP_DAYS:
      return makeFlatVectorFromDevice<int32_t>(
          col, pool, DATE(), stream, mr);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return makeFlatVectorFromDevice<Timestamp>(
          col, pool, TIMESTAMP(), stream, mr);
    // ========================
    // LIST → ArrayVector
    // ========================
    case cudf::type_id::LIST:
      return makeArrayVectorFromDevice(col, pool, meta, stream, mr);
    // ========================
    // STRUCT → RowVector
    // ========================
    case cudf::type_id::STRUCT:
      return makeRowVectorFromDevice(col, pool, meta, stream, mr);

    default:
      VELOX_FAIL("Unsupported cudf type: {}", static_cast<int>(col.type().id()));
  }
}

// ========================
// Entry point
// ========================
RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<VectorPtr> children;
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (size_t i = 0; i < table.num_columns(); ++i) {
    const auto& col = table.column(i);

    std::string name =
        (i < metadata.size() && !metadata[i].name.empty())
        ? metadata[i].name
        : "c" + std::to_string(i);

    auto vec = toVeloxColumn(
        col,
        pool,
        i < metadata.size() ? metadata[i] : cudf::column_metadata{},
        stream,
        mr);

    children.push_back(vec);
    names.push_back(name);
    types.push_back(vec->type());
  }
  stream.synchronize();

  auto rowType = std::make_shared<RowType>(std::move(names), std::move(types));

  return std::make_shared<RowVector>(
      pool,
      rowType,
      nullptr,
      table.num_rows(),
      std::move(children));
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
      auto* rawOffsets = arrayVector->rawOffsets();
      auto* rawSizes = arrayVector->rawSizes();
      int32_t totalElements = 0;
      std::vector<int32_t> offsets(numRows + 1);
      offsets[0] = 0;
      // Step 1: build compact offsets
      for (int i = 0; i < numRows; ++i) {
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
        for (int j = 0; j < size; ++j) {
           gatherIndices.push_back(offset + j);
        }
      }
      // Wrap indices into a Velox buffer/vector
      auto indexBuffer = AlignedBuffer::allocate<vector_size_t>( gatherIndices.size(), pool);
      std::memcpy( indexBuffer->asMutable<vector_size_t>(), gatherIndices.data(), gatherIndices.size() * sizeof(vector_size_t));
      // Use Velox dictionary to compact
      auto compactElements = BaseVector::wrapInDictionary( BufferPtr(nullptr), indexBuffer, gatherIndices.size(), elements);
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
