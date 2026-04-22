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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/exec/util/InteropUtil.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

#include <cstdlib>
#include <cstring>

namespace facebook::velox::cudf_velox {

cudf::type_id veloxToCudfTypeId(const TypePtr& type) {
  // Legacy helper retained for compatibility. Note: returning cudf::type_id
  // discards decimal scale; prefer veloxToCudfDataType when scale matters.
  return veloxToCudfDataType(type).id();
}

cudf::data_type veloxToCudfDataType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::data_type{cudf::type_id::BOOL8};
    case TypeKind::TINYINT:
      return cudf::data_type{cudf::type_id::INT8};
    case TypeKind::SMALLINT:
      return cudf::data_type{cudf::type_id::INT16};
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
      }
      return cudf::data_type{cudf::type_id::INT32};
    case TypeKind::BIGINT:
      // BIGINT is used for both INT64 and DECIMAL64
      if (type->isDecimal()) {
        auto const decimalType =
            std::dynamic_pointer_cast<const ShortDecimalType>(type);
        VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
        auto const cudfScale = numeric::scale_type{-decimalType->scale()};
        return cudf::data_type{cudf::type_id::DECIMAL64, cudfScale};
      }
      return cudf::data_type{cudf::type_id::INT64};
    case TypeKind::HUGEINT: {
      // HUGEINT is used only for DECIMAL128
      // per facebookincubator/velox PR 4434 (May 2, 2023)
      // although see commented-out HUGEINT -> DURATION_DAYS below
      VELOX_CHECK(
          type->isDecimal(), "HUGEINT should only be used for DECIMAL128");
      auto const decimalType =
          std::dynamic_pointer_cast<const LongDecimalType>(type);
      VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
      auto const cudfScale = numeric::scale_type{-decimalType->scale()};
      return cudf::data_type{cudf::type_id::DECIMAL128, cudfScale};
    }
    case TypeKind::REAL:
      return cudf::data_type{cudf::type_id::FLOAT32};
    case TypeKind::DOUBLE:
      return cudf::data_type{cudf::type_id::FLOAT64};
    case TypeKind::VARCHAR:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::VARBINARY:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::TIMESTAMP:
      return cudf::data_type{CudfConfig::getInstance().timestampUnit};
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    case TypeKind::ARRAY:
      return cudf::data_type{cudf::type_id::LIST};
    case TypeKind::ROW:
      return cudf::data_type{cudf::type_id::STRUCT};
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      break;
  }
  CUDF_FAIL(
      "Unsupported Velox type: " +
      std::string(TypeKindName::toName(type->kind())));
  return cudf::data_type{cudf::type_id::EMPTY};
}

namespace with_arrow {
namespace {

void setArrowFormatBackToVarbinary(ArrowSchema* schema, const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ROW: {
      if (schema->n_children != static_cast<int64_t>(type->size())) {
        break;
      }
      for (size_t i = 0; i < type->size(); ++i) {
        setArrowFormatBackToVarbinary(schema->children[i], type->childAt(i));
      }
      break;
    }
    case TypeKind::VARBINARY: {
      // Replace any format string with "z" to indicate VARBINARY.
      static constexpr const char* kVarbinaryArrowFormat = "z";
      if (schema->format != nullptr) {
        std::free(const_cast<char*>(schema->format));
        schema->format = nullptr;
      }
      const size_t bufferLen = std::strlen(kVarbinaryArrowFormat) + 1;
      auto* buffer = static_cast<char*>(std::malloc(bufferLen));
      VELOX_CHECK_NOT_NULL(buffer);
      std::memcpy(buffer, kVarbinaryArrowFormat, bufferLen);
      schema->format = buffer;
      break;
    }
    default:
      break;
  }
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    const RowTypePtr* outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // To avoid ownership issues, we make copies of the Arrow objects
  // returned from cuDF as unique_ptrs, then mark the originals as
  // released so their destructors don't try to free the resources.
  //
  // A better solution would be alternative versions of the cuDF
  // to_arrow_host functions that return Arrow objects by value
  // or populate objects passed by reference, but that would require
  // changes to cuDF.
  //
  // seves 1/17/26

  auto arrowDeviceArray = cudf::to_arrow_host(table, stream, mr);
  ArrowArray arrayCopy = arrowDeviceArray->array;
  arrowDeviceArray->array.release = nullptr;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  ArrowSchema schemaCopy = *arrowSchema;
  arrowSchema->release = nullptr;

  // Override schema type recursively with outputType if provided. This is
  // needed for some types like VARBINARY which are exported as STRING (the
  // format is overridden to "z" when the exportVarbinaryAsString option is set
  // to true in the exportToArrow() call) because cuDF does not have a VARBINARY
  // type. This code implements the other side of the conversion, to change the
  // format back to "z" so that the data re-imports as VARBINARY.
  if (outputType) {
    setArrowFormatBackToVarbinary(&schemaCopy, *outputType);
  }

  auto veloxTable = importFromArrowAsOwner(schemaCopy, arrayCopy, pool);
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

// Recursively generate metadata using exact names from Velox RowType.
cudf::column_metadata getMetadataWithName(
    const facebook::velox::TypePtr& type,
    const std::string& name) {
  cudf::column_metadata meta(name);
  if (type->kind() == facebook::velox::TypeKind::ROW) {
    auto rowType =
        std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
    for (size_t i = 0; i < rowType->size(); ++i) {
      meta.children_meta.push_back(
          getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
    }
  } else if (type->kind() == facebook::velox::TypeKind::ARRAY) {
    // cudf::lists_column_view::child_column_index is 1, the first metadata is
    // offsets
    meta.children_meta.emplace_back(cudf::column_metadata(name + "_offsets"));
    meta.children_meta.push_back(
        getMetadataWithName(type->childAt(0), "element"));
  }
  return meta;
}

std::vector<cudf::column_metadata> getMetadataWithName(
    const RowTypePtr& rowType) {
  std::vector<cudf::column_metadata> metadata;
  for (size_t i = 0; i < rowType->size(); ++i) {
    metadata.push_back(
        getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
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
  return toVeloxColumn(table, pool, metadata, nullptr, stream, mr);
}

// New overload: Accepts a Velox TypePtr for recursive metadata construction.
RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Recursively generate metadata using Velox type names for all columns.
  // This assumes 'type' is a RowType and its children match the cudf table
  // columns.
  auto rowType =
      std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
  VELOX_CHECK_NOT_NULL(rowType);
  auto metadata = getMetadataWithName(rowType);
  return toVeloxColumn(table, pool, metadata, &rowType, stream, mr);
}

} // namespace with_arrow

namespace {
std::unique_ptr<cudf::table> toCudfTableArrow(
    const VectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone) {
  TimestampUnit unit;
  switch (CudfConfig::getInstance().timestampUnit) {
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      unit = TimestampUnit::kNano;
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      unit = TimestampUnit::kMicro;
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      unit = TimestampUnit::kMilli;
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
      unit = TimestampUnit::kSecond;
      break;
    default:
      VELOX_UNSUPPORTED();
  }
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask. Also, libcudf does not support Arrow binary, so we export
  // VARBINARY as UTF-8.
  ArrowOptions arrowOptions{
      .flattenDictionary = true,
      .flattenConstant = true,
      .timestampUnit = unit,
      .timestampTimeZone = timestampTimeZone,
      .exportVarbinaryAsString = true,
      .useDecimalTypeWidth = true};
  ArrowArray arrowArray;
  exportToArrow(veloxTable, arrowArray, pool, arrowOptions);
  ArrowSchema arrowSchema;
  exportToArrow(veloxTable, arrowSchema, arrowOptions);
  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream, mr);

  // Synchronize before releasing Arrow resources.  cudf::from_arrow uses
  // cudaMemcpyBatchAsync (CUDA 13.0+) with cudaMemcpySrcAccessOrderStream,
  // which defers reading the host source buffers until the stream reaches
  // each copy.  The Arrow arrays must therefore stay alive until the stream
  // has executed those copies.
  stream.synchronize();

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
  std::optional<std::string> timestampTimeZone;

  /// Copies the Velox null bitmask to a GPU device buffer. Velox and cudf both
  /// use a validity bitmask where bit=1 means valid (not null), so no
  /// conversion is needed. Returns an empty buffer if there are no nulls.
  inline std::unique_ptr<rmm::device_buffer> copyNullMask(
      const BaseVector& input,
      int32_t nullCount) const {
    if (nullCount == 0) {
      return std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    // Velox stores null bits as uint64_t words; compute byte size needed.
    const auto* rawNulls = input.rawNulls();

    auto const numRows = static_cast<cudf::size_type>(input.size());
    auto const paddedWords = cudf::bitmask_allocation_size_bytes(numRows) /
        sizeof(cudf::bitmask_type);
    auto const copySize = bits::nbytes(numRows);

    auto mask =
        rmm::device_uvector<cudf::bitmask_type>(paddedWords, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        mask.data(), rawNulls, copySize, cudaMemcpyDefault, stream.value()));
    return std::make_unique<rmm::device_buffer>(mask.release());
  }

  /// Returns the null count for a Velox vector. Uses Velox's cached value if
  /// available, otherwise counts from the raw bitmap.
  static cudf::size_type getNullCount(const BaseVector& vector) {
    if (vector.nulls()) {
      return 0;
    }
    if (auto nullCount = vector.getNullCount()) {
      return *nullCount;
    }
    return BaseVector::countNulls(vector.nulls(), vector.size()); 
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
    auto nullCount = getNullCount(vector);
    auto nullBuf = copyNullMask(vector, nullCount);

    auto cudfType = veloxToCudfDataType(type);
    return std::make_unique<cudf::column>(
        cudfType, numRows, std::move(dataBuf), std::move(*nullBuf), nullCount);
  }

  // Converts a Velox FlatVector<StringView> to a cudf STRING column.
  // Because we don't know the StringView is in which stringBuffers, so we
  // cannot implement as Arrow to no copy. Velox must copy all the CPU buffers
  // to GPU buffers, and then make string column.
  std::unique_ptr<cudf::column> readStringColumn(
      const BaseVector& vector) const {
    const auto numRows = vector.size();

    auto* flatVector = vector.as<FlatVector<StringView>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<StringView> for direct conversion");

    // Build offsets (int32_t[numRows+1]) and concatenated chars on CPU.
    // Use shared_ptr for automatic lifetime management with CUDA stream.
    auto offsets = std::make_shared<std::vector<int32_t>>(numRows + 1);
    int64_t totalChars = 0;
    for (int32_t i = 0; i < numRows; ++i) {
      (*offsets)[i] = static_cast<int32_t>(totalChars);
      if (!vector.isNullAt(i)) {
        totalChars += flatVector->valueAt(i).size();
      }
    }
    VELOX_CHECK_LE(
        totalChars,
        std::numeric_limits<int32_t>::max(),
        "Total string length exceeds int32 max");
    (*offsets)[numRows] = totalChars;

    // Build contiguous chars buffer.
    auto chars = std::make_shared<std::vector<char>>(totalChars);
    for (int32_t i = 0; i < numRows; ++i) {
      if (!vector.isNullAt(i)) {
        auto sv = flatVector->valueAt(i);
        std::memcpy(chars->data() + (*offsets)[i], sv.data(), sv.size());
      }
    }

    rmm::device_buffer offsetBuf(offsets->size() * sizeof(int32_t), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsetBuf.data(),
        offsets->data(),
        offsets->size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.value()));

    // Schedule cleanup of offsets buffer after copy completes
    CUDF_CUDA_TRY(cudaLaunchHostFunc(
        stream.value(),
        [](void* p) {
          auto sp = static_cast<std::shared_ptr<void>*>(p);
          delete sp; // refcount drops here
        },
        new std::shared_ptr<void>(offsets)));

    auto offsetsColumn = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets->size()),
        std::move(offsetBuf),
        rmm::device_buffer{0, stream, mr},
        0);

    // Copy chars to GPU.
    rmm::device_buffer charsBuf(totalChars, stream, mr);
    if (totalChars > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          charsBuf.data(),
          chars->data(),
          totalChars,
          cudaMemcpyHostToDevice,
          stream.value()));

      // Schedule cleanup of chars buffer after copy completes
      CUDF_CUDA_TRY(cudaLaunchHostFunc(
          stream.value(),
          [](void* p) {
            auto sp = static_cast<std::shared_ptr<void>*>(p);
            delete sp; // refcount drops here
          },
          new std::shared_ptr<void>(chars)));
    }

    // No need to synchronize - cudaLaunchHostFunc ensures buffers stay alive

    // Copy null mask and get null count.
    auto nullCount = getNullCount(vector);
    auto nullBuf = copyNullMask(vector, nullCount);

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
  /// Arrow-style cumulative offsets (length = numRows + 1). We compute the
  /// cumulative offsets from Velox's (offset, size) pairs and recursively
  /// convert the elements child vector.
  std::unique_ptr<cudf::column> readArrayColumn(
      const VectorPtr& vector,
      const TypePtr& type,
      memory::MemoryPool* pool) const {
    const auto numRows = vector->size();
    auto* arrayVector = vector->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(
        arrayVector, "Expected ArrayVector for direct conversion");
    if (!isCompact(*arrayVector) &&
        arrayVector->elements()->type()->isPrimitiveType()) {
      auto* rawOffsets = arrayVector->rawOffsets();
      auto* rawSizes = arrayVector->rawSizes();

      // Step 1: build compact offsets using GPU-accelerated prefix sum
      auto [offsetsColumn, totalElements] = makeOffsetsColumnFromSizes(rawSizes, numRows, stream, mr);

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
      auto indexBuffer =
          AlignedBuffer::allocate<vector_size_t>(gatherIndices.size(), pool);
      std::memcpy(
          indexBuffer->asMutable<vector_size_t>(),
          gatherIndices.data(),
          gatherIndices.size() * sizeof(vector_size_t));
      // Use Velox dictionary to compact
      auto compactElements = BaseVector::wrapInDictionary(
          BufferPtr(nullptr), indexBuffer, gatherIndices.size(), elements);
      // Now convert compacted child
      BaseVector::flattenVector(compactElements);
      auto childColumn = convertColumn(compactElements, type->childAt(0), pool);
      auto nullCount = getNullCount(*vector);
      auto nullBuf = copyNullMask(*vector, nullCount);
      return cudf::make_lists_column(
          numRows,
          std::move(offsetsColumn),
          std::move(childColumn),
          nullCount,
          std::move(*nullBuf));
    }

    if (!isCompact(*arrayVector)) {
      // Non-compact: accumulate only for non-null rows, always set offset for
      // each row. Need to reorder all the row, fallback to with arrow
      // conversion Wrap the vector in a RowVector for Arrow conversion
      auto rowType = ROW({type});
      std::vector<VectorPtr> children = {vector};
      auto rowVector = std::make_shared<RowVector>(
          pool, rowType, nullptr, vector->size(), children);
      auto columns =
          toCudfTableArrow(rowVector, pool, stream, mr, timestampTimeZone)
              ->release();
      return std::move(columns[0]);
    }
    // If compact, use GPU-accelerated offsets generation from sizes
    auto offsetsColumn =
        makeOffsetsColumnFromSizes(arrayVector->rawSizes(), numRows, stream, mr).first;
    auto elements = arrayVector->elements();
    auto childColumn = convertColumn(elements, type->childAt(0), pool);

    auto nullCount = getNullCount(*vector);
    auto nullBuf = copyNullMask(*vector, nullCount);
    return cudf::make_lists_column(
        numRows,
        std::move(offsetsColumn),
        std::move(childColumn),
        nullCount,
        std::move(*nullBuf));
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

    auto nullCount = getNullCount(vector);
    auto nullBuf = copyNullMask(vector, nullCount);

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
      case TypeKind::HUGEINT:
        return readColumn<TypeKind::HUGEINT>(*vector, type);
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
  /// arrays, and cudf BOOL8 stores them as one byte per value (int8_t).
  /// Optimized GPU-accelerated version using cuDF's mask_to_bools to expand
  /// bit-packed representation directly on GPU.
  std::unique_ptr<cudf::column> readBooleanColumn(
      const BaseVector& vector,
      const TypePtr& type) const {
    const auto numRows = vector.size();
    auto* flatVector = vector.as<FlatVector<bool>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<bool> for direct conversion");
    // Copy null mask to device (bit-packed boolean data)
    auto nullCount = getNullCount(vector);
    auto nullBuf = copyNullMask(vector, nullCount);
    auto dataBuffer = flatVector->rawValues<uint8_t>();
    auto const dataWords = cudf::num_bitmask_words(numRows);
    auto const copySize = bits::nbytes(numRows);

    auto data = rmm::device_uvector<cudf::bitmask_type>(dataWords, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        data.data(), dataBuffer, copySize, cudaMemcpyDefault, stream.value()));

    // Use cudf::mask_to_bools to expand bit-packed mask to BOOL8 column
    auto boolColumn = cudf::mask_to_bools(
        static_cast<cudf::bitmask_type*>(data.data()),
        0,
        static_cast<cudf::size_type>(numRows),
        stream,
        mr);

    // Set null mask for nulls in the original vector
    boolColumn->set_null_mask(std::move(*nullBuf), nullCount);
    return boolColumn;
  }

  /// Fully GPU-accelerated version: copies Timestamp array to GPU and uses
  /// Thrust transform to convert to nanoseconds directly on device.
  std::unique_ptr<cudf::column> readTimestampColumn(
      const BaseVector& vector,
      const TypePtr& type) const {
    const auto numRows = vector.size();
    auto* flatVector = vector.as<FlatVector<Timestamp>>();
    VELOX_CHECK_NOT_NULL(
        flatVector, "Expected FlatVector<Timestamp> for direct conversion");

    // Copy Timestamp array to device
    const auto* rawTimestamps = flatVector->rawValues();
    auto dTimestamps = rmm::device_uvector<Timestamp>(numRows, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dTimestamps.data(),
        rawTimestamps,
        numRows * sizeof(Timestamp),
        cudaMemcpyHostToDevice,
        stream.value()));

    // Copy null mask to device
    auto nullCount = getNullCount(vector);
    auto nullBuf = copyNullMask(vector, nullCount);
    auto dMask = static_cast<cudf::bitmask_type*>(nullBuf->data());

    auto dValues = rmm::device_uvector<int64_t>(numRows, stream, mr);

    convertTimestamps(
        dTimestamps.data(),
        dMask,
        dValues.data(),
        numRows,
        CudfConfig::getInstance().timestampUnit,
        timestampTimeZone ? std::optional<std::string_view>(*timestampTimeZone)
                          : std::nullopt,
        stream);

    auto cudfType = veloxToCudfDataType(type);
    return std::make_unique<cudf::column>(
        cudfType, numRows, dValues.release(), std::move(*nullBuf), nullCount);
  }
};

} // namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone) {
  const auto numColumns = veloxTable->childrenSize();
  const auto numRows = veloxTable->size();

  // Flatten dictionary and constant encodings to get flat vectors.
  // This matches what the Arrow path does with ArrowOptions{true, true}.
  for (auto& child : veloxTable->children()) {
    facebook::velox::BaseVector::flattenVector(child);
    if (child->isLazy()) {
      child = child->as<facebook::velox::LazyVector>()->loadedVectorShared();
      VELOX_DCHECK_NOT_NULL(child);
    }
    // In case of output from Limit, RowVector size can be smaller than its children size.
    if (child->size() > numRows) {
      child = child->slice(0, numRows);
    }
  }

  const auto& rowType = veloxTable->type()->as<TypeKind::ROW>();

  DispatchColumn dispatcher{stream, mr, std::move(timestampTimeZone)};

  std::vector<std::unique_ptr<cudf::column>> cudfColumns;
  cudfColumns.reserve(numColumns);

  for (auto i = 0; i < numColumns; ++i) {
    auto child = veloxTable->childAt(i);
    const auto& colType = rowType.childAt(i);
    auto column = dispatcher.convertColumn(child, colType, pool);
    cudfColumns.push_back(std::move(column));
  }
  stream.synchronize();

  return std::make_unique<cudf::table>(std::move(cudfColumns));
}

namespace with_arrow {
std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone) {
  return toCudfTableArrow(veloxTable, pool, stream, mr, timestampTimeZone);
  }
}

} // namespace facebook::velox::cudf_velox
