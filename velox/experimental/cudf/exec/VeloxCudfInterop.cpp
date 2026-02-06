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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

#include <cstdlib>
#include <cstring>

namespace facebook::velox::cudf_velox {

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
        auto const decimalType = std::dynamic_pointer_cast<const ShortDecimalType>(type);
        VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
        auto const cudfScale = numeric::scale_type{-decimalType->scale()};
        return cudf::data_type{cudf::type_id::DECIMAL64, cudfScale};
      }
      return cudf::data_type{cudf::type_id::INT64};
    case TypeKind::HUGEINT: {
      // HUGEINT is used only for DECIMAL128
      // per facebookincubator/velox PR 4434 (May 2, 2023)
      // although see commented-out HUGEINT -> DURATION_DAYS below
      VELOX_CHECK(type->isDecimal(), "HUGEINT should only be used for DECIMAL128");
      auto const decimalType = std::dynamic_pointer_cast<const LongDecimalType>(type);
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
      return cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
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
  CUDF_FAIL("Unsupported Velox type: " + std::string(TypeKindName::toName(type->kind())));
  return cudf::data_type{cudf::type_id::EMPTY};
}

namespace with_arrow {

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask.
  ArrowOptions arrowOptions{true, true};
  // libcudf does not support Arrow binary; export VARBINARY as UTF-8.
  arrowOptions.exportVarbinaryAsString = true;
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
  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream);

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

void setArrowSchemaFormat(ArrowSchema* schema, const char* format) {
  if (!schema) {
    return;
  }
  if (schema->format != nullptr) {
    std::free(const_cast<char*>(schema->format));
    schema->format = nullptr;
  }
  if (format != nullptr) {
    const size_t size = std::strlen(format) + 1;
    auto* buffer = static_cast<char*>(std::malloc(size));
    VELOX_CHECK_NOT_NULL(buffer);
    std::memcpy(buffer, format, size);
    schema->format = buffer;
  }
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    const RowTypePtr* expectedType,
    rmm::cuda_stream_view stream) {

  // To avoid ownership issues, we make copies of the Arrow objects
  // returned from CUDF as unique_ptrs, then mark the originals as
  // released so their destructors don't try to free the resources.
  //
  // A better solution would be alternative versions of the CUDF
  // to_arrow_host functions that return Arrow objects by value
  // or populate objects passed by reference, but that would require
  // changes to CUDF.
  //
  // seves 1/17/26

  auto arrowDeviceArray = cudf::to_arrow_host(table, stream);
  ArrowArray arrayCopy = arrowDeviceArray->array;
  arrowDeviceArray->array.release = nullptr;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  ArrowSchema schemaCopy = *arrowSchema;
  arrowSchema->release = nullptr;

  if (expectedType) {
    auto applyExpectedArrowFormat =
        [&](auto&& self, ArrowSchema* schema, const TypePtr& type) -> void {
      if (!schema || !schema->format) {
        return;
      }
      switch (type->kind()) {
        case TypeKind::ROW: {
          if (schema->n_children != static_cast<int64_t>(type->size())) {
            return;
          }
          for (size_t i = 0; i < type->size(); ++i) {
            self(self, schema->children[i], type->childAt(i));
          }
          return;
        }
        case TypeKind::ARRAY: {
          if (schema->n_children < 1) {
            return;
          }
          self(self, schema->children[0], type->childAt(0));
          return;
        }
        case TypeKind::MAP: {
          if (schema->n_children < 1) {
            return;
          }
          auto* entry = schema->children[0];
          if (!entry || entry->n_children < 2) {
            return;
          }
          self(self, entry->children[0], type->childAt(0));
          self(self, entry->children[1], type->childAt(1));
          return;
        }
        case TypeKind::VARBINARY: {
          setArrowSchemaFormat(schema, "z");
          return;
        }
        default:
          return;
      }
    };
    applyExpectedArrowFormat(
        applyExpectedArrowFormat, &schemaCopy, *expectedType);
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

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, nullptr, stream);
}

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    const facebook::velox::RowTypePtr& expectedType,
    std::string namePrefix,
    rmm::cuda_stream_view stream) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, &expectedType, stream);
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return toVeloxColumn(table, pool, metadata, nullptr, stream);
}

} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
