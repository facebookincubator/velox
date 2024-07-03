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

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include "velox/vector/tests/utils/VectorMaker.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <nvtx3/nvtx3.hpp>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "VeloxCudfInterop.h"

namespace facebook::velox::cudf_velox {

// Velox type to CUDF type
/*
template<TypeKind::KIND kind>
struct VeloxToCudfType {
  using type = typename TypeTraits<kind>::NativeType;
  static constexpr cudf::type_id id = cudf::type_id::EMPTY;
  //cudf::type_to_id<type>();
};

#define VELOX_TO_CUDF_TYPE(CUDF_KIND, VELOX_KIND)   \
template <>                                         \
struct TypeTraits<TypeKind::VELOX_KIND> {           \
using type = typename TypeTraits<kind>::NativeType; \
static constexpr cudf::type_id id = CUDF_KIND;      \
};

VELOX_TO_CUDF_TYPE(cudf::type_id::BOOL8, BOOLEAN)
VELOX_TO_CUDF_TYPE(cudf::type_id::INT8, TINYINT)
VELOX_TO_CUDF_TYPE(cudf::type_id::INT16, SMALLINT)
VELOX_TO_CUDF_TYPE(cudf::type_id::INT32, INTEGER)
VELOX_TO_CUDF_TYPE(cudf::type_id::INT64, BIGINT)
VELOX_TO_CUDF_TYPE(cudf::type_id::FLOAT32, REAL)
VELOX_TO_CUDF_TYPE(cudf::type_id::FLOAT64, DOUBLE)
VELOX_TO_CUDF_TYPE(cudf::type_id::STRING, VARCHAR)
VELOX_TO_CUDF_TYPE(cudf::type_id::STRING, VARBINARY)
VELOX_TO_CUDF_TYPE(cudf::type_id::TIMESTAMP_NANOSECONDS, TIMESTAMP)
VELOX_TO_CUDF_TYPE(cudf::type_id::DURATION_DAYS, DATE)
// VELOX_TO_CUDF_TYPE(IntervalDayTime, INTERVAL_DAY_TIME)
VELOX_TO_CUDF_TYPE(cudf::type_id::DECIMAL64, SHORT_DECIMAL)
VELOX_TO_CUDF_TYPE(cudf::type_id::DECIMAL128, LONG_DECIMAL)
// VELOX_TO_CUDF_TYPE(Array, ARRAY)
// VELOX_TO_CUDF_TYPE(Map, MAP)
// VELOX_TO_CUDF_TYPE(Row, ROW)
// VELOX_TO_CUDF_TYPE(Opaque, OPAQUE)
// VELOX_TO_CUDF_TYPE(UnKnown, UNKNOWN)
*/

cudf::type_id velox_to_cudf_type_id(TypeKind kind) {
  switch (kind) {
    case TypeKind::BOOLEAN:
      return cudf::type_id::BOOL8;
    case TypeKind::TINYINT:
      return cudf::type_id::INT8;
    case TypeKind::SMALLINT:
      return cudf::type_id::INT16;
    case TypeKind::INTEGER:
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
    // case TypeKind::ARRAY: return cudf::type_id::EMPTY;
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    case TypeKind::ROW:
      return cudf::type_id::STRUCT;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      return cudf::type_id::EMPTY;
  }
}

TypeKind cudf_to_velox_type_id(cudf::type_id kind) {
  switch (kind) {
    case cudf::type_id::BOOL8:
      return TypeKind::BOOLEAN;
    case cudf::type_id::INT8:
      return TypeKind::TINYINT;
    case cudf::type_id::INT16:
      return TypeKind::SMALLINT;
    case cudf::type_id::INT32:
      return TypeKind::INTEGER;
    case cudf::type_id::INT64:
      return TypeKind::BIGINT;
    case cudf::type_id::FLOAT32:
      return TypeKind::REAL;
    case cudf::type_id::FLOAT64:
      return TypeKind::DOUBLE;
    case cudf::type_id::STRING:
      return TypeKind::VARCHAR;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return TypeKind::TIMESTAMP;
    // TODO: DATE is now a logical type
    // case cudf::type_id::DURATION_DAYS: return TypeKind::DATE;
    // case cudf::type_id::EMPTY: return TypeKind::INTERVAL_DAY_TIME;
    // TODO: DECIMAL is now a logical type
    // case cudf::type_id::DECIMAL64: return TypeKind::SHORT_DECIMAL;
    // case cudf::type_id::DECIMAL128: return TypeKind::LONG_DECIMAL;
    // case cudf::type_id::EMPTY: return TypeKind::ARRAY;
    // case cudf::type_id::EMPTY: return TypeKind::MAP;
    case cudf::type_id::STRUCT:
      return TypeKind::ROW;
    // case cudf::type_id::EMPTY: return TypeKind::OPAQUE;
    // case cudf::type_id::EMPTY: return TypeKind::UNKNOWN;
    default:
      return TypeKind::UNKNOWN;
  }
}

// Convert a Velox vector to a CUDF column
struct copy_to_device {
  rmm::cuda_stream_view stream;
  template <typename T>
  static constexpr bool is_supported() {
    return cudf::is_rep_layout_compatible<T>();
  }
  // Fixed width types
  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(VectorPtr& h_vec) const {
    auto velox_data = h_vec->as<FlatVector<T>>();
    auto velox_data_ptr = velox_data->rawValues();
    cudf::host_span<const T> velox_host_span(
        velox_data_ptr, int{h_vec->size()});
    auto d_v = cudf::detail::make_device_uvector_sync(
        velox_host_span, stream, rmm::mr::get_current_device_resource());
    return std::make_unique<cudf::column>(
        std::move(d_v), rmm::device_buffer{}, 0);
  }

  template <
      typename T,
      typename... Args,
      std::enable_if_t<not is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(Args... args) const {
    CUDF_FAIL("Unsupported type for to_cudf conversion");
  }
};

// Row vector to table
// Vector to column
// template<bool is_table=true>
std::unique_ptr<cudf::table> to_cudf_table(const RowVectorPtr& leftBatch) {
  NVTX3_FUNC_RANGE();
  // cudf type dispatcher to copy data from velox vector to cudf column
  using cudf_col_ptr = std::unique_ptr<cudf::column>;
  std::vector<cudf_col_ptr> cudf_columns;
  auto copier = copy_to_device{cudf::get_default_stream()};
  for (auto& h_vec : leftBatch->children()) {
    auto cudf_kind =
        cudf::data_type{velox_to_cudf_type_id(h_vec->type()->kind())};
    auto cudf_column = cudf::type_dispatcher(cudf_kind, copier, h_vec);
    cudf_columns.push_back(std::move(cudf_column));
  }
  return std::make_unique<cudf::table>(std::move(cudf_columns));
}

// Convert a CUDF column to a Velox vector
struct copy_to_host {
  rmm::cuda_stream_view stream;
  memory::MemoryPool* pool_;

  template <typename T>
  static constexpr bool is_supported() {
    // return cudf::is_rep_layout_compatible<T>();
    return cudf::is_numeric<T>() and not std::is_same<T, bool>::value;
  }
  // Fixed width types
  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  VectorPtr operator()(TypePtr velox_type, cudf::column_view const& col) const {
    // auto velox_col = BaseVector::create(velox_type, col.size(), pool_);
    // auto velox_col = BaseVector::create<FlatVector<T> >(velox_type,
    // col.size(), pool_);
    auto velox_col = test::VectorMaker{pool_}.flatVector<T>(col.size());
    // auto velox_data = velox_col->as<FlatVector<T>>();
    auto velox_data_ptr = velox_col->mutableRawValues();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        velox_data_ptr,
        col.data<T>(),
        col.size() * sizeof(T),
        cudaMemcpyDefault,
        stream.value()));
    stream.synchronize();
    return velox_col;
  }

  template <
      typename T,
      typename... Args,
      std::enable_if_t<not is_supported<T>()>* = nullptr>
  VectorPtr operator()(Args... args) const {
    CUDF_FAIL("Unsupported type for to_velox conversion");
  }
};

VectorPtr to_velox_column(
    const cudf::column_view& col,
    memory::MemoryPool* pool) {
  NVTX3_FUNC_RANGE();
  auto velox_kind = cudf_to_velox_type_id(col.type().id());
  auto velox_type = createScalarType(velox_kind);
  // cudf type dispatcher to copy data from cudf column to velox vector
  auto copier = copy_to_host{cudf::get_default_stream(), pool};
  return cudf::type_dispatcher(col.type(), copier, velox_type, col);
}

RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    std::string name_prefix) {
  NVTX3_FUNC_RANGE();
  std::vector<VectorPtr> children;
  std::vector<std::string> names;
  for (auto& col : table) {
    auto velox_col = to_velox_column(col, pool);
    children.push_back(std::move(velox_col));
    names.push_back(name_prefix + std::to_string(names.size()));
  }
  auto vcol =
      test::VectorMaker{pool}.rowVector(std::move(names), std::move(children));
  return vcol;
}

} // namespace facebook::velox::cudf_velox
                                          