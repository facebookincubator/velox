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
#include "velox/vector/arrow/Bridge.h"

#include "velox/vector/tests/utils/VectorMaker.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <nvtx3/nvtx3.hpp>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <functional>
#include <numeric>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

namespace facebook::velox::cudf_velox {

namespace {

template <typename Value, typename Functor, typename... Ts>
constexpr decltype(auto)
vector_encoding_dispatcher(VectorPtr vec, Functor f, Ts&&... args) {
  using facebook::velox::VectorEncoding::Simple;
  switch (vec->encoding()) {
    case Simple::FLAT:
      return f(vec->as<FlatVector<Value>>(), std::forward<Ts>(args)...);
    case Simple::DICTIONARY:
      return f(vec->as<DictionaryVector<Value>>(), std::forward<Ts>(args)...);
    default: {
      if (cudfDebugEnabled()) {
        std::cout << "Unsupported Velox encoding: " << vec->encoding()
                  << std::endl;
      }
      CUDF_FAIL("Unsupported Velox encoding");
    }
  }
}

// TODO: dispatch other duration/timestamp types!
template <typename T>
using cudf_storage_type_t = std::conditional_t<
    std::is_same_v<cudf::timestamp_D, T>,
    cudf::timestamp_D::rep,
    cudf::device_storage_type_t<T>>;

} // namespace

cudf::type_id velox_to_cudf_type_id(const TypePtr& type) {
  if (cudfDebugEnabled()) {
    std::cout << "Converting Velox type " << type->toString() << " to cudf"
              << std::endl;
  }
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
    // case TypeKind::ARRAY: return cudf::type_id::EMPTY;
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    case TypeKind::ROW:
      return cudf::type_id::STRUCT;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      CUDF_FAIL("Unsupported Velox type");
      return cudf::type_id::EMPTY;
  }
}

TypePtr cudf_type_id_to_velox_type(cudf::type_id type_id) {
  switch (type_id) {
    case cudf::type_id::BOOL8:
      return BOOLEAN();
    case cudf::type_id::INT8:
      return TINYINT();
    case cudf::type_id::INT16:
      return SMALLINT();
    case cudf::type_id::INT32:
      return INTEGER();
    case cudf::type_id::INT64:
      return BIGINT();
    case cudf::type_id::FLOAT32:
      return REAL();
    case cudf::type_id::FLOAT64:
      return DOUBLE();
    case cudf::type_id::STRING:
      return VARCHAR();
    case cudf::type_id::TIMESTAMP_DAYS:
      return DATE();
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return TIMESTAMP();
    // TODO: DATE is now a logical type
    // case cudf::type_id::DURATION_DAYS: return ???;
    // case cudf::type_id::EMPTY: return TypeKind::INTERVAL_DAY_TIME;
    // TODO: DECIMAL is now a logical type
    // case cudf::type_id::DECIMAL64: return TypeKind::SHORT_DECIMAL;
    // case cudf::type_id::DECIMAL128: return TypeKind::LONG_DECIMAL;
    // case cudf::type_id::EMPTY: return TypeKind::ARRAY;
    // case cudf::type_id::EMPTY: return TypeKind::MAP;
    // case cudf::type_id::STRUCT:
    //   // TODO: Need parametric type support?
    //   return ROW();
    // case cudf::type_id::EMPTY: return TypeKind::OPAQUE;
    // case cudf::type_id::EMPTY: return TypeKind::UNKNOWN;
    default:
      return UNKNOWN();
  }
}

// Convert a Velox vector to a CUDF column
struct copy_to_device {
  rmm::cuda_stream_view stream;

  // Fixed width types
  template <
      typename T,
      std::enable_if_t<cudf::is_rep_layout_compatible<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(VectorPtr const& h_vec) const {
    VELOX_CHECK_NOT_NULL(h_vec);
    using velox_T = cudf_storage_type_t<T>;
    if (cudfDebugEnabled()) {
      std::cout << "Converting fixed width column" << std::endl;
      std::cout << "Encoding: " << h_vec->encoding() << std::endl;
      std::cout << "Type: " << h_vec->type()->toString() << std::endl;
      std::cout << "velox_T: " << typeid(velox_T{}).name() << std::endl;
    }
    auto velox_data = h_vec->as<FlatVector<velox_T>>();
    VELOX_CHECK_NOT_NULL(velox_data);
    auto velox_data_ptr = velox_data->rawValues();
    cudf::host_span<const velox_T> velox_host_span(
        velox_data_ptr, int{h_vec->size()});
    auto d_v = cudf::detail::make_device_uvector_sync(
        velox_host_span, stream, rmm::mr::get_current_device_resource());
    return std::make_unique<cudf::column>(
        std::move(d_v), rmm::device_buffer{}, 0);
  }

  // Strings
  template <
      typename T,
      std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(VectorPtr const& h_vec) const {
    if (cudfDebugEnabled()) {
      std::cout << "Converting string column" << std::endl;
    }

    auto const num_rows = h_vec->size();
    auto h_offsets = std::vector<int64_t>(num_rows + 1);
    h_offsets[0] = 0;
    auto make_offsets = [&](auto const& vec) {
      VELOX_CHECK_NOT_NULL(vec);
      if (cudfDebugEnabled()) {
        std::cout << "Starting offset calculation" << std::endl;
      }
      for (auto i = 0; i < num_rows; i++) {
        h_offsets[i + 1] = h_offsets[i] + vec->valueAt(i).size();
      }
    };
    vector_encoding_dispatcher<StringView>(h_vec, make_offsets);

    auto d_offsets = cudf::detail::make_device_uvector_sync(
        h_offsets, stream, rmm::mr::get_current_device_resource());

    auto chars_size = h_offsets[num_rows];
    auto h_chars = std::vector<char>(chars_size);

    auto make_chars = [&](auto vec) {
      VELOX_CHECK_NOT_NULL(vec);
      for (auto i = 0; i < num_rows; i++) {
        auto const string_view = vec->valueAt(i);
        auto const size = string_view.size();
        auto const offset = h_offsets[i];
        std::copy(
            string_view.data(),
            string_view.data() + size,
            h_chars.begin() + offset);
      }
    };
    vector_encoding_dispatcher<StringView>(h_vec, make_chars);

    auto d_chars = cudf::detail::make_device_uvector_sync(
        h_chars, stream, rmm::mr::get_current_device_resource());

    return cudf::make_strings_column(
        num_rows,
        std::make_unique<cudf::column>(
            std::move(d_offsets), rmm::device_buffer{}, 0),
        d_chars.release(),
        0,
        rmm::device_buffer{});
  }

  template <
      typename T,
      typename... Args,
      std::enable_if_t<
          not(cudf::is_rep_layout_compatible<T>() or
              std::is_same_v<T, cudf::string_view>)>* = nullptr>
  std::unique_ptr<cudf::column> operator()(VectorPtr const& h_vec) const {
    if (cudfDebugEnabled()) {
      std::string error_message = "Unsupported type for to_cudf conversion: ";
      error_message += h_vec->type()->toString();
      std::cout << error_message << std::endl;
    }
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
  for (auto const& h_vec : leftBatch->children()) {
    auto cudf_kind = cudf::data_type{velox_to_cudf_type_id(h_vec->type())};
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
    auto velox_buffer = AlignedBuffer::allocate<T>(col.size(), pool_);
    auto velox_col = std::make_shared<FlatVector<T>>(
        pool_,
        velox_type,
        nullptr,
        col.size(),
        velox_buffer,
        std::vector<BufferPtr>{});
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
  auto velox_type = cudf_type_id_to_velox_type(col.type().id());
  if (cudfDebugEnabled()) {
    std::cout << "Converting to_velox_column: " << velox_type->toString()
              << std::endl;
  }
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

namespace with_arrow {

std::unique_ptr<cudf::table> to_cudf_table(
    const facebook::velox::RowVectorPtr& veloxTable, // BaseVector or RowVector?
    facebook::velox::memory::MemoryPool* pool) {
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
  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray);

  // Release Arrow resources
  if (arrowArray.release) {
    arrowArray.release(&arrowArray);
  }
  if (arrowSchema.release) {
    arrowSchema.release(&arrowSchema);
  }
  return tbl;
}

void to_signed_int_format(char* format) {
  VELOX_CHECK_NOT_NULL(format);
  switch (format[0]) {
    case 'C':
      format[0] = 'c';
      break;
    case 'S':
      format[0] = 's';
      break;
    case 'I':
      format[0] = 'i';
      break;
    case 'L':
      format[0] = 'l';
      break;
    default:
      return;
  }
  printf(
      "Warning: arrowSchema.format: %s, unsigned is treated as signed indices\n",
      format);
}

// Changes all unsigned indices to signed indices for dictionary columns from
// cudf which uses unsigned indices, but velox uses signed indices.
void fix_dictionary_indices(ArrowSchema& arrowSchema) {
  if (arrowSchema.dictionary != nullptr) {
    to_signed_int_format(const_cast<char*>(arrowSchema.format));
    fix_dictionary_indices(*arrowSchema.dictionary);
  }
  for (size_t i = 0; i < arrowSchema.n_children; ++i) {
    VELOX_CHECK_NOT_NULL(arrowSchema.children[i]);
    fix_dictionary_indices(*arrowSchema.children[i]);
  }
}

facebook::velox::RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string name_prefix) {
  auto arrowDeviceArray = cudf::to_arrow_host(table);
  auto& arrowArray = arrowDeviceArray->array;

  std::vector<cudf::column_metadata> metadata;
  for (auto i = 0; i < table.num_columns(); i++) {
    metadata.push_back(cudf::column_metadata(name_prefix + std::to_string(i)));
  }
  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  // Hack to convert unsigned indices to signed indices for dictionary columns
  fix_dictionary_indices(*arrowSchema);

  auto veloxTable = importFromArrowAsOwner(*arrowSchema, arrowArray, pool);
  // BaseVector to RowVector
  auto casted_ptr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(casted_ptr);
  return casted_ptr;
}
} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
