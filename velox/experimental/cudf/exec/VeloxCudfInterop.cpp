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
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/arrow/Bridge.h"

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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <functional>
#include <numeric>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

namespace facebook::velox::cudf_velox {

namespace with_arrow {

std::unique_ptr<cudf::table> to_cudf_table(
    const facebook::velox::RowVectorPtr& veloxTable, // BaseVector or RowVector?
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
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

RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    rmm::cuda_stream_view stream) {
  auto arrowDeviceArray = cudf::to_arrow_host(table, stream);
  auto& arrowArray = arrowDeviceArray->array;

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

template <typename Iterator>
std::vector<cudf::column_metadata>
get_metadata(Iterator begin, Iterator end, const std::string& name_prefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(name_prefix + std::to_string(i)));
    metadata.back().children_meta = get_metadata(
        c->child_begin(), c->child_end(), name_prefix + std::to_string(i));
    i++;
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string name_prefix,
    rmm::cuda_stream_view stream) {
  auto metadata = get_metadata(table.begin(), table.end(), name_prefix);
  return to_velox_column(table, pool, metadata, stream);
}

RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return to_velox_column(table, pool, metadata, stream);
}

} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
