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
#include "common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include "TableWithNames.hpp"

/**
 * Calculate the table sizes in bytes.
 *
 * Note: This function only supports tables with fixed-width and string columns.
 */
int64_t calculate_table_size(cudf::table_view input_table) {
  int64_t table_size = 0;

  int i = 0;
  for (auto& current_column : input_table) {
    try {
      cudf::data_type dtype = current_column.type();
      if (cudf::is_fixed_width(dtype)) {
        table_size += (cudf::size_of(dtype) * current_column.size());
      } else {
        assert(dtype.id() == cudf::type_id::STRING);
        cudf::strings_column_view str_column_view(current_column);
        cudaStream_t s = 0;
        table_size += str_column_view.chars_size(rmm::cuda_stream_view(s));
        // also take into account the offsets.
        auto offset_column = str_column_view.offsets();
        table_size +=
            (cudf::size_of(offset_column.type()) * offset_column.size());
      }
    } catch (std::exception& e) {
      std::cout << "Error computing size of column " << i << ": " << e.what()
                << std::endl;
    }
    ++i;
  }
  return table_size;
}

/// @brief Returns the index of to the named column given the list of columns.
/// @param col_name The name of the column
/// @param columns The list of the columns that are present in the table.
/// @return A column index.
int32_t get_col_idx(
    std::string col_name,
    std::vector<std::string> const& columns) {
  return std::distance(
      columns.begin(), std::find(columns.begin(), columns.end(), col_name));
}

std::vector<std::string> getStringCol(
    const cudf::strings_column_view& str_column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream) {
  max_rows =
      str_column_view.size() < max_rows ? str_column_view.size() : max_rows;
  auto offset_view = str_column_view.offsets();
  const cudf::size_type* ptr_offsets_data =
      offset_view.template data<cudf::size_type>();
  auto const h_offsets = cudf::detail::make_host_vector(
      cudf::device_span<cudf::size_type const>(ptr_offsets_data, max_rows + 1),
      stream);
  const cudf::size_type* host_offsets = h_offsets.data();

  auto const total_num_bytes = std::distance(
      str_column_view.chars_begin(stream), str_column_view.chars_end(stream));
  char const* ptr_all_bytes = str_column_view.chars_begin(stream);
  // copy the bytes to host
  auto const h_bytes = cudf::detail::make_host_vector(
      cudf::device_span<char const>(ptr_all_bytes, total_num_bytes), stream);
  const char* str_ptr = h_bytes.data();

  std::vector<std::string> str_vec;
  for (cudf::size_type i = 0; i < max_rows; ++i) {
    std::string str(str_ptr + host_offsets[i], str_ptr + host_offsets[i + 1]);
    str_vec.push_back(str);
  }
  return str_vec; // rely on the compiler's Return-Value-Optimization to avoid a
                  // vector copy.
}

void dumpStringCol(
    const cudf::strings_column_view& str_column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream) {
  std::cout << "string column view has: " << str_column_view.size()
            << " elements." << std::endl;

  auto const str_vec = getStringCol(str_column_view, max_rows, stream);
  for (auto& str : str_vec) {
    std::cout << str << std::endl;
  }
}
