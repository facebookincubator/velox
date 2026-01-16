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
#pragma once

#include <cuda_runtime.h>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <iostream>
#include <numeric>

#include "TableWithNames.hpp"

/// @brief Computes the data storage size of a table. The table cannot contain
/// nested structures.
/// @param input_table The input table.
/// @return The size in bytes.
int64_t calculate_table_size(cudf::table_view input_table);

/// @brief Returns the index of to the named column given the list of columns.
/// @param col_name The name of the column
/// @param columns The list of the columns that are present in the table.
/// @return A column index.
int32_t get_col_idx(
    std::string col_name,
    std::vector<std::string> const& columns);

/// @brief Debug utility for dumping the contents of a string column.
/// @param str_column_view The string column view to be dumped.
/// @param stream The cuda stream.
std::vector<std::string> getStringCol(
    const cudf::strings_column_view& str_column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream);

/// @brief Debug utility for dumping the contents of a string column.
/// @param str_column_view The string column view to be dumped.
/// @param stream The cuda stream.
void dumpStringCol(
    const cudf::strings_column_view& str_column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream);

/// @brief Template function for retrieving the contents of a fixed-size column.
/// @param column_view The column view to be dumped.
/// @param max_rows The maximum number of rows to be retrieved.
/// @param stream The cude stream.
template <typename T>
std::vector<T> getColVector(
    const cudf::column_view& column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream) {
  max_rows = column_view.size() < max_rows ? column_view.size() : max_rows;
  const T* ptr_data = column_view.template data<T>();
  auto host_vec = cudf::detail::make_host_vector(
      cudf::device_span<T const>(ptr_data, max_rows), stream);
  std::vector<T> vec(max_rows);
  std::copy(host_vec.begin(), host_vec.end(), vec.begin());
  return vec;
}

/// @brief Template function for dumping the contents of a fixed-size column
/// @param column_view The column view to be dumped.
/// @param max_rows The maximum number of rows to be dumped.
/// @param stream The cude stream.
template <typename T>
void dumpCol(
    const cudf::column_view& column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream) {
  std::cout << "column view has: " << column_view.size() << " elements."
            << std::endl;

  auto const h_data = getColVector<T>(column_view, max_rows, stream);
  for (const auto& element : h_data) {
    std::cout << "Data: " << element << std::endl;
  }
}
