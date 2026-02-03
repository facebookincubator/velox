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

#include <cudf/table/table.hpp>
#include <vector>

/**
 * @brief A class to represent a table with column names attached
 */
class table_with_names {
 public:
  table_with_names(
      std::unique_ptr<cudf::table> tbl,
      std::vector<std::string> col_names)
      : tbl(std::move(tbl)), col_names(col_names){};
  /**
   * @brief Return the table view
   */
  [[nodiscard]] cudf::table_view table() const;

  /**
   * @brief Return the table view, compatible with cudf::table.
   */
  [[nodiscard]] cudf::table_view view() const;

  /**
   * @brief Return the column view for a given column name
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::column_view column(std::string const& col_name) const;
  /**
   * @param Return the column names of the table
   */
  [[nodiscard]] std::vector<std::string> const& column_names() const;
  /**
   * @brief Translate a column name to a column index
   *
   * @param col_name The name of the column
   */
  [[nodiscard]] cudf::size_type column_id(std::string const& col_name) const;
  /**
   * @brief Append a column to the table
   *
   * @param col The column to append
   * @param col_name The name of the appended column
   */
  table_with_names& append(
      std::unique_ptr<cudf::column>& col,
      std::string const& col_name);
  /**
   * @brief Replace a column to the table
   *
   * @param col The column to replace the existing column with the given name.
   * @param col_name The name of the column to replace
   */
  table_with_names& replace(
      std::unique_ptr<cudf::column>& col,
      std::string const& col_name);
  /**
   * @brief Rename a column to the table
   *
   * @param col The column to rename
   * @param col_name The new name of the column
   */
  table_with_names& rename(
      std::string const& col_name,
      std::string const& col_new_name);
  /**
   * @brief Drop a column to the table
   *
   * @param col_name The name of the column to drop
   */
  table_with_names& drop(std::string const& col_name);
  /**
   * @brief Generic projection: Re-organizes the table such that it contains
   * only the columns specified in the new_col_names vector. Columns that are
   * not contained in this vector are dropped. If a name from new_vector doesn't
   * exist, an exception is thrown.
   * @param new_col_names The new column names of the table that defines the
   * order of the columns in the table.
   */
  table_with_names& project(std::vector<std::string> const& new_col_names);
  /**
   * @brief Select a subset of columns from the table
   *
   * @param col_names The names of the columns to select
   */
  [[nodiscard]] cudf::table_view select(
      std::vector<std::string> const& col_names) const;

  void dump(int32_t max_rows, rmm::cuda_stream_view stream) const;

  std::unique_ptr<cudf::table> extract_top_n(
      int32_t limit,
      rmm::cuda_stream_view stream);

 private:
  std::unique_ptr<cudf::table> tbl;
  std::vector<std::string> col_names;
};
