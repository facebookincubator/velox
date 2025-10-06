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
#include "TableWithNames.hpp"

#include <cudf/detail/gather.hpp>
#include <cudf/table/table_view.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include "common.hpp"

cudf::table_view table_with_names::table() const {
  return tbl->view();
}
cudf::table_view table_with_names::view() const {
  return tbl->view();
}

cudf::column_view table_with_names::column(std::string const& col_name) const {
  return tbl->view().column(column_id(col_name));
}

std::vector<std::string> const& table_with_names::column_names() const {
  return col_names;
}

cudf::size_type table_with_names::column_id(std::string const& col_name) const {
  auto it = std::find(col_names.begin(), col_names.end(), col_name);
  if (it == col_names.end()) {
    std::string err_msg = "Column `" + col_name + "` not found";
    throw std::runtime_error(err_msg);
  }
  return std::distance(col_names.begin(), it);
}

table_with_names& table_with_names::append(
    std::unique_ptr<cudf::column>& col,
    std::string const& col_name) {
  auto cols = tbl->release();
  cols.push_back(std::move(col));
  tbl = std::make_unique<cudf::table>(std::move(cols));
  col_names.push_back(col_name);
  return (*this);
}

table_with_names& table_with_names::replace(
    std::unique_ptr<cudf::column>& col,
    std::string const& col_name) {
  auto cols = tbl->release();
  auto col_id = column_id(col_name);
  cols[col_id] = std::move(col);
  tbl = std::make_unique<cudf::table>(std::move(cols));
  return (*this);
}

table_with_names& table_with_names::rename(
    std::string const& col_name,
    std::string const& col_new_name) {
  auto col_id = column_id(col_name);
  col_names[col_id] = col_new_name;
  return (*this);
}

table_with_names& table_with_names::drop(std::string const& col_name) {
  auto cols = tbl->release();
  auto col_id = column_id(col_name);
  cols.erase(cols.begin() + col_id);
  tbl = std::make_unique<cudf::table>(std::move(cols));
  col_names.erase(col_names.begin() + col_id);
  return (*this);
}

table_with_names& table_with_names::project(
    std::vector<std::string> const& new_col_names) {
  // create a map to store the association of column ids and existing names
  std::map<std::string, cudf::size_type> col_id_map;
  for (cudf::size_type col = 0; col < tbl->num_columns(); col++) {
    col_id_map[col_names[col]] = col;
  }
  auto cols = tbl->release();
  std::vector<std::unique_ptr<cudf::column>> new_cols;
  for (std::size_t col = 0; col < new_col_names.size(); col++) {
    auto it = col_id_map.find(new_col_names[col]);
    if (it != col_id_map.end()) {
      auto old_col_id = it->second;
      new_cols.push_back(std::move(cols[old_col_id]));
    } else {
      std::string err_msg = "Column `" + new_col_names[col] + "` not found !";
      throw std::runtime_error(err_msg);
    }
  }
  tbl = std::make_unique<cudf::table>(std::move(new_cols));
  col_names = new_col_names;
  return (*this);
}

cudf::table_view table_with_names::select(
    std::vector<std::string> const& col_names) const {
  std::vector<cudf::size_type> col_indices;
  for (auto const& col_name : col_names) {
    col_indices.push_back(column_id(col_name));
  }
  return tbl->select(col_indices);
}

void table_with_names::dump(int32_t max_rows, rmm::cuda_stream_view stream)
    const {
  // compute the maximum width needed for each column.
  if (max_rows > tbl->num_rows()) {
    max_rows = tbl->num_rows();
  }
  std::cout << "dumping " << max_rows << " from total " << tbl->num_rows()
            << std::endl;
  std::vector<uint32_t> widths;
  widths.reserve(tbl->num_columns());
  for (int32_t col = 0; col < tbl->num_columns(); col++) {
    uint32_t width = 0;
    switch (tbl->get_column(col).type().id()) {
      case cudf::type_id::BOOL8:
        width = 7;
        break;
      case cudf::type_id::INT16:
        width = 6;
        break;
      case cudf::type_id::INT32:
        width = 11;
        break;
      case cudf::type_id::INT64:
        width = 20;
        break;
      case cudf::type_id::FLOAT32:
        width = 11;
        break;
      case cudf::type_id::FLOAT64:
        width = 18;
        break;
      case cudf::type_id::STRING: {
        width = 0;
        auto str_vec =
            getStringCol(tbl->get_column(col).view(), max_rows, stream);
        for (auto& str : str_vec) {
          if (str.length() > width) {
            width = str.length();
          }
        }
        width += 2;
        break;
      }
      default:
        break;
    }
    widths[col] = width;
  }
  // print the column names
  for (int32_t col = 0; col < tbl->num_columns(); col++) {
    std::cout << std::setw(widths[col]) << col_names[col] << " | ";
  }
  std::cout << std::endl;
  // print a separating line
  for (int32_t col = 0; col < tbl->num_columns(); col++) {
    std::string dashes(widths[col], '-');
    std::cout << std::setw(widths[col]) << dashes << "-+-";
  }
  std::cout << std::endl;

  // print the column data
  struct data_vecs {
    std::vector<bool> bool_vec;
    std::vector<int16_t> short_vec;
    std::vector<int32_t> int_vec;
    std::vector<int64_t> long_vec;
    std::vector<float> float_vec;
    std::vector<double> dbl_vec;
    std::vector<std::string> str_vec;
  };
  std::vector<data_vecs> data;
  for (int32_t col = 0; col < tbl->num_columns(); col++) {
    switch (tbl->get_column(col).type().id()) {
      case cudf::type_id::BOOL8: {
        data_vecs d;
        d.bool_vec =
            getColVector<bool>(tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);
      } break;
      case cudf::type_id::INT16: {
        data_vecs d;
        d.short_vec = getColVector<int16_t>(
            tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);
      } break;
      case cudf::type_id::INT32: {
        data_vecs d;
        d.int_vec = getColVector<int32_t>(
            tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);
      } break;
      case cudf::type_id::INT64: {
        data_vecs d;
        d.long_vec = getColVector<int64_t>(
            tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);

      } break;
      case cudf::type_id::FLOAT32: {
        data_vecs d;
        d.float_vec =
            getColVector<float>(tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);

      } break;

      case cudf::type_id::FLOAT64: {
        data_vecs d;
        d.dbl_vec =
            getColVector<double>(tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);

      } break;
      case cudf::type_id::STRING: {
        data_vecs d;
        d.str_vec = getStringCol(tbl->get_column(col).view(), max_rows, stream);
        data.push_back(d);
      } break;
      default:

        break;
    }
  }
  for (int32_t row = 0; row < max_rows; row++) {
    for (int32_t col = 0; col < tbl->num_columns(); col++) {
      switch (tbl->get_column(col).type().id()) {
        case cudf::type_id::BOOL8: {
          std::cout << std::setw(widths[col]) << std::right
                    << data[col].bool_vec[row] << " | ";
        } break;
        case cudf::type_id::INT16: {
          std::cout << std::setw(widths[col]) << std::right
                    << data[col].short_vec[row] << " | ";
        } break;
        case cudf::type_id::INT32: {
          std::cout << std::setw(widths[col]) << std::right
                    << data[col].int_vec[row] << " | ";
        } break;
        case cudf::type_id::INT64: {
          std::cout << std::setw(widths[col]) << std::right
                    << data[col].long_vec[row] << " | ";
        } break;
        case cudf::type_id::FLOAT32: {
          std::cout << std::setw(widths[col]) << data[col].float_vec[row]
                    << " | ";
        } break;
        case cudf::type_id::FLOAT64: {
          std::cout << std::setw(widths[col]) << data[col].dbl_vec[row]
                    << " | ";
        } break;
        case cudf::type_id::STRING: {
          std::cout << std::setw(widths[col]) << data[col].str_vec[row]
                    << " | ";
        } break;
        default:
          break;
      }
    }
    std::cout << std::endl;
  }
}

std::unique_ptr<cudf::table> table_with_names::extract_top_n(
    int32_t limit,
    rmm::cuda_stream_view stream) {
  // --- Create the indices for the limit
  uint32_t indices[limit];
  for (int i = 0; i < limit; i++)
    indices[i] = i;

  // --- Extract those indices from the table
  rmm::device_uvector<int32_t> dev_vec(limit, stream);
  cudaMemcpy(
      dev_vec.data(),
      indices,
      sizeof(uint32_t) * limit,
      cudaMemcpyHostToDevice);

  // Create a cudf::device_span from the device vector
  auto const indices_span = cudf::device_span<cudf::size_type const>{dev_vec};

  // Create a cudf::column_view from the device span
  auto const gather_map = cudf::column_view{indices_span};

  // Use cudf::gather to extract the top N rows
  std::unique_ptr<cudf::table> result_table =
      cudf::gather(this->table(), gather_map);

  return result_table;
}
