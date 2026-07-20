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
/*
 * A standalone table_view schema printer for cudf::table_view.
 * Can be copied to any project that uses cudf table_view.
 *
 * Usage:
 *   cudf::table_view const& table = ...;
 *   cudf::print_table_schema(table);
 *   // or
 *   std::string str = cudf::table_schema_to_string(table);
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace cudf {

/**
 * @brief Get the string representation of a cudf::type_id
 */
inline std::string type_id_to_string(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::EMPTY:
      return "EMPTY";
    case cudf::type_id::INT8:
      return "INT8";
    case cudf::type_id::INT16:
      return "INT16";
    case cudf::type_id::INT32:
      return "INT32";
    case cudf::type_id::INT64:
      return "INT64";
    case cudf::type_id::UINT8:
      return "UINT8";
    case cudf::type_id::UINT16:
      return "UINT16";
    case cudf::type_id::UINT32:
      return "UINT32";
    case cudf::type_id::UINT64:
      return "UINT64";
    case cudf::type_id::FLOAT32:
      return "FLOAT32";
    case cudf::type_id::FLOAT64:
      return "FLOAT64";
    case cudf::type_id::BOOL8:
      return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS:
      return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS:
      return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS:
      return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS:
      return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS:
      return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS:
      return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS:
      return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32:
      return "DICTIONARY32";
    case cudf::type_id::STRING:
      return "STRING";
    case cudf::type_id::LIST:
      return "LIST";
    case cudf::type_id::DECIMAL32:
      return "DECIMAL32";
    case cudf::type_id::DECIMAL64:
      return "DECIMAL64";
    case cudf::type_id::DECIMAL128:
      return "DECIMAL128";
    case cudf::type_id::STRUCT:
      return "STRUCT";
    default:
      return "UNKNOWN_TYPE";
  }
}

/**
 * @brief Recursively print column type information
 *
 * @param col The column view to print
 * @param os Output stream to write to
 * @param depth Current depth in the tree (for indentation)
 * @param indent_str String to use for each level of indentation
 * @param column_name Optional name for the column
 */
inline void print_column_type_impl(
    column_view const& col,
    std::ostream& os,
    int depth,
    std::string const& indent_str,
    std::string const& column_name = "") {
  // Print indentation
  for (int i = 0; i < depth; ++i) {
    os << indent_str;
  }

  // Print column name if provided
  if (!column_name.empty()) {
    os << column_name << ": ";
  }

  auto const type_id = col.type().id();
  os << type_id_to_string(type_id);

  // Print additional info for fixed-point types (scale)
  if (type_id == cudf::type_id::DECIMAL32 ||
      type_id == cudf::type_id::DECIMAL64 ||
      type_id == cudf::type_id::DECIMAL128) {
    os << "(scale=" << col.type().scale() << ")";
  }

  // Print size info
  os << " [size=" << col.size();
  if (col.nullable()) {
    os << ", nulls=" << col.null_count();
  }
  os << "]";

  // Handle nested types
  auto const num_children = col.num_children();
  if (num_children > 0) {
    os << " {\n";

    if (type_id == cudf::type_id::LIST) {
      // LIST has 1 child: offsets (internal) and the actual list elements
      // The child at index 0 is the offsets column (INT32)
      // The child at index 1 is the element column
      if (num_children >= 1) {
        print_column_type_impl(
            col.child(0), os, depth + 1, indent_str, "offsets");
      }
      if (num_children >= 2) {
        print_column_type_impl(
            col.child(1), os, depth + 1, indent_str, "elements");
      }
    } else if (type_id == cudf::type_id::STRING) {
      // STRING has 2 children: offsets and chars
      if (num_children >= 1) {
        print_column_type_impl(
            col.child(0), os, depth + 1, indent_str, "offsets");
      }
      if (num_children >= 2) {
        print_column_type_impl(
            col.child(1), os, depth + 1, indent_str, "chars");
      }
    } else if (type_id == cudf::type_id::STRUCT) {
      // STRUCT has N children, one for each field
      for (size_type i = 0; i < num_children; ++i) {
        print_column_type_impl(
            col.child(i),
            os,
            depth + 1,
            indent_str,
            "field[" + std::to_string(i) + "]");
      }
    } else if (type_id == cudf::type_id::DICTIONARY32) {
      // DICTIONARY has 2 children: indices and keys
      if (num_children >= 1) {
        print_column_type_impl(
            col.child(0), os, depth + 1, indent_str, "indices");
      }
      if (num_children >= 2) {
        print_column_type_impl(col.child(1), os, depth + 1, indent_str, "keys");
      }
    } else {
      // Generic handling for other types with children
      for (size_type i = 0; i < num_children; ++i) {
        print_column_type_impl(
            col.child(i),
            os,
            depth + 1,
            indent_str,
            "child[" + std::to_string(i) + "]");
      }
    }

    for (int i = 0; i < depth; ++i) {
      os << indent_str;
    }
    os << "}\n";
  } else {
    os << "\n";
  }
}

/**
 * @brief Print the schema/type information of a table_view to an output stream
 *
 * Recursively prints the type hierarchy for all columns including nested types
 * (LIST, STRUCT, STRING, DICTIONARY).
 *
 * @param table The table view to print
 * @param os Output stream (defaults to std::cout)
 * @param indent_str Indentation string for each level
 */
inline void print_table_schema(
    table_view const& table,
    std::ostream& os = std::cout,
    std::string const& indent_str = "  ") {
  os << "=== Table Schema ===\n";
  os << "Columns: " << table.num_columns() << ", Rows: " << table.num_rows()
     << "\n";
  os << "---\n";

  for (size_type i = 0; i < table.num_columns(); ++i) {
    print_column_type_impl(
        table.column(i),
        os,
        0,
        indent_str,
        "column[" + std::to_string(i) + "]");
  }

  os << "===================\n";
}

/**
 * @brief Convert the schema/type information of a table_view to a string
 *
 * @param table The table view to convert
 * @param indent_str Indentation string for each level
 * @return String representation of the table schema
 */
inline std::string table_schema_to_string(
    table_view const& table,
    std::string const& indent_str = "  ") {
  std::ostringstream oss;
  print_table_schema(table, oss, indent_str);
  return oss.str();
}

/**
 * @brief Print the schema/type information of a single column_view
 *
 * @param col The column view to print
 * @param os Output stream (defaults to std::cout)
 * @param indent_str Indentation string for each level
 * @param column_name Optional name for the column
 */
inline void print_column_schema(
    column_view const& col,
    std::ostream& os = std::cout,
    std::string const& indent_str = "  ",
    std::string const& column_name = "column") {
  os << "=== Column Schema ===\n";
  print_column_type_impl(col, os, 0, indent_str, column_name);
  os << "====================\n";
}

/**
 * @brief Convert the schema/type information of a column_view to a string
 *
 * @param col The column view to convert
 * @param indent_str Indentation string for each level
 * @param column_name Optional name for the column
 * @return String representation of the column schema
 */
inline std::string column_schema_to_string(
    column_view const& col,
    std::string const& indent_str = "  ",
    std::string const& column_name = "column") {
  std::ostringstream oss;
  print_column_schema(col, oss, indent_str, column_name);
  return oss.str();
}

/**
 * @brief Get a compact one-line type representation of a column
 *
 * Example outputs:
 *   - "INT32"
 *   - "LIST<FLOAT64>"
 *   - "STRUCT<INT32, STRING, LIST<INT64>>"
 *
 * @param col The column view
 * @return Compact type string
 */
inline std::string column_type_to_compact_string(column_view const& col) {
  auto const type_id = col.type().id();
  std::string result = type_id_to_string(type_id);

  if (type_id == cudf::type_id::DECIMAL32 ||
      type_id == cudf::type_id::DECIMAL64 ||
      type_id == cudf::type_id::DECIMAL128) {
    result += "(scale=" + std::to_string(col.type().scale()) + ")";
  }

  auto const num_children = col.num_children();

  if (type_id == cudf::type_id::LIST && num_children >= 2) {
    // For LIST, show element type
    result += "<" + column_type_to_compact_string(col.child(1)) + ">";
  } else if (type_id == cudf::type_id::STRUCT && num_children > 0) {
    // For STRUCT, show all field types
    result += "<";
    for (size_type i = 0; i < num_children; ++i) {
      if (i > 0)
        result += ", ";
      result += column_type_to_compact_string(col.child(i));
    }
    result += ">";
  } else if (type_id == cudf::type_id::DICTIONARY32 && num_children >= 2) {
    // For DICTIONARY, show keys type
    result += "<keys=" + column_type_to_compact_string(col.child(1)) + ">";
  }

  return result;
}

/**
 * @brief Get a compact one-line schema representation of a table
 *
 * Example output: "(INT32, STRING, LIST<FLOAT64>, STRUCT<INT32, STRING>)"
 *
 * @param table The table view
 * @return Compact schema string
 */
inline std::string table_schema_to_compact_string(table_view const& table) {
  std::string result = "(";
  for (size_type i = 0; i < table.num_columns(); ++i) {
    if (i > 0)
      result += ", ";
    result += column_type_to_compact_string(table.column(i));
  }
  result += ")";
  return result;
}

/**
 * @brief Print a compact one-line schema of a table
 *
 * @param table The table view to print
 * @param os Output stream (defaults to std::cout)
 */
inline void print_table_schema_compact(
    table_view const& table,
    std::ostream& os = std::cout) {
  os << "Table" << table_schema_to_compact_string(table) << " ["
     << table.num_rows() << " rows]\n";
}

} // namespace cudf
