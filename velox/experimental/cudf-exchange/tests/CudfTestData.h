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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <stddef.h>

#include <memory>
#include <string>
#include <vector>
#include "velox/type/Type.h"

namespace facebook::velox::cudf_exchange {

/// @brief Enum to identify different table types for testing
enum class TableType {
  NARROW, // Original CudfTestData structure (INT32, FLOAT64, VARCHAR)
  WIDE, // Numeric types only (no STRING or STRUCT)
  WIDE_COMPLEX // All cudf data types including STRING and STRUCT
};

/// @brief Abstract base class for generating test tables with cudf data.
/// Provides common helper methods for creating cudf columns and tables.
class BaseTableGenerator {
 public:
  // Make a constant to avoid too many variables
  static const int STRING_LENGTH = 4;

  BaseTableGenerator() = default;
  virtual ~BaseTableGenerator() = default;

  /// @brief Initialize the test data with the specified number of rows.
  virtual void initialize(size_t numRows) = 0;

  /// @brief Get the row type for this table generator.
  virtual facebook::velox::RowTypePtr getRowType() const = 0;

  /// @brief Get the number of rows in the generated data.
  virtual size_t getNumRows() const = 0;

  /// @brief Create a cudf table from the generated data.
  /// @param stream The CUDA stream to use.
  /// @return A unique pointer to the created cudf table.
  virtual std::unique_ptr<cudf::table> makeTable(
      rmm::cuda_stream_view stream) = 0;

  /// @brief Verify that the received table matches the generated data.
  /// @param table The table to verify.
  /// @param startRow The starting row index in the generated data.
  /// @param numRows The number of rows to verify.
  /// @param stream The CUDA stream to use.
  /// @return True if the data matches, false otherwise.
  virtual bool verifyTable(
      const cudf::table_view& table,
      size_t startRow,
      size_t numRows,
      rmm::cuda_stream_view stream) = 0;

  // ----- Helper methods for creating cudf columns -----

  /// @brief Generate a random string of the specified length.
  static std::string genRandomStr(size_t len);

  /// @brief Create a numeric column from a vector of values.
  template <typename T>
  static std::unique_ptr<cudf::column> makeNumericColumn(
      const std::vector<T>& hostValues,
      rmm::cuda_stream_view stream);

  /// @brief Create a strings column from a vector of host strings.
  static std::unique_ptr<cudf::column> makeStringsColumn(
      const std::vector<std::string>& hostStrings);

  /// @brief Create a struct column from child columns.
  static std::unique_ptr<cudf::column> makeStructColumn(
      std::vector<std::unique_ptr<cudf::column>> children,
      cudf::size_type numRows);

  /// @brief Retrieve values from a numeric column to host memory.
  template <typename T>
  static std::vector<T> getColVector(
      const cudf::column_view& columnView,
      cudf::size_type maxRows,
      rmm::cuda_stream_view stream);

  /// @brief Retrieve strings from a strings column to host memory.
  static std::vector<std::string> getStringCol(
      const cudf::column_view& columnView,
      cudf::size_type maxRows,
      rmm::cuda_stream_view stream);
};

/// @brief Original test data structure with narrow schema (INT32, FLOAT64,
/// VARCHAR). Preserves compatibility with existing tests.
class CudfTestData : public BaseTableGenerator {
 public:
  inline static const std::vector<std::string> kTestColumnNames = {
      "c0",
      "c1",
      "c2"};
  inline const static std::vector<TypePtr> kTestColumnTypes = {
      INTEGER(),
      DOUBLE(),
      VARCHAR()};
  inline const static facebook::velox::RowTypePtr kTestRowType =
      ROW(kTestColumnNames, kTestColumnTypes);

  CudfTestData() = default;

  void initialize(size_t numRows) override {
    initialize(numRows, STRING_LENGTH, STRING_LENGTH);
  }

  void
  initialize(size_t numRows, size_t minStringLength, size_t maxStringLength);

  facebook::velox::RowTypePtr getRowType() const override {
    return kTestRowType;
  }

  size_t getNumRows() const override {
    return numRows_;
  }

  std::unique_ptr<cudf::table> makeTable(rmm::cuda_stream_view stream) override;

  bool verifyTable(
      const cudf::table_view& table,
      size_t startRow,
      size_t numRows,
      rmm::cuda_stream_view stream) override;

  // Legacy accessors for backward compatibility with existing tests
  std::shared_ptr<std::vector<std::string>> getStrings() {
    return strings_;
  }

  std::shared_ptr<std::vector<uint32_t>> getIntegers() {
    return integers_;
  }

  std::shared_ptr<std::vector<float>> getDoubles() {
    return doubles_;
  }

  /// @brief Sets the data directly (used for creating partitioned test data).
  void setData(
      std::shared_ptr<std::vector<uint32_t>> integers,
      std::shared_ptr<std::vector<float>> doubles,
      std::shared_ptr<std::vector<std::string>> strings) {
    integers_ = std::move(integers);
    doubles_ = std::move(doubles);
    strings_ = std::move(strings);
    numRows_ = integers_->size();
  }

 protected:
  std::shared_ptr<std::vector<std::string>> strings_;
  std::shared_ptr<std::vector<uint32_t>> integers_;
  std::shared_ptr<std::vector<float>> doubles_;
  size_t numRows_ = 0;
};

/// @brief Wide table generator with numeric cudf data types only.
/// Used to test that numeric column types transfer correctly through the
/// exchange. Does NOT include STRING or STRUCT columns - use
/// WideComplexTestTable for those.
class WideTestTable : public BaseTableGenerator {
 public:
  // Column names for the wide table (numeric types only)
  inline static const std::vector<std::string> kColumnNames = {
      "int8_col", // INT8
      "int16_col", // INT16
      "int32_col", // INT32
      "int64_col", // INT64
      "uint8_col", // UINT8
      "uint16_col", // UINT16
      "uint32_col", // UINT32
      "uint64_col", // UINT64
      "float32_col", // FLOAT32
      "float64_col", // FLOAT64
      "bool_col" // BOOL8
  };

  // Column types for Velox (numeric types only)
  inline static const std::vector<TypePtr> kColumnTypes = {
      TINYINT(), // INT8
      SMALLINT(), // INT16
      INTEGER(), // INT32
      BIGINT(), // INT64
      TINYINT(), // UINT8 (mapped to TINYINT)
      SMALLINT(), // UINT16 (mapped to SMALLINT)
      INTEGER(), // UINT32 (mapped to INTEGER)
      BIGINT(), // UINT64 (mapped to BIGINT)
      REAL(), // FLOAT32
      DOUBLE(), // FLOAT64
      BOOLEAN()}; // BOOL8

  inline static const facebook::velox::RowTypePtr kRowType =
      ROW(kColumnNames, kColumnTypes);

  WideTestTable() = default;

  void initialize(size_t numRows) override;

  facebook::velox::RowTypePtr getRowType() const override {
    return kRowType;
  }

  size_t getNumRows() const override {
    return numRows_;
  }

  std::unique_ptr<cudf::table> makeTable(rmm::cuda_stream_view stream) override;

  bool verifyTable(
      const cudf::table_view& table,
      size_t startRow,
      size_t numRows,
      rmm::cuda_stream_view stream) override;

 protected:
  /// @brief Helper to add numeric columns to a column vector.
  /// Can be used by derived classes to build tables with additional columns.
  void addNumericColumns(
      std::vector<std::unique_ptr<cudf::column>>& columns,
      rmm::cuda_stream_view stream);

  /// @brief Helper to verify numeric columns in a table.
  /// @param table The table view to verify.
  /// @param startRow Starting row in the reference data.
  /// @param numRows Number of rows to verify.
  /// @param stream CUDA stream.
  /// @return True if all numeric columns match, false otherwise.
  bool verifyNumericColumns(
      const cudf::table_view& table,
      size_t startRow,
      size_t numRows,
      rmm::cuda_stream_view stream);

  // Data storage for numeric column types
  std::vector<int8_t> int8Data_;
  std::vector<int16_t> int16Data_;
  std::vector<int32_t> int32Data_;
  std::vector<int64_t> int64Data_;
  std::vector<uint8_t> uint8Data_;
  std::vector<uint16_t> uint16Data_;
  std::vector<uint32_t> uint32Data_;
  std::vector<uint64_t> uint64Data_;
  std::vector<float> float32Data_;
  std::vector<double> float64Data_;
  std::vector<int8_t> boolData_; // stored as int8_t for BOOL8

  size_t numRows_ = 0;
};

/// @brief Wide table generator that extends WideTestTable with STRING and
/// STRUCT columns. Used to test that all column types including complex types
/// transfer correctly.
class WideComplexTestTable : public WideTestTable {
 public:
  // Column names: base numeric columns + string + struct
  inline static const std::vector<std::string> kColumnNames = {
      "int8_col", // INT8
      "int16_col", // INT16
      "int32_col", // INT32
      "int64_col", // INT64
      "uint8_col", // UINT8
      "uint16_col", // UINT16
      "uint32_col", // UINT32
      "uint64_col", // UINT64
      "float32_col", // FLOAT32
      "float64_col", // FLOAT64
      "bool_col", // BOOL8
      "string_col", // STRING
      "struct_col" // STRUCT<int64, float64>
  };

  // Column types for Velox: base numeric types + string + struct
  inline static const std::vector<TypePtr> kColumnTypes = {
      TINYINT(), // INT8
      SMALLINT(), // INT16
      INTEGER(), // INT32
      BIGINT(), // INT64
      TINYINT(), // UINT8 (mapped to TINYINT)
      SMALLINT(), // UINT16 (mapped to SMALLINT)
      INTEGER(), // UINT32 (mapped to INTEGER)
      BIGINT(), // UINT64 (mapped to BIGINT)
      REAL(), // FLOAT32
      DOUBLE(), // FLOAT64
      BOOLEAN(), // BOOL8
      VARCHAR(), // STRING
      ROW({"field1", "field2"}, // STRUCT
          {BIGINT(), DOUBLE()})};

  inline static const facebook::velox::RowTypePtr kRowType =
      ROW(kColumnNames, kColumnTypes);

  WideComplexTestTable() = default;

  void initialize(size_t numRows) override;

  facebook::velox::RowTypePtr getRowType() const override {
    return kRowType;
  }

  std::unique_ptr<cudf::table> makeTable(rmm::cuda_stream_view stream) override;

  bool verifyTable(
      const cudf::table_view& table,
      size_t startRow,
      size_t numRows,
      rmm::cuda_stream_view stream) override;

 protected:
  // Additional data storage for complex column types
  std::vector<std::string> stringData_;
  // Struct children data
  std::vector<int64_t> structField1Data_;
  std::vector<double> structField2Data_;
};

/// @brief Factory function to create a table generator based on TableType.
inline std::shared_ptr<BaseTableGenerator> createTableGenerator(
    TableType type) {
  switch (type) {
    case TableType::NARROW:
      return std::make_shared<CudfTestData>();
    case TableType::WIDE:
      return std::make_shared<WideTestTable>();
    case TableType::WIDE_COMPLEX:
      return std::make_shared<WideComplexTestTable>();
    default:
      return std::make_shared<CudfTestData>();
  }
}

} // namespace facebook::velox::cudf_exchange
