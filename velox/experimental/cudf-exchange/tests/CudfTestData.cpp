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

#include "velox/experimental/cudf-exchange/tests/CudfTestData.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <rmm/device_buffer.hpp>

#include <functional>
#include <random>

namespace facebook::velox::cudf_exchange {

static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

// ----- BaseTableGenerator static helper methods -----

std::string BaseTableGenerator::genRandomStr(size_t len) {
  std::string rStr;
  rStr.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    rStr += alphanum[rand() % (sizeof(alphanum) - 1)];
  }
  return rStr;
}

template <typename T>
std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<T>& hostValues,
    rmm::cuda_stream_view stream) {
  size_t numRows = hostValues.size();

  // Allocate a device buffer of the correct size
  rmm::device_buffer data(
      numRows * sizeof(T), stream, rmm::mr::get_current_device_resource());

  // Copy host -> device
  cudaMemcpyAsync(
      data.data(),
      hostValues.data(),
      numRows * sizeof(T),
      cudaMemcpyHostToDevice,
      stream.value());

  // Build the cudf::column from the device buffer
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_to_id<T>()},
      numRows,
      std::move(data),
      rmm::device_buffer{}, // no null mask
      0);                   // no nulls
}

// Explicit template instantiations
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<int8_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<int16_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<int32_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<int64_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<uint8_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<uint16_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<uint32_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<uint64_t>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<float>& hostValues,
    rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> BaseTableGenerator::makeNumericColumn(
    const std::vector<double>& hostValues,
    rmm::cuda_stream_view stream);

std::unique_ptr<cudf::column> BaseTableGenerator::makeStringsColumn(
    const std::vector<std::string>& hostStrings) {
  auto numRows = hostStrings.size();

  // --- Create offsets array ---
  std::vector<int32_t> hOffsets(numRows + 1);
  hOffsets[0] = 0;
  for (size_t i = 0; i < numRows; ++i) {
    hOffsets[i + 1] = hOffsets[i] + static_cast<int32_t>(hostStrings[i].size());
  }

  // Copy offsets to device
  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      cudf::mask_state::UNALLOCATED);

  cudaMemcpy(
      offsetsCol->mutable_view().data<int32_t>(),
      hOffsets.data(),
      sizeof(int32_t) * (numRows + 1),
      cudaMemcpyHostToDevice);

  // --- Create chars buffer ---
  size_t totalBytes = 0;
  for (auto const& s : hostStrings)
    totalBytes += s.size();

  rmm::device_buffer charsBuffer(
      totalBytes,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

  std::vector<char> hostConcat;
  hostConcat.reserve(totalBytes);
  for (auto const& s : hostStrings)
    hostConcat.insert(hostConcat.end(), s.begin(), s.end());

  cudaMemcpy(
      charsBuffer.data(),
      hostConcat.data(),
      totalBytes,
      cudaMemcpyHostToDevice);

  // --- Build strings column ---
  return cudf::make_strings_column(
      numRows,
      std::move(offsetsCol),
      std::move(charsBuffer),
      0,                    // null_count
      rmm::device_buffer{}); // null mask
}

std::unique_ptr<cudf::column> BaseTableGenerator::makeStructColumn(
    std::vector<std::unique_ptr<cudf::column>> children,
    cudf::size_type numRows) {
  return cudf::make_structs_column(
      numRows,
      std::move(children),
      0,                    // null_count
      rmm::device_buffer{}); // null_mask
}

template <typename T>
std::vector<T> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream) {
  maxRows = columnView.size() < maxRows ? columnView.size() : maxRows;
  const T* ptrData = columnView.data<T>();
  auto hostVec = cudf::detail::make_host_vector_async(
      cudf::device_span<T const>(ptrData, maxRows), stream);
  std::vector<T> vec(maxRows);
  std::copy(hostVec.begin(), hostVec.end(), vec.begin());
  return vec;
}

// Explicit template instantiations for getColVector
template std::vector<int8_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<int16_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<int32_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<int64_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<uint8_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<uint16_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<uint32_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<uint64_t> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<float> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);
template std::vector<double> BaseTableGenerator::getColVector(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream);

std::vector<std::string> BaseTableGenerator::getStringCol(
    const cudf::column_view& columnView,
    cudf::size_type maxRows,
    rmm::cuda_stream_view stream) {
  cudf::strings_column_view strColView{columnView};
  maxRows = strColView.size() < maxRows ? strColView.size() : maxRows;

  auto offsetView = strColView.offsets();
  const cudf::size_type* ptrOffsetsData =
      offsetView.data<cudf::size_type>();
  auto const hOffsets = cudf::detail::make_host_vector_async(
      cudf::device_span<cudf::size_type const>(ptrOffsetsData, maxRows + 1),
      stream);
  const cudf::size_type* hostOffsets = hOffsets.data();

  auto const totalNumBytes = std::distance(
      strColView.chars_begin(stream), strColView.chars_end(stream));
  char const* ptrAllBytes = strColView.chars_begin(stream);
  auto const hBytes = cudf::detail::make_host_vector_async(
      cudf::device_span<char const>(ptrAllBytes, totalNumBytes), stream);
  const char* strPtr = hBytes.data();

  std::vector<std::string> strVec;
  for (cudf::size_type i = 0; i < maxRows; ++i) {
    std::string str(strPtr + hostOffsets[i], strPtr + hostOffsets[i + 1]);
    strVec.push_back(str);
  }
  return strVec;
}

// ----- CudfTestData implementation -----

void CudfTestData::initialize(
    size_t numRows,
    size_t minStringLength,
    size_t maxStringLength) {
  VLOG(3) << "+ CudfTestData::initialize numRows:" << numRows
          << " stringLength:[" << minStringLength << ".." << maxStringLength
          << "]";
  numRows_ = numRows;
  strings_ = std::make_shared<std::vector<std::string>>();
  integers_ = std::make_shared<std::vector<uint32_t>>();
  doubles_ = std::make_shared<std::vector<float>>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(minStringLength, maxStringLength);
  std::hash<std::string> hasher;

  for (size_t i = 0; i < numRows_; i++) {
    int strLength = dist(gen);
    std::string str = genRandomStr(strLength);
    double hashValue = hasher(str);

    strings_->push_back(str);
    integers_->push_back(strLength);
    doubles_->push_back(hashValue);
  }

  for (size_t i = 0; i < numRows; i++) {
    VLOG(4) << "In dataTest Generated data String: " << strings_->at(i)
            << " Integer: " << integers_->at(i)
            << " Double: " << doubles_->at(i);
  }

  VLOG(3) << "- CudfTestData::initialize";
}

std::unique_ptr<cudf::table> CudfTestData::makeTable(
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> columns;

  // Column 0: INT32 (integers)
  columns.push_back(makeNumericColumn(*integers_, stream));

  // Column 1: FLOAT64 (doubles stored as float)
  columns.push_back(makeNumericColumn(*doubles_, stream));

  // Column 2: STRING
  columns.push_back(makeStringsColumn(*strings_));

  return std::make_unique<cudf::table>(std::move(columns));
}

bool CudfTestData::verifyTable(
    const cudf::table_view& table,
    size_t startRow,
    size_t numRows,
    rmm::cuda_stream_view stream) {
  if (table.num_columns() != 3) {
    VLOG(0) << "CudfTestData::verifyTable: expected 3 columns, got "
            << table.num_columns();
    return false;
  }

  // Get the data from the received table
  auto receivedInts = getColVector<uint32_t>(table.column(0), numRows, stream);
  auto receivedDoubles = getColVector<float>(table.column(1), numRows, stream);
  auto receivedStrings = getStringCol(table.column(2), numRows, stream);

  // Compare with expected data
  for (size_t i = 0; i < numRows; ++i) {
    size_t srcIdx = startRow + i;
    if (srcIdx >= integers_->size()) {
      VLOG(0) << "CudfTestData::verifyTable: srcIdx " << srcIdx
              << " out of range " << integers_->size();
      return false;
    }

    if (receivedInts[i] != (*integers_)[srcIdx]) {
      VLOG(0) << "CudfTestData::verifyTable: int mismatch at row " << i
              << ": expected " << (*integers_)[srcIdx] << ", got "
              << receivedInts[i];
      return false;
    }

    if (receivedDoubles[i] != (*doubles_)[srcIdx]) {
      VLOG(0) << "CudfTestData::verifyTable: double mismatch at row " << i
              << ": expected " << (*doubles_)[srcIdx] << ", got "
              << receivedDoubles[i];
      return false;
    }

    if (receivedStrings[i] != (*strings_)[srcIdx]) {
      VLOG(0) << "CudfTestData::verifyTable: string mismatch at row " << i
              << ": expected '" << (*strings_)[srcIdx] << "', got '"
              << receivedStrings[i] << "'";
      return false;
    }
  }

  return true;
}

// ----- WideTestTable implementation -----
// Contains only numeric columns (no STRING or STRUCT)

void WideTestTable::initialize(size_t numRows) {
  VLOG(3) << "+ WideTestTable::initialize numRows:" << numRows;
  numRows_ = numRows;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> intDist(-1000000, 1000000);
  std::uniform_real_distribution<double> doubleDist(-1000.0, 1000.0);
  std::uniform_int_distribution<> boolDist(0, 1);

  int8Data_.resize(numRows);
  int16Data_.resize(numRows);
  int32Data_.resize(numRows);
  int64Data_.resize(numRows);
  uint8Data_.resize(numRows);
  uint16Data_.resize(numRows);
  uint32Data_.resize(numRows);
  uint64Data_.resize(numRows);
  float32Data_.resize(numRows);
  float64Data_.resize(numRows);
  boolData_.resize(numRows);

  for (size_t i = 0; i < numRows; ++i) {
    int64_t randInt = intDist(gen);
    double randDouble = doubleDist(gen);

    int8Data_[i] = static_cast<int8_t>(randInt % 128);
    int16Data_[i] = static_cast<int16_t>(randInt % 32768);
    int32Data_[i] = static_cast<int32_t>(randInt);
    int64Data_[i] = randInt;
    uint8Data_[i] = static_cast<uint8_t>(std::abs(randInt) % 256);
    uint16Data_[i] = static_cast<uint16_t>(std::abs(randInt) % 65536);
    uint32Data_[i] = static_cast<uint32_t>(std::abs(randInt));
    uint64Data_[i] = static_cast<uint64_t>(std::abs(randInt));
    float32Data_[i] = static_cast<float>(randDouble);
    float64Data_[i] = randDouble;
    boolData_[i] = boolDist(gen);
  }

  VLOG(3) << "- WideTestTable::initialize";
}

void WideTestTable::addNumericColumns(
    std::vector<std::unique_ptr<cudf::column>>& columns,
    rmm::cuda_stream_view stream) {
  columns.push_back(makeNumericColumn(int8Data_, stream));
  columns.push_back(makeNumericColumn(int16Data_, stream));
  columns.push_back(makeNumericColumn(int32Data_, stream));
  columns.push_back(makeNumericColumn(int64Data_, stream));
  columns.push_back(makeNumericColumn(uint8Data_, stream));
  columns.push_back(makeNumericColumn(uint16Data_, stream));
  columns.push_back(makeNumericColumn(uint32Data_, stream));
  columns.push_back(makeNumericColumn(uint64Data_, stream));
  columns.push_back(makeNumericColumn(float32Data_, stream));
  columns.push_back(makeNumericColumn(float64Data_, stream));
  columns.push_back(makeNumericColumn(boolData_, stream));
}

std::unique_ptr<cudf::table> WideTestTable::makeTable(
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  addNumericColumns(columns, stream);
  return std::make_unique<cudf::table>(std::move(columns));
}

bool WideTestTable::verifyNumericColumns(
    const cudf::table_view& table,
    size_t startRow,
    size_t numRows,
    rmm::cuda_stream_view stream) {
  // Verify numeric columns (columns 0-10)
  auto rxInt8 = getColVector<int8_t>(table.column(0), numRows, stream);
  auto rxInt16 = getColVector<int16_t>(table.column(1), numRows, stream);
  auto rxInt32 = getColVector<int32_t>(table.column(2), numRows, stream);
  auto rxInt64 = getColVector<int64_t>(table.column(3), numRows, stream);
  auto rxUint8 = getColVector<uint8_t>(table.column(4), numRows, stream);
  auto rxUint16 = getColVector<uint16_t>(table.column(5), numRows, stream);
  auto rxUint32 = getColVector<uint32_t>(table.column(6), numRows, stream);
  auto rxUint64 = getColVector<uint64_t>(table.column(7), numRows, stream);
  auto rxFloat32 = getColVector<float>(table.column(8), numRows, stream);
  auto rxFloat64 = getColVector<double>(table.column(9), numRows, stream);
  auto rxBool = getColVector<int8_t>(table.column(10), numRows, stream);

  for (size_t i = 0; i < numRows; ++i) {
    size_t srcIdx = startRow + i;
    if (srcIdx >= numRows_) {
      VLOG(0) << "verifyNumericColumns: srcIdx " << srcIdx
              << " out of range " << numRows_;
      return false;
    }

    if (rxInt8[i] != int8Data_[srcIdx] || rxInt16[i] != int16Data_[srcIdx] ||
        rxInt32[i] != int32Data_[srcIdx] || rxInt64[i] != int64Data_[srcIdx] ||
        rxUint8[i] != uint8Data_[srcIdx] ||
        rxUint16[i] != uint16Data_[srcIdx] ||
        rxUint32[i] != uint32Data_[srcIdx] ||
        rxUint64[i] != uint64Data_[srcIdx] ||
        rxFloat32[i] != float32Data_[srcIdx] ||
        rxFloat64[i] != float64Data_[srcIdx] ||
        rxBool[i] != boolData_[srcIdx]) {
      VLOG(0) << "verifyNumericColumns: data mismatch at row " << i;
      return false;
    }
  }
  return true;
}

bool WideTestTable::verifyTable(
    const cudf::table_view& table,
    size_t startRow,
    size_t numRows,
    rmm::cuda_stream_view stream) {
  if (table.num_columns() != 11) {
    VLOG(0) << "WideTestTable::verifyTable: expected 11 columns, got "
            << table.num_columns();
    return false;
  }
  return verifyNumericColumns(table, startRow, numRows, stream);
}

// ----- WideComplexTestTable implementation -----
// Extends WideTestTable with STRING and STRUCT columns

void WideComplexTestTable::initialize(size_t numRows) {
  // First initialize the base class (numeric columns)
  WideTestTable::initialize(numRows);

  VLOG(3) << "+ WideComplexTestTable::initialize (adding complex columns)";

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> intDist(-1000000, 1000000);
  std::uniform_real_distribution<double> doubleDist(-1000.0, 1000.0);
  std::uniform_int_distribution<> strLenDist(1, 20);

  stringData_.resize(numRows);
  structField1Data_.resize(numRows);
  structField2Data_.resize(numRows);

  for (size_t i = 0; i < numRows; ++i) {
    int64_t randInt = intDist(gen);
    double randDouble = doubleDist(gen);

    stringData_[i] = genRandomStr(strLenDist(gen));
    structField1Data_[i] = randInt * 2;
    structField2Data_[i] = randDouble * 2.0;
  }

  VLOG(3) << "- WideComplexTestTable::initialize";
}

std::unique_ptr<cudf::table> WideComplexTestTable::makeTable(
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> columns;

  // Add numeric columns from base class
  addNumericColumns(columns, stream);

  // Add string column
  columns.push_back(makeStringsColumn(stringData_));

  // Add struct column
  std::vector<std::unique_ptr<cudf::column>> structChildren;
  structChildren.push_back(makeNumericColumn(structField1Data_, stream));
  structChildren.push_back(makeNumericColumn(structField2Data_, stream));
  columns.push_back(
      makeStructColumn(std::move(structChildren), static_cast<cudf::size_type>(numRows_)));

  return std::make_unique<cudf::table>(std::move(columns));
}

bool WideComplexTestTable::verifyTable(
    const cudf::table_view& table,
    size_t startRow,
    size_t numRows,
    rmm::cuda_stream_view stream) {
  if (table.num_columns() != 13) {
    VLOG(0) << "WideComplexTestTable::verifyTable: expected 13 columns, got "
            << table.num_columns();
    return false;
  }

  // First verify numeric columns using base class helper
  if (!verifyNumericColumns(table, startRow, numRows, stream)) {
    return false;
  }

  // Verify string column (column 11)
  auto rxStrings = getStringCol(table.column(11), numRows, stream);

  // Verify struct column children (column 12)
  cudf::structs_column_view structView{table.column(12)};
  auto rxStructField1 =
      getColVector<int64_t>(structView.child(0), numRows, stream);
  auto rxStructField2 =
      getColVector<double>(structView.child(1), numRows, stream);

  for (size_t i = 0; i < numRows; ++i) {
    size_t srcIdx = startRow + i;
    if (rxStrings[i] != stringData_[srcIdx] ||
        rxStructField1[i] != structField1Data_[srcIdx] ||
        rxStructField2[i] != structField2Data_[srcIdx]) {
      VLOG(0) << "WideComplexTestTable::verifyTable: complex column mismatch at row " << i;
      return false;
    }
  }

  return true;
}

} // namespace facebook::velox::cudf_exchange
