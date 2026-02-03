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
#include "velox/experimental/cudf-exchange/tests/CudfTestHelpers.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

namespace facebook::velox::cudf_exchange {

std::shared_ptr<Task> createSourceTask(
    const std::string& taskId,
    std::shared_ptr<memory::MemoryPool> pool,
    RowTypePtr rowType,
    uint64_t kMaxOutputBufferSize) {
  VLOG(3) << "Testing SourceTask";
  const size_t vectorSize = 10;

  auto typeParams = rowType->parameters();
  std::vector<VectorPtr> vecPtrs;
  for (auto& typeParam : typeParams) {
    vecPtrs.emplace_back(
        BaseVector::create(typeParam.type, vectorSize, pool.get()));
  }

  // Wrap the vector (column) in a RowVector.
  auto rowVector = std::make_shared<RowVector>(
      pool.get(), // pool where allocations will be made.
      rowType, // input row type.
      BufferPtr(nullptr), // no nulls on this example.
      vectorSize, // length of the vectors.
      vecPtrs); // the input vector data.

  auto planFragment =
      exec::test::PlanBuilder().values({rowVector}).planFragment();

  std::shared_ptr<folly::Executor> executor(
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency()));

  std::unordered_map<std::string, std::string> configSettings{
      {velox::core::QueryConfig::kMaxOutputBufferSize,
       std::to_string(kMaxOutputBufferSize)}};

  auto queryCtx = core::QueryCtx::create(
      executor.get(), core::QueryConfig(std::move(configSettings)));

  auto task = Task::create(
      taskId,
      std::move(planFragment),
      0, // partition number, irrelevant here; will be set by the test.
      std::move(queryCtx),
      Task::ExecutionMode::kParallel);

  return task;
}

std::shared_ptr<facebook::velox::exec::Task> createExchangeTask(
    const std::string& taskId,
    facebook::velox::RowTypePtr rowType,
    int partitionId,
    core::PlanNodeId& exchangeNodeId) {
  auto planFragment = exec::test::PlanBuilder()
                          .exchange(rowType, VectorSerde::Kind::kCompactRow)
                          .capturePlanNodeId(exchangeNodeId)
                          .planFragment();

  std::unordered_map<std::string, std::string> configSettings;
  auto queryCtx = core::QueryCtx::create(
      nullptr, core::QueryConfig(std::move(configSettings)));

  auto task = Task::create(
      taskId,
      std::move(planFragment),
      partitionId,
      std::move(queryCtx),
      Task::ExecutionMode::kParallel);
  return task;
}

std::shared_ptr<Task> createPartitionedOutputTask(
    const std::string& taskId,
    std::shared_ptr<memory::MemoryPool> pool,
    RowTypePtr rowType,
    int numPartitions,
    const std::vector<std::string>& partitionKeys,
    uint64_t kMaxOutputBufferSize) {
  VLOG(3) << "Creating PartitionedOutput task with " << numPartitions
          << " partitions";

  const size_t vectorSize = 10;

  // Create a dummy row vector for the Values node (required by PlanBuilder)
  auto typeParams = rowType->parameters();
  std::vector<VectorPtr> vecPtrs;
  for (auto& typeParam : typeParams) {
    vecPtrs.emplace_back(
        BaseVector::create(typeParam.type, vectorSize, pool.get()));
  }

  auto rowVector = std::make_shared<RowVector>(
      pool.get(),
      rowType,
      BufferPtr(nullptr),
      vectorSize,
      vecPtrs);

  // Build the plan: Values -> PartitionedOutput
  auto planFragment = exec::test::PlanBuilder()
                          .values({rowVector})
                          .partitionedOutput(partitionKeys, numPartitions)
                          .planFragment();

  std::shared_ptr<folly::Executor> executor(
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency()));

  std::unordered_map<std::string, std::string> configSettings{
      {velox::core::QueryConfig::kMaxOutputBufferSize,
       std::to_string(kMaxOutputBufferSize)}};

  auto queryCtx = core::QueryCtx::create(
      executor.get(), core::QueryConfig(std::move(configSettings)));

  auto task = Task::create(
      taskId,
      std::move(planFragment),
      0, // partition number
      std::move(queryCtx),
      Task::ExecutionMode::kParallel);

  return task;
}

std::shared_ptr<cudf_velox::CudfVector> makeCudfVector(
    memory::MemoryPool* pool,
    size_t numRows,
    RowTypePtr rowType,
    std::shared_ptr<BaseTableGenerator> tableGenerator,
    rmm::cuda_stream_view stream) {
  // Create table using either makeTable or tableGenerator->makeTable()
  std::unique_ptr<cudf::table> table;
  if (tableGenerator == nullptr) {
    table = makeTable(numRows, rowType, stream);
  } else {
    table = tableGenerator->makeTable(stream);
  }

  // Sync the stream before creating CudfVector
  stream.synchronize();

  // Create and return CudfVector
  return std::make_shared<cudf_velox::CudfVector>(
      pool,
      rowType,
      static_cast<vector_size_t>(numRows),
      std::move(table),
      stream);
}

template <typename T>
std::unique_ptr<cudf::column> make_numeric_column_from_vector(
    const std::vector<T>& host_values,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr =
        rmm::mr::get_current_device_resource()) {
  size_t num_rows = host_values.size();

  // Allocate a device buffer of the correct size
  rmm::device_buffer data(num_rows * sizeof(T), stream, mr);

  // Copy host -> device
  cudaMemcpyAsync(
      data.data(),
      host_values.data(),
      num_rows * sizeof(T),
      cudaMemcpyHostToDevice,
      stream.value());

  // Build the cudf::column from the device buffer
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_to_id<T>()}, // e.g. cudf::type_id::FLOAT64
      num_rows,
      std::move(data),
      rmm::device_buffer{}, // no null mask
      0); // no nulls
}

// Creates device buffer with concatenated string bytes from host vector
rmm::device_buffer make_chars_buffer_from_host(
    const std::vector<std::string>& host_strings,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr =
        rmm::mr::get_current_device_resource()) {
  // Compute total bytes needed
  size_t total_bytes = 0;
  for (auto const& s : host_strings)
    total_bytes += s.size();

  // Allocate device buffer of appropriate size
  rmm::device_buffer chars_buffer(total_bytes, stream, mr);

  // Copy host string data into one contiguous host buffer first
  std::vector<char> host_concat;
  host_concat.reserve(total_bytes);
  for (auto const& s : host_strings)
    host_concat.insert(host_concat.end(), s.begin(), s.end());

  // Copy host -> device
  cudaMemcpyAsync(
      chars_buffer.data(),
      host_concat.data(),
      total_bytes,
      cudaMemcpyHostToDevice,
      stream.value());

  return chars_buffer;
}

std::unique_ptr<cudf::column> make_strings_column_from_host(
    const std::vector<std::string>& host_strings) {
  auto num_rows = host_strings.size();

  // --- Create offsets array ---
  std::vector<int32_t> h_offsets(num_rows + 1);
  h_offsets[0] = 0;
  for (size_t i = 0; i < num_rows; ++i)
    h_offsets[i + 1] =
        h_offsets[i] + static_cast<int32_t>(host_strings[i].size());

  // Copy offsets to device
  auto offsets_col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      num_rows + 1,
      cudf::mask_state::UNALLOCATED);

  cudaMemcpy(
      offsets_col->mutable_view().data<int32_t>(),
      h_offsets.data(),
      sizeof(int32_t) * (num_rows + 1),
      cudaMemcpyHostToDevice);

  // --- Create chars buffer ---
  auto chars_buffer = make_chars_buffer_from_host(host_strings);

  // --- Build strings column ---
  return cudf::make_strings_column(
      num_rows,
      std::move(offsets_col),
      std::move(chars_buffer),
      0, // null_count
      rmm::device_buffer{}); // null mask
}

std::unique_ptr<cudf::table> makeTable(
    std::size_t numRows,
    facebook::velox::RowTypePtr rowType,
    rmm::cuda_stream_view stream) {
  // Create table with default values
  std::vector<std::unique_ptr<cudf::column>> columns;

  for (size_t i = 0; i < rowType->size(); ++i) {
    const std::string& name = rowType->nameOf(i);
    const auto& type = rowType->childAt(i);

    cudf::type_id cudfType;
    std::unique_ptr<cudf::column> col;

    switch (type->kind()) {
      case TypeKind::INTEGER: {
        cudfType = cudf::type_id::INT32;
        std::vector<uint32_t> values(numRows);
        col = make_numeric_column_from_vector(values);
        break;
      }
      case TypeKind::DOUBLE: {
        cudfType = cudf::type_id::FLOAT64;
        std::vector<float> values(numRows);
        col = make_numeric_column_from_vector(values);

        break;
      }
      case TypeKind::VARCHAR: {
        std::vector<std::string> myStrings(numRows);
        col = make_strings_column_from_host(myStrings);
        break;
      }
      default:
        VLOG(0) << "Unhandled type " << type->kind();
        break;
    }

    columns.push_back(std::move(col));
  }

  return std::make_unique<cudf::table>(std::move(columns));
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
  auto const h_offsets = cudf::detail::make_host_vector_async(
      cudf::device_span<cudf::size_type const>(ptr_offsets_data, max_rows + 1),
      stream);
  const cudf::size_type* host_offsets = h_offsets.data();

  auto const total_num_bytes = std::distance(
      str_column_view.chars_begin(stream), str_column_view.chars_end(stream));
  char const* ptr_all_bytes = str_column_view.chars_begin(stream);
  // copy the bytes to host
  auto const h_bytes = cudf::detail::make_host_vector_async(
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

} // namespace facebook::velox::cudf_exchange
