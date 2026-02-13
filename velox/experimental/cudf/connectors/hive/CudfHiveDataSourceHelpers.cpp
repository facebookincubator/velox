/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.hpp"

#include "velox/dwio/common/BufferedInput.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <folly/futures/Future.h>

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back(
      [dst, size, sharedStream](rmm::cuda_stream_view stream) {
        std::vector<uint8_t> buffer(size);
        sharedStream->readFully(reinterpret_cast<char*>(buffer.data()), size);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst, buffer.data(), size, cudaMemcpyHostToDevice, stream.value()));
      });
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);
  for (auto& deviceLoad : pendingDeviceLoads_) {
    deviceLoad(stream);
  }
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyHostToDevice,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

referenceToNameConverter::referenceToNameConverter(
    std::optional<std::reference_wrapper<const cudf::ast::expression>> expr,
    const std::vector<cudf::io::parquet::SchemaElement>& schemaTree,
    cudf::host_span<const std::string> readColumnNames) {
  if (not expr.has_value()) {
    return;
  }
  // Map column indices to their names
  if (not readColumnNames.empty()) {
    std::transform(
        readColumnNames.begin(),
        readColumnNames.end(),
        thrust::counting_iterator<cudf::size_type>(0),
        std::inserter(indicesToColumnNames_, indicesToColumnNames_.end()),
        [](const auto& columnName, const auto colIndex) {
          return std::make_pair(colIndex, columnName);
        });
  } else {
    const auto& root = schemaTree.front();
    std::for_each(
        thrust::counting_iterator(0),
        thrust::counting_iterator<cudf::size_type>(root.children_idx.size()),
        [&](int32_t colIndex) {
          const auto schemaIdx = root.children_idx[colIndex];
          indicesToColumnNames_.insert({colIndex, schemaTree[schemaIdx].name});
        });
  }

  expr.value().get().accept(*this);
}

std::reference_wrapper<const cudf::ast::expression>
referenceToNameConverter::visit(const cudf::ast::literal& expr) {
  return expr;
}

std::reference_wrapper<const cudf::ast::expression>
referenceToNameConverter::visit(const cudf::ast::column_reference& expr) {
  const auto columnIdx = expr.get_column_index();
  const auto columnName = indicesToColumnNames_.find(columnIdx);
  VELOX_CHECK(
      columnName != indicesToColumnNames_.end(), "Column index not found");
  convertedExpr_.push(cudf::ast::column_name_reference{columnName->second});
  return std::reference_wrapper<const cudf::ast::expression>(
      convertedExpr_.back());
}

std::reference_wrapper<const cudf::ast::expression>
referenceToNameConverter::visit(const cudf::ast::column_name_reference& expr) {
  return expr;
}

std::reference_wrapper<const cudf::ast::expression>
referenceToNameConverter::visit(const cudf::ast::operation& expr) {
  const auto operands = expr.get_operands();
  auto op = expr.get_operator();
  auto newOperands = visitOperands(operands);
  const auto operatorArity = cudf::ast::detail::ast_operator_arity(op);
  if (operatorArity == 2) {
    convertedExpr_.push(
        cudf::ast::operation{op, newOperands.front(), newOperands.back()});
  } else if (operatorArity == 1) {
    convertedExpr_.push(cudf::ast::operation{op, newOperands.front()});
  }
  return convertedExpr_.back();
}

std::reference_wrapper<const cudf::ast::expression>
referenceToNameConverter::convertedExpression() const {
  return convertedExpr_.back();
}

std::vector<std::reference_wrapper<const cudf::ast::expression>>
referenceToNameConverter::visitOperands(
    cudf::host_span<const std::reference_wrapper<const cudf::ast::expression>>
        operands) {
  std::vector<std::reference_wrapper<const cudf::ast::expression>>
      transformedOperands;
  for (const auto& operand : operands) {
    const auto newOperand = operand.get().accept(*this);
    transformedOperands.push_back(newOperand);
  }
  return transformedOperands;
}

std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource) {
  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len = sizeof(file_ender_s);
  const size_t len = dataSource->size();

  const auto header_buffer = dataSource->host_read(0, header_len);
  const auto ender_buffer = dataSource->host_read(len - ender_len, ender_len);
  const auto header =
      reinterpret_cast<const file_header_s*>(header_buffer->data());
  const auto ender =
      reinterpret_cast<const file_ender_s*>(ender_buffer->data());
  VELOX_CHECK(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic =
      (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  VELOX_CHECK(
      header->magic == parquet_magic && ender->magic == parquet_magic,
      "Corrupted header or footer");
  VELOX_CHECK(
      ender->footer_len != 0 &&
          ender->footer_len <= (len - header_len - ender_len),
      "Incorrect footer length");

  return dataSource->host_read(
      len - ender->footer_len - ender_len, ender->footer_len);
}

std::vector<std::unique_ptr<cudf::io::datasource>>
makeDataSourcesFromSourceInfo(
    const cudf::io::source_info& info,
    size_t offset,
    size_t maxSizeEstimate) {
  switch (info.type()) {
    case cudf::io::io_type::FILEPATH: {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.reserve(info.filepaths().size());
      for (auto const& filepath : info.filepaths()) {
        sources.emplace_back(
            cudf::io::datasource::create(filepath, offset, maxSizeEstimate));
      }
      return sources;
    }
    case cudf::io::io_type::HOST_BUFFER:
      return cudf::io::datasource::create(info.host_buffers());
    case cudf::io::io_type::DEVICE_BUFFER:
      return cudf::io::datasource::create(info.device_buffers());
    case cudf::io::io_type::USER_IMPLEMENTED:
      return cudf::io::datasource::create(info.user_sources());
    default:
      CUDF_FAIL("Unsupported source type");
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive
