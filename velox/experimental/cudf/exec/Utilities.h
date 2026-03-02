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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

// Concatenate a vector of cuDF tables into a single table
[[nodiscard]] std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Concatenate a vector of cuDF tables into a single table.
// This function joins the streams owned by individual tables on the passed
// stream. Inputs are not safe to use after calling this function.
[[nodiscard]] std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Concatenates multiple CUDF tables with automatic batching based on
 * size limits.
 *
 * This function concatenates a vector of CUDF tables while respecting size
 * limits imposed by cudf::size_type i.e. 32-bit signed integer. Unlike
 * getConcatenatedTable that returns a single concatenated table, this batched
 * version splits the concatenation into multiple output tables when the total
 * number of rows would exceeds ~2.1 billion, the maximum value representable by
 * cudf::size_type
 *
 * The function is stream-safe and handles proper stream synchronization. All
 * input streams from individual tables are collected and joined on the provided
 * output stream. Tables that may have been created on different CUDA streams
 * are also properly synchronized.
 *
 * @param tables Input vector of CUDF tables to concatenate (consumed during
 * operation)
 * @param tableType Velox type representation for creating empty tables when
 * needed
 * @param stream CUDA stream for asynchronous operations and memory management
 * @return Vector of concatenated tables (multiple if input exceeded size
 * limits)
 *
 */
[[nodiscard]] std::vector<std::unique_ptr<cudf::table>>
getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Wrapper for CUDA events used for stream synchronization.
 *
 * CudaEvent provides a safe, move-only wrapper around cudaEvent_t that
 * automatically manages the event lifecycle.
 *
 * @note This class is non-copyable but move-constructible, ensuring exclusive
 * ownership of the underlying CUDA event.
 *
 * Example usage:
 * @code
 *   // Create an event
 *   CudaEvent event;
 *
 *   // Record the event on stream1 after some work
 *   // ... launch kernels on stream1 ...
 *   event.recordFrom(stream1);
 *
 *   // Make stream2 wait for the event before proceeding
 *   event.waitOn(stream2);
 *   // ... launch kernels on stream2 (will wait for stream1 to reach event) ...
 * @endcode
 */
class CudaEvent {
 public:
  /**
   * @brief Constructs a CUDA event with the specified flags.
   *
   * @param flags Optional flags for event creation (default: 0).
   *              Common flags include:
   *              - cudaEventDefault: Default event creation flag
   *              - cudaEventBlockingSync: Use blocking synchronization
   *              - cudaEventDisableTiming: Event will not record timing data
   *              - cudaEventInterprocess: Event may be used for interprocess
   *
   * @throws May throw if CUDA event creation fails
   */
  explicit CudaEvent(unsigned int flags = 0);

  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;

  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept;

  /**
   * @brief Records this event in the specified CUDA stream.
   *
   * The event will be recorded after all previously issued operations in the
   * stream have completed. This does not block the host thread or the stream.
   *
   * @param stream The CUDA stream in which to record the event
   * @return Reference to this CudaEvent for method chaining
   */
  const CudaEvent& recordFrom(rmm::cuda_stream_view stream) const;

  /**
   * @brief Makes the specified stream wait until this event has been recorded.
   *
   * All future operations in the specified stream will wait until this event
   * has been recorded (i.e., all operations before the recordFrom() call have
   * completed). This enables synchronization between different CUDA streams
   * without blocking the host thread.
   *
   * @param stream The CUDA stream that should wait for this event
   * @return Reference to this CudaEvent for method chaining
   */
  const CudaEvent& waitOn(rmm::cuda_stream_view stream) const;

 private:
  cudaEvent_t event_{};
};
} // namespace facebook::velox::cudf_velox
