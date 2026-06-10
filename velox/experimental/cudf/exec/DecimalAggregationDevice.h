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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace facebook::velox::cudf_velox::detail {

template <typename T>
inline constexpr bool isDecimalSumStorageType =
    std::is_same_v<T, int64_t> || std::is_same_v<T, __int128_t>;

template <typename T>
inline constexpr bool isOffsetStorageType =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

// Size in bytes of each row's packed decimal SUM intermediate state in the
// strings payload (count, overflow placeholder, and 128-bit sum split into
// words).
constexpr size_t kDecimalSumStateSize = 32;

/**
 * Writes strings-style prefix offsets: offset[i] == i * kDecimalSumStateSize.
 *
 * @param offsetsView output offsets column of numRows + 1 elements.
 * @param numRows number of payload rows.
 * @param stream CUDA stream for the launch.
 */
struct fillOffsetsForDecimalSumState {
  template <
      typename OffsetT,
      std::enable_if_t<isOffsetStorageType<OffsetT>, int> = 0>
  void operator()(
      cudf::mutable_column_view offsetsView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;

  template <
      typename OffsetT,
      std::enable_if_t<!isOffsetStorageType<OffsetT>, int> = 0>
  void operator()(
      cudf::mutable_column_view offsetsView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const {}
};

/**
 * Encodes each row's partial sum and count into the fixed-width device layout
 * used for VARBINARY interchange.
 *
 * @param sumCol per-row sums.
 * @param counts per-row int64 counts.
 * @param offsetsView per-row byte offsets into chars.
 * @param chars output payload buffer.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
struct packDecimalSumState {
  template <
      typename SumT,
      std::enable_if_t<isDecimalSumStorageType<SumT>, int> = 0>
  void operator()(
      cudf::column_view sumCol,
      const int64_t* counts,
      cudf::column_view offsetsView,
      uint8_t* chars,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;

  template <
      typename SumT,
      std::enable_if_t<!isDecimalSumStorageType<SumT>, int> = 0>
  void operator()(
      cudf::column_view sumCol,
      const int64_t* counts,
      cudf::column_view offsetsView,
      uint8_t* chars,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const {}
};

/**
 * Inverse of packDecimalSumState.
 *
 * @param offsetsView per-row byte offsets into chars.
 * @param chars packed payload buffer.
 * @param sumView output per-row DECIMAL128 sums.
 * @param countView output per-row counts.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
struct unpackDecimalSumState {
  template <
      typename OffsetT,
      std::enable_if_t<isOffsetStorageType<OffsetT>, int> = 0>
  void operator()(
      cudf::column_view offsetsView,
      const uint8_t* chars,
      cudf::mutable_column_view sumView,
      cudf::mutable_column_view countView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;

  template <
      typename OffsetT,
      std::enable_if_t<!isOffsetStorageType<OffsetT>, int> = 0>
  void operator()(
      cudf::column_view offsetsView,
      const uint8_t* chars,
      cudf::mutable_column_view sumView,
      cudf::mutable_column_view countView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const {}
};

/**
 * Per-row half-up integer divide of sum by count; count == 0 writes zero
 * (validity is applied separately).
 *
 * @param sumCol per-row sums.
 * @param counts per-row counts.
 * @param outView output per-row averages.
 * @param numRows number of rows.
 * @param stream CUDA stream for the launch.
 */
struct averageRoundDecimalSum {
  template <
      typename SumT,
      std::enable_if_t<isDecimalSumStorageType<SumT>, int> = 0>
  void operator()(
      cudf::column_view sumCol,
      const int64_t* counts,
      cudf::mutable_column_view outView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const;

  template <
      typename SumT,
      std::enable_if_t<!isDecimalSumStorageType<SumT>, int> = 0>
  void operator()(
      cudf::column_view sumCol,
      const int64_t* counts,
      cudf::mutable_column_view outView,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream) const {}
};

/**
 * Builds a null mask for rows where sum and count are both valid and count is
 * non-zero, for serializing state or finalizing averages.
 *
 * @param sumCol decoded sum column.
 * @param countCol decoded count column.
 * @param stream CUDA stream for the launch.
 * @param mr memory resource for the returned mask.
 * @return {null mask buffer, null count}.
 */
std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::detail
