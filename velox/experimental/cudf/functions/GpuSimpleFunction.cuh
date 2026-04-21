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

// Bridges GpuVectorFunction with GpuSimpleFunctionAdapter: takes cuDF columns,
// constructs readers/writers, combines null masks, launches the kernel, and
// returns a cuDF output column.
#pragma once

#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/device_uvector.hpp>

namespace facebook::velox::gpu {

namespace detail {

template <typename T>
cudf::type_id typeToId();

template <>
inline cudf::type_id typeToId<bool>() {
  return cudf::type_id::BOOL8;
}
template <>
inline cudf::type_id typeToId<int8_t>() {
  return cudf::type_id::INT8;
}
template <>
inline cudf::type_id typeToId<int16_t>() {
  return cudf::type_id::INT16;
}
template <>
inline cudf::type_id typeToId<int32_t>() {
  return cudf::type_id::INT32;
}
template <>
inline cudf::type_id typeToId<int64_t>() {
  return cudf::type_id::INT64;
}
template <>
inline cudf::type_id typeToId<float>() {
  return cudf::type_id::FLOAT32;
}
template <>
inline cudf::type_id typeToId<double>() {
  return cudf::type_id::FLOAT64;
}

__global__ void combineNullMasks(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* const* masks,
    int numMasks,
    cudf::size_type numRows) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numRows)
    return;

  cudf::size_type wordIdx = row / 32;
  cudf::size_type bitIdx = row % 32;
  cudf::bitmask_type bit = cudf::bitmask_type{1} << bitIdx;

  for (int m = 0; m < numMasks; ++m) {
    if (masks[m] && !(masks[m][wordIdx] & bit)) {
      output[wordIdx] &= ~bit;
      return;
    }
  }
}

template <typename T>
GpuColumnReader<T> makeReader(cudf::column_view col) {
  auto data = cudf::device_span<const T>(col.data<T>(), col.size());
  return GpuColumnReader<T>{data};
}

template <typename ReturnType, typename... ArgTypes, size_t... Is>
auto makeReaders(
    const std::vector<cudf::column_view>& inputs,
    std::index_sequence<Is...>) {
  return std::make_tuple(makeReader<ArgTypes>(inputs[Is])...);
}

} // namespace detail

template <typename FnType, typename ReturnType, typename... ArgTypes>
class GpuSimpleFunction : public GpuVectorFunction {
 public:
  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type numRows,
      const cudf::bitmask_type* activeRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto outputCol = cudf::make_fixed_width_column(
        cudf::data_type{detail::typeToId<ReturnType>()},
        numRows,
        cudf::mask_state::ALL_VALID,
        stream,
        mr);

    auto* outData =
        outputCol->mutable_view().template data<ReturnType>();
    auto* outNull = outputCol->mutable_view().null_mask();

    GpuColumnWriter<ReturnType> writer{outData, outNull};

    auto readers = detail::makeReaders<ReturnType, ArgTypes...>(
        inputs, std::index_sequence_for<ArgTypes...>{});

    const cudf::bitmask_type* combinedNull = nullptr;
    rmm::device_uvector<cudf::bitmask_type> combinedBuf(0, stream, mr);

    auto hasAnyNulls = buildCombinedNullMask(
        inputs, numRows, stream, mr, combinedBuf);
    if (hasAnyNulls) {
      combinedNull = combinedBuf.data();
    }

    launchKernel(
        writer,
        readers,
        combinedNull,
        activeRows,
        numRows,
        stream,
        std::index_sequence_for<ArgTypes...>{});

    stream.synchronize();
    return outputCol;
  }

 private:
  bool buildCombinedNullMask(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      rmm::device_uvector<cudf::bitmask_type>& buf) {
    std::vector<const cudf::bitmask_type*> masks;
    for (auto& col : inputs) {
      if (col.nullable()) {
        masks.push_back(col.null_mask());
      }
    }
    if (masks.empty()) {
      return false;
    }
    if (masks.size() == 1) {
      buf.resize(cudf::bitmask_allocation_size_bytes(numRows) /
                     sizeof(cudf::bitmask_type),
                 stream);
      cudaMemcpyAsync(
          buf.data(),
          masks[0],
          cudf::bitmask_allocation_size_bytes(numRows),
          cudaMemcpyDeviceToDevice,
          stream.value());
      return true;
    }

    size_t words = cudf::bitmask_allocation_size_bytes(numRows) /
        sizeof(cudf::bitmask_type);
    buf.resize(words, stream);
    cudaMemsetAsync(buf.data(), 0xFF, words * sizeof(cudf::bitmask_type),
                    stream.value());

    rmm::device_uvector<const cudf::bitmask_type*> dMasks(
        masks.size(), stream, mr);
    cudaMemcpyAsync(
        dMasks.data(),
        masks.data(),
        masks.size() * sizeof(const cudf::bitmask_type*),
        cudaMemcpyHostToDevice,
        stream.value());

    int blocks = (numRows + 255) / 256;
    detail::combineNullMasks<<<blocks, 256, 0, stream.value()>>>(
        buf.data(), dMasks.data(), masks.size(), numRows);
    return true;
  }

  template <typename ReaderTuple, size_t... Is>
  void launchKernel(
      GpuColumnWriter<ReturnType> writer,
      ReaderTuple& readers,
      const cudf::bitmask_type* combinedNull,
      const cudf::bitmask_type* activeRows,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      std::index_sequence<Is...>) {
    GpuSimpleFunctionAdapter<FnType, ReturnType, ArgTypes...>::apply(
        writer,
        std::get<Is>(readers)...,
        combinedNull,
        activeRows,
        numRows,
        stream.value());
  }
};

template <typename FnType, typename ReturnType, typename... ArgTypes>
void registerGpuFunction(const std::string& name) {
  GpuFunctionKey key{
      name,
      detail::typeToId<ReturnType>(),
      {detail::typeToId<ArgTypes>()...}};
  GpuFunctionRegistry::instance().registerFunction(
      std::move(key),
      []() -> std::unique_ptr<GpuVectorFunction> {
        return std::make_unique<
            GpuSimpleFunction<FnType, ReturnType, ArgTypes...>>();
      });
}

} // namespace facebook::velox::gpu
