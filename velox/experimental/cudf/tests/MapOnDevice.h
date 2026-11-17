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

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <vector>

namespace facebook::velox::cudf_velox {

/// Runs `kernel` over a device copy of `input` and brings the results back.
/// `kernel` receives the device input, a device output buffer of the same
/// length, and that length, and launches whatever it needs on them.
///
/// Shared by the device tests because each has the same shape -- copy up, run,
/// copy down -- and because a hand-rolled cudaMalloc per test is where leaks
/// and missing synchronization come from.
template <typename TIn, typename TOut, typename Kernel>
std::vector<TOut> mapOnDevice(const std::vector<TIn>& input, Kernel kernel) {
  const int count = static_cast<int>(input.size());
  TIn* deviceInput{nullptr};
  TOut* deviceOutput{nullptr};
  EXPECT_EQ(cudaMalloc(&deviceInput, count * sizeof(TIn)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&deviceOutput, count * sizeof(TOut)), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          deviceInput,
          input.data(),
          count * sizeof(TIn),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  kernel(deviceInput, deviceOutput, count);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << cudaGetErrorString(cudaGetLastError());

  std::vector<TOut> output(count);
  EXPECT_EQ(
      cudaMemcpy(
          output.data(),
          deviceOutput,
          count * sizeof(TOut),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cudaFree(deviceInput), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceOutput), cudaSuccess);
  return output;
}

} // namespace facebook::velox::cudf_velox
