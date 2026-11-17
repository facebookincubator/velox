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

#include "velox/experimental/torchwave/Model.h"

#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/tests/ExecutorTestBase.h"

namespace torch::wave {
namespace {

class ModelTest : public ExecutorTestBase {};

// Verifies the public TorchWaveModel load()/run() whole-graph entry point - the
// API a serving caller uses instead of an AOTInductor model container -
// produces the same outputs as the saved reference.
TEST_F(ModelTest, loadAndRun) {
  const auto pt2Path = getDataFilePath(dataDir(), "data/cat_2d_reuse_test.pt2");
  const auto expected = loadReferenceValues(
      getDataFilePath(dataDir(), "data/cat_2d_reuse_test_results.pt"));

  // Sample inputs come from the same archive (reuse the fixture loader).
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  auto inputs = loadSampleInputs(*fixture);

  auto model = TorchWaveModel::load(pt2Path);
  ASSERT_NE(model, nullptr);
  auto outputs = model->run(inputs);

  verifyOutputs(outputsToHost(outputs, "model"), expected, "TorchWaveModel");
}

// Runs with the inputs still being written on a side stream, as a trainer that
// preprocesses on its own stream would call it: the run must wait for that
// stream rather than for the legacy default stream.
TEST_F(ModelTest, runOnCallerStream) {
  const auto pt2Path = getDataFilePath(dataDir(), "data/cat_2d_reuse_test.pt2");
  const auto expected = loadReferenceValues(
      getDataFilePath(dataDir(), "data/cat_2d_reuse_test_results.pt"));
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  const auto sampleInputs = loadSampleInputs(*fixture);
  auto model = TorchWaveModel::load(pt2Path);
  ASSERT_NE(model, nullptr);

  auto sideStream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard cudaStreamGuard(sideStream);
  TorchStreamGuard torchStreamGuard(sideStream.stream());
  std::vector<c10::IValue> inputs;
  for (const auto& input : sampleInputs) {
    if (!input.isTensor()) {
      inputs.push_back(input);
      continue;
    }
    const auto source = input.toTensor().to(at::kCUDA);
    sideStream.synchronize();
    auto delayed = at::zeros_like(source);
    // About a second on current GPUs, far longer than the run takes to start.
    at::cuda::sleep(2'000'000'000);
    delayed.copy_(source);
    inputs.push_back(delayed);
  }
  auto outputs = model->run(inputs);

  verifyOutputs(outputsToHost(outputs, "model"), expected, "callerStream");
}

} // namespace
} // namespace torch::wave
