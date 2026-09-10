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

#include <gtest/gtest.h>

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

} // namespace
} // namespace torch::wave
