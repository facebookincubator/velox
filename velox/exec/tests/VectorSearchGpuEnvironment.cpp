/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <gtest/gtest.h>

#include "velox/common/vectorindex/GpuLocalVectorIndex.h"

namespace facebook::velox::exec {
namespace {

using namespace common::vectorindex;

// Runs the vector search tests with every local index they build and read on
// the GPU, so they check that the GPU gives the CPU's results.
class VectorSearchGpuEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    if (gpuLocalVectorIndexDevices() == 0) {
      GTEST_SKIP() << "No CUDA device";
    }
    setLocalVectorIndexAccelerator(makeGpuLocalVectorIndexAccelerator({
        .minSearchSize = 0,
        .build = true,
        .minBuildSize = 0,
    }));
  }

  void TearDown() override {
    setLocalVectorIndexAccelerator(nullptr);
  }
};

const auto* const kEnvironment =
    testing::AddGlobalTestEnvironment(new VectorSearchGpuEnvironment());

} // namespace
} // namespace facebook::velox::exec
