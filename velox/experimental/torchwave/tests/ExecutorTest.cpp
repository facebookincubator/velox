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

#include "velox/experimental/torchwave/tests/ExecutorTestBase.h"

#include <unistd.h>

#include <cuda_runtime.h> // @manual
#include <fmt/format.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "velox/experimental/torchwave/WaveConfig.h"

DEFINE_string(
    custom,
    "",
    "Custom test model base name (without .pt2/.pt extension)");
DECLARE_string(reference_frame);

namespace torch::wave {
namespace {

class ExecutorTest : public ExecutorTestBase {};

TEST_F(ExecutorTest, elementTest) {
  runTest("data/element_test.pt2", "data/element_test_results.pt");
}

TEST_F(ExecutorTest, maskedSelectTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/masked_select_test.pt2",
      "data/masked_select_test_results.pt",
      "single");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/masked_select_test.pt2",
      "data/masked_select_test_results.pt",
      "3 step");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/masked_select_test.pt2",
      "data/masked_select_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, sumTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/sum_test.pt2", "data/sum_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest("data/sum_test.pt2", "data/sum_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/sum_test.pt2", "data/sum_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, cumsumTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/cumsum_test.pt2", "data/cumsum_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cumsum_test.pt2", "data/cumsum_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cumsum_test.pt2", "data/cumsum_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, exclusiveSumTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/exclusive_sum_test.pt2",
      "data/exclusive_sum_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/exclusive_sum_test.pt2",
      "data/exclusive_sum_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/exclusive_sum_test.pt2",
      "data/exclusive_sum_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, repeatInterleaveTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/repeat_interleave_test.pt2",
      "data/repeat_interleave_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/repeat_interleave_test.pt2",
      "data/repeat_interleave_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/repeat_interleave_test.pt2",
      "data/repeat_interleave_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, catTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, catTest2) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/cat_test2.pt2", "data/cat_test2_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest("data/cat_test2.pt2", "data/cat_test2_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cat_test2.pt2", "data/cat_test2_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, arangeTest) {
  runTest("data/arange_test.pt2", "data/arange_test_results.pt");
}

TEST_F(ExecutorTest, elementShapeTest) {
  runTest("data/element_shape_test.pt2", "data/element_shape_test_results.pt");
}

TEST_F(ExecutorTest, elementShapeNcTest) {
  runTest(
      "data/element_shape_nc_test.pt2",
      "data/element_shape_nc_test_results.pt");
}

TEST_F(ExecutorTest, elementTest2) {
  runTest("data/element_test2.pt2", "data/element_test2_results.pt");
}

TEST_F(ExecutorTest, isinTest) {
  runTest("data/isin_test.pt2", "data/isin_test_results.pt");
}

TEST_F(ExecutorTest, largeElementTest) {
  runTest("data/large_element_test.pt2", "data/large_element_test_results.pt");
}

TEST_F(ExecutorTest, referenceFrame) {
  auto pt2Path = getDataFilePath(dataDir(), "data/element_test.pt2");
  auto resultsPath = getDataFilePath(dataDir(), "data/element_test_results.pt");

  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);

  auto expected = loadReferenceValues(resultsPath);
  setGraphDevice(fixture->model.graph.get(), true);

  auto refPath =
      fmt::format("/tmp/torchwave_ref_frame_{}.pt", static_cast<int>(getpid()));

  // First wave run: save intermediates as the reference frame.
  WaveConfig::get().saveReferenceFramePath = refPath;
  runWave(*fixture, expected);

  // Second wave run: verify intermediates match the reference.
  FLAGS_reference_frame = refPath;
  runWave(*fixture, expected);
  FLAGS_reference_frame = "";

  LOG(INFO) << "Reference frame: " << lastRefTensorsChecked_ << " tensors, "
            << lastRefNodesChecked_ << " nodes checked";
  EXPECT_GT(lastRefTensorsChecked_, 0);
  EXPECT_GT(lastRefNodesChecked_, 0);

  std::remove(refPath.c_str());
}

TEST_F(ExecutorTest, custom) {
  if (FLAGS_custom.empty()) {
    return;
  }
  runTest(FLAGS_custom + ".pt2", FLAGS_custom + "_results.pt");
}

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  if (int device; cudaGetDevice(&device) != cudaSuccess) {
    LOG(WARNING) << "No CUDA detected, skipping all tests";
    return 0;
  }
  return RUN_ALL_TESTS();
}
