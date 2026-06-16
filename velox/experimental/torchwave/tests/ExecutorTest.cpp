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

// aten.tensor / scalar_tensor (0-d tensor from a symbolic size) feeding an
// add.tensor, plus _to_copy applied to the input tensor and to the 0-d tensor
// with a dtype change.
TEST_F(ExecutorTest, tensorTest) {
  runTest("data/tensor_test.pt2", "data/tensor_test_results.pt");
}

// aten.sym_numel (element count of a dynamic tensor) used two ways: fed through
// scalar_tensor into an add (broadcast), and returned directly to host as an
// int.
TEST_F(ExecutorTest, numelTest) {
  runTest("data/numel_test.pt2", "data/numel_test_results.pt");
}

// Fused elementwise ops interleaved with view-like breaks (view, slice,
// select.int), like the ROO dense-feature preproc chain. The view/slice/select
// ops break the fused kernels and run host-side, so the wave executor emits one
// node with many steps alternating fused code and view breaks. The [:, :K]
// slices feed the next kernel non-contiguously and (via reshape) a
// clone-then-view break.
TEST_F(ExecutorTest, viewInterleaveTest) {
  runTest(
      "data/view_interleave_test.pt2", "data/view_interleave_test_results.pt");
}

// In-place mutation through views and clones. Validates that torchwave honors
// the imperative order of in-place ops (add_) on aliased storage and keeps
// clones from being eliminated when their source is mutated later.
TEST_F(ExecutorTest, inPlaceTest) {
  runTest("data/in_place_test.pt2", "data/in_place_test_results.pt");
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

TEST_F(ExecutorTest, cat2dViewTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cat_2d_view_test.pt2",
      "data/cat_2d_view_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_2d_view_test.pt2",
      "data/cat_2d_view_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cat_2d_view_test.pt2", "data/cat_2d_view_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, cat2dReuseTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cat_2d_reuse_test.pt2",
      "data/cat_2d_reuse_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_2d_reuse_test.pt2",
      "data/cat_2d_reuse_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cat_2d_reuse_test.pt2", "data/cat_2d_reuse_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, arangeTest) {
  runTest("data/arange_test.pt2", "data/arange_test_results.pt");
}

TEST_F(ExecutorTest, indexTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/index_test.pt2", "data/index_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest("data/index_test.pt2", "data/index_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/index_test.pt2", "data/index_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;

  // Error injection: set one index out of range and verify error is reported.
  {
    WaveConfig::get().throwOnError = false;
    auto baseDir = dataDir();
    auto pt2Path = getDataFilePath(baseDir, "data/index_test.pt2");
    auto fixture = ModelFixture::load(pt2Path);
    WaveGraphExecutor waveExec(fixture->makeModelContext());
    auto& graph = waveExec.graph();
    auto pooledFrame = waveExec.getFrame();
    auto inputs = loadSampleInputs(*fixture);
    auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);
    fillWaveFrame(graph, *pooledFrame, deviceInputs);
    auto values = graph.userInputs();
    auto names = graph.signature().userInputs();
    for (size_t i = 0; i < values.size() && i < names.size(); ++i) {
      if (names[i].find("idx") == std::string::npos) {
        continue;
      }
      auto& iv = pooledFrame->getIValue(values[i]->id());
      if (iv.isTensor()) {
        auto& tensor = iv.toTensor();
        if (tensor.dim() == 1 && tensor.device().is_cuda()) {
          tensor[0] = 999'999;
          break;
        }
      }
    }
    waveExec.executeWithPrefilledFrame(*pooledFrame);
    waveExec.returnFrame(std::move(pooledFrame));
    auto& errors = waveThreadInfo().errors;
    EXPECT_FALSE(errors.empty()) << "Expected error from out-of-range index";
    LOG(INFO) << "index_put error injection result:\n" << errors;
    WaveConfig::get().throwOnError = true;
  }
}

TEST_F(ExecutorTest, elementShapeTest) {
  runTest("data/element_shape_test.pt2", "data/element_shape_test_results.pt");
}

TEST_F(ExecutorTest, elementShapeNcTest) {
  runTest(
      "data/element_shape_nc_test.pt2",
      "data/element_shape_nc_test_results.pt");
}

// Three-way broadcast across different ranks: [100] + [20,1] + [10,1,1].
TEST_F(ExecutorTest, elementShapeTest3) {
  runTest(
      "data/element_shape_test3.pt2", "data/element_shape_test3_results.pt");
}

TEST_F(ExecutorTest, elementTest2) {
  runTest("data/element_test2.pt2", "data/element_test2_results.pt");
}

TEST_F(ExecutorTest, isinTest) {
  runTest("data/isin_test.pt2", "data/isin_test_results.pt");
}

TEST_F(ExecutorTest, nonzeroTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/nonzero_test.pt2", "data/nonzero_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/nonzero_test.pt2", "data/nonzero_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/nonzero_test.pt2", "data/nonzero_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, maskedPutTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/masked_put_test.pt2",
      "data/masked_put_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/masked_put_test.pt2",
      "data/masked_put_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/masked_put_test.pt2", "data/masked_put_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, indexGetTest) {
  runTest("data/index_get_test.pt2", "data/index_get_test_results.pt");
}

TEST_F(ExecutorTest, dedupTest) {
  runTest("data/dedup_test.pt2", "data/dedup_test_results.pt");
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
  // Reload fixture since makeModelContext moves the graph.
  fixture = ModelFixture::load(pt2Path);
  setGraphDevice(fixture->model.graph.get(), true);
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

// Per-fix isolating tests for the ads-preproc torchwave fixes (each fails when
// its fix is reverted).  Plain torch.export models exercised via runTest.
// Mixed-dtype min/max operations: int32 and int64 inputs.
TEST_F(ExecutorTest, mixedTypeMinMaxTest) {
  runTest(
      "data/mixed_type_minmax_test.pt2",
      "data/mixed_type_minmax_test_results.pt");
}

// Clone with contiguous memory_format must not be elided.
TEST_F(ExecutorTest, cloneContiguousTest) {
  runTest(
      "data/clone_contiguous_test.pt2",
      "data/clone_contiguous_test_results.pt");
}

// Empty-tensor broadcast: [8,1] + [1,0] -> [8,0] without reading null storage.
TEST_F(ExecutorTest, broadcastEmptyTest) {
  runTest(
      "data/broadcast_empty_test.pt2", "data/broadcast_empty_test_results.pt");
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
