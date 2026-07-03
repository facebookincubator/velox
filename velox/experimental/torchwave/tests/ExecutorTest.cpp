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
DEFINE_string(
    save_model,
    "",
    "With --custom: save the graph as <save_model>.pt2 and a synthetic-data "
    "spec as <save_model>.spec");
DEFINE_string(
    run_synthetic,
    "",
    "Load the graph from <run_synthetic>.pt2 and the spec from "
    "<run_synthetic>.spec, generate synthetic data, and run nativert-GPU "
    "reference vs wave");
DEFINE_int64(
    synthetic_seed,
    0,
    "Seed for --run_synthetic data generation (deterministic)");
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

// slice_scatter on 2-D tensors along dim 0 and dim 1 with a runtime (symint)
// start, step > 1, lowered to a clone + fused in-place tw.slice_scatter_.
TEST_F(ExecutorTest, scatterTest) {
  runTest("data/scatter_test.pt2", "data/scatter_test_results.pt");

  // Error injection: corrupt the dim-0 slice start (read via .item() into the
  // scatter start arg) to an out-of-range value and verify the device-side
  // bounds check in __slice_scatter fires and is reported as "Bad idx".
  {
    auto pt2Path = getDataFilePath(dataDir(), "data/scatter_test.pt2");
    auto fixture = ModelFixture::load(pt2Path);
    // Capture the 'start0' input (a 0-D int tensor) before the graph is moved
    // into the executor; the alterInputs callback only receives the frame.
    int32_t startValueId = -1;
    {
      auto values = fixture->model.graph->userInputs();
      auto names = fixture->model.graph->signature().userInputs();
      for (size_t i = 0; i < values.size() && i < names.size(); ++i) {
        if (names[i].find("start0") != std::string::npos) {
          startValueId = values[i]->id();
          break;
        }
      }
    }
    ASSERT_GE(startValueId, 0) << "No start0 input found in scatter_test graph";
    auto errors = runWaveExpectError(
        *fixture, [startValueId](nativert::ExecutionFrame& frame) {
          auto& iv = frame.getIValue(startValueId);
          if (iv.isTensor()) {
            iv.toTensor().fill_(999'999);
          }
        });
    EXPECT_NE(errors.find("Bad idx"), std::string::npos)
        << "Expected a 'Bad idx' device error, got:\n"
        << errors;
  }
}

// logit (inverse sigmoid), with eps=None and with an eps clamp, as a fused
// elementwise op.
TEST_F(ExecutorTest, logitTest) {
  runTest("data/logit_test.pt2", "data/logit_test_results.pt");
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

  // Error injection: corrupt one index tensor per case so its values fall out
  // of range, and verify the device-side bounds check in __index_put_elt_*
  // reports the correct dimension. Corrupting the 1-D index errors on dim 0;
  // corrupting only the dim-1 index of the 2-D case errors on dim 1; corrupting
  // only the dim-2 index of the 3-D case errors on dim 2 (the other indices
  // stay in range, so __index_put_elt_two/three pick the corrupted dimension).
  // errorString() formats each erroring block as "... <dim> <badValue> Bad
  // idx".
  constexpr int32_t kBadIndex = 999'999;
  struct IndexErrorCase {
    std::string indexInput;
    int32_t expectedDim;
  };
  const std::vector<IndexErrorCase> indexErrorCases = {
      {"idx1d_0", 0},
      {"idx2d_1", 1},
      {"idx3d_2", 2},
  };
  for (const auto& errorCase : indexErrorCases) {
    auto pt2Path = getDataFilePath(dataDir(), "data/index_test.pt2");
    auto fixture = ModelFixture::load(pt2Path);
    // Capture the index input to corrupt before the graph is moved into the
    // executor; the alterInputs callback only receives the frame.
    int32_t idxValueId = -1;
    {
      auto values = fixture->model.graph->userInputs();
      auto names = fixture->model.graph->signature().userInputs();
      for (size_t i = 0; i < values.size() && i < names.size(); ++i) {
        if (names[i].find(errorCase.indexInput) != std::string::npos) {
          idxValueId = values[i]->id();
          break;
        }
      }
    }
    ASSERT_GE(idxValueId, 0)
        << "No " << errorCase.indexInput << " input found in index_test graph";
    auto errors = runWaveExpectError(
        *fixture, [idxValueId, kBadIndex](nativert::ExecutionFrame& frame) {
          auto& iv = frame.getIValue(idxValueId);
          if (iv.isTensor()) {
            iv.toTensor().fill_(kBadIndex);
          }
        });
    // Assert the bounds check fired on the expected dimension with the bad
    // index value echoed, not just that some error occurred.
    const std::string expected = std::to_string(errorCase.expectedDim) + " " +
        std::to_string(kBadIndex) + " Bad idx";
    EXPECT_NE(errors.find(expected), std::string::npos)
        << "Case " << errorCase.indexInput << ": expected '" << expected
        << "' in device errors, got:\n"
        << errors;
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

// Saves the --custom graph as <save_model>.pt2 and a synthetic-data spec as
// <save_model>.spec. The spec is analyzed from the model's sample inputs and
// weights so a later --run_synthetic can reproduce a same-shape dataset.
TEST_F(ExecutorTest, saveModel) {
  if (FLAGS_save_model.empty()) {
    return;
  }
  ASSERT_FALSE(FLAGS_custom.empty()) << "--save_model requires --custom";
  auto pt2Path = FLAGS_custom.front() == '/'
      ? FLAGS_custom + ".pt2"
      : getDataFilePath(dataDir(), FLAGS_custom + ".pt2");
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  auto inputs = loadSampleInputs(*fixture);
  saveSyntheticModel(*fixture, inputs, FLAGS_save_model);
}

// Loads a saved graph + spec, generates synthetic data, and checks wave against
// the nativert-GPU reference (outputs and reference frame).
TEST_F(ExecutorTest, runSynthetic) {
  if (FLAGS_run_synthetic.empty()) {
    return;
  }
  runSynthetic(
      FLAGS_run_synthetic,
      std::optional<uint64_t>(static_cast<uint64_t>(FLAGS_synthetic_seed)));
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
