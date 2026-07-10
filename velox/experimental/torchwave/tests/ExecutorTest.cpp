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
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <torch/nativert/graph/Graph.h>

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
DECLARE_string(save_reference_frame);

namespace torch::wave {
namespace {

class ExecutorTest : public ExecutorTestBase {};

// --- Programmatic-graph helpers for the per-fix multi-block tests below ---
//
// These tests build a small nativert graph in memory with stringToGraph (no
// checked-in .pt2), attach the dtype/rank metadata the wave compiler requires,
// and run it through the wave executor on GPU, comparing against an ATen
// reference. stringToGraph yields topology only, so setTensorValuesMeta must
// supply dtype (mandatory) and rank; concrete sizes come from the runtime
// frame tensors, so the size entries here are placeholders.

torch::_export::ScalarType toExportScalarType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Long:
      return torch::_export::ScalarType::LONG;
    case c10::ScalarType::Int:
      return torch::_export::ScalarType::INT;
    case c10::ScalarType::Float:
      return torch::_export::ScalarType::FLOAT;
    case c10::ScalarType::Double:
      return torch::_export::ScalarType::DOUBLE;
    case c10::ScalarType::Bool:
      return torch::_export::ScalarType::BOOL;
    default:
      TORCH_CHECK(false, "unsupported dtype ", static_cast<int>(dtype));
  }
}

torch::_export::TensorMeta
makeTensorMeta(c10::ScalarType dtype, int64_t rank, int64_t sizeValue = 1) {
  torch::_export::TensorMeta meta;
  meta.set_dtype(toExportScalarType(dtype));
  meta.set_layout(torch::_export::Layout::Strided);
  meta.set_requires_grad(false);
  torch::_export::Device device;
  device.set_type("cuda");
  device.set_index(0);
  meta.set_device(std::move(device));
  torch::_export::SymInt zero;
  zero.set_as_int(0);
  meta.set_storage_offset(std::move(zero));
  // The number of size entries sets the rank. For a materialized input the
  // concrete sizes come from the runtime frame tensor, so 'sizeValue' is a
  // placeholder; it matters only for a None operand whose extent wave reads
  // from this metadata (e.g. a size-0 empty cat operand needs sizeValue == 0).
  std::vector<torch::_export::SymInt> sizes;
  sizes.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    torch::_export::SymInt dim;
    dim.set_as_int(sizeValue);
    sizes.push_back(std::move(dim));
  }
  meta.set_sizes(std::move(sizes));
  return meta;
}

// Runs 'graph' (with 'meta' applied) through the wave executor on GPU once per
// entry in 'runs' (each a positional CPU-tensor input set by graph user-input
// order), reusing one executor and its pooled frame/state across runs so a
// multi-run call exercises state reuse (and grid-choice caching) across
// executions. Returns the last run's host output tensors.
std::vector<at::Tensor> runWaveProgrammatic(
    std::unique_ptr<nativert::Graph> graph,
    const std::unordered_map<std::string, torch::_export::TensorMeta>& meta,
    const std::vector<std::vector<at::Tensor>>& runs) {
  graph->setTensorValuesMeta(meta);
  setGraphDevice(graph.get(), /*isCuda=*/true);

  auto ctx = std::make_unique<ModelContext>();
  ctx->weights = std::make_shared<nativert::Weights>(graph.get());
  ctx->graph = std::move(graph);

  WaveGraphExecutor exec(std::move(ctx));
  const auto& runGraph = exec.graph();
  const auto& inputNames = runGraph.signature().userInputs();

  std::vector<at::Tensor> hostOutputs;
  for (const auto& inputs : runs) {
    // Move only defined inputs to device; an undefined (default-constructed)
    // input is left as None in the frame, so it reaches an op as a null-storage
    // operand -- the way an empty operand arrives in the ads graph.
    std::vector<c10::IValue> definedInputs;
    std::vector<size_t> definedPositions;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].defined()) {
        definedInputs.emplace_back(inputs[i]);
        definedPositions.push_back(i);
      }
    }
    auto [deviceInputs, transferUs] = inputsToDevice(definedInputs);

    auto frame = exec.getFrame();
    TORCH_CHECK(frame != nullptr, "null frame");
    for (size_t j = 0; j < definedPositions.size(); ++j) {
      size_t i = definedPositions[j];
      TORCH_CHECK(i < inputNames.size(), "input index out of range");
      auto* value = runGraph.tryGetValue(inputNames[i]);
      TORCH_CHECK(value != nullptr, "missing input value ", inputNames[i]);
      frame->setIValue(value->id(), deviceInputs[j]);
    }
    auto outputs = exec.executeWithPrefilledFrame(*frame);
    exec.returnFrame(std::move(frame));

    auto host = outputsToHost(outputs, "programmatic");
    hostOutputs.clear();
    for (auto& iv : host) {
      hostOutputs.push_back(iv.toTensor());
    }
  }
  return hostOutputs;
}

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

// Explicit coverage for the two cooperative-grid launch paths, run on the large
// (100k-element) cumsum graph forced onto the cooperative grid.  The scan
// lowers to tw.cumsum_cg -- one cooperative step whose cumsum_head (a
// multi-block producer that writes the per-block prefix `counts[]` to global
// memory) is read back by cumsum_final across an opBarrier, spread over ~100
// blocks.
//
// Two configurations exercise the code the two fixes touch, both asserting
// output == reference:
//   (a) isCg exercises the NORMAL cooperative launch, on which the opBarrier
//       acquire __threadfence() (Core.cuh / Headers.h) sits on the hot path
//       guarding the cross-block `counts` read.
//   (b) isCg + debugSingleOps exercises the DEBUG single-step launch, on which
//       groupAndCooperative (CompiledOp.cpp) launches every op of a cooperative
//       step cooperatively rather than per-op via the regular path.
//
// This pins the isCg + debugSingleOps combination that no other test exercises
// explicitly and guards against a deterministic regression of either path (a
// crash or always-wrong result).  It is NOT a probabilistic race catcher: on
// the A100 (sm_80) the underlying memory-ordering race and the debug mis-launch
// did not reproduce a deterministic failure when either fix was reverted --
// both were deterministic only on the original ROO graph and hardware.  See the
// diff's test plan for the measured revert data.
TEST_F(ExecutorTest, cgBarrierRegressionTest) {
  // Reset all overrides even if an assertion below aborts the body, so a
  // failure here cannot leak cg/debug state into later tests.
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().isCg = std::nullopt;
    WaveConfig::get().debugSingleOps = false;
    WaveConfig::get().useSingleBlock = std::nullopt;
  });

  auto resultsPath = getDataFilePath(dataDir(), "data/cumsum_test_results.pt");
  auto expected = loadReferenceValues(resultsPath);
  ASSERT_FALSE(expected.empty());
  auto pt2Path = getDataFilePath(dataDir(), "data/cumsum_test.pt2");

  // (a) Normal cooperative launch (the FIX #2 acquire fence is on this path).
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  WaveConfig::get().debugSingleOps = false;
  {
    auto fixture = ModelFixture::load(pt2Path);
    ASSERT_NE(fixture, nullptr);
    setGraphDevice(fixture->model.graph.get(), true);
    runWave(*fixture, expected);
  }

  // (b) Debug single-step launch under a cooperative grid (the FIX #1
  // groupAndCooperative path).
  WaveConfig::get().debugSingleOps = true;
  {
    auto fixture = ModelFixture::load(pt2Path);
    ASSERT_NE(fixture, nullptr);
    setGraphDevice(fixture->model.graph.get(), true);
    runWave(*fixture, expected);
  }
}

// Repro candidate for the ads cross-composite cumsum bug (value %3187): the
// cumsum reads a select-view of a MULTI-CONSUMER wave-produced cast (so the
// cast materializes as a standalone placed after the scan, unlike a fused
// single-consumer cast), feeds an exclusive-prefix cat([zeros[1],
// cumsum[:-1]]), alongside a co-located cumsum(new_ones) range-gen -- the three
// structural ingredients prior synthetics lacked.
TEST_F(ExecutorTest, cumsumOffsetsReproTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cumsum_offsets_repro_test.pt2",
      "data/cumsum_offsets_repro_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cumsum_offsets_repro_test.pt2",
      "data/cumsum_offsets_repro_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cumsum_offsets_repro_test.pt2",
      "data/cumsum_offsets_repro_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

// Regression guard for grid-choice caching across a size change in the pooled
// ExecutionState (the path CompiledOp.cpp's gridChoices reset touches). A
// scanOutputReturnBarrier scan (cumsum) feeding a consumer runs with auto grid
// selection (useSingleBlock = nullopt) on a SMALL input then a LARGE input,
// reusing one executor and its pooled state. Run 1's small input makes the
// grid-choice kernel pick the single-block variant; run 2's large input should
// be multi-block. Verifies run 2 is correct after run 1 cached a single-block
// choice. Note: reverting the gridChoices reset alone does not
// deterministically break this (the grid-swap bounds guard in gatherLaunches
// compensates on run 2), so this exercises the reuse path rather than isolating
// that one fix. Programmatic graph, no external .pt2.
TEST_F(ExecutorTest, scanRepeatGridChoiceTest) {
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().useSingleBlock = std::nullopt;
    WaveConfig::get().scanOutputReturnBarrier = true;
  });
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().scanOutputReturnBarrier = true;

  auto graph = nativert::stringToGraph(R"(graph(%x):
%cs = torch.ops.aten.cumsum.default(self=%x, dim=0)
%o = torch.ops.aten.add.Tensor(self=%cs, other=%cs)
return(%o)
)");
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["x"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["cs"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);

  auto small = at::arange(64, at::kLong) % 7;
  auto large = at::arange(200000, at::kLong) % 7;
  auto outputs =
      runWaveProgrammatic(std::move(graph), meta, {{small}, {large}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = at::cumsum(large, 0) * 2;
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << "run 2 (pooled-state reuse after a smaller run 1) mismatch: "
      << firstDifference(outputs[0], reference);
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

// Multi-block cat with an empty (null-storage) leading operand. The empty
// operand arrives as an undefined tensor with null storage; on the device
// Tensor::init then makes such a rank-0 operand numEl==1 (the empty product),
// so __copy / __copyConvert (Core.cuh / Headers.h) iterate once and, without
// the null-storage guard, dereference null -- a CUDA illegal memory access. The
// guard skips the empty operand so the kernel completes. This test asserts
// crash-avoidance: without the guard runWaveProgrammatic throws (illegal memory
// access); with it the run completes. The two non-empty operands force a real
// multi-block copy. Output correctness is not asserted because injecting the
// null-storage operand via a None input also zeroes the launch grid (a harness
// limitation, not a wave bug), so only the crash-avoidance contract is checked.
// Programmatic graph, no external .pt2.
TEST_F(ExecutorTest, catEmptyOperandTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().useSingleBlock = std::nullopt; });
  WaveConfig::get().useSingleBlock = false;

  auto graph = nativert::stringToGraph(R"(graph(%empty, %a, %b):
%list[] = prim.ListPack(l0=%empty, l1=%a, l2=%b)
%o = torch.ops.aten.cat.default(tensors=%list, dim=0)
return(%o)
)");
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  // The empty operand contributes zero elements to the cat layout (sizeValue 0)
  // but is still copied by __copy at kernel time as a null-storage operand.
  meta["empty"] = makeTensorMeta(c10::ScalarType::Long, 1, /*sizeValue=*/0);
  meta["a"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["b"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);

  // An undefined tensor is left as None in the frame, so the cat operand
  // arrives with null storage.
  auto empty = at::Tensor();
  auto a = at::arange(100000, at::kLong);
  auto b = at::arange(100000, at::kLong) + 7;
  EXPECT_NO_THROW(
      { runWaveProgrammatic(std::move(graph), meta, {{empty, a, b}}); });
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

// A rank-4 tensor exercised through a wave elementwise op. kMaxDims must be >=
// 4 (KernelParams.h / Headers.h). A transposed (non-contiguous) rank-4 view
// keeps four distinct strides that cannot collapse to a lower rank, so the
// device Tensor holds four dims/strides; with kMaxDims == 3 those fixed-size
// arrays overflow. Programmatic graph, no external .pt2.
TEST_F(ExecutorTest, rank4Test) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().useSingleBlock = std::nullopt; });
  WaveConfig::get().useSingleBlock = false;

  auto graph = nativert::stringToGraph(R"(graph(%x):
%t = torch.ops.aten.transpose.int(self=%x, dim0=1, dim1=3)
%o = torch.ops.aten.add.Tensor(self=%t, other=%t)
return(%o)
)");
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["x"] = makeTensorMeta(c10::ScalarType::Long, 4);
  meta["t"] = makeTensorMeta(c10::ScalarType::Long, 4);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 4);

  auto x = at::arange(8 * 8 * 8 * 2048, at::kLong).reshape({8, 8, 8, 2048});
  auto outputs = runWaveProgrammatic(std::move(graph), meta, {{x}});
  ASSERT_EQ(outputs.size(), 1);
  auto transposed = x.transpose(1, 3);
  auto reference = transposed + transposed;
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
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

// Reference-verify round trip on a graph that produces a shape-only meta
// output. index_get with arithmetic on the indices (clamp(idx * 3 + 2, ...))
// yields an index that a gather consumes only as a register argument, so wave
// never materializes it into a real frame tensor -- it allocates a meta
// placeholder for shape inference only.  The authoritative reference is saved
// from the serial CPU run (real tensors), so the wave-side meta is not
// serialized (that would crash in serializeReferenceFrame).  The wave run then
// verifies its intermediates against the reference: without the CompiledOp
// meta-skip in verifyAgainstReference the meta output is compared element-wise
// and aborts in firstDifference ("Cannot copy out of meta tensor; no data!").
// The fix skips an intentional shape-only meta, whose correctness is covered by
// verifying its data-consumer's output.
TEST_F(ExecutorTest, shapeOnlyMetaReferenceFrame) {
  auto pt2Path = getDataFilePath(dataDir(), "data/index_get_test.pt2");
  auto resultsPath =
      getDataFilePath(dataDir(), "data/index_get_test_results.pt");
  auto expected = loadReferenceValues(resultsPath);

  auto refPath = fmt::format(
      "/tmp/torchwave_shape_only_meta_ref_{}.pt", static_cast<int>(getpid()));

  // Save the authoritative reference from the serial CPU run (real tensors).
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  FLAGS_save_reference_frame = refPath;
  runSerial(*fixture, expected);
  FLAGS_save_reference_frame = "";

  // Verify the wave run (which holds the shape-only meta) against the
  // reference. Reload the fixture since makeModelContext moves the graph.
  fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  setGraphDevice(fixture->model.graph.get(), true);
  FLAGS_reference_frame = refPath;
  runWave(*fixture, expected);
  FLAGS_reference_frame = "";

  LOG(INFO) << "Reference frame: " << lastRefTensorsChecked_ << " tensors, "
            << lastRefNodesChecked_ << " nodes checked";
  EXPECT_GT(lastRefTensorsChecked_, 0);

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

// clamp(x, min=None, max=5): a None min is an absent optional carried as a
// None-typed input value.  It must select __clamp<false, true> (no lower
// bound) rather than __clamp<true, true> with a None->0 min, otherwise negative
// inputs are wrongly clamped up to 0.
TEST_F(ExecutorTest, clampNoneMinTest) {
  runTest(
      "data/clamp_none_min_test.pt2", "data/clamp_none_min_test_results.pt");
}

// cumsum(x[:, 1], dim=0): the input is a non-contiguous column view (select
// dim=1, stride = ncols).  The scan must honor the stride rather than reading
// the backing storage contiguously.
TEST_F(ExecutorTest, cumsumSelectTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cumsum_select_test.pt2",
      "data/cumsum_select_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cumsum_select_test.pt2",
      "data/cumsum_select_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cumsum_select_test.pt2",
      "data/cumsum_select_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

// cumsum over a doubly-strided view (x[:, 0, 1] via two chained selects on a 3D
// tensor), matching the ads-preproc range pattern.  The view must not be
// mistaken for contiguous; otherwise the scan reads x's row-major storage.
TEST_F(ExecutorTest, cumsumSelect3dTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cumsum_select3d_test.pt2",
      "data/cumsum_select3d_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cumsum_select3d_test.pt2",
      "data/cumsum_select3d_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cumsum_select3d_test.pt2",
      "data/cumsum_select3d_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;
}

// exclusive_sum (cat(zeros[1], cumsum(x))) over a non-contiguous select-column
// view: cumsum(select(x, dim=1, index=1)). The exclusive_sum rewrite feeds the
// select view directly to exclusive_sum, whose multi-block final stage
// (Scan.cuh exclusive_sum_final / exclusive_sum) must read it through
// complexIdx to honor the stride; a flat read sums the wrong storage. This is
// the exclusive_sum analog of cumsumSelectTest. Programmatic graph, no external
// .pt2.
TEST_F(ExecutorTest, exclusiveSumSelectTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().useSingleBlock = std::nullopt; });
  WaveConfig::get().useSingleBlock = false;

  auto graph = nativert::stringToGraph(R"(graph(%x):
%sel = torch.ops.aten.select.int(self=%x, dim=1, index=1)
%cs = torch.ops.aten.cumsum.default(self=%sel, dim=0)
%z = torch.ops.aten.zeros.default(size=[1])
%list[] = prim.ListPack(l0=%z, l1=%cs)
%o = torch.ops.aten.cat.default(tensors=%list, dim=0)
return(%o)
)");
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["x"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["sel"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["cs"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["z"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);

  auto x = at::arange(100000 * 4, at::kLong).reshape({100000, 4});
  auto outputs = runWaveProgrammatic(std::move(graph), meta, {{x}});
  ASSERT_EQ(outputs.size(), 1);
  auto sel = x.select(1, 1);
  auto reference = at::cat({at::zeros({1}, at::kLong), at::cumsum(sel, 0)}, 0);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
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
