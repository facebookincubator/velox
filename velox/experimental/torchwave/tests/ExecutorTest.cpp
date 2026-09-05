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

// compilePlans for a programmatic graph: same steps as the .pt2 path, so a
// graph built here can assert plan structure rather than only results.
CompiledPlan compilePlanProgrammatic(
    std::unique_ptr<nativert::Graph> graph,
    const std::unordered_map<std::string, torch::_export::TensorMeta>& meta,
    CompiledPlan::Mode mode) {
  graph->setTensorValuesMeta(meta);
  setGraphDevice(graph.get(), /*isCuda=*/true);
  auto ctx = std::make_unique<ModelContext>();
  ctx->weights = std::make_shared<nativert::Weights>(graph.get());
  ctx->graph = std::move(graph);
  // Constructing the executor compiles and places the graph, which is what
  // fills the grids CompiledPlan reads.
  WaveGraphExecutor exec(std::move(ctx));
  return CompiledPlan::from(*exec.waveGraph(), mode);
}

// Ordering guard for an in-place write over a buffer that a DIFFERENT kernel op
// filled. c[:, 0] = v lowers to a copy into a select view plus a select_scatter
// that wave rewrites into an in-place write; the buffer it overwrites is the
// clone, produced by an earlier kernel op. The write must not land in the same
// kernel as the op that fills that buffer -- in multi-kernel mode that means a
// kernel boundary between them (in cg / single-block it would be an
// intra-kernel opBarrier instead). kernelBoundaryBetween also requires both ops
// to appear, so the assertion fails rather than passes vacuously if the
// in-place rewrite stops firing and the shape under test disappears.
// Programmatic graph, no external .pt2.
TEST_F(ExecutorTest, inPlaceWriteOverEarlierKernelBufferTest) {
  const char* kGraph = R"(graph(%x, %v):
%c = torch.ops.aten.cumsum.default(self=%x, dim=1)
%col = torch.ops.aten.select.int(self=%c, dim=1, index=0)
%w = torch.ops.aten.copy.default(self=%col, src=%v)
%sc = torch.ops.aten.select_scatter.default(self=%c, src=%w, dim=1, index=0)
%other = torch.ops.aten.select.int(self=%sc, dim=1, index=1)
%o = torch.ops.aten.mul.Tensor(self=%other, other=%other)
return(%o)
)";
  auto meta = [] {
    std::unordered_map<std::string, torch::_export::TensorMeta> m;
    m["x"] = makeTensorMeta(c10::ScalarType::Float, 2);
    m["v"] = makeTensorMeta(c10::ScalarType::Float, 1);
    m["c"] = makeTensorMeta(c10::ScalarType::Float, 2);
    m["col"] = makeTensorMeta(c10::ScalarType::Float, 1);
    m["w"] = makeTensorMeta(c10::ScalarType::Float, 1);
    m["sc"] = makeTensorMeta(c10::ScalarType::Float, 2);
    m["other"] = makeTensorMeta(c10::ScalarType::Float, 1);
    m["o"] = makeTensorMeta(c10::ScalarType::Float, 1);
    return m;
  }();

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(
      plan.kernelBoundaryBetween("aten.clone.default", "tw.select_scatter"));

  constexpr int32_t kRows = 512;
  constexpr int32_t kCols = 8;
  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto v = at::arange(kRows, at::kFloat) * 3;
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{x, v}});
  ASSERT_EQ(outputs.size(), 1);
  auto scatter = at::cumsum(x, 1);
  scatter.select(1, 0).copy_(v);
  auto reference = scatter.select(1, 1) * scatter.select(1, 1);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// A cumsum whose input is another cumsum. The second scan reads the first's
// output from memory, so every mode has to order the two, and each does it
// differently: multi-kernel ends the first scan's kernel, cg co-fuses them
// behind opBarriers, single-block behind __syncthreads. That makes this the
// shape to soak (--gtest_filter plus --gtest_repeat) when a barrier or
// kernel-boundary rule changes -- a dropped one shows up as wrong sums, not as
// a crash. One test per mode, so a failure names the mode in the gtest summary,
// which is all a repeated run leaves behind. Programmatic graph, no .pt2.
const char* kDoubleCumsumGraph = R"(graph(%x):
%c1 = torch.ops.aten.cumsum.default(self=%x, dim=0)
%c2 = torch.ops.aten.cumsum.default(self=%c1, dim=0)
return(%c2)
)";

std::unordered_map<std::string, torch::_export::TensorMeta> doubleCumsumMeta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "c1", "c2"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  return meta;
}

// Compiles the graph with the mode choice left open: the per-mode variant grids
// are only populated then, since forcing a mode puts that one grid in the
// main slot instead.
CompiledPlan doubleCumsumPlan(CompiledPlan::Mode mode) {
  return compilePlanProgrammatic(
      nativert::stringToGraph(kDoubleCumsumGraph), doubleCumsumMeta(), mode);
}

// 100k elements so the multi-block path really is multi-block. Values are kept
// small (% 7) because the result is a sum of sums.
void runDoubleCumsum() {
  auto input = at::arange(100000, at::kLong) % 7;
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kDoubleCumsumGraph),
      doubleCumsumMeta(),
      {{input}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = at::cumsum(at::cumsum(input, 0), 0);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

TEST_F(ExecutorTest, doubleCumsumMultiBlockTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().useSingleBlock = std::nullopt; });

  // The second scan's head must not land in the first scan's kernel: the
  // launch boundary is what orders the read. Both ops appear, so this fails
  // rather than passes vacuously if the lowering changes.
  auto plan = doubleCumsumPlan(CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(
      plan.kernelBoundaryBetween("aten.cumsum.default", "tw.cumsum_head"));

  WaveConfig::get().useSingleBlock = false;
  runDoubleCumsum();
}

TEST_F(ExecutorTest, doubleCumsumSingleBlockTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().useSingleBlock = std::nullopt; });

  // Single-block orders the two scans with __syncthreads, which allocates no
  // barrier counter and so is invisible to the plan. Assert the shape the mode
  // promises -- both scans in ONE kernel, not split -- and leave the ordering
  // itself to the numeric check below.
  auto plan = doubleCumsumPlan(CompiledPlan::Mode::kSingleBlock);
  int32_t fusedKernels = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (!kernel.standalone) {
          ++fusedKernels;
        }
      }
    }
  }
  EXPECT_EQ(fusedKernels, 1);

  WaveConfig::get().useSingleBlock = true;
  runDoubleCumsum();
}

TEST_F(ExecutorTest, doubleCumsumCgTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().isCg = std::nullopt; });

  // cg keeps both scans in one cooperative kernel, so the ordering has to be an
  // intra-kernel opBarrier. barrierBetween checks both: co-fused, and that
  // kernel carries at least one barrier.
  auto plan = doubleCumsumPlan(CompiledPlan::Mode::kCG);
  EXPECT_TRUE(plan.barrierBetween("aten.cumsum.default", "tw.cumsum_cg"));

  WaveConfig::get().isCg = true;
  runDoubleCumsum();
}

// A RANDOM-ACCESS read of a scan's output inside the same cooperative kernel.
// aten.index.Tensor carries no barrier of its own, and the index is REVERSED,
// so block 0 reads what the last block wrote: that is what makes a missing
// barrier observable. A second cumsum does not -- its
// reader is aligned 1:1 with the writer and never crosses a block boundary, so
// it stays correct even with the ordering removed. Anything that changes when
// callNeedsBarrier fires should be soaked against THIS shape
// (--gtest_filter='*cumsumIndexTensorCg*' --gtest_repeat=N), not against a
// chain of scans. Programmatic graph, no external .pt2.
TEST_F(ExecutorTest, cumsumIndexTensorCgTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().isCg = std::nullopt; });

  const char* kGraph = R"(graph(%x, %idx):
%c = torch.ops.aten.cumsum.default(self=%x, dim=0)
%list[] = prim.ListPack(l0=%idx)
%o = torch.ops.aten.index.Tensor(self=%c, indices=%list)
return(%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["x"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["c"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["idx"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);

  // Large enough that the scan spans many blocks and the last of them is still
  // writing when the first starts gathering. Reversed rather than a scattered
  // permutation on purpose: measured against a stubbed callNeedsBarrier,
  // reversed gives ~27k wrong elements and a permutation only ~2.9k, because
  // what matters is the time between the write and the read, and reversing
  // maximizes it. A permutation mostly reads locations already written.
  constexpr int64_t kSize = 1 << 20;
  auto x = at::arange(kSize, at::kLong) % 7;
  auto idx = kSize - 1 - at::arange(kSize, at::kLong);

  WaveConfig::get().isCg = true;
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{x, idx}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = at::index_select(at::cumsum(x, 0), 0, idx);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
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

// The same graph with the reuse passes on, which is what reaches the
// pre-partition clone elision in elideReadOnlyClones. Nothing else runs that
// pass -- enableReuse is off by default -- so without this the whole-graph
// clone work is untested.
//
// This graph is the adversarial case for the clone CSE step: 'd' and 'e' are
// both a.clone() with the same (absent) memory_format, so they collide on the
// CSE key, yet they are distinct snapshots -- 'a' is mutated by va.add_(b)
// between them -- and both are returned. Merging them is wrong twice over: the
// values differ (d == a0+3, e == a0+3+b), and redirecting a clone that feeds a
// graph output leaves that output unproduced, because replaceAllUses rewires
// users and the output node is not one of them.
TEST_F(ExecutorTest, inPlaceCloneElisionTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  const bool savedElide = WaveConfig::get().elideClones;
  WaveConfig::get().enableReuse = true;
  WaveConfig::get().elideClones = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
    WaveConfig::get().elideClones = savedElide;
  };

  // Merging the two collapses output 4 ('e') to a non-tensor, because the
  // value its output slot names is left with a dead producer.
  runTest("data/in_place_test.pt2", "data/in_place_test_results.pt");
}

// Column writes through `.copy_`, the shape the ROO preproc graph produces:
// select -> copy -> select_scatter, chained over four columns. The functional
// copy carries the destination only for its shape and dtype, so it lowers to a
// register-valued elementwise and the whole chain lands in one kernel. Before
// copy was registered it ran as an eager standalone and split each write into
// its own step.
TEST_F(ExecutorTest, copyColumnTest) {
  runTest("data/copy_column_test.pt2", "data/copy_column_test_results.pt");

  auto plans = compilePlans("data/copy_column_test.pt2");
  EXPECT_TRUE(plans.multiKernel.fuses(
      {"aten.copy.default", "aten.select.int", "tw.select_scatter"}));
}

// copy with a source that broadcasts and/or converts to the destination:
// a size-1 dim broadcast over rows, a one-element source, an int64 source cast
// to float, an explicit aten.expand feeding the copy (the ROO form), and a
// lower-rank source. All five fuse; expand comes along as an elementwise
// identity rather than a separate host-side view.
TEST_F(ExecutorTest, copyBroadcastTest) {
  runTest(
      "data/copy_broadcast_test.pt2", "data/copy_broadcast_test_results.pt");

  auto plans = compilePlans("data/copy_broadcast_test.pt2");
  EXPECT_TRUE(
      plans.multiKernel.fuses({"aten.copy.default", "aten.expand.default"}));
}

// copy whose source aliases the buffer the enclosing scatter writes, shifted by
// one element. Clone elision drops the snapshot (the source's previous value is
// dead), so a register-valued copy would read and write one buffer in a single
// fused loop with no ordering between lanes. The overlap check must retarget
// these to tw.copy_out, which materializes the read into its own buffer.
//
// The plan assertion below, not the output comparison, is what catches a
// regression here. The wrong lowering is a race, and it does not reliably
// produce wrong numbers: within a warp every lane loads before any lane stores,
// so corruption needs an unlucky schedule across warps. Removing the
// copyMayOverlap check leaves the outputs correct on this hardware and fails
// only the plan.
TEST_F(ExecutorTest, copyOverlapTest) {
  runTest("data/copy_overlap_test.pt2", "data/copy_overlap_test_results.pt");

  auto plans = compilePlans("data/copy_overlap_test.pt2");
  // The overlapping copies take the materializing variant, not the register
  // one. Its output is a real buffer, which puts the scatter that reads it in a
  // later step -- so the whole read is behind a kernel boundary, not merely a
  // barrier, and the register form (which would inline the read into the
  // scatter's write) is never generated.
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.copy_out"}));
  EXPECT_TRUE(plans.multiKernel.inLaterStep("tw.slice_scatter", "tw.copy_out"));
  EXPECT_FALSE(plans.multiKernel.fuses({"aten.copy.default"}));
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

// An integer gather g=base[gather_idx] (tw.index_elt_one) reused as an index in
// two places, so it materializes in an earlier ProjectNode and both later uses
// import it through a prim.ListPack. One use is a tw.index_put_elt_one fused
// with a downstream elementwise, whose self clone and ListPack index element
// come from earlier kernels. Regresses the elementwise leaf-collection gap
// where such a cross-kernel value was dropped from the ElementExpr inputs (the
// fused index_put self or ListPack index reached codegen with no param slot).
TEST_F(ExecutorTest, indexListpackReuseTest) {
  runTest(
      "data/index_listpack_reuse_test.pt2",
      "data/index_listpack_reuse_test_results.pt");
}

// Number of aten.clone.default nodes that still have a live user after the
// optimizer has run on 'pt2File'. Clone-elision proof for the fused in-place
// scatter tests: each builds a fresh executor because runTest consumes its own
// fixture, and counts the clones the pass did not drop.
int countLiveClonesAfterOpt(const std::string& pt2Path) {
  auto fixture = ModelFixture::load(pt2Path);
  EXPECT_NE(fixture, nullptr);
  if (fixture == nullptr) {
    return -1;
  }
  WaveGraphExecutor exec(fixture->makeModelContext());
  int liveClones = 0;
  for (const auto& node : exec.graph().nodes()) {
    if (node.target() != "torch.ops.aten.clone.default") {
      continue;
    }
    for (const auto* out : node.outputs()) {
      if (out != nullptr && !out->users().empty()) {
        ++liveClones;
        break;
      }
    }
  }
  return liveClones;
}

// select_scatter as a fused in-place elementwise op along dim 0 and dim 1,
// covering the in-place and copy scenarios. The aten.select_scatter rewrite
// inserts clone(self) + the in-place tw.select_scatter; clone-elision (under
// enableReuse) drops the clone when self is a dead intermediate. out0's base
// (a0+a1) is dead, so its clone is elided and the write lands in base0's buffer
// in place; out1's base (b0+b1) is also a graph output, so its clone is kept (a
// defensive copy) and base1 is preserved. Verifies correctness for both dims
// and asserts exactly one clone was elided (out0 in place, out1 copy).
TEST_F(ExecutorTest, selectScatterTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  WaveConfig::get().enableReuse = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };

  // Correctness for both dims: out0 (in-place over the dead intermediate base0)
  // and out1 (copy: base1 stays live as a graph output) must match eager, and
  // the returned base1 must be preserved.
  runTest(
      "data/select_scatter_test.pt2", "data/select_scatter_test_results.pt");

  // In-place proof: after optimization exactly one aten.clone survives --
  // out1's defensive copy, kept because base1 is a graph output. out0's clone
  // is elided because base0 is a dead intermediate, so its select_scatter
  // writes base0 in place. Build a fresh executor to inspect its optimized
  // graph (runTest consumed its own fixture).
  int liveClones = countLiveClonesAfterOpt(
      getDataFilePath(dataDir(), "data/select_scatter_test.pt2"));
  EXPECT_EQ(liveClones, 1)
      << "expected one surviving clone (out1 copy); out0's clone should be "
         "elided so its select_scatter writes base0 in place";
}

// slice_scatter as a fused in-place elementwise op with an INTERMEDIATE base,
// covering the in-place and copy scenarios (scatterTest only covers graph-input
// bases, whose clone is always kept). out0's base (a0+a1) is a dead
// intermediate, so its clone is elided and the write lands in base0's buffer in
// place -- the elided-clone / intra-op-materialized-self path where the scatter
// output must alias self at its full shape, not the (smaller) src shape. out1's
// base (b0+b1) is also a graph output, so its clone is kept (a defensive copy).
// Verifies correctness for both dims and asserts exactly one clone survived.
TEST_F(ExecutorTest, sliceScatterInPlaceTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  WaveConfig::get().enableReuse = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };

  // Correctness for both dims: out0 (in-place over the dead intermediate base0)
  // and out1 (copy: base1 stays live as a graph output) must match eager, and
  // the returned base1 must be preserved.
  runTest(
      "data/slice_scatter_inplace_test.pt2",
      "data/slice_scatter_inplace_test_results.pt");

  // In-place proof: after optimization exactly one aten.clone survives --
  // out1's defensive copy, kept because base1 is a graph output. out0's clone
  // is elided because base0 is a dead intermediate, so its slice_scatter writes
  // base0 in place. Build a fresh executor to inspect its optimized graph
  // (runTest consumed its own fixture).
  int liveClones = countLiveClonesAfterOpt(
      getDataFilePath(dataDir(), "data/slice_scatter_inplace_test.pt2"));
  EXPECT_EQ(liveClones, 1)
      << "expected one surviving clone (out1 copy); out0's clone should be "
         "elided so its slice_scatter writes base0 in place";
}

// scatter.src as a fused in-place elementwise op along dim 0 and dim 1. The
// aten.scatter.src rewrite inserts clone(self) + the in-place tw.scatter, which
// scatters each src element to the destination whose 'dim' coordinate is read
// from the index tensor. The index is a permutation along 'dim', so the
// parallel overwrite is deterministic and matches eager. Both bases are dead
// intermediates with an independent src, so both clones are elided and each
// scatter writes its base in place.
TEST_F(ExecutorTest, scatterSrcTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  WaveConfig::get().enableReuse = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };

  runTest("data/scatter_src_test.pt2", "data/scatter_src_test_results.pt");

  int liveClones = countLiveClonesAfterOpt(
      getDataFilePath(dataDir(), "data/scatter_src_test.pt2"));
  EXPECT_EQ(liveClones, 0)
      << "both scatter.src bases are dead intermediates with independent src, "
         "so both clones should be elided";
}

// scatter_add as a fused in-place elementwise op along dim 0 and dim 1,
// accumulating with an atomic add so duplicate destination indices sum (the
// parallel wave result matches eager). Covers clone-elision in both directions:
// out0's base (aa0+aa1) is a dead intermediate with an independent src, so its
// clone is elided and the accumulation lands in base0 in place; out1's src is
// base1 itself (shares base1's storage), so its clone is KEPT -- accumulating
// in place would read partially updated values. Asserts exactly one clone
// survives.
TEST_F(ExecutorTest, scatterAddTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  WaveConfig::get().enableReuse = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };

  runTest("data/scatter_add_test.pt2", "data/scatter_add_test_results.pt");

  int liveClones = countLiveClonesAfterOpt(
      getDataFilePath(dataDir(), "data/scatter_add_test.pt2"));
  EXPECT_EQ(liveClones, 1)
      << "expected one surviving clone: out1's clone kept because src shares a "
         "base with self; out0's clone elided (dead base, independent src)";
}

// A fused tw.scatter_add accumulating [4096] src/index into a [256] tensor,
// feeding an elementwise consumer whose operands are all [256]. Forced onto the
// cooperative grid, where both land in one kernel op: multi-block ends the
// producer's kernel (readsFusedElementwiseProducerFromMemory), so only cg keeps
// the border inside a kernel and only cg can mis-size across it.
//
// Guards two defects that made the ROO preproc fault under --cg 1:
//   (a) the size walk recursed through the elementwise border, sizing the
//       consumer's output by the scatter's [4096] src/index instead of the
//       materialized [256] output -- a 16x shape divergence, and the device
//       loop (sized by the output) then read every [256] operand far past its
//       end.
//   (b) 'index' is read again by the gather limit[index] in a later expression
//       of the same kernel op, which allocates an alt Tensor copy. Its own-dims
//       primary (forced by the scatter) used to make the emitter drop that
//       copy, leaving it at the host's zero fill -- the gather then read
//       through null storage.
// (b) faults on its own, so the output comparison catches it. (a) does not:
// the gather reads only limit[0..255], which the over-long loop still computes
// correctly, and the garbage tail is never consumed -- so the buffer's size is
// asserted directly.
TEST_F(ExecutorTest, scatterAddCgConsumerTest) {
  const bool savedFree = WaveConfig::get().freeIntermediates;
  auto resetConfig = folly::makeGuard([savedFree] {
    WaveConfig::get().isCg = std::nullopt;
    WaveConfig::get().useSingleBlock = std::nullopt;
    WaveConfig::get().freeIntermediates = savedFree;
  });
  WaveConfig::get().useSingleBlock = false;
  WaveConfig::get().isCg = true;
  // Keep intermediates so the consumer's buffer can be inspected after the run.
  WaveConfig::get().freeIntermediates = false;

  auto pt2Path =
      getDataFilePath(dataDir(), "data/scatter_add_cg_consumer_test.pt2");
  auto resultsPath = getDataFilePath(
      dataDir(), "data/scatter_add_cg_consumer_test_results.pt");
  auto expected = loadReferenceValues(resultsPath);
  ASSERT_FALSE(expected.empty());

  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  setGraphDevice(fixture->model.graph.get(), true);

  WaveGraphExecutor exec(fixture->makeModelContext());
  auto& graph = exec.graph();

  // The consumer of the scatter's materialized output.
  const nativert::Value* consumerOutput = nullptr;
  for (const auto& node : graph.nodes()) {
    if (node.target() == "torch.ops.aten.minimum.default") {
      ASSERT_FALSE(node.outputs().empty());
      consumerOutput = node.outputs()[0];
      break;
    }
  }
  ASSERT_NE(consumerOutput, nullptr) << "fixture must keep the minimum node";

  auto frame = exec.getFrame();
  ASSERT_NE(frame, nullptr);
  auto inputs = loadSampleInputs(*fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);
  fillWaveFrame(graph, *frame, deviceInputs);
  auto outputs = exec.executeWithPrefilledFrame(*frame);

  const auto& consumerValue = frame->getIValue(consumerOutput->id());
  ASSERT_TRUE(consumerValue.isTensor());
  EXPECT_EQ(consumerValue.toTensor().numel(), 256)
      << "the consumer of a materialized elementwise border is sized by that "
         "border's output, not by the scatter's 4096-element src/index";

  auto hostOutputs = outputsToHost(outputs, "cg");
  verifyOutputs(hostOutputs, expected, "cg");
  exec.returnFrame(std::move(frame));
}

// Repro for the fused tw.slice_scatter dim=1 multi-row failure (ROO batch=768).
// out0 scatters a contiguous src (control); out1 scatters a NON-CONTIGUOUS src
// (a dim=1 view fed through clamp) into a dead-intermediate base -- the pattern
// whose rows past the first came back as uninitialized garbage. 64 rows > 32
// inner extent so a per-row failure shows. Both outputs compared against eager.
TEST_F(ExecutorTest, sliceScatterDim1ViewTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  WaveConfig::get().enableReuse = true;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };
  runTest(
      "data/slice_scatter_dim1_view_test.pt2",
      "data/slice_scatter_dim1_view_test_results.pt");
}

// Column assignment out[:, c] = w[:, c], which functionalizes to
// select_scatter(dim=1) inside slice_scatter(dim=0, start=0, end=2**63-1). The
// open-ended slice's int64 sentinel end must reach the device function
// unnarrowed: at 32 bits it reads as -1, the slice length collapses to 0 and
// every element scatters onto row 0, leaving the output at its base value.
// Every other slice_scatter fixture uses a small literal end, so only this one
// covers the sentinel. Run with reuse both on and off: with reuse on the base
// clone is elided and the scatter writes in place, with it off the scatter
// writes a distinct clone, and the end sentinel has to hold in both.
TEST_F(ExecutorTest, sliceScatterOpenEndTest) {
  const bool savedReuse = WaveConfig::get().enableReuse;
  SCOPE_EXIT {
    WaveConfig::get().enableReuse = savedReuse;
  };
  for (const bool reuse : {true, false}) {
    WaveConfig::get().enableReuse = reuse;
    runTest(
        "data/slice_scatter_open_end_test.pt2",
        "data/slice_scatter_open_end_test_results.pt",
        reuse ? "reuse" : "no-reuse");
  }
}

// logit (inverse sigmoid), with eps=None and with an eps clamp, as a fused
// elementwise op.
TEST_F(ExecutorTest, logitTest) {
  runTest("data/logit_test.pt2", "data/logit_test_results.pt");
}

// bucketize / searchsorted as fused elementwise binary searches: float and int
// value dtypes, right=False/True, side="right", int32/int64 output, a bucketize
// fused with a downstream add, and a 2-D (multi-row) searchsorted where each
// query row searches the matching sorted row.
TEST_F(ExecutorTest, searchOpsTest) {
  runTest("data/search_ops_test.pt2", "data/search_ops_test_results.pt");
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

// masked_select over a dynamically-empty input: the graph slices data/flags to
// [:end] where end is a runtime scalar, so the sliced tensors reserve the full
// length but are empty at runtime (end=0). The output must be [0], not the
// reserved capacity. Regression for all three modes leaving the output size
// unset when the element loop runs zero iterations.
TEST_F(ExecutorTest, maskedSelectEmptyTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/masked_select_empty_test.pt2",
      "data/masked_select_empty_test_results.pt",
      "multi-kernel");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/masked_select_empty_test.pt2",
      "data/masked_select_empty_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/masked_select_empty_test.pt2",
      "data/masked_select_empty_test_results.pt",
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

TEST_F(ExecutorTest, bincountTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/bincount_test.pt2", "data/bincount_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/bincount_test.pt2",
      "data/bincount_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/bincount_test.pt2", "data/bincount_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, cumsumTest) {
  // The mode reaches verifyOutputs as the display name, but a scoped trace also
  // attaches it to failures raised anywhere else under it -- worth having when
  // the only record of a rare failure is a truncated log.
  {
    SCOPED_TRACE("multi-block");
    WaveConfig::get().useSingleBlock = false;
    runTest(
        "data/cumsum_test.pt2", "data/cumsum_test_results.pt", "multi-block");
  }
  {
    SCOPED_TRACE("single-block");
    WaveConfig::get().useSingleBlock = true;
    runTest(
        "data/cumsum_test.pt2", "data/cumsum_test_results.pt", "single-block");
  }
  {
    SCOPED_TRACE("cg");
    WaveConfig::get().useSingleBlock = std::nullopt;
    WaveConfig::get().isCg = true;
    runTest("data/cumsum_test.pt2", "data/cumsum_test_results.pt", "cg");
    WaveConfig::get().isCg = std::nullopt;
  }
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

// Multi-dimensional and strided repeat_interleave along an explicit dim, plus
// the dim=None (flatten) case. Programmatic graphs compared against eager
// at::repeat_interleave. Covers 1/2/3-D inputs, each axis, extra size-1 dims
// (the [1,N] "run a 1-D op under a fake batch dim" pattern), and a
// non-contiguous (transposed) input. Per-element repeats (1-D counts); the
// wave repeat_interleave gathers each output element from its source segment.
TEST_F(ExecutorTest, repeatInterleaveMultiDimTest) {
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().useSingleBlock = std::nullopt;
    WaveConfig::get().isCg = std::nullopt;
  });

  // Runs repeat_interleave(self, repeats[, dim]) on the wave engine and checks
  // it against eager. 'dimArg' is "" (dim=None flatten) or ", dim=N".
  auto check = [&](const at::Tensor& self,
                   const at::Tensor& repeats,
                   const std::string& dimArg,
                   std::optional<int64_t> dim,
                   const char* label) {
    std::string graphStr = std::string("graph(%x, %r):\n") +
        "%o = torch.ops.aten.repeat_interleave.self_Tensor(self=%x, "
        "repeats=%r" +
        dimArg + ")\nreturn(%o)\n";
    auto graph = nativert::stringToGraph(graphStr);
    std::unordered_map<std::string, torch::_export::TensorMeta> meta;
    meta["x"] = makeTensorMeta(self.scalar_type(), self.dim());
    meta["r"] = makeTensorMeta(repeats.scalar_type(), repeats.dim());
    meta["o"] =
        makeTensorMeta(self.scalar_type(), dim.has_value() ? self.dim() : 1);
    auto outputs =
        runWaveProgrammatic(std::move(graph), meta, {{self, repeats}});
    ASSERT_EQ(outputs.size(), 1) << label;
    auto reference = dim.has_value() ? at::repeat_interleave(self, repeats, dim)
                                     : at::repeat_interleave(self, repeats);
    EXPECT_TRUE(tensorsMatch(outputs[0], reference))
        << label << ": " << firstDifference(outputs[0], reference);
  };

  // Deterministic, varied per-segment counts in [1, 3].
  auto counts = [](int64_t n) { return at::arange(n, at::kLong) % 3 + 1; };

  WaveConfig::get().useSingleBlock = false;

  // 1-D: dim=None (flatten) and explicit dim=0.
  {
    auto x = at::arange(64, at::kFloat);
    check(x, counts(64), "", std::nullopt, "1d-flatten");
    check(x, counts(64), ", dim=0", 0, "1d-dim0");
  }
  // 2-D flatten (dim=None) over a multi-row input -> 1-D output.
  {
    auto x = at::arange(3 * 8, at::kFloat).reshape({3, 8});
    check(x, counts(3 * 8), "", std::nullopt, "2d-flatten");
  }
  // 2-D along each axis, all dims > 1 (primary validation).
  {
    auto x = at::arange(5 * 32, at::kFloat).reshape({5, 32});
    check(x, counts(32), ", dim=1", 1, "2d-dim1");
  }
  {
    auto x = at::arange(7 * 20, at::kFloat).reshape({7, 20});
    check(x, counts(7), ", dim=0", 0, "2d-dim0");
  }
  // 3-D along the middle axis, all dims > 1.
  {
    auto x = at::arange(3 * 8 * 6, at::kFloat).reshape({3, 8, 6});
    check(x, counts(8), ", dim=1", 1, "3d-dim1");
  }
  // Extra size-1 dims (the ROO [1,N] pattern and its trailing-1 mirror).
  {
    auto x = at::arange(48, at::kFloat).reshape({1, 48});
    check(x, counts(48), ", dim=1", 1, "leading-1-dim1");
  }
  {
    auto x = at::arange(40, at::kFloat).reshape({40, 1});
    check(x, counts(40), ", dim=0", 0, "trailing-1-dim0");
  }
  // Non-contiguous (transposed) source: [6,10] -> [10,6] view, interleave dim1.
  {
    auto x = at::arange(6 * 10, at::kFloat).reshape({6, 10}).transpose(0, 1);
    check(x, counts(6), ", dim=1", 1, "strided-dim1");
  }

  // Broadcast: a 0-D repeat count applies uniformly to every segment.
  {
    auto x2d = at::arange(5 * 12, at::kFloat).reshape({5, 12});
    auto r3 = at::scalar_tensor(3, at::kLong); // 0-dim
    check(x2d, r3, ", dim=1", 1, "bcast-2d-dim1");
    check(x2d, r3, ", dim=0", 0, "bcast-2d-dim0");
    check(x2d, r3, "", std::nullopt, "bcast-2d-flatten");
    auto x1d = at::arange(30, at::kFloat);
    check(x1d, at::scalar_tensor(2, at::kLong), ", dim=0", 0, "bcast-1d-dim0");
  }

  // Zero repeats. A segment with count 0 contributes nothing to the output, so
  // the interleave axis can keep its original length (or shrink) while every
  // other axis is unchanged -- the case where inferring the axis from the
  // input/output shapes finds no difference and would silently fall back to
  // axis 0. The kernel takes the axis from the host instead, so these must
  // hold for a non-zero interleave axis.
  {
    // sum(repeats) == size(input, dim): output shape equals input shape.
    auto x = at::arange(4 * 3, at::kFloat).reshape({4, 3});
    auto sameLen = at::tensor({0, 2, 1}, at::kLong);
    check(x, sameLen, ", dim=1", 1, "zero-repeat-same-length-dim1");
    // All-ones is the other shape-preserving case (identity).
    check(x, at::ones({3}, at::kLong), ", dim=1", 1, "identity-dim1");
    // Leading zero, and a zero in the middle, with a changed output length.
    check(x, at::tensor({0, 1, 2}, at::kLong), ", dim=1", 1, "zero-first-dim1");
    check(
        x, at::tensor({2, 0, 2}, at::kLong), ", dim=1", 1, "zero-middle-dim1");
    // Same on axis 0 of a 3-D input, where a wrong axis is unambiguous.
    auto x3 = at::arange(3 * 2 * 5, at::kFloat).reshape({3, 2, 5});
    check(x3, at::tensor({1, 0, 2}, at::kLong), ", dim=0", 0, "zero-3d-dim0");
    // All repeats zero: an empty output.
    check(x, at::zeros({3}, at::kLong), ", dim=1", 1, "all-zero-dim1");
    // Broadcast zero: every segment drops out.
    check(x, at::scalar_tensor(0, at::kLong), ", dim=1", 1, "bcast-zero-dim1");
    // Flatten with zeros.
    check(
        x,
        at::tensor({2, 0, 1, 0, 3, 1, 0, 2, 1, 1, 0, 2}, at::kLong),
        "",
        std::nullopt,
        "zero-flatten");
  }

  // The primary 2-D case across all three grid variants.
  auto x2 = at::arange(5 * 32, at::kFloat).reshape({5, 32});
  WaveConfig::get().useSingleBlock = true;
  check(x2, counts(32), ", dim=1", 1, "2d-dim1-single-block");
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  check(x2, counts(32), ", dim=1", 1, "2d-dim1-cg");
  WaveConfig::get().isCg = std::nullopt;
}

TEST_F(ExecutorTest, repeatTest) {
  runTest("data/repeat_test.pt2", "data/repeat_test_results.pt");
}

TEST_F(ExecutorTest, catTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  WaveConfig::get().useSingleBlock = false;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "cg");

  // The allocation-group mode. o1 joins five operands, two of which an earlier
  // node produces, so this is the graph where a concat group places a result
  // whose operands were written before the concat's kernel ran.
  config.freeIntermediates = true;
  runTest("data/cat_test.pt2", "data/cat_test_results.pt", "cg groups");
  config.freeIntermediates = savedFree;
  WaveConfig::get().isCg = std::nullopt;

  // Plan structure. o3 = cat([ms1, ms2, ms3]) joins three operands, which is
  // the allocation group's path: the result is laid out on the host, so every
  // operand's extent has to be known before the concat sizes anything. A
  // masked_select settles its extent on device, so it now ends its own kernel
  // first rather than fusing into the cat -- there is no serial fill to fall
  // back on that could discover the extent as it goes. The multi-kernel grid
  // already decomposed it, and its compaction step's extent is read back
  // before the cat, so that one still fuses.
  auto plans = compilePlans("data/cat_test.pt2");
  if (WaveConfig::get().singlePass) {
    // --tw_single_pass replaces both decompositions with one look-back op.
    // Like the cg variant it sets the length on device, so the concat group
    // cannot lay the result out around it and the cat fuses neither form.
    EXPECT_FALSE(
        plans.cg.fuses({"aten.cat.default", "tw.masked_select_1pass"}));
    EXPECT_FALSE(plans.multiKernel.fuses(
        {"aten.cat.default", "tw.masked_select_1pass"}));
  } else {
    EXPECT_FALSE(plans.cg.fuses({"aten.cat.default", "tw.masked_select_cg"}));
    EXPECT_TRUE(plans.multiKernel.fuses(
        {"aten.cat.default", "tw.masked_select_final"}));
  }
  EXPECT_FALSE(plans.singleBlock.fuses(
      {"aten.cat.default", "aten.masked_select.default"}));
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

// Cats of 2-D and 3-D operands along every dimension. Only dim 0 leaves an
// operand's region of the result contiguous; every other dim makes it a
// strided band, written either through the host-made view the producing
// expression fills or by __concatCopy for an operand the kernel only copies.
// o8's operand is a gather, which decomposes the output index itself instead
// of writing through the view, so it has to map that index through the band's
// strides.
TEST_F(ExecutorTest, catNdTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  WaveConfig::get().useSingleBlock = false;
  runTest("data/cat_nd_test.pt2", "data/cat_nd_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_nd_test.pt2", "data/cat_nd_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cat_nd_test.pt2", "data/cat_nd_test_results.pt", "cg");

  // The allocation-group mode, which needs the freeing on. o7 joins four
  // operands on a strided axis, so this is where the wider concats meet the
  // pass that would place them.
  config.freeIntermediates = true;
  runTest("data/cat_nd_test.pt2", "data/cat_nd_test_results.pt", "cg groups");
  config.freeIntermediates = savedFree;
  WaveConfig::get().isCg = std::nullopt;

  // The wider cats are fused rather than handed to the eager op, and an
  // operand the graph computes lands in the cat's own kernel.
  auto plans = compilePlans("data/cat_nd_test.pt2");
  EXPECT_FALSE(plans.multiKernel.standalone("aten.cat.default"));
  EXPECT_TRUE(plans.multiKernel.fuses({"aten.cat.default", "aten.add.Tensor"}));
}

// Concats whose operands are produced by kernels of their own -- the shape of
// the ROO preproc graph's final concat. Each operand would otherwise allocate a
// buffer that the concat immediately copies into its result; the concat
// allocation group places the result at the step that makes the operands and
// hands each of them the region it occupies, so the producing kernel writes in
// place and __concatCopy finds source and destination already the same memory.
//
// The values have to come out the same whether the group runs or not, so the
// graph is checked on the ordinary path first and on the mode after.
TEST_F(ExecutorTest, catAllocGroupTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  const auto savedGroup = config.enableAllocGroup;
  const auto savedConcat = config.enableConcatAllocGroup;
  auto resetConfig = folly::makeGuard([&, savedFree, savedGroup, savedConcat] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
    config.enableAllocGroup = savedGroup;
    config.enableConcatAllocGroup = savedConcat;
  });

  const std::string pt2 = "data/cat_alloc_group_test.pt2";
  const std::string results = "data/cat_alloc_group_test_results.pt";

  config.useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  config.useSingleBlock = true;
  runTest(pt2, results, "single-block");
  config.useSingleBlock = std::nullopt;

  // The mode needs the cooperative grid, which fixes the steps a group's point
  // is expressed in, and the freeing, without which no group buffer is ever
  // released.
  config.isCg = true;
  config.freeIntermediates = true;
  config.enableAllocGroup = true;

  config.enableConcatAllocGroup = false;
  runTest(pt2, results, "cg alloc groups, concat groups off");
  config.enableConcatAllocGroup = true;
  runTest(pt2, results, "cg alloc groups, concat groups on");

  // What the pass made of it. 'wide' places all four of its gathers and 'mixed'
  // the two around the graph input it cannot place, so six operand allocations
  // and six concat copies go away, and with the two results eight values stop
  // being the lifetime grouping's to place.
  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatGroups, 2);
  EXPECT_EQ(stats.numConcatMembers, 6);
  EXPECT_EQ(stats.numInConcatGroup, stats.numConcatMembers + 2);
  // 'pair' is below the threshold. 'nd' and 'scaled' have every operand
  // computed by the concat's own kernel behind a reserveShape, so no earlier
  // point knows their extents and the layout cannot be laid out ahead of them.
  // 'scaled' is the case an operand's own descriptor does not show: the
  // elementwise op between the gather and the concat gives the operand a size
  // expression of its own, and only a walk up its producer chain finds the
  // reserve behind it. Placed anyway, the regions are laid out from extents
  // that are not there yet and overlap.
  EXPECT_EQ(stats.numConcatTooFew, 1);
  EXPECT_EQ(stats.numConcatNoMembers, 0);
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 2);

  // Nothing is placed with the mode's concat half switched off, which is what
  // makes the arm above an A/B rather than two runs of the same thing.
  config.enableConcatAllocGroup = false;
  auto without = allocGroupStats(pt2);
  EXPECT_EQ(without.numConcatGroups, 0);
  EXPECT_EQ(without.numInConcatGroup, 0);
}

// The same shapes joined along a new dimension instead of an existing one, up
// to the rank-4 limit of the kernel tensor descriptor. Every stack operand
// occupies a single position along the new dim, which is a strided slice
// unless that dim is the outermost. o7, o8 and o9 fill such a slice from a
// gather and from a scan, the two kinds of producer that place their own
// writes rather than writing through the view they are handed.
TEST_F(ExecutorTest, stackNdTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/stack_nd_test.pt2", "data/stack_nd_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/stack_nd_test.pt2",
      "data/stack_nd_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/stack_nd_test.pt2", "data/stack_nd_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;

  auto plans = compilePlans("data/stack_nd_test.pt2");
  EXPECT_FALSE(plans.multiKernel.standalone("aten.stack.default"));
  EXPECT_TRUE(
      plans.multiKernel.fuses({"aten.stack.default", "aten.add.Tensor"}));
}

// A cat whose operand length is decided on device, at rank 1 and at rank 2.
// The 1-D cat absorbs its masked_select: the same kernel patches the following
// operands' view bases once the length is known. A wider cat cannot -- the
// host allocates the result and hands each operand a strided view of it, which
// needs every shape up front -- so its device-sized operand ends its kernel
// first and the length is read back before the cat's launch.
TEST_F(ExecutorTest, catDynamicNdTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/cat_dynamic_nd_test.pt2",
      "data/cat_dynamic_nd_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_dynamic_nd_test.pt2",
      "data/cat_dynamic_nd_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/cat_dynamic_nd_test.pt2",
      "data/cat_dynamic_nd_test_results.pt",
      "cg");
  WaveConfig::get().isCg = std::nullopt;

  auto plans = compilePlans("data/cat_dynamic_nd_test.pt2");
  // Single-block is where the contrast shows: nothing else would end
  // nonzero's kernel there (its multiBlockReturnBarrier does not apply), so
  // the boundary before the 2-D cat is the concat's own operand pushdown,
  // while the 1-D cat still fuses its masked_select.
  EXPECT_TRUE(plans.singleBlock.fuses(
      {"aten.cat.default", "aten.masked_select.default"}));
  EXPECT_TRUE(plans.singleBlock.kernelBoundaryBetween(
      "aten.cat.default", "aten.nonzero.default"));
  EXPECT_TRUE(plans.singleBlock.inLaterStep(
      "aten.cat.default", "aten.nonzero.default"));
  EXPECT_TRUE(plans.cg.kernelBoundaryBetween(
      "aten.cat.default", "aten.nonzero.default"));
}

// Composed masked_selects feeding an elementwise add:
//   masked_select(masked_select(stuff, f1), f2) * 2 + masked_select(stuff,
//   comp)
// where f1 then f2 selects the same elements as comp, so both add operands have
// equal length. A masked_select sets its length on device; fusing it into the
// elementwise consumer (whose loop bound is that length) reads the length
// before the masked_select kernel writes it, giving a too-long result -- unless
// placeKernels puts a boundary between them (the fix).
TEST_F(ExecutorTest, maskedSelectComposeTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest(
      "data/masked_select_compose_test.pt2",
      "data/masked_select_compose_test_results.pt",
      "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/masked_select_compose_test.pt2",
      "data/masked_select_compose_test_results.pt",
      "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(
      "data/masked_select_compose_test.pt2",
      "data/masked_select_compose_test_results.pt",
      "cg");
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

// aten.cat and its aten.concat alias over the same pair of 2-D inputs on
// dim 1, so the two must lower identically.
TEST_F(ExecutorTest, cat2dTest) {
  WaveConfig::get().useSingleBlock = false;
  runTest("data/cat_2d_test.pt2", "data/cat_2d_test_results.pt", "multi-block");

  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_2d_test.pt2", "data/cat_2d_test_results.pt", "single-block");

  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest("data/cat_2d_test.pt2", "data/cat_2d_test_results.pt", "cg");
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

// Exercises _local_scalar_dense (Tensor.item()) feeding dynamic sizes: an
// arange whose end is item(sum(lengths)), and a ones whose size dims are
// item(lengths[i]). Validates that item() produces a scalar (fused like
// aten.item), not a zero-dim tensor, so the fused arange/ones size correctly.
TEST_F(ExecutorTest, dynamicShapeTest) {
  runTest("data/dynamic_shape_test.pt2", "data/dynamic_shape_test_results.pt");
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

  // Plan structure: advanced indexing that gives an index tensor for *every*
  // source dim lowers to the fused tw.index_elt_{one,two} -- a 1-D-source
  // gather (source_a[idx]) and a 2-D coordinate gather (matrix[row, col]); the
  // latter also fuses its trailing arithmetic (matrix[row, col] * 2 + 1).
  auto plans = compilePlans("data/index_get_test.pt2");
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.index_elt_one"}));
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.index_elt_two"}));
  EXPECT_TRUE(plans.multiKernel.fuses(
      {"tw.index_elt_two", "aten.mul.Scalar", "aten.add.Scalar"}));
}

// aten.index.Tensor over cat(var, factory): cat([var, zeros])[idx] -- two
// operands, the second a constant-fill factory -- rewrites to the fused
// tw.index_elt_one_default (default 0, an out-of-range index reads the default
// instead of erroring). cat([var, var2, zeros])[idx] has three operands, so it
// does not match the pattern and stays a plain index over the materialized cat.
TEST_F(ExecutorTest, indexEltDefaultTest) {
  runTest(
      "data/index_elt_default_test.pt2",
      "data/index_elt_default_test_results.pt");

  auto plans = compilePlans("data/index_elt_default_test.pt2");
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.index_elt_one_default"}));
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.index_elt_one"}));
}

TEST_F(ExecutorTest, indexSelectTest) {
  runTest("data/index_select_test.pt2", "data/index_select_test_results.pt");
}

// Advanced indexing (aten.index.Tensor): a single 1-D integer index selecting
// one dimension is rewritten to the fused index_select; a separated multi-index
// case (x[i, :, k]) falls back to the eager op.
TEST_F(ExecutorTest, indexTensorTest) {
  runTest("data/index_tensor_test.pt2", "data/index_tensor_test_results.pt");

  // Prove the rewrite fired: corrupt one converted index per case out of range
  // and confirm the device-side bounds check in __index_select reports the
  // matching dimension. Only the fused index_select performs this check, so a
  // "<dim> <badValue>" error means x[sel] took the fused path (not the eager
  // fallback). sel0/sel1/sel2 select dims 0/1/2 respectively. The site reports
  // the bound and the operand shapes after those two, so match on the pair
  // that identifies the failure rather than on the whole line.
  constexpr int32_t kBadIndex = 999'999;
  struct IndexErrorCase {
    std::string indexInput;
    int32_t expectedDim;
  };
  const std::vector<IndexErrorCase> indexErrorCases = {
      {"sel0", 0},
      {"sel1", 1},
      {"sel2", 2},
  };
  for (const auto& errorCase : indexErrorCases) {
    auto pt2Path = getDataFilePath(dataDir(), "data/index_tensor_test.pt2");
    auto fixture = ModelFixture::load(pt2Path);
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
    ASSERT_GE(idxValueId, 0) << "No " << errorCase.indexInput
                             << " input found in index_tensor_test graph";
    auto errors = runWaveExpectError(
        *fixture, [idxValueId, kBadIndex](nativert::ExecutionFrame& frame) {
          auto& iv = frame.getIValue(idxValueId);
          if (iv.isTensor()) {
            iv.toTensor().fill_(kBadIndex);
          }
        });
    const std::string expected =
        std::to_string(errorCase.expectedDim) + " " + std::to_string(kBadIndex);
    EXPECT_NE(errors.find(expected), std::string::npos)
        << "Case " << errorCase.indexInput << ": expected '" << expected
        << "' in device errors, got:\n"
        << errors;
    EXPECT_NE(errors.find("Bad idx"), std::string::npos)
        << "Case " << errorCase.indexInput
        << ": expected a 'Bad idx' report, got:\n"
        << errors;
  }

  // Plan structure: how aten.index.Tensor lowers, and how the placement of the
  // bool-mask (masked_select) case differs across modes. Only the multi-kernel
  // grid holds every op; the single-block and cg grids hold just the
  // masked_select, which has those variants.
  auto plans = compilePlans("data/index_tensor_test.pt2");

  // Multi-kernel: a single-dim integer index (a=x[sel0], b=x[:,sel1],
  // c=x[:,:,sel2]) fuses to tw.index_select. The separated-dim case
  // (d=x[sep0,:,sep2]) cannot, so it stays the eager aten.index.Tensor behind a
  // kernel boundary from the fused index_selects. The bool-mask case
  // (e=x1d[mask]) becomes a multi-step masked_select: head -> add_sizes ->
  // final.
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.index_select"}));
  EXPECT_TRUE(plans.multiKernel.standalone("aten.index.Tensor"));
  EXPECT_TRUE(plans.multiKernel.kernelBoundaryBetween(
      "aten.index.Tensor", "tw.index_select"));
  if (WaveConfig::get().singlePass) {
    // --tw_single_pass replaces the head/add_sizes/final decomposition and the
    // cg variant with one look-back op, so there is nothing to sequence.
    EXPECT_TRUE(plans.multiKernel.fuses({"tw.masked_select_1pass"}));
  } else {
    EXPECT_TRUE(plans.multiKernel.fuses({"tw.masked_select_head"}));
    EXPECT_TRUE(plans.multiKernel.inLaterStep(
        "tw.masked_select_final", "tw.masked_select_head"));
  }

  // Single-block: masked_select fuses into one kernel instead of decomposing.
  EXPECT_TRUE(plans.singleBlock.fuses({"aten.masked_select.default"}));

  // Cooperative grid: masked_select uses its dedicated multi-block variant.
  EXPECT_TRUE(plans.cg.fuses(
      {WaveConfig::get().singlePass ? "tw.masked_select_1pass"
                                    : "tw.masked_select_cg"}));
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

// Reference-verify round trip on a graph whose clones the in-place pass elides.
// An index_put_ chain lowers to a defensive clone before each in-place write.
// The reference is saved from a run with reuse off, which keeps the clones, so
// each clone's input still holds its pre-mutation value. With reuse on,
// rewriteInPlace elides those clones and the writer mutates the input's buffer,
// so the input now holds the post-mutation value. The divergence is intended:
// verifyAgainstReference must skip elided clone inputs, or the run aborts with
// a reference mismatch that is not a bug.
TEST_F(ExecutorTest, elidedCloneInputReferenceFrame) {
  auto pt2Path = getDataFilePath(dataDir(), "data/index_put_chain_test.pt2");
  auto resultsPath =
      getDataFilePath(dataDir(), "data/index_put_chain_test_results.pt");
  auto expected = loadReferenceValues(resultsPath);

  auto refPath = fmt::format(
      "/tmp/torchwave_elided_clone_ref_{}.pt", static_cast<int>(getpid()));

  const bool savedEnableReuse = WaveConfig::get().enableReuse;
  SCOPE_EXIT {
    FLAGS_reference_frame = "";
    WaveConfig::get().saveReferenceFramePath = "";
    WaveConfig::get().enableReuse = savedEnableReuse;
    std::remove(refPath.c_str());
  };

  // Reference from a run with the clones intact.
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  setGraphDevice(fixture->model.graph.get(), true);
  WaveConfig::get().enableReuse = false;
  WaveConfig::get().saveReferenceFramePath = refPath;
  runWave(*fixture, expected);

  // Verify the elided run, whose clone inputs are overwritten in place. Reload
  // the fixture since makeModelContext moves the graph.
  fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);
  setGraphDevice(fixture->model.graph.get(), true);
  WaveConfig::get().enableReuse = true;
  FLAGS_reference_frame = refPath;
  runWave(*fixture, expected);

  LOG(INFO) << "Reference frame: " << lastRefTensorsChecked_ << " tensors, "
            << lastRefNodesChecked_ << " nodes checked";
  EXPECT_GT(lastRefTensorsChecked_, 0);
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
