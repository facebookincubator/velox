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

#include <map>

#include <cuda_runtime.h> // @manual
#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <functional>
#include <numeric>

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/StaticDims.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/ValueTable.h"
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
  ScopedPlanMode openMode;
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

// Two independent scans feeding one add. In cg both tw.cumsum_cg calls land in
// one op and share its __shared__ scratch (size, rounded, counter, reduction
// temps), which each call's thread 0 rewrites on entry. Warps of the same
// block still testing the first scan's loop bound then read the second scan's
// size, run past the first input and corrupt the block's partial sums. Nothing
// orders the two calls across blocks, so the separation is a block barrier,
// not an opBarrier.
const char* kTwoCumsumGraph = R"(graph(%x, %y):
%c1 = torch.ops.aten.cumsum.default(self=%x, dim=0)
%c2 = torch.ops.aten.cumsum.default(self=%y, dim=0)
%r = torch.ops.aten.add.Tensor(self=%c1, other=%c2)
return(%r)
)";

std::unordered_map<std::string, torch::_export::TensorMeta> twoCumsumMeta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "y", "c1", "c2", "r"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  return meta;
}

TEST_F(ExecutorTest, twoCumsumCgScratchTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().isCg = std::nullopt; });

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kTwoCumsumGraph),
      twoCumsumMeta(),
      CompiledPlan::Mode::kCG);
  int32_t sharedOps = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        const auto first = kernel.code.find("cumsum_cg<");
        if (first == std::string::npos) {
          continue;
        }
        const auto second = kernel.code.find("cumsum_cg<", first + 1);
        if (second == std::string::npos) {
          continue;
        }
        ++sharedOps;
        const auto between = kernel.code.substr(first, second - first);
        EXPECT_NE(between.find("__syncthreads();"), std::string::npos)
            << kernel.code;
      }
    }
  }
  // Fails rather than passes vacuously if the scans stop sharing an op.
  EXPECT_EQ(sharedOps, 1) << plan.describe();

  WaveConfig::get().isCg = true;
  auto x = at::arange(100000, at::kLong) % 7;
  auto y = at::arange(70000, at::kLong) % 5;
  y = at::cat({y, y.narrow(0, 0, 30000)});
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kTwoCumsumGraph), twoCumsumMeta(), {{x, y}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = at::cumsum(x, 0) + at::cumsum(y, 0);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// A broadcast whose operands reach it through unsqueeze views: [R,C,1] plus
// [1,1,C] is [R,C,C], the index pattern of the ads id-matching decomposition.
// The launch must cover R*C*C elements, not the R*C of the larger view base.
const char* kViewBroadcastGraph = R"(graph(%a, %b):
%u1 = torch.ops.aten.unsqueeze.default(self=%a, dim=2)
%u0 = torch.ops.aten.unsqueeze.default(self=%b, dim=0)
%u2 = torch.ops.aten.unsqueeze.default(self=%u0, dim=0)
%s = torch.ops.aten.add.Tensor(self=%u1, other=%u2)
%c = torch.ops.aten.clamp.default(self=%s, min=0, max=100000)
return(%c)
)";

std::unordered_map<std::string, torch::_export::TensorMeta>
viewBroadcastMeta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["a"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["b"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["u0"] = makeTensorMeta(c10::ScalarType::Long, 2);
  for (const auto* name : {"u1", "u2", "s", "c"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 3);
  }
  return meta;
}

TEST_F(ExecutorTest, viewBroadcastSizingTest) {
  constexpr int64_t kRows = 1000;
  constexpr int64_t kCols = 200;
  auto a = at::arange(kRows * kCols, at::kLong).reshape({kRows, kCols}) % 997;
  auto b = at::arange(kCols, at::kLong);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kViewBroadcastGraph),
      viewBroadcastMeta(),
      {{a, b}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference =
      at::clamp(a.unsqueeze(2) + b.unsqueeze(0).unsqueeze(0), 0, 100000);
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// The ads id-matching index: clamp(add(unsqueeze(a), unsqueeze(b))) flattened
// to the index list of a gather from cat(src, zeros). The gather becomes
// tw.index_elt_one_default. The flattening view moves from the index to the
// gather's output, so the clamp is computed in the gather's loop and the
// [R,C,C] index is never written. The launch has to cover the [R,C,C]
// broadcast; sized by its largest input it covered the [R,C,1] side alone.
const char* kViewBroadcastGatherGraph = R"(graph(%src, %a, %b):
%z = torch.ops.aten.new_zeros.default(self=%src, size=[1])
%l[] = prim.ListPack(l0=%src, l1=%z)
%cat = torch.ops.aten.cat.default(tensors=%l, dim=0)
%u1 = torch.ops.aten.unsqueeze.default(self=%a, dim=2)
%u0 = torch.ops.aten.unsqueeze.default(self=%b, dim=0)
%u2 = torch.ops.aten.unsqueeze.default(self=%u0, dim=0)
%s = torch.ops.aten.add.Tensor(self=%u1, other=%u2)
%c = torch.ops.aten.clamp.default(self=%s, min=0, max=300000)
%f = torch.ops.aten.view.default(self=%c, size=[40000000])
%il[] = prim.ListPack(l0=%f)
%g = torch.ops.aten.index.Tensor(self=%cat, indices=%il)
%o = torch.ops.aten.view.default(self=%g, size=[1000, 40000])
return(%o)
)";

std::unordered_map<std::string, torch::_export::TensorMeta>
viewBroadcastGatherMeta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"src", "z", "cat", "b", "f", "g"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  for (const auto* name : {"a", "u0", "o"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  for (const auto* name : {"u1", "u2", "s", "c"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 3);
  }
  return meta;
}

// The gather indexes by the clamp itself: the flattening view of the index
// moves to the gather's output, so no view separates the index expression from
// the gather that consumes it.
TEST_F(ExecutorTest, flattenedIndexMovesToGatherOutputTest) {
  auto graph = nativert::stringToGraph(kViewBroadcastGatherGraph);
  graph->setTensorValuesMeta(viewBroadcastGatherMeta());
  setGraphDevice(graph.get(), /*isCuda=*/true);
  auto ctx = std::make_unique<ModelContext>();
  ctx->weights = std::make_shared<nativert::Weights>(graph.get());
  ctx->graph = std::move(graph);
  WaveGraphExecutor exec(std::move(ctx));

  std::vector<std::string> indexProducers;
  for (const auto& node : exec.graph().nodes()) {
    if (node.target() == "tw.index_elt_one_default") {
      const auto* pack = node.inputs()[1].value->producer();
      indexProducers.emplace_back(
          pack->inputs()[0].value->producer()->target());
    }
  }
  EXPECT_THAT(
      indexProducers, testing::ElementsAre("torch.ops.aten.clamp.default"));
}

// Sizes come from an arange bound, a factory size list, a view with an
// inferred dim and an expand of an input of unknown length. The optimizer
// carries them through unsqueeze and broadcast; the input's own length stays
// unknown.
TEST_F(ExecutorTest, staticDimsTest) {
  const char* kGraph = R"(graph(%x):
%r = torch.ops.aten.arange.default(end=200)
%u = torch.ops.aten.unsqueeze.default(self=%r, dim=0)
%o = torch.ops.aten.new_ones.default(self=%x, size=[1000, 1])
%s = torch.ops.aten.add.Tensor(self=%o, other=%u)
%v = torch.ops.aten.view.default(self=%s, size=[-1, 40])
%c = torch.ops.aten.unsqueeze.default(self=%x, dim=1)
%e = torch.ops.aten.expand.default(self=%c, size=[-1, 200])
return(%v, %e)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "r"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  for (const auto* name : {"u", "o", "s", "v", "c", "e"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  auto graph = nativert::stringToGraph(kGraph);
  graph->setTensorValuesMeta(meta);
  setGraphDevice(graph.get(), /*isCuda=*/true);
  auto ctx = std::make_unique<ModelContext>();
  ctx->weights = std::make_shared<nativert::Weights>(graph.get());
  ctx->graph = std::move(graph);
  WaveGraphExecutor exec(std::move(ctx));

  const auto& types = exec.waveGraph()->types();
  const auto outputs = exec.graph().outputs();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(staticDimsOf(types, outputs[0]), StaticDims({5000, 40}));
  EXPECT_EQ(staticDimsOf(types, outputs[1]), StaticDims({-1, 200}));
}

// A broadcast add whose operand shapes come from view sizes reads each operand
// at an index with literal divisors, not through its runtime descriptor, and
// loops to a literal bound.
TEST_F(ExecutorTest, staticLeafIndexTest) {
  const char* kGraph = R"(graph(%x, %y):
%a = torch.ops.aten.view.default(self=%x, size=[100, 20, 1])
%b = torch.ops.aten.view.default(self=%y, size=[1, 1, 20])
%s = torch.ops.aten.add.Tensor(self=%a, other=%b)
return(%s)
)";
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().inputContiguous = false; });
  WaveConfig::get().inputContiguous = true;
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "y"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  for (const auto* name : {"a", "b", "s"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 3);
  }
  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  int32_t addKernels = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has("aten.add.Tensor")) {
          ++addKernels;
          EXPECT_NE(kernel.code.find("[(idx / 20u)]"), std::string::npos)
              << kernel.code;
          EXPECT_NE(kernel.code.find("[(idx % 20u)]"), std::string::npos)
              << kernel.code;
          EXPECT_NE(kernel.code.find("size = 40000u;"), std::string::npos)
              << kernel.code;
          EXPECT_EQ(
              kernel.code.find("complexIdx(isFastPath"), std::string::npos)
              << kernel.code;
        }
      }
    }
  }
  EXPECT_EQ(addKernels, 1) << plan.describe();

  auto x = at::arange(2000, at::kLong);
  auto y = at::arange(20, at::kLong) * 10000;
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{x, y}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = x.view({100, 20, 1}) + y.view({1, 1, 20});
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// A launch compiled for static shapes fails before running when a tensor's
// actual shape differs, or when a tensor read at literal indices is strided.
TEST_F(ExecutorTest, staticShapeCheckTest) {
  ValueEntry entry;
  entry.kind = ValueEntry::kTensor;
  entry.t.rank = 2;
  entry.t.dims[0] = 1000;
  entry.t.dims[1] = 200;
  entry.t.strides[0] = 200;
  entry.t.strides[1] = 1;
  ValueTable table{.entries = &entry, .size = 1};
  StepProgram program;

  program.shapeChecks = {
      {.valueId = 0, .dims = {1000, 200}, .contiguous = true}};
  runShapeChecks(program, table);

  program.shapeChecks = {{.valueId = 0, .dims = {1000, 40}}};
  EXPECT_THAT(
      [&] { runShapeChecks(program, table); },
      testing::ThrowsMessage<c10::Error>(testing::HasSubstr(
          "static shape its kernel was compiled for: value %0, compiled for "
          "[1000, 40], actual [1000, 200]")));

  // A band of a wider tensor: right dims, strided rows.
  entry.t.strides[0] = 400;
  program.shapeChecks = {
      {.valueId = 0, .dims = {1000, 200}, .contiguous = true}};
  EXPECT_THAT(
      [&] { runShapeChecks(program, table); },
      testing::ThrowsMessage<c10::Error>(testing::HasSubstr(
          "dense layout its kernel was compiled for: value %0, dim 0 has "
          "stride 400 instead of 200")));
}

// A sizing call folded to a constant at bind is still run with
// --verify_static_sizes, which fails when its result differs from the
// constant.
TEST_F(ExecutorTest, constantSizeVerifyTest) {
  ValueEntry entry;
  entry.kind = ValueEntry::kTensor;
  entry.t.rank = 1;
  entry.t.dims[0] = 7;
  ValueTable table{.entries = &entry, .size = 1};
  int32_t ids[] = {0};
  Span groups[] = {{0, 1}};
  Shape constantShape;
  StepProgram program;
  program.sizes.emplace_back();
  program.code.ids = ids;
  program.code.spans = groups;
  program.code.shapes = &constantShape;
  const SizeOp size{
      .fn =
          +[](const ReserveArgs& args, Shape* out, int32_t /*numOut*/) {
            out[0].setLike(args.tensor(0));
            return true;
          },
      .groups = {0, 1},
      .shapes = {0, 1},
      .source = 0,
      .constant = true};

  constantShape.setVector(7);
  verifyConstantSize(size, program, table);

  constantShape.setVector(5);
  EXPECT_THAT(
      [&] { verifyConstantSize(size, program, table); },
      testing::ThrowsMessage<c10::Error>(testing::HasSubstr(
          "Static size differs from the reserve: built-in operands %0, "
          "size 0 shape 0, static [5], reserve [7]")));
}

TEST_F(ExecutorTest, viewBroadcastGatherSizingTest) {
  const auto meta = viewBroadcastGatherMeta();
  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kViewBroadcastGatherGraph),
      meta,
      CompiledPlan::Mode::kMultiKernel);
  int32_t clampKernels = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has("aten.clamp.default")) {
          ++clampKernels;
          EXPECT_TRUE(kernel.has("tw.index_elt_one_default"))
              << plan.describe();
          EXPECT_TRUE(kernel.broadcastCount) << plan.describe();
        }
      }
    }
  }
  EXPECT_EQ(clampKernels, 1) << plan.describe();

  auto src = at::arange(300000, at::kLong) % 991;
  auto a = (at::arange(200000, at::kLong) * 7 % 299800).reshape({1000, 200});
  auto b = at::arange(200, at::kLong);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kViewBroadcastGatherGraph), meta, {{src, a, b}});
  ASSERT_EQ(outputs.size(), 1);
  auto index =
      at::clamp(a.unsqueeze(2) + b.unsqueeze(0).unsqueeze(0), 0, 300000);
  auto padded = at::cat({src, at::zeros({1}, at::kLong)});
  auto reference =
      padded.index_select(0, index.reshape({-1})).view({1000, 40000});
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// The same gather with its offsets made by a 1-D cumsum, which builds a cg
// variant. There a materialized clamp would be its own loop writing the
// [R,C,C] index, read by the gather after an opBarrier.
const char* kCumsumBroadcastGatherGraph = R"(graph(%src, %d, %b):
%e = torch.ops.aten.cumsum.default(self=%d, dim=0)
%a = torch.ops.aten.view.default(self=%e, size=[1000, 200])
%z = torch.ops.aten.new_zeros.default(self=%src, size=[1])
%l[] = prim.ListPack(l0=%src, l1=%z)
%cat = torch.ops.aten.cat.default(tensors=%l, dim=0)
%u1 = torch.ops.aten.unsqueeze.default(self=%a, dim=2)
%u0 = torch.ops.aten.unsqueeze.default(self=%b, dim=0)
%u2 = torch.ops.aten.unsqueeze.default(self=%u0, dim=0)
%s = torch.ops.aten.add.Tensor(self=%u1, other=%u2)
%c = torch.ops.aten.clamp.default(self=%s, min=0, max=300000)
%f = torch.ops.aten.view.default(self=%c, size=[40000000])
%il[] = prim.ListPack(l0=%f)
%g = torch.ops.aten.index.Tensor(self=%cat, indices=%il)
%o = torch.ops.aten.view.default(self=%g, size=[1000, 40000])
return(%o)
)";

TEST_F(ExecutorTest, cumsumBroadcastGatherCgTest) {
  auto resetConfig =
      folly::makeGuard([] { WaveConfig::get().isCg = std::nullopt; });
  auto meta = viewBroadcastGatherMeta();
  meta["d"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["e"] = makeTensorMeta(c10::ScalarType::Long, 1);
  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kCumsumBroadcastGatherGraph),
      meta,
      CompiledPlan::Mode::kCG);
  int32_t gatherKernels = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has("tw.index_elt_one_default")) {
          ++gatherKernels;
          EXPECT_EQ(
              kernel.code.find("// torch.ops.aten.clamp.default("),
              std::string::npos)
              << kernel.code;
          EXPECT_EQ(kernel.numBarriers, 0) << kernel.code;
        }
      }
    }
  }
  EXPECT_EQ(gatherKernels, 1) << plan.describe();

  WaveConfig::get().isCg = true;
  auto src = at::arange(300000, at::kLong) % 991;
  auto d = at::arange(200000, at::kLong) % 3;
  auto b = at::arange(200, at::kLong);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kCumsumBroadcastGatherGraph),
      meta,
      {{src, d, b}});
  ASSERT_EQ(outputs.size(), 1);
  auto index = at::clamp(
      at::cumsum(d, 0).view({1000, 200}).unsqueeze(2) +
          b.unsqueeze(0).unsqueeze(0),
      0,
      300000);
  auto padded = at::cat({src, at::zeros({1}, at::kLong)});
  auto reference =
      padded.index_select(0, index.reshape({-1})).view({1000, 40000});
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
}

// The ads id-matching search: eq(searchsorted(sorted, q), vals) reshaped to
// rank 3 and masked. All shapes are fixed by the views, so the view is
// computed through: the eq is evaluated in the masking loop at its index and
// never written. The searchsorted picks its sorted row from the eq's shape,
// [R, C * C], which differs from the loop's [R, C, C].
const char* kViewFusionGraph = R"(graph(%z, %x, %y, %m):
%q = torch.ops.aten.view.default(self=%x, size=[100, 400])
%r = torch.ops.aten.view.default(self=%y, size=[100, 400])
%s = torch.ops.aten.searchsorted.Tensor(sorted_sequence=%z, self=%q)
%e = torch.ops.aten.eq.Tensor(self=%s, other=%r)
%v = torch.ops.aten.view.default(self=%e, size=[100, 20, 20])
%w = torch.ops.aten.view.default(self=%m, size=[1, 1, 20])
%a = torch.ops.aten.bitwise_and.Tensor(self=%v, other=%w)
return(%a)
)";

std::unordered_map<std::string, torch::_export::TensorMeta> viewFusionMeta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["z"] = makeTensorMeta(c10::ScalarType::Long, 2);
  for (const auto* name : {"x", "y"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  meta["m"] = makeTensorMeta(c10::ScalarType::Bool, 1);
  for (const auto* name : {"q", "r", "s"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  meta["e"] = makeTensorMeta(c10::ScalarType::Bool, 2);
  for (const auto* name : {"v", "w", "a"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Bool, 3);
  }
  return meta;
}

TEST_F(ExecutorTest, viewFusionTest) {
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().isCg = std::nullopt;
    WaveConfig::get().fuseThroughViews = true;
  });
  const auto meta = viewFusionMeta();
  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kViewFusionGraph),
      meta,
      CompiledPlan::Mode::kMultiKernel);
  int32_t maskKernels = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has("aten.bitwise_and.Tensor")) {
          ++maskKernels;
          EXPECT_TRUE(kernel.has("aten.eq.Tensor")) << plan.describe();
          EXPECT_EQ(
              kernel.code.find("// torch.ops.aten.eq.Tensor("),
              std::string::npos)
              << kernel.code;
          EXPECT_EQ(kernel.numBarriers, 0) << kernel.code;
        }
      }
    }
  }
  EXPECT_EQ(maskKernels, 1) << plan.describe();

  auto z = std::get<0>(at::sort(at::randint(0, 1000, {100, 50}, at::kLong), 1));
  auto x = at::randint(-5, 1005, {40000}, at::kLong);
  auto m = at::arange(20, at::kLong) % 3 != 0;
  auto q = x.view({100, 400});
  auto s = at::searchsorted(z, q);
  // About a third of the positions match.
  auto y = at::where(
               at::arange(40000, at::kLong).view({100, 400}) % 3 == 0, s, s + 1)
               .reshape({-1});
  auto reference =
      (s == y.view({100, 400})).view({100, 20, 20}) & m.view({1, 1, 20});
  for (bool fuse : {true, false}) {
    for (bool cg : {false, true}) {
      SCOPED_TRACE(fmt::format("fuse {} cg {}", fuse, cg));
      WaveConfig::get().fuseThroughViews = fuse;
      WaveConfig::get().isCg = cg;
      auto outputs = runWaveProgrammatic(
          nativert::stringToGraph(kViewFusionGraph), meta, {{z, x, y, m}});
      ASSERT_EQ(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], reference))
          << firstDifference(outputs[0], reference);
    }
  }
}

// any() over the trailing dims of a fused elementwise expression is evaluated
// in the reduction's row loop, and the expression is never written. Covers
// the ads search shape (static, a warp per row of 20), and runtime shapes
// with rows of 300 (a warp) and 3000 (a block).
TEST_F(ExecutorTest, rowReductionFusionTest) {
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().isCg = std::nullopt;
    WaveConfig::get().fuseRowReductions = true;
  });
  auto expectFused = [](const CompiledPlan& plan) {
    int32_t anyKernels = 0;
    for (const auto& node : plan.nodes()) {
      for (const auto& step : node.steps) {
        for (const auto& kernel : step.kernels) {
          if (kernel.has("tw.any_rows")) {
            ++anyKernels;
            EXPECT_EQ(kernel.code.find("any_rows<"), std::string::npos)
                << kernel.code;
            EXPECT_EQ(kernel.numBarriers, 0) << kernel.code;
          }
        }
      }
    }
    EXPECT_EQ(anyKernels, 1) << plan.describe();
  };
  auto runModes =
      [](const char* graph,
         const std::unordered_map<std::string, torch::_export::TensorMeta>&
             meta,
         const std::vector<at::Tensor>& inputs,
         const at::Tensor& reference) {
        for (bool fuse : {true, false}) {
          for (bool cg : {false, true}) {
            SCOPED_TRACE(fmt::format("fuse {} cg {}", fuse, cg));
            WaveConfig::get().fuseRowReductions = fuse;
            WaveConfig::get().isCg = cg;
            auto outputs = runWaveProgrammatic(
                nativert::stringToGraph(graph), meta, {inputs});
            ASSERT_EQ(outputs.size(), 1);
            EXPECT_TRUE(tensorsMatch(outputs[0], reference))
                << firstDifference(outputs[0], reference);
          }
        }
        WaveConfig::get().fuseRowReductions = true;
        WaveConfig::get().isCg = std::nullopt;
      };

  {
    const std::string graph =
        std::string(kViewFusionGraph)
            .replace(
                std::string(kViewFusionGraph).find("return(%a)"),
                10,
                "%o = torch.ops.aten.any.dim(self=%a, "
                "dim=2)\nreturn(%o)");
    auto meta = viewFusionMeta();
    meta["o"] = makeTensorMeta(c10::ScalarType::Bool, 2);
    expectFused(compilePlanProgrammatic(
        nativert::stringToGraph(graph),
        meta,
        CompiledPlan::Mode::kMultiKernel));
    auto z =
        std::get<0>(at::sort(at::randint(0, 1000, {100, 50}, at::kLong), 1));
    auto x = at::randint(-5, 1005, {40000}, at::kLong);
    auto m = at::arange(20, at::kLong) % 7 == 0;
    auto s = at::searchsorted(z, x.view({100, 400}));
    // A match in about one row in ten.
    auto y =
        at::where(
            at::arange(40000, at::kLong).view({100, 400}) % 97 == 0, s, s + 1)
            .reshape({-1});
    auto reference =
        ((s == y.view({100, 400})).view({100, 20, 20}) & m.view({1, 1, 20}))
            .any(2);
    runModes(graph.c_str(), meta, {z, x, y, m}, reference);
  }

  const char* kRuntimeShapeGraph = R"(graph(%x, %y):
%l = torch.ops.aten.lt.Tensor(self=%x, other=%y)
%o = torch.ops.aten.any.dim(self=%l, dim=1)
return(%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "y"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  meta["l"] = makeTensorMeta(c10::ScalarType::Bool, 2);
  meta["o"] = makeTensorMeta(c10::ScalarType::Bool, 1);
  expectFused(compilePlanProgrammatic(
      nativert::stringToGraph(kRuntimeShapeGraph),
      meta,
      CompiledPlan::Mode::kMultiKernel));
  for (int64_t rowLen : {300, 3000}) {
    SCOPED_TRACE(rowLen);
    auto x = at::randint(0, 1000, {500, rowLen}, at::kLong);
    // Rows with an odd index have no element where x < y.
    auto y = at::where(
        at::arange(500, at::kLong).view({500, 1}) % 2 == 1,
        at::zeros_like(x),
        at::randint(0, 3, {500, rowLen}, at::kLong));
    runModes(kRuntimeShapeGraph, meta, {x, y}, at::lt(x, y).any(1));
  }
}

// x[mask] with a boolean mask of x's own rank is masked_select. The ads
// id-matching graph compacts the matched positions this way, over an expanded
// [1, C] -> [R, C] source, so the source is a stride-0 view.
const char* kFullRankMaskIndexGraph = R"(graph(%r, %m):
%u = torch.ops.aten.unsqueeze.default(self=%r, dim=0)
%e = torch.ops.aten.expand.default(self=%u, size=[1000, 200])
%l[] = prim.ListPack(l0=%m)
%o = torch.ops.aten.index.Tensor(self=%e, indices=%l)
return(%o)
)";

TEST_F(ExecutorTest, fullRankMaskIndexTest) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["r"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["m"] = makeTensorMeta(c10::ScalarType::Bool, 2);
  meta["u"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["e"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);
  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kFullRankMaskIndexGraph),
      meta,
      CompiledPlan::Mode::kMultiKernel);
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.standalone && kernel.has("aten.index.Tensor"))
            << plan.describe();
      }
    }
  }

  auto r = at::arange(200, at::kLong) * 3;
  auto m = (at::arange(200000, at::kLong) * 7919 % 13 < 5).reshape({1000, 200});
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kFullRankMaskIndexGraph), meta, {{r, m}});
  ASSERT_EQ(outputs.size(), 1);
  auto reference = r.unsqueeze(0).expand({1000, 200}).index({m});
  EXPECT_TRUE(tensorsMatch(outputs[0], reference))
      << firstDifference(outputs[0], reference);
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

// In-place aten.copy_, as a non-functionalized export keeps it, for each
// pairing of contiguous and strided source and destination: whole tensor,
// transposed source, column destination, and column to column chained over
// four columns. copy_ is an in-place elementwise that writes self through its
// strides and reads src through its own, so every write fuses; the transpose
// and the column selects stay host-side views it reads through.
TEST_F(ExecutorTest, copyInPlaceLayoutTest) {
  runTest(
      "data/copy_inplace_layout_test.pt2",
      "data/copy_inplace_layout_test_results.pt");

  auto plans = compilePlans("data/copy_inplace_layout_test.pt2");
  EXPECT_TRUE(
      plans.multiKernel.fuses({"aten.copy_.default", "aten.mul.Scalar"}));
  EXPECT_TRUE(
      plans.multiKernel.fuses({"aten.copy_.default", "aten.select.int"}));
  EXPECT_FALSE(plans.multiKernel.standalone("aten.copy_.default"));
}

// In-place aten.copy_ whose source broadcasts (a size-1 dim, an aten.expand of
// one element) or converts (int64 into a float column), and a shift by one
// that reads a clone of the buffer it writes. The clone is a required snapshot
// because its base is overwritten afterwards, so it must survive and be fully
// written before copy_ reads it.
TEST_F(ExecutorTest, copyInPlaceConvertTest) {
  runTest(
      "data/copy_inplace_convert_test.pt2",
      "data/copy_inplace_convert_test_results.pt");

  auto plans = compilePlans("data/copy_inplace_convert_test.pt2");
  EXPECT_TRUE(
      plans.multiKernel.fuses({"aten.copy_.default", "aten.expand.default"}));
  EXPECT_FALSE(plans.multiKernel.standalone("aten.copy_.default"));
}

// Column copies into one target, the shape the ROO preproc graph uses to fill a
// ones_like column by column. Each copy_ mutates the target, so without
// WaveConfig::recognizeDisjointCopies every copy is ordered after the previous
// one and the graph splits into a chain of layers. The rewrite merges them into
// one copy_ of a stack of the sources. Two graphs:
//   merged   four copies into columns 0..3, the target read only after them
//   blocked  the same with a read of the target between the second and third
//            copy, which must keep the copies apart
// Each runs with the flag on and off and is checked against ATen.
TEST_F(ExecutorTest, disjointColumnCopiesTest) {
  constexpr int32_t kRows = 512;
  constexpr int32_t kCols = 4;
  auto graphString = [&](bool readBetween) {
    std::string str = "graph(%x, %s):\n";
    str += "%t = torch.ops.aten.ones_like.default(self=%x)\n";
    str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
    for (int32_t i = 0; i < kCols; ++i) {
      str += fmt::format(
          "%sl{0} = torch.ops.aten.slice.Tensor(self=%t, dim=0, start=0, "
          "end=9223372036854775807)\n"
          "%d{0} = torch.ops.aten.select.int(self=%sl{0}, dim=1, index={0})\n"
          "%c{0} = torch.ops.aten.select.int(self=%s2, dim=1, index={0})\n"
          "%e{0} = torch.ops.aten.expand.default(self=%c{0}, size=[{1}])\n"
          "%w{0} = torch.ops.aten.copy_.default(self=%d{0}, src=%e{0})\n",
          i,
          kRows);
      if (readBetween && i == 1) {
        str += "%mid = torch.ops.aten.mul.Tensor(self=%t, other=%x)\n";
      }
    }
    str += "%o = torch.ops.aten.add.Tensor(self=%t, other=%x)\n";
    str += readBetween ? "return(%o, %mid)\n" : "return(%o)\n";
    return str;
  };
  auto meta = [&] {
    std::unordered_map<std::string, torch::_export::TensorMeta> m;
    for (const auto* name : {"x", "s", "t", "s2", "o", "mid"}) {
      m[name] = makeTensorMeta(c10::ScalarType::Float, 2);
    }
    for (int32_t i = 0; i < kCols; ++i) {
      m[fmt::format("sl{}", i)] = makeTensorMeta(c10::ScalarType::Float, 2);
      for (const auto* prefix : {"d", "c", "e", "w"}) {
        m[fmt::format("{}{}", prefix, i)] =
            makeTensorMeta(c10::ScalarType::Float, 1);
      }
    }
    return m;
  }();
  // copy_ nodes left after the executor has compiled the graph.
  auto countCopies = [&](const std::string& str) {
    auto graph = nativert::stringToGraph(str);
    graph->setTensorValuesMeta(meta);
    setGraphDevice(graph.get(), /*isCuda=*/true);
    auto ctx = std::make_unique<ModelContext>();
    ctx->weights = std::make_shared<nativert::Weights>(graph.get());
    ctx->graph = std::move(graph);
    WaveGraphExecutor exec(std::move(ctx));
    int32_t copies = 0;
    for (const auto& node : exec.graph().nodes()) {
      copies += node.target() == "torch.ops.aten.copy_.default" ? 1 : 0;
    }
    return copies;
  };

  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  // Every column of the target is overwritten, so it ends up as 2 * s.
  auto filled = s * 2;
  auto halfFilled = at::ones_like(x);
  halfFilled.narrow(1, 0, 2).copy_(filled.narrow(1, 0, 2));

  const bool saved = WaveConfig::get().recognizeDisjointCopies;
  SCOPE_EXIT {
    WaveConfig::get().recognizeDisjointCopies = saved;
  };
  for (const bool readBetween : {false, true}) {
    const auto str = graphString(readBetween);
    for (const bool recognize : {true, false}) {
      SCOPED_TRACE(
          fmt::format("readBetween={} recognize={}", readBetween, recognize));
      WaveConfig::get().recognizeDisjointCopies = recognize;
      EXPECT_EQ(countCopies(str), recognize && !readBetween ? 1 : kCols);

      auto outputs =
          runWaveProgrammatic(nativert::stringToGraph(str), meta, {{x, s}});
      ASSERT_EQ(outputs.size(), readBetween ? 2 : 1);
      auto expected = filled + x;
      EXPECT_TRUE(tensorsMatch(outputs[0], expected))
          << firstDifference(outputs[0], expected);
      if (readBetween) {
        auto expectedMid = halfFilled * x;
        EXPECT_TRUE(tensorsMatch(outputs[1], expectedMid))
            << firstDifference(outputs[1], expectedMid);
      }
    }
  }
}

// Counts the nodes of each of 'targets' left in the graph 'str' once the
// executor has compiled it. A target counts every node whose target contains
// it, so "select_scatter" counts the aten op and its tw form.
std::map<std::string, int32_t> compiledTargetCounts(
    const std::string& str,
    const std::unordered_map<std::string, torch::_export::TensorMeta>& meta,
    const std::vector<std::string>& targets) {
  auto graph = nativert::stringToGraph(str);
  graph->setTensorValuesMeta(meta);
  setGraphDevice(graph.get(), /*isCuda=*/true);
  auto ctx = std::make_unique<ModelContext>();
  ctx->weights = std::make_shared<nativert::Weights>(graph.get());
  ctx->graph = std::move(graph);
  WaveGraphExecutor exec(std::move(ctx));
  std::map<std::string, int32_t> counts;
  for (const auto& target : targets) {
    counts[target] = 0;
  }
  for (const auto& node : exec.graph().nodes()) {
    for (const auto& target : targets) {
      counts[target] += node.target().find(target) != std::string::npos ? 1 : 0;
    }
  }
  return counts;
}

// Number of steps of 'plan' that run 'op' in some kernel.
int32_t stepsRunning(const CompiledPlan& plan, std::string_view op) {
  int32_t steps = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      bool runs = false;
      for (const auto& kernel : step.kernels) {
        runs |= kernel.has(op);
      }
      steps += runs ? 1 : 0;
    }
  }
  return steps;
}

// The column source of disjointColumnCopiesTest: column i of %s2, expanded to
// a column of the target, as %e<i>.
std::string columnSource(int32_t column, int32_t rows) {
  return fmt::format(
      "%c{0} = torch.ops.aten.select.int(self=%s2, dim=1, index={0})\n"
      "%e{0} = torch.ops.aten.expand.default(self=%c{0}, size=[{1}])\n",
      column,
      rows);
}

// The functional form of "%b<link>[:, column].copy_(src)": the copy into the
// column, a select_scatter writing the column back, and a slice_scatter
// writing that over the whole tensor, giving %b<link + 1>.
std::string
functionalColumnWrite(int32_t link, int32_t column, const std::string& src) {
  return fmt::format(
      "%va{0} = torch.ops.aten.slice.Tensor(self=%b{0}, dim=0, start=0, "
      "end=9223372036854775807)\n"
      "%sel{0} = torch.ops.aten.select.int(self=%va{0}, dim=1, index={1})\n"
      "%cp{0} = torch.ops.aten.copy.default(self=%sel{0}, src={2})\n"
      "%vb{0} = torch.ops.aten.slice.Tensor(self=%b{0}, dim=0, start=0, "
      "end=9223372036854775807)\n"
      "%ss{0} = torch.ops.aten.select_scatter.default(self=%vb{0}, "
      "src=%cp{0}, dim=1, index={1})\n"
      "%b{3} = torch.ops.aten.slice_scatter.default(self=%b{0}, src=%ss{0}, "
      "dim=0, start=0, end=9223372036854775807)\n",
      link,
      column,
      src,
      link + 1);
}

// The same for "%b<link>[:, start:end].copy_(src)" with a slice_scatter of
// columns start..end-1 in place of the select_scatter.
std::string functionalSliceWrite(
    int32_t link,
    int32_t start,
    int32_t end,
    const std::string& src) {
  return fmt::format(
      "%va{0} = torch.ops.aten.slice.Tensor(self=%b{0}, dim=0, start=0, "
      "end=9223372036854775807)\n"
      "%sel{0} = torch.ops.aten.slice.Tensor(self=%va{0}, dim=1, start={1}, "
      "end={2})\n"
      "%cp{0} = torch.ops.aten.copy.default(self=%sel{0}, src={3})\n"
      "%vb{0} = torch.ops.aten.slice.Tensor(self=%b{0}, dim=0, start=0, "
      "end=9223372036854775807)\n"
      "%ss{0} = torch.ops.aten.slice_scatter.default(self=%vb{0}, "
      "src=%cp{0}, dim=1, start={1}, end={2})\n"
      "%b{4} = torch.ops.aten.slice_scatter.default(self=%b{0}, src=%ss{0}, "
      "dim=0, start=0, end=9223372036854775807)\n",
      link,
      start,
      end,
      src,
      link + 1);
}

// TensorMeta for the values the helpers above name, over 'links' links: the
// chain values are 2-D, the per-column values 1-D. A slice link's per-link
// values are 2-D; 'sliceLinks' lists them.
std::unordered_map<std::string, torch::_export::TensorMeta> chainMeta(
    int32_t links,
    int32_t columns,
    const std::vector<int32_t>& sliceLinks,
    const std::vector<std::string>& extra2d) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto& name : extra2d) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
  }
  for (int32_t i = 0; i < columns; ++i) {
    meta[fmt::format("c{}", i)] = makeTensorMeta(c10::ScalarType::Float, 1);
    meta[fmt::format("e{}", i)] = makeTensorMeta(c10::ScalarType::Float, 1);
  }
  for (int32_t i = 0; i <= links; ++i) {
    meta[fmt::format("b{}", i)] = makeTensorMeta(c10::ScalarType::Float, 2);
  }
  for (int32_t i = 0; i < links; ++i) {
    const bool isSlice =
        std::find(sliceLinks.begin(), sliceLinks.end(), i) != sliceLinks.end();
    for (const auto* name : {"va", "vb", "ss"}) {
      meta[fmt::format("{}{}", name, i)] =
          makeTensorMeta(c10::ScalarType::Float, 2);
    }
    for (const auto* name : {"sel", "cp"}) {
      meta[fmt::format("{}{}", name, i)] =
          makeTensorMeta(c10::ScalarType::Float, isSlice ? 2 : 1);
    }
  }
  return meta;
}

// Sets the three rewrite switches for the scope of a test and restores them.
class RewriteSwitches {
 public:
  RewriteSwitches()
      : reinplace_{WaveConfig::get().reinplaceScatters},
        recognize_{WaveConfig::get().recognizeDisjointCopies},
        ordering_{WaveConfig::get().disjointWriteOrdering} {}

  ~RewriteSwitches() {
    WaveConfig::get().reinplaceScatters = reinplace_;
    WaveConfig::get().recognizeDisjointCopies = recognize_;
    WaveConfig::get().disjointWriteOrdering = ordering_;
  }

  void set(bool reinplace, bool recognize, bool ordering) {
    WaveConfig::get().reinplaceScatters = reinplace;
    WaveConfig::get().recognizeDisjointCopies = recognize;
    WaveConfig::get().disjointWriteOrdering = ordering;
  }

 private:
  const bool reinplace_;
  const bool recognize_;
  const bool ordering_;
};

// Column copies into runs of columns: {0, 1}, {3, 4} and the single column 6.
// rewriteDisjointCopies merges each run of two or more into one copy and
// leaves the single one. A group that writes column 1 twice is not merged, and
// the later copy wins.
TEST_F(ExecutorTest, disjointColumnRunsTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 7;
  auto graphString = [&](const std::vector<int32_t>& columns) {
    std::string str = "graph(%x, %s):\n";
    str += "%t = torch.ops.aten.ones_like.default(self=%x)\n";
    str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
    for (size_t i = 0; i < columns.size(); ++i) {
      str += fmt::format(
          "%sl{0} = torch.ops.aten.slice.Tensor(self=%t, dim=0, start=0, "
          "end=9223372036854775807)\n"
          "%d{0} = torch.ops.aten.select.int(self=%sl{0}, dim=1, index={1})\n"
          "%c{0} = torch.ops.aten.select.int(self=%s2, dim=1, index={0})\n"
          "%e{0} = torch.ops.aten.expand.default(self=%c{0}, size=[{2}])\n"
          "%w{0} = torch.ops.aten.copy_.default(self=%d{0}, src=%e{0})\n",
          i,
          columns[i],
          kRows);
    }
    str += "%o = torch.ops.aten.add.Tensor(self=%t, other=%x)\n";
    str += "return(%o)\n";
    return str;
  };
  auto metaFor = [&](size_t numCopies) {
    std::unordered_map<std::string, torch::_export::TensorMeta> meta;
    for (const auto* name : {"x", "s", "t", "s2", "o"}) {
      meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
    }
    for (size_t i = 0; i < numCopies; ++i) {
      meta[fmt::format("sl{}", i)] = makeTensorMeta(c10::ScalarType::Float, 2);
      for (const auto* prefix : {"d", "c", "e", "w"}) {
        meta[fmt::format("{}{}", prefix, i)] =
            makeTensorMeta(c10::ScalarType::Float, 1);
      }
    }
    return meta;
  };
  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  // Copy i writes column i of 2 * s into column columns[i] of the target.
  auto expectedFor = [&](const std::vector<int32_t>& columns) {
    auto target = at::ones_like(x);
    for (size_t i = 0; i < columns.size(); ++i) {
      target.select(1, columns[i])
          .copy_(s.select(1, static_cast<int64_t>(i)) * 2);
    }
    return target + x;
  };

  RewriteSwitches switches;
  for (const bool recognize : {true, false}) {
    switches.set(/*reinplace=*/true, recognize, /*ordering=*/true);
    {
      SCOPED_TRACE(fmt::format("runs recognize={}", recognize));
      const std::vector<int32_t> columns{0, 1, 3, 4, 6};
      const auto str = graphString(columns);
      const auto meta = metaFor(columns.size());
      EXPECT_EQ(
          compiledTargetCounts(str, meta, {"copy_"})["copy_"],
          recognize ? 3 : 5);
      auto outputs =
          runWaveProgrammatic(nativert::stringToGraph(str), meta, {{x, s}});
      ASSERT_EQ(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], expectedFor(columns)))
          << firstDifference(outputs[0], expectedFor(columns));
    }
    {
      SCOPED_TRACE(fmt::format("duplicate recognize={}", recognize));
      const std::vector<int32_t> columns{0, 1, 1};
      const auto str = graphString(columns);
      const auto meta = metaFor(columns.size());
      EXPECT_EQ(compiledTargetCounts(str, meta, {"copy_"})["copy_"], 3);
      auto outputs =
          runWaveProgrammatic(nativert::stringToGraph(str), meta, {{x, s}});
      ASSERT_EQ(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], expectedFor(columns)))
          << firstDifference(outputs[0], expectedFor(columns));
    }
  }
}

// The functional form of the column-copy loop, as functionalization writes it:
// per column a copy, a select_scatter writing it back and a slice_scatter
// writing that over the whole tensor, each link reading the previous one's
// result. reinplaceScatterChains writes the chain into the ones_like in place;
// rewriteDisjointCopies then merges it into one copy as it does the in-place
// loop. Every combination of the two is checked against ATen.
TEST_F(ExecutorTest, functionalColumnChainTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 4;
  std::string str = "graph(%x, %s):\n";
  str += "%b0 = torch.ops.aten.ones_like.default(self=%x)\n";
  str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
  for (int32_t i = 0; i < kCols; ++i) {
    str += columnSource(i, kRows);
    str += functionalColumnWrite(i, i, fmt::format("%e{}", i));
  }
  str += fmt::format(
      "%o = torch.ops.aten.add.Tensor(self=%b{}, other=%x)\n", kCols);
  str += "return(%o)\n";
  const auto meta = chainMeta(kCols, kCols, {}, {"x", "s", "s2", "o"});

  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  const auto expected = s * 2 + x;

  RewriteSwitches switches;
  for (const bool reinplace : {true, false}) {
    for (const bool recognize : {true, false}) {
      SCOPED_TRACE(
          fmt::format("reinplace={} recognize={}", reinplace, recognize));
      switches.set(reinplace, recognize, /*ordering=*/true);
      auto counts =
          compiledTargetCounts(str, meta, {"copy_", "select_scatter"});
      EXPECT_EQ(counts["select_scatter"], reinplace ? 0 : kCols);
      if (reinplace) {
        EXPECT_EQ(counts["copy_"], recognize ? 1 : kCols);
      }
      auto outputs =
          runWaveProgrammatic(nativert::stringToGraph(str), meta, {{x, s}});
      ASSERT_EQ(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], expected))
          << firstDifference(outputs[0], expected);
    }
  }
}

// Functional chains that must not be written in place, each checked to keep
// its scatters and to produce the functional result:
//   intermediate  an intermediate result is also returned
//   input         the root is a graph input, which in place would be modified
//   readBetween   the root is read between two links
//   srcFromRoot   a link writes a column of the root, which the earlier links
//                 overwrite in place
TEST_F(ExecutorTest, functionalColumnChainBlockedTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 4;
  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  auto ones = at::ones_like(x);
  // The root with columns 0..k-1 written with 2 * s.
  auto writtenUpTo = [&](int64_t k) {
    auto target = at::ones_like(x);
    target.narrow(1, 0, k).copy_((s * 2).narrow(1, 0, k));
    return target;
  };

  RewriteSwitches switches;
  switches.set(/*reinplace=*/true, /*recognize=*/true, /*ordering=*/true);
  for (const std::string variant :
       {"intermediate", "input", "readBetween", "srcFromRoot"}) {
    SCOPED_TRACE(variant);
    const bool rootIsInput = variant == "input";
    std::string str =
        rootIsInput ? "graph(%x, %s, %b0):\n" : "graph(%x, %s):\n";
    if (!rootIsInput) {
      str += "%b0 = torch.ops.aten.ones_like.default(self=%x)\n";
    }
    str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
    std::vector<std::string> extra{"x", "s", "s2", "o"};
    for (int32_t i = 0; i < kCols; ++i) {
      str += columnSource(i, kRows);
      std::string src = fmt::format("%e{}", i);
      if (variant == "srcFromRoot" && i == 1) {
        str +=
            "%rootColumn = torch.ops.aten.select.int(self=%b0, dim=1, "
            "index=0)\n";
        src = "%rootColumn";
      }
      str += functionalColumnWrite(i, i, src);
      if (variant == "readBetween" && i == 1) {
        str += "%mid = torch.ops.aten.mul.Tensor(self=%b0, other=%x)\n";
      }
    }
    str += fmt::format(
        "%o = torch.ops.aten.add.Tensor(self=%b{}, other=%x)\n", kCols);
    if (variant == "intermediate") {
      str += "return(%o, %b2)\n";
    } else if (variant == "readBetween") {
      str += "return(%o, %mid)\n";
      extra.emplace_back("mid");
    } else {
      str += "return(%o)\n";
    }
    auto meta = chainMeta(kCols, kCols, {}, extra);
    meta["rootColumn"] = makeTensorMeta(c10::ScalarType::Float, 1);

    EXPECT_EQ(
        compiledTargetCounts(str, meta, {"select_scatter"})["select_scatter"],
        kCols);
    std::vector<at::Tensor> inputs{x, s};
    if (rootIsInput) {
      inputs.push_back(ones.clone());
    }
    auto outputs =
        runWaveProgrammatic(nativert::stringToGraph(str), meta, {inputs});
    auto expected = writtenUpTo(kCols);
    if (variant == "srcFromRoot") {
      // Column 1 is the root's original column 0, a column of ones.
      expected.select(1, 1).fill_(1);
    }
    ASSERT_GE(outputs.size(), 1);
    EXPECT_TRUE(tensorsMatch(outputs[0], expected + x))
        << firstDifference(outputs[0], expected + x);
    if (variant == "intermediate") {
      ASSERT_EQ(outputs.size(), 2);
      EXPECT_TRUE(tensorsMatch(outputs[1], writtenUpTo(2)))
          << firstDifference(outputs[1], writtenUpTo(2));
    }
    if (variant == "readBetween") {
      ASSERT_EQ(outputs.size(), 2);
      EXPECT_TRUE(tensorsMatch(outputs[1], x))
          << firstDifference(outputs[1], x);
    }
  }
}

// A functional chain of mixed links: columns 0 and 1, a slice link writing
// columns 3..4, and column 6. Written in place, the run 0..1 merges into one
// copy and the other two stay copies of their own, three in all.
TEST_F(ExecutorTest, functionalMixedChainTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 7;
  std::string str = "graph(%x, %s):\n";
  str += "%b0 = torch.ops.aten.ones_like.default(self=%x)\n";
  str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
  str += columnSource(0, kRows) + functionalColumnWrite(0, 0, "%e0");
  str += columnSource(1, kRows) + functionalColumnWrite(1, 1, "%e1");
  str +=
      "%band = torch.ops.aten.slice.Tensor(self=%s2, dim=1, start=3, "
      "end=5)\n";
  str += functionalSliceWrite(2, 3, 5, "%band");
  str += columnSource(6, kRows) + functionalColumnWrite(3, 6, "%e6");
  str += "%o = torch.ops.aten.add.Tensor(self=%b4, other=%x)\n";
  str += "return(%o)\n";
  const auto meta = chainMeta(4, kCols, {2}, {"x", "s", "s2", "o", "band"});

  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  auto expected = at::ones_like(x);
  for (const int64_t column : {0, 1, 3, 4, 6}) {
    expected.select(1, column).copy_(s.select(1, column) * 2);
  }
  expected += x;

  RewriteSwitches switches;
  for (const bool reinplace : {true, false}) {
    SCOPED_TRACE(fmt::format("reinplace={}", reinplace));
    switches.set(reinplace, /*recognize=*/true, /*ordering=*/true);
    auto counts = compiledTargetCounts(str, meta, {"copy_", "select_scatter"});
    EXPECT_EQ(counts["select_scatter"], reinplace ? 0 : 3);
    if (reinplace) {
      EXPECT_EQ(counts["copy_"], 3);
    }
    auto outputs =
        runWaveProgrammatic(nativert::stringToGraph(str), meta, {{x, s}});
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(tensorsMatch(outputs[0], expected))
        << firstDifference(outputs[0], expected);
  }
}

// A functional chain that mixes column writes with index-tensor writes:
// column 0, an index_put of whole rows, column 1, then a scatter_add along
// dim 0. The rows the index_put writes cross both columns, so the order of the
// writes decides the result. Written in place, the index_put and scatter_add
// become their in-place forms on the ones_like, no link copies the tensor, and
// every write stays ordered against the others. A second graph checks a
// one-link chain whose write is the fused tw.scatter_add.
TEST_F(ExecutorTest, functionalIndexedChainTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 4;
  constexpr int32_t kPut = 16;
  std::string str = "graph(%x, %s, %idx, %vals, %sidx, %ssrc):\n";
  str += "%b0 = torch.ops.aten.ones_like.default(self=%x)\n";
  str += "%s2 = torch.ops.aten.mul.Scalar(self=%s, other=2)\n";
  str += columnSource(0, kRows) + functionalColumnWrite(0, 0, "%e0");
  str +=
      "%l[] = prim.ListPack(l0=%idx)\n"
      "%b2 = torch.ops.aten.index_put.default(self=%b1, indices=%l, "
      "values=%vals)\n";
  str += columnSource(1, kRows) + functionalColumnWrite(2, 1, "%e1");
  str +=
      "%b4 = torch.ops.aten.scatter_add.default(self=%b3, dim=0, "
      "index=%sidx, src=%ssrc)\n";
  str += "%o = torch.ops.aten.add.Tensor(self=%b4, other=%x)\n";
  str += "return(%o)\n";
  auto meta = chainMeta(4, kCols, {}, {"x", "s", "s2", "o", "vals", "ssrc"});
  meta["idx"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["sidx"] = makeTensorMeta(c10::ScalarType::Long, 2);

  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto s = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) * 3;
  auto idx = at::arange(kPut, at::kLong) * 5;
  auto vals = at::arange(kPut * kCols, at::kFloat).reshape({kPut, kCols}) - 50;
  auto sidx = (at::arange(kRows * kCols, at::kLong) % kRows)
                  .flip(0)
                  .reshape({kRows, kCols});
  auto ssrc = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols});
  auto expected = at::ones_like(x);
  expected.select(1, 0).copy_(s.select(1, 0) * 2);
  expected.index_put_({idx}, vals);
  expected.select(1, 1).copy_(s.select(1, 1) * 2);
  expected.scatter_add_(0, sidx, ssrc);
  expected += x;

  RewriteSwitches switches;
  for (const bool reinplace : {true, false}) {
    SCOPED_TRACE(fmt::format("reinplace={}", reinplace));
    switches.set(reinplace, /*recognize=*/true, /*ordering=*/true);
    auto counts = compiledTargetCounts(
        str, meta, {"select_scatter", "index_put.default", "clone"});
    EXPECT_EQ(counts["select_scatter"], reinplace ? 0 : 2);
    if (reinplace) {
      EXPECT_EQ(counts["index_put.default"], 0);
      EXPECT_EQ(counts["clone"], 0);
    }
    auto outputs = runWaveProgrammatic(
        nativert::stringToGraph(str), meta, {{x, s, idx, vals, sidx, ssrc}});
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(tensorsMatch(outputs[0], expected))
        << firstDifference(outputs[0], expected);
  }

  // A chain of one fused index-tensor write, a histogram into a fresh tensor:
  // the in-place tw.scatter_add must stay reachable from the chain's readers.
  const char* kHistogram = R"(graph(%hsrc, %hidx):
%z = torch.ops.aten.new_zeros.default(self=%hsrc, size=[64])
%h = torch.ops.aten.scatter_add.default(self=%z, dim=0, index=%hidx, src=%hsrc)
%ho = torch.ops.aten.mul.Scalar(self=%h, other=2)
return(%ho)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> histogramMeta;
  for (const auto* name : {"hsrc", "z", "h", "ho"}) {
    histogramMeta[name] = makeTensorMeta(c10::ScalarType::Float, 1);
  }
  histogramMeta["hidx"] = makeTensorMeta(c10::ScalarType::Long, 1);
  auto hsrc = at::arange(1024, at::kFloat);
  auto hidx = at::arange(1024, at::kLong) % 64;
  const auto histogram =
      at::zeros({64}, at::kFloat).scatter_add(0, hidx, hsrc) * 2;
  for (const bool reinplace : {true, false}) {
    SCOPED_TRACE(fmt::format("histogram reinplace={}", reinplace));
    switches.set(reinplace, /*recognize=*/true, /*ordering=*/true);
    EXPECT_EQ(
        compiledTargetCounts(
            kHistogram, histogramMeta, {"tw.scatter_add"})["tw.scatter_add"],
        1);
    auto outputs = runWaveProgrammatic(
        nativert::stringToGraph(kHistogram), histogramMeta, {{hsrc, hidx}});
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(tensorsMatch(outputs[0], histogram))
        << firstDifference(outputs[0], histogram);
  }
}

// The functional form of "t.view(128, 16)[:, :5] += y" followed by an index_put
// into the same view: a slice_scatter and an index_put whose root is a view of
// t. Written in place, both write t through the view. The variants that must
// keep the chain functional:
//   baseRead     t itself is also returned, so it must keep its values
//   baseInput    the view is of a graph input, which in place would be modified
//   valsFromBase the index_put writes a view of t, taken before the chain,
//                which the slice update would already have changed in place
TEST_F(ExecutorTest, functionalViewRootTest) {
  constexpr int32_t kPut = 8;
  auto x = at::arange(256 * 8, at::kFloat).reshape({256, 8}) / 7;
  auto y = at::arange(128 * 5, at::kFloat).reshape({128, 5}) * 3;
  auto idx = at::arange(kPut, at::kLong) * 9;
  auto vals = at::arange(kPut * 16, at::kFloat).reshape({kPut, 16}) - 100;

  RewriteSwitches switches;
  for (const std::string variant :
       {"inPlace", "baseRead", "baseInput", "valsFromBase"}) {
    for (const bool reinplace : {true, false}) {
      SCOPED_TRACE(fmt::format("{} reinplace={}", variant, reinplace));
      switches.set(reinplace, /*recognize=*/true, /*ordering=*/true);
      const bool baseInput = variant == "baseInput";
      const bool valsFromBase = variant == "valsFromBase";
      std::string str = "graph(%x, %y, %idx, %vals):\n";
      str += baseInput ? "%t = torch.ops.aten.view.default(self=%x, "
                         "size=[256, 8])\n"
                       : "%t = torch.ops.aten.mul.Scalar(self=%x, other=3)\n";
      if (valsFromBase) {
        str +=
            "%v3 = torch.ops.aten.view.default(self=%t, size=[128, 16])\n"
            "%pv = torch.ops.aten.slice.Tensor(self=%v3, dim=0, start=0, "
            "end=8)\n";
      }
      str +=
          "%v1 = torch.ops.aten.view.default(self=%t, size=[128, 16])\n"
          "%v2 = torch.ops.aten.view.default(self=%t, size=[128, 16])\n"
          "%sl = torch.ops.aten.slice.Tensor(self=%v2, dim=1, start=0, end=5)\n"
          "%a = torch.ops.aten.add.Tensor(self=%sl, other=%y)\n"
          "%u = torch.ops.aten.slice_scatter.default(self=%v1, src=%a, dim=1, "
          "start=0, end=5)\n"
          "%l[] = prim.ListPack(l0=%idx)\n";
      str += fmt::format(
          "%p = torch.ops.aten.index_put.default(self=%u, indices=%l, "
          "values={})\n",
          valsFromBase ? "%pv" : "%vals");
      str +=
          "%w = torch.ops.aten.view.default(self=%p, size=[256, 8])\n"
          "%o = torch.ops.aten.mul.Scalar(self=%w, other=2)\n";
      str += variant == "baseRead" ? "return(%o, %t)\n" : "return(%o)\n";
      std::unordered_map<std::string, torch::_export::TensorMeta> meta;
      for (const auto* name :
           {"x",
            "y",
            "vals",
            "t",
            "v1",
            "v2",
            "v3",
            "sl",
            "a",
            "u",
            "pv",
            "p",
            "w",
            "o"}) {
        meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
      }
      meta["idx"] = makeTensorMeta(c10::ScalarType::Long, 1);

      const bool blocked = variant != "inPlace";
      auto counts = compiledTargetCounts(str, meta, {"slice_scatter", "clone"});
      if (reinplace) {
        EXPECT_EQ(counts["slice_scatter"], blocked ? 1 : 0);
        if (!blocked) {
          EXPECT_EQ(counts["clone"], 0);
        }
      }

      auto base = baseInput ? x.clone() : x * 3;
      auto view = base.clone().view({128, 16});
      view.narrow(1, 0, 5).add_(y);
      view.index_put_(
          {idx}, valsFromBase ? base.view({128, 16}).narrow(0, 0, kPut) : vals);
      auto expected = view.view({256, 8}) * 2;
      auto xInput = x.clone();
      auto outputs = runWaveProgrammatic(
          nativert::stringToGraph(str), meta, {{xInput, y, idx, vals}});
      ASSERT_GE(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], expected))
          << firstDifference(outputs[0], expected);
      if (variant == "baseRead") {
        ASSERT_EQ(outputs.size(), 2);
        EXPECT_TRUE(tensorsMatch(outputs[1], base))
            << firstDifference(outputs[1], base);
      }
      if (baseInput) {
        EXPECT_TRUE(tensorsMatch(xInput, x)) << firstDifference(xInput, x);
      }
    }
  }
}

// In-place writes into one tensor that cannot be merged into one copy: two
// columns from plain tensors and a band of two columns. With
// WaveConfig::disjointWriteOrdering their regions are provably disjoint, so
// they run in one step; without it each is ordered after the previous one.
// Writes that overlap (column 0 twice, then a band over columns 0..1) stay
// ordered either way, and the last one wins.
TEST_F(ExecutorTest, disjointWriteOrderingTest) {
  constexpr int32_t kRows = 256;
  constexpr int32_t kCols = 4;
  auto graphString = [](bool overlap) {
    const int32_t first = 0;
    const int32_t second = overlap ? 0 : 1;
    const int32_t bandStart = overlap ? 0 : 2;
    return fmt::format(
        "graph(%x, %a, %b, %w):\n"
        "%t = torch.ops.aten.ones_like.default(self=%x)\n"
        "%d0 = torch.ops.aten.select.int(self=%t, dim=1, index={0})\n"
        "%w0 = torch.ops.aten.copy_.default(self=%d0, src=%a)\n"
        "%d1 = torch.ops.aten.select.int(self=%t, dim=1, index={1})\n"
        "%w1 = torch.ops.aten.copy_.default(self=%d1, src=%b)\n"
        "%d2 = torch.ops.aten.slice.Tensor(self=%t, dim=1, start={2}, "
        "end={3})\n"
        "%w2 = torch.ops.aten.copy_.default(self=%d2, src=%w)\n"
        "%o = torch.ops.aten.add.Tensor(self=%t, other=%x)\n"
        "return(%o)\n",
        first,
        second,
        bandStart,
        bandStart + 2);
  };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"x", "w", "t", "d2", "w2", "o"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
  }
  for (const auto* name : {"a", "b", "d0", "w0", "d1", "w1"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 1);
  }
  auto x = at::arange(kRows * kCols, at::kFloat).reshape({kRows, kCols}) / 7;
  auto a = at::arange(kRows, at::kFloat) * 3;
  auto b = at::arange(kRows, at::kFloat) * 5;
  auto w = at::arange(kRows * 2, at::kFloat).reshape({kRows, 2}) * 11;

  RewriteSwitches switches;
  for (const bool overlap : {false, true}) {
    const auto str = graphString(overlap);
    auto expected = at::ones_like(x);
    expected.select(1, 0).copy_(a);
    expected.select(1, overlap ? 0 : 1).copy_(b);
    expected.narrow(1, overlap ? 0 : 2, 2).copy_(w);
    expected += x;
    for (const bool ordering : {true, false}) {
      SCOPED_TRACE(fmt::format("overlap={} ordering={}", overlap, ordering));
      switches.set(/*reinplace=*/true, /*recognize=*/true, ordering);
      auto plan = compilePlanProgrammatic(
          nativert::stringToGraph(str), meta, CompiledPlan::Mode::kMultiKernel);
      EXPECT_EQ(
          stepsRunning(plan, "aten.copy_.default"),
          ordering && !overlap ? 1 : 3)
          << plan.describe();
      auto outputs = runWaveProgrammatic(
          nativert::stringToGraph(str), meta, {{x, a, b, w}});
      ASSERT_EQ(outputs.size(), 1);
      EXPECT_TRUE(tensorsMatch(outputs[0], expected))
          << firstDifference(outputs[0], expected);
    }
  }
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

// aten.scatter_add_, the in-place form a non-functionalized export keeps, along
// dim 0 and dim 1 and as int64 counts into zeros. These are retargeted to
// tw.scatter_add on self itself, with no clone of self inserted. The fourth
// writes into a strided column view, which tw.scatter_add does not write
// through, so it stays the eager standalone and must still be correct.
TEST_F(ExecutorTest, scatterAddInPlaceTest) {
  runTest(
      "data/scatter_add_inplace_test.pt2",
      "data/scatter_add_inplace_test_results.pt");

  auto plans = compilePlans("data/scatter_add_inplace_test.pt2");
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.scatter_add"}));
  EXPECT_TRUE(plans.multiKernel.standalone("aten.scatter_add_.default"));
  EXPECT_EQ(
      countLiveClonesAfterOpt(
          getDataFilePath(dataDir(), "data/scatter_add_inplace_test.pt2")),
      0);
}

// tw.scatter_add on an index sorted into runs, as ROO's per-row counts are. A
// warp whose active lanes share a destination adds with one atomic, so the run
// boundaries and the end of the loop are what can go wrong: the runs total 1045
// elements, so the last warp has 21 active lanes, all inside the final run, and
// one run boundary falls inside a warp. 'counts' computes src from flags (the
// operand is evaluated only for lanes in range, the scatter on every lane);
// 'sums' reads src from memory. Checked on all three grids.
TEST_F(ExecutorTest, scatterAddRunsTest) {
  auto resetConfig = folly::makeGuard([] {
    WaveConfig::get().isCg = std::nullopt;
    WaveConfig::get().useSingleBlock = std::nullopt;
  });
  const std::string pt2 = "data/scatter_add_runs_test.pt2";
  const std::string results = "data/scatter_add_runs_test_results.pt";
  WaveConfig::get().useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  WaveConfig::get().useSingleBlock = true;
  runTest(pt2, results, "single-block");
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  runTest(pt2, results, "cg");

  auto plans = compilePlans(pt2);
  EXPECT_TRUE(plans.multiKernel.fuses({"tw.scatter_add"}));
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
  int64_t consumerNumel = -1;
  if (consumerValue.isTensor()) {
    consumerNumel = consumerValue.toTensor().numel();
  } else {
    const auto sizes = exec.waveGraph()->lastValueSizes(consumerOutput->id());
    ASSERT_FALSE(sizes.empty()) << "the consumer's output was not produced";
    consumerNumel = std::accumulate(
        sizes.begin(), sizes.end(), int64_t{1}, std::multiplies<>());
  }
  EXPECT_EQ(consumerNumel, 256)
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

// searchsorted with a rank-3 sorted sequence, each [i, j] row searched by
// the matching query row, read through a transposed view so the searched
// dim has a stride other than 1. Both sides, against eager. Programmatic
// graph, no .pt2.
TEST_F(ExecutorTest, searchsortedRank3StridedTest) {
  const char* kGraph = R"(graph(%s, %q):
%t = torch.ops.aten.transpose.int(self=%s, dim0=1, dim1=2)
%l = torch.ops.aten.searchsorted.Tensor(sorted_sequence=%t, self=%q)
%r = torch.ops.aten.searchsorted.Tensor(sorted_sequence=%t, self=%q, side="right")
return (%l, %r)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["s"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["t"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["q"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["l"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["r"] = makeTensorMeta(c10::ScalarType::Long, 3);
  // Sorted along dim 1, so the transpose's last dim is sorted with stride 4.
  auto s = std::get<0>(at::sort(at::randint(0, 100, {3, 50, 4}, at::kLong), 1));
  auto q = at::randint(-5, 105, {3, 4, 70}, at::kLong);
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{s, q}});
  ASSERT_EQ(outputs.size(), 2);
  auto t = s.transpose(1, 2);
  auto left = at::searchsorted(t, q);
  auto right = at::searchsorted(t, q, /*out_int32=*/false, /*right=*/true);
  EXPECT_TRUE(tensorsMatch(outputs[0], left))
      << firstDifference(outputs[0], left);
  EXPECT_TRUE(tensorsMatch(outputs[1], right))
      << firstDifference(outputs[1], right);
}

// Fused aten.gather (tw.gather): dims 0, 1 and -1 at ranks 2 and 3, and the
// ads id-matching chain eq(gather(sorted, 1, clamp(slice(pos))), vals)
// viewed to rank 3 and masked, so the gather's index producers fuse into it
// and a reshaping view sits between it and the expression output. Checks
// that no gather runs standalone and the results match eager. Programmatic
// graph, no .pt2.
TEST_F(ExecutorTest, gatherFusedTest) {
  const char* kGraph = R"(graph(%a, %i0, %b, %i1, %i2, %srt, %pos, %vals, %m):
%g0 = torch.ops.aten.gather.default(self=%a, dim=0, index=%i0)
%g1 = torch.ops.aten.gather.default(self=%b, dim=1, index=%i1)
%g2 = torch.ops.aten.gather.default(self=%b, dim=-1, index=%i2)
%sl = torch.ops.aten.slice.Tensor(self=%pos, dim=0, start=0, end=100)
%cl = torch.ops.aten.clamp.default(self=%sl, max=49)
%g3 = torch.ops.aten.gather.default(self=%srt, dim=1, index=%cl)
%e = torch.ops.aten.eq.Tensor(self=%g3, other=%vals)
%v = torch.ops.aten.view.default(self=%e, size=[100, 30, 40])
%h = torch.ops.aten.bitwise_and.Tensor(self=%v, other=%m)
return (%g0, %g1, %g2, %h)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["a"] = makeTensorMeta(c10::ScalarType::Float, 2);
  meta["i0"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["b"] = makeTensorMeta(c10::ScalarType::Float, 3);
  meta["i1"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["i2"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["srt"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["pos"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["vals"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["m"] = makeTensorMeta(c10::ScalarType::Bool, 3);
  for (const char* name : {"sl", "cl", "g3"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  meta["g0"] = makeTensorMeta(c10::ScalarType::Float, 2);
  meta["g1"] = makeTensorMeta(c10::ScalarType::Float, 3);
  meta["g2"] = makeTensorMeta(c10::ScalarType::Float, 3);
  meta["e"] = makeTensorMeta(c10::ScalarType::Bool, 2);
  meta["v"] = makeTensorMeta(c10::ScalarType::Bool, 3);
  meta["h"] = makeTensorMeta(c10::ScalarType::Bool, 3);

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  bool sawGather = false;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.standalone && kernel.has("aten.gather.default"))
            << plan.describe();
        if (kernel.has("tw.gather") && kernel.has("aten.clamp.default")) {
          sawGather = true;
        }
      }
    }
  }
  EXPECT_TRUE(sawGather) << plan.describe();

  auto a = at::randn({20, 7});
  auto i0 = at::randint(0, 20, {15, 7}, at::kLong);
  auto b = at::randn({4, 9, 6});
  auto i1 = at::randint(0, 9, {4, 9, 5}, at::kLong);
  auto i2 = at::randint(0, 6, {4, 9, 5}, at::kLong);
  auto srt = std::get<0>(at::sort(at::randint(0, 60, {100, 50}, at::kLong), 1));
  auto pos = at::randint(0, 51, {132, 1200}, at::kLong);
  auto vals = at::randint(0, 60, {100, 1200}, at::kLong);
  auto m = at::randint(0, 2, {100, 30, 40}, at::kLong).to(at::kBool);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kGraph),
      meta,
      {{a, i0, b, i1, i2, srt, pos, vals, m}});
  ASSERT_EQ(outputs.size(), 4);
  std::vector<at::Tensor> expected{
      at::gather(a, 0, i0),
      at::gather(b, 1, i1),
      at::gather(b, -1, i2),
      at::eq(at::gather(srt, 1, pos.narrow(0, 0, 100).clamp_max(49)), vals)
              .view({100, 30, 40}) &
          m};
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_TRUE(tensorsMatch(outputs[i], expected[i]))
        << i << ": " << firstDifference(outputs[i], expected[i]);
  }
}

// A gather whose output broadcasts into a larger expression:
// add(gather(a, 1, i)[20, 1], x[20, 30]). Against eager in every mode.
// Programmatic graph, no .pt2.
TEST_F(ExecutorTest, gatherBroadcastTest) {
  const char* kGraph = R"(graph(%a, %i, %x):
%g = torch.ops.aten.gather.default(self=%a, dim=1, index=%i)
%o = torch.ops.aten.add.Tensor(self=%g, other=%x)
return (%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["a"] = makeTensorMeta(c10::ScalarType::Float, 2);
  meta["i"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["x"] = makeTensorMeta(c10::ScalarType::Float, 2);
  meta["g"] = makeTensorMeta(c10::ScalarType::Float, 2);
  meta["o"] = makeTensorMeta(c10::ScalarType::Float, 2);
  auto a = at::randn({20, 7});
  auto i = at::randint(0, 7, {20, 1}, at::kLong);
  auto x = at::randn({20, 30});
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{a, i, x}});
  ASSERT_EQ(outputs.size(), 1);
  auto expected = at::gather(a, 1, i) + x;
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
}

// A gather feeding the source of an in-place scatter into a larger tensor,
// as in the ads graph. The loop runs over the source, so the gather's own
// elements, not the scatter output's. Against eager in every mode.
// Programmatic graph, no .pt2.
TEST_F(ExecutorTest, gatherIntoScatterTest) {
  const char* kGraph = R"(graph(%z, %si, %a, %i):
%c = torch.ops.aten.clone.default(self=%z)
%g = torch.ops.aten.gather.default(self=%a, dim=1, index=%i)
%m = torch.ops.aten.mul.Tensor(self=%g, other=%g)
%o = torch.ops.aten.scatter.src(self=%c, dim=1, index=%si, src=%m)
return (%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"z", "a", "c", "g", "m", "o"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  meta["si"] = makeTensorMeta(c10::ScalarType::Long, 2);
  meta["i"] = makeTensorMeta(c10::ScalarType::Long, 2);
  auto z = at::zeros({40, 50}, at::kLong);
  // Distinct targets per row so the scatter result does not depend on order.
  auto si = std::get<1>(at::sort(at::rand({40, 50}), 1))
                .narrow(1, 0, 30)
                .contiguous();
  auto a = at::randint(0, 1000, {40, 9}, at::kLong);
  auto i = at::randint(0, 9, {40, 30}, at::kLong);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kGraph), meta, {{z, si, a, i}});
  ASSERT_EQ(outputs.size(), 1);
  auto g = at::gather(a, 1, i);
  auto expected = z.clone().scatter(1, si, g * g);
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
}

// An in-place scatter whose source broadcasts one of its operands:
// scatter(clone(z), 1, si, add(a[40, 30], b[40, 1])). The loop runs over the
// source, so b must be read broadcast against the source's shape. Against
// eager in every mode. Programmatic graph, no .pt2.
TEST_F(ExecutorTest, scatterBroadcastSrcTest) {
  const char* kGraph = R"(graph(%z, %si, %a, %b):
%c = torch.ops.aten.clone.default(self=%z)
%s = torch.ops.aten.add.Tensor(self=%a, other=%b)
%o = torch.ops.aten.scatter.src(self=%c, dim=1, index=%si, src=%s)
return (%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"z", "si", "a", "b", "c", "s", "o"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  auto z = at::zeros({40, 50}, at::kLong);
  auto si = std::get<1>(at::sort(at::rand({40, 50}), 1))
                .narrow(1, 0, 30)
                .contiguous();
  auto a = at::randint(0, 1000, {40, 30}, at::kLong);
  auto b = at::randint(0, 1000, {40, 1}, at::kLong);
  auto outputs = runWaveProgrammatic(
      nativert::stringToGraph(kGraph), meta, {{z, si, a, b}});
  ASSERT_EQ(outputs.size(), 1);
  auto expected = z.clone().scatter(1, si, a + b);
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
}

// TensorMeta with the given static sizes, for rewrites that read them.
torch::_export::TensorMeta makeSizedTensorMeta(
    c10::ScalarType dtype,
    const std::vector<int64_t>& sizes) {
  auto meta = makeTensorMeta(dtype, static_cast<int64_t>(sizes.size()));
  std::vector<torch::_export::SymInt> symSizes;
  for (auto size : sizes) {
    torch::_export::SymInt dim;
    dim.set_as_int(size);
    symSizes.push_back(std::move(dim));
  }
  meta.set_sizes(std::move(symSizes));
  return meta;
}

// The ads id-matching search pads the sorted rows and the queries with cat
// and slices the padding rows off the result. The search is row by row, so
// the padding is dead: the plan must search the unpadded tensors (no cat
// left) and match eager. Programmatic graph, no .pt2.
TEST_F(ExecutorTest, searchsortedRowPaddingTest) {
  const char* kGraph = R"(graph(%s, %q):
%sp = torch.ops.aten.new_full.default(self=%s, size=[8, 50], fill_value=9223372036854775807)
%sl[] = prim.ListPack(l0=%s, l1=%sp)
%sc = torch.ops.aten.cat.default(tensors=%sl)
%qp = torch.ops.aten.new_zeros.default(self=%q, size=[8, 1200])
%ql[] = prim.ListPack(l0=%q, l1=%qp)
%qc = torch.ops.aten.cat.default(tensors=%ql)
%p = torch.ops.aten.searchsorted.Tensor(sorted_sequence=%sc, self=%qc)
%ps = torch.ops.aten.slice.Tensor(self=%p, dim=0, start=0, end=100)
%c = torch.ops.aten.clamp.default(self=%ps, max=49)
%g = torch.ops.aten.gather.default(self=%s, dim=1, index=%c)
%e = torch.ops.aten.eq.Tensor(self=%g, other=%q)
return (%e)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["s"] = makeSizedTensorMeta(c10::ScalarType::Long, {100, 50});
  meta["sp"] = makeSizedTensorMeta(c10::ScalarType::Long, {8, 50});
  meta["sc"] = makeSizedTensorMeta(c10::ScalarType::Long, {108, 50});
  meta["q"] = makeSizedTensorMeta(c10::ScalarType::Long, {100, 1200});
  meta["qp"] = makeSizedTensorMeta(c10::ScalarType::Long, {8, 1200});
  meta["qc"] = makeSizedTensorMeta(c10::ScalarType::Long, {108, 1200});
  meta["p"] = makeSizedTensorMeta(c10::ScalarType::Long, {108, 1200});
  meta["ps"] = makeSizedTensorMeta(c10::ScalarType::Long, {100, 1200});
  meta["c"] = makeSizedTensorMeta(c10::ScalarType::Long, {100, 1200});
  meta["g"] = makeSizedTensorMeta(c10::ScalarType::Long, {100, 1200});
  meta["e"] = makeSizedTensorMeta(c10::ScalarType::Bool, {100, 1200});

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  bool sawSearch = false;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.has("aten.cat.default")) << plan.describe();
        sawSearch = sawSearch || kernel.has("tw.searchsorted");
      }
    }
  }
  EXPECT_TRUE(sawSearch) << plan.describe();

  auto s = std::get<0>(at::sort(at::randint(0, 60, {100, 50}, at::kLong), 1));
  auto q = at::randint(0, 60, {100, 1200}, at::kLong);
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{s, q}});
  ASSERT_EQ(outputs.size(), 1);
  auto expected =
      at::eq(at::gather(s, 1, at::searchsorted(s, q).clamp_max(49)), q);
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
}

// any over trailing dims as the fused row reduction tw.any_rows: any.dim on
// the last dim (also given as -1) of a fused bool producer, and any.dims over
// [1, 2] and [2]. No any may run standalone, and the results match eager.
// Programmatic graph, no .pt2.
TEST_F(ExecutorTest, anyRowsTest) {
  const char* kGraph = R"(graph(%a, %b, %m):
%e = torch.ops.aten.eq.Tensor(self=%a, other=%b)
%h = torch.ops.aten.bitwise_and.Tensor(self=%e, other=%m)
%r0 = torch.ops.aten.any.dim(self=%h, dim=2)
%r1 = torch.ops.aten.any.dim(self=%h, dim=-1)
%r2 = torch.ops.aten.any.dims(self=%h, dim=[1, 2])
%r3 = torch.ops.aten.any.dims(self=%m, dim=[2])
return (%r0, %r1, %r2, %r3)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["a"] = makeTensorMeta(c10::ScalarType::Long, 3);
  meta["b"] = makeTensorMeta(c10::ScalarType::Long, 3);
  for (const char* name : {"m", "e", "h"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Bool, 3);
  }
  for (const char* name : {"r0", "r1", "r3"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Bool, 2);
  }
  meta["r2"] = makeTensorMeta(c10::ScalarType::Bool, 1);

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.standalone && kernel.has("aten.any"))
            << plan.describe();
      }
    }
  }

  auto a = at::randint(0, 50, {20, 30, 40}, at::kLong);
  auto b = at::randint(0, 50, {20, 30, 40}, at::kLong);
  auto m = at::rand({20, 30, 40}) < 0.3;
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{a, b, m}});
  ASSERT_EQ(outputs.size(), 4);
  auto h = at::eq(a, b) & m;
  std::vector<at::Tensor> expected{
      h.any(2), h.any(-1), h.any(at::IntArrayRef{1, 2}), m.any(2)};
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_TRUE(tensorsMatch(outputs[i], expected[i]))
        << i << ": " << firstDifference(outputs[i], expected[i]);
  }
}

// cumsum(new_ones(x, [n])) with a dynamic n, the range generator of the ads
// graph, expanded to [30, n] and added to a per-row operand. It rewrites to a
// materialized tw.arange_mem plus one, so no new_ones runs standalone, and
// the result matches eager in every mode. The text format cannot type a
// SymInt list, so the graph is parsed with a static-size new_ones and its
// size is then replaced by [sym_size(y, 1)] of a 2-D y (the size of a 1-D
// tensor would take the new_ones -> ones_like rewrite instead). Programmatic
// graph, no .pt2.
std::unique_ptr<nativert::Graph> makeDynamicOnesRangeGraph() {
  const char* kGraph = R"(graph(%x, %y, %b):
%o = torch.ops.aten.new_ones.default(self=%x, size=[17])
%c = torch.ops.aten.cumsum.default(self=%o, dim=0)
%r = torch.ops.aten.sub.Tensor(self=%c, other=1)
%u = torch.ops.aten.unsqueeze.default(self=%r, dim=0)
%e = torch.ops.aten.expand.default(self=%u, size=[30, -1])
%bu = torch.ops.aten.unsqueeze.default(self=%b, dim=1)
%a = torch.ops.aten.add.Tensor(self=%bu, other=%e)
%v = torch.ops.aten.view.default(self=%a, size=[-1])
return (%v)
)";
  auto graph = nativert::stringToGraph(kGraph);
  nativert::Node* ones = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.target() == "torch.ops.aten.new_ones.default") {
      ones = &node;
    }
  }
  auto* y = graph->tryGetValue("y");
  auto* size = graph->createNode("torch.ops.aten.sym_size.int", {{"self", y}});
  size->addAttribute({"dim", int64_t{1}});
  auto* n = size->addOutput("n", nativert::Type(nativert::Type::Kind::SymInt));
  graph->insertBefore(size, ones);
  auto* pack =
      graph->createListPack({n}, nativert::Type(nativert::Type::Kind::SymInt));
  graph->insertBefore(pack, ones);
  auto* dynamicOnes = graph->createNode(
      "torch.ops.aten.new_ones.default",
      {{"self", ones->inputs()[0].value}, {"size", pack->outputs()[0]}});
  auto* onesOut = dynamicOnes->addOutput(
      "o2", nativert::Type(nativert::Type::Kind::Tensor));
  graph->insertBefore(dynamicOnes, ones);
  graph->replaceAllUses(ones->outputs()[0], onesOut);
  graph->removeNode(ones);
  graph->renumberValues();
  return graph;
}

TEST_F(ExecutorTest, dynamicOnesRangeTest) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"x", "b", "o", "o2", "c", "r"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  for (const char* name : {"y", "u", "e", "bu", "a"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 2);
  }
  meta["v"] = makeTensorMeta(c10::ScalarType::Long, 1);

  auto plan = compilePlanProgrammatic(
      makeDynamicOnesRangeGraph(), meta, CompiledPlan::Mode::kMultiKernel);
  bool sawRange = false;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.has("aten.new_ones.default")) << plan.describe();
        sawRange = sawRange || kernel.has("tw.arange_mem");
      }
    }
  }
  EXPECT_TRUE(sawRange) << plan.describe();

  auto x = at::randint(0, 10, {30}, at::kLong);
  auto y = at::randint(0, 10, {3, 17}, at::kLong);
  auto b = at::randint(0, 1000, {30}, at::kLong) * 100;
  auto outputs =
      runWaveProgrammatic(makeDynamicOnesRangeGraph(), meta, {{x, y, b}});
  ASSERT_EQ(outputs.size(), 1);
  auto range = at::arange(17, at::kLong);
  auto expected =
      (b.unsqueeze(1) + range.unsqueeze(0).expand({30, 17})).view({-1});
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
}

// cat([x, new_full(x, size=[1], fill_value=n)]) with a dynamic n, as the ads
// graph appends a total to offsets. The one-element fill rewrites to a fused
// unsqueeze(scalar_tensor(n)), so no new_full runs standalone, and the result
// matches eager in every mode. The graph is parsed with a constant fill that
// is then replaced by sym_size(y, 1), a SymInt the text format cannot type.
// Programmatic graph, no .pt2.
std::unique_ptr<nativert::Graph> makeDynamicFillGraph() {
  const char* kGraph = R"(graph(%x, %y):
%f = torch.ops.aten.new_full.default(self=%x, size=[1], fill_value=7)
%l[] = prim.ListPack(l0=%x, l1=%f)
%c = torch.ops.aten.cat.default(tensors=%l)
return (%c)
)";
  auto graph = nativert::stringToGraph(kGraph);
  nativert::Node* full = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.target() == "torch.ops.aten.new_full.default") {
      full = &node;
    }
  }
  auto* size = graph->createNode(
      "torch.ops.aten.sym_size.int", {{"self", graph->tryGetValue("y")}});
  size->addAttribute({"dim", int64_t{1}});
  auto* n = size->addOutput("n", nativert::Type(nativert::Type::Kind::SymInt));
  graph->insertBefore(size, full);
  auto* dynamicFull = graph->createNode(
      "torch.ops.aten.new_full.default",
      {{"self", full->inputs()[0].value}, {"fill_value", n}});
  dynamicFull->addAttribute({"size", std::vector<int64_t>{1}});
  auto* fullOut = dynamicFull->addOutput(
      "f2", nativert::Type(nativert::Type::Kind::Tensor));
  graph->insertBefore(dynamicFull, full);
  graph->replaceAllUses(full->outputs()[0], fullOut);
  graph->removeNode(full);
  graph->renumberValues();
  return graph;
}

TEST_F(ExecutorTest, dynamicFillTest) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"x", "f", "f2", "c"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Long, 1);
  }
  meta["y"] = makeTensorMeta(c10::ScalarType::Long, 2);
  auto plan = compilePlanProgrammatic(
      makeDynamicFillGraph(), meta, CompiledPlan::Mode::kMultiKernel);
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        EXPECT_FALSE(kernel.has("aten.new_full.default")) << plan.describe();
      }
    }
  }
  auto x = at::randint(0, 100, {40}, at::kLong);
  auto y = at::zeros({3, 23}, at::kLong);
  auto outputs = runWaveProgrammatic(makeDynamicFillGraph(), meta, {{x, y}});
  ASSERT_EQ(outputs.size(), 1);
  auto expected = at::cat({x, at::full({1}, 23, at::kLong)});
  EXPECT_TRUE(tensorsMatch(outputs[0], expected))
      << firstDifference(outputs[0], expected);
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

// A launch runs at the largest block size any of its ops asks for, and every
// device function in it takes that as its block size. tw.sum2 is aten.sum's
// single-block form with a block size of 1024; the two aten.sum ops beside it
// ask for none, yet share its launch, so they have to run correctly at 1024 as
// well. In the multi-kernel form each aten.sum ends in a tw.sum_final one step
// later, in the same node, and that step runs at the default again. A device
// function compiled for
// one block size and launched at another reduces the wrong number of lanes and
// gets the sums wrong. Programmatic graph, no .pt2.
void registerSum2() {
  if (Registry::metadata("tw.sum2") != nullptr) {
    return;
  }
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.sum2",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .argumentMeta({{.isRegister = true}})
      .returnMeta(
          {{.isRegister = false,
            .reserve =
                [](const ReserveArgs&, Shape* out, int32_t) {
                  out[0].rank = 0;
                  return true;
                }}})
      .headerFile("velox/experimental/torchwave/Scan.cuh")
      .deviceFunc("tw_sum")
      .sharedDecls({{"Int32X32", "warpSums"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .blockSize(1024)
      .registerOp();
}

const char* kSum2Graph = R"(graph(%a, %b, %c):
%a2 = torch.ops.aten.mul.Tensor(self=%a, other=%a)
%s1 = torch.ops.aten.sum.default(self=%a2)
%s2 = tw.sum2(input=%b, dtype="Float")
%s3 = torch.ops.aten.sum.default(self=%c)
%d = torch.ops.aten.sub.Tensor(self=%c, other=%s3)
%s4 = torch.ops.aten.sum.default(self=%d)
return(%s1, %s2, %s3, %s4)
)";

std::unordered_map<std::string, torch::_export::TensorMeta> sum2Meta() {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto* name : {"a", "b", "c", "a2", "d"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 1);
  }
  for (const auto* name : {"s1", "s2", "s3", "s4"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 0);
  }
  return meta;
}

TEST_F(ExecutorTest, blockSizeFromOpTest) {
  registerSum2();
  for (const auto mode :
       {CompiledPlan::Mode::kMultiKernel, CompiledPlan::Mode::kSingleBlock}) {
    auto plan = compilePlanProgrammatic(
        nativert::stringToGraph(kSum2Graph), sum2Meta(), mode);
    EXPECT_TRUE(plan.runsAtBlockSize("tw.sum2", 1024));
    EXPECT_TRUE(plan.runsAtBlockSize("aten.mul.Tensor", 1024));
  }
  // The aten.sum ops split into a head and a final. The heads share
  // tw.sum2's step, the finals come a step later in the same node, at the
  // default size again.
  auto multiPlan = compilePlanProgrammatic(
      nativert::stringToGraph(kSum2Graph),
      sum2Meta(),
      CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(multiPlan.fuses({"aten.mul.Tensor", "tw.sum_head"}));
  EXPECT_TRUE(multiPlan.inLaterStep("tw.sum_final", "tw.sum2"));
  EXPECT_TRUE(multiPlan.runsAtBlockSize("tw.sum_final", 256));
  // tw.sum2 has no cooperative-grid form, so the cg view of the plan does not
  // show it beside the tw.sum_cg ops; the cg run below checks that case.

  // Small integers, so every summation order gives the same float.
  auto a = (at::arange(3000, at::kFloat).remainder(5));
  auto b = (at::arange(5000, at::kFloat).remainder(7));
  auto c = (at::arange(700, at::kFloat).remainder(3));
  auto check = [&](const char* label) {
    SCOPED_TRACE(label);
    auto outputs = runWaveProgrammatic(
        nativert::stringToGraph(kSum2Graph), sum2Meta(), {{a, b, c}});
    ASSERT_EQ(outputs.size(), 4);
    EXPECT_TRUE(tensorsMatch(outputs[0], (a * a).sum()))
        << firstDifference(outputs[0], (a * a).sum());
    EXPECT_TRUE(tensorsMatch(outputs[1], b.sum()))
        << firstDifference(outputs[1], b.sum());
    EXPECT_TRUE(tensorsMatch(outputs[2], c.sum()))
        << firstDifference(outputs[2], c.sum());
    EXPECT_TRUE(tensorsMatch(outputs[3], (c - c.sum()).sum()))
        << firstDifference(outputs[3], (c - c.sum()).sum());
  };
  SCOPE_EXIT {
    WaveConfig::get().useSingleBlock = std::nullopt;
    WaveConfig::get().isCg = std::nullopt;
  };
  WaveConfig::get().useSingleBlock = false;
  check("multi-block");
  WaveConfig::get().useSingleBlock = true;
  check("single-block");
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = true;
  check("cg");
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
// choice. Programmatic graph, no external .pt2.
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
  const auto savedSingleBlock = config.useSingleBlock;
  const auto savedCg = config.isCg;
  auto resetConfig =
      folly::makeGuard([&, savedFree, savedSingleBlock, savedCg] {
        config.useSingleBlock = savedSingleBlock;
        config.isCg = savedCg;
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

  // Allocation groups over the single-block grid, which they are on for
  // whenever isCg is set. There the cumsum feeding o2 is computed inside the
  // concat's own kernel and sized only by that launch, while another operand
  // is written a node earlier. The group must not lay the result out before
  // the cumsum is sized.
  WaveConfig::get().isCg = false;
  WaveConfig::get().useSingleBlock = true;
  runTest(
      "data/cat_test.pt2", "data/cat_test_results.pt", "single-block groups");
  WaveConfig::get().useSingleBlock = std::nullopt;
  WaveConfig::get().isCg = std::nullopt;

  // Plan structure. o3 = cat([ms1, ms2, ms3]) joins three operands, which is
  // the allocation group's path: the result is laid out on the host, so every
  // operand's extent has to be known before the concat sizes anything, and
  // there is no serial fill to fall back on that could discover an extent as
  // it goes. A masked_select's extent is out of reach either way -- the cg and
  // single-pass forms settle it on device, and the decomposed form's compaction
  // step knows it only through its own reserve -- so in every grid variant the
  // operand ends its kernel and is materialized before the cat rather than
  // fusing into it. That read-back is the point: it is what makes the extent an
  // ordinary frame tensor's by the time the concat lays the result out.
  auto plans = compilePlans("data/cat_test.pt2");
  if (WaveConfig::get().singlePass) {
    // --tw_single_pass replaces both decompositions with one look-back op.
    EXPECT_FALSE(
        plans.cg.fuses({"aten.cat.default", "tw.masked_select_1pass"}));
    EXPECT_FALSE(plans.multiKernel.fuses(
        {"aten.cat.default", "tw.masked_select_1pass"}));
  } else {
    EXPECT_FALSE(plans.cg.fuses({"aten.cat.default", "tw.masked_select_cg"}));
    EXPECT_FALSE(plans.multiKernel.fuses(
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

// A copy into a pitched band. catNdTest covers the operand an expression
// computes, which writes through the view the concat hands it; this one covers
// the operand that is copied. A clone reads its source at the source's layout
// and writes one element per element, so given a band with the result's row
// pitch it has to map the write through the destination's strides -- indexing
// the destination linearly walks out of the band and into the next operand's.
// Every copy here therefore has a computed operand beside it, so that a linear
// write shows up as a corrupted neighbour and not only as a short tail.
//
// The concat allocation group is what hands a copy its band, and it only does
// so for a producer declaring ArgumentMeta::mayWriteStrided. So the numbers
// alone are not the assertion: a copy refused the band still comes out right,
// __concatCopy just moves it in afterwards. numConcatStridedBand is what
// separates the two.
TEST_F(ExecutorTest, catStridedBandTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  const std::string pt2 = "data/cat_strided_band_test.pt2";
  const std::string results = "data/cat_strided_band_test_results.pt";

  config.useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  config.useSingleBlock = true;
  runTest(pt2, results, "single-block");
  config.useSingleBlock = std::nullopt;

  config.isCg = true;
  runTest(pt2, results, "cg");

  // The arm that actually carves the bands: the group needs the cooperative
  // grid and the freeing.
  config.freeIntermediates = true;
  runTest(pt2, results, "cg groups");

  // Ten operands carved over four concats: the copies -- three each in o1, o2
  // and o4, and o3's middle one.
  //
  // The count is what says the copies wrote through the destination's strides,
  // because seven of the ten bands are pitched; only o4's three are a run. A
  // copy that cannot write a pitched band is refused it and moved in afterwards
  // by __concatCopy, which leaves the values right and the count at three.
  //
  // Ten and not eleven because concatOperandsInPlace is on: o3's computed
  // operand used to be carved as an eleventh member, writing through the view
  // the concat handed it. It is now fused into the concat's own kernel and
  // writes its band directly, which is a member the group no longer has to
  // bind.
  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatMembers, 10);
  EXPECT_EQ(stats.numConcatGroups, 4);
  EXPECT_EQ(stats.numConcatTooFew, 0);
  EXPECT_EQ(stats.numConcatNoMembers, 0);
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 0);
  EXPECT_EQ(stats.numInConcatGroup, stats.numConcatMembers + 4);
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
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
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
  runTest(pt2, results, "cg alloc groups");

  // What the pass made of it. Every cat above the threshold forms a group, and
  // nothing is left to the offset walk: an operand reaches its region one of
  // two ways, and the counts below separate them. 'wide' places all four of its
  // gathers and 'nd' all three -- the cat is a kernel break, so the elementwise
  // op between each gather and the concat no longer fuses into the concat's
  // kernel and each operand becomes a launch that writes its region. 'mixed's
  // 'plain' is the other way: the graph hands it to us, no launch writes it, so
  // the group cannot carve it and a copy op of its own fills its band instead.
  // That copy is outside the group, which is why 'mixed' carves 2 of 3.
  //
  // 'scaled' is a third way. Under concatOperandsInPlace its operands are not
  // pushed into launches of their own at all: the concat's own kernel computes
  // them straight into their bands, so the group binds none of them and the
  // total is 9.
  //
  // The run above is what proves the copy happened. A band nothing writes is
  // not empty, it is whatever the fresh allocation held, so 'mixed' comes back
  // wrong from element 127 -- where 'plain's region starts -- if the copy is
  // dropped or lands in a buffer of its own.
  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatGroups, 4);
  // 'wide' 4, 'mixed' 2, 'nd' 3, and 'scaled' none -- see below.
  EXPECT_EQ(stats.numConcatMembers, 4 + 2 + 3 + 0);
  // Each group also takes its own result out of the lifetime grouping.
  EXPECT_EQ(stats.numInConcatGroup, stats.numConcatMembers + 4);
  // 'pair' is below the threshold, so it keeps the concat's own fill.
  EXPECT_EQ(stats.numConcatTooFew, 1);
  // 'scaled' carves nothing, and that is the concatOperandsInPlace state: its
  // operands are sized where the concat's own kernel computes them, so the
  // group owns the result's buffer and lays the regions out but has no write to
  // redirect. With the pushdown instead, each of its three became a launch of
  // its own and a member the group had to bind.
  EXPECT_EQ(stats.numConcatNoMembers, 1);
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 0);

  // The same mode on the multi-kernel grid. Nothing about a lifetime or a band
  // needs the cooperative grid -- both grids are settled by compilation -- so
  // the plan is built against whichever one the config names, and the values
  // have to come out the same on either.
  config.isCg = false;
  runTest(pt2, results, "multi-kernel alloc groups");

  auto mk = allocGroupStats(pt2, /*cg=*/false);
  // Every concat that grouped under the cooperative grid still groups here: a
  // group's members are decided by where the operands are written relative to
  // where the result is laid out, and the multi-kernel grid orders those the
  // same way.
  EXPECT_EQ(mk.numConcatGroups, 4);
  EXPECT_EQ(mk.numConcatMembers, 4 + 2 + 3 + 0);
  EXPECT_EQ(mk.numInConcatGroup, mk.numConcatMembers + 4);
  EXPECT_EQ(mk.numConcatTooFew, 1);
  EXPECT_EQ(mk.numConcatNoMembers, 1);
  EXPECT_EQ(mk.numConcatUnplaceableOperand, 0);
}

// The rule the parallel fill exists for: a cat of more than two operands never
// reaches the serial fill. 'handed' is the case catAllocGroupTest does not
// have -- six operands the graph hands over, so no launch writes any of them,
// the allocation group carves none and every band is filled by a copy op of its
// own. Those copies have to run in the concat's own step, side by side. One
// after another is the chain the rule forbids, and it is not visible in the
// values: a serial fill computes the same result, slowly.
//
// The other half is the kernel boundary. The bands are written by launches
// beside the concat rather than inside it, so a reader of the result cannot
// fuse into the concat's kernel and has to land in a later step.
TEST_F(ExecutorTest, catParallelFillTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  const std::string pt2 = "data/cat_parallel_fill_test.pt2";
  const std::string results = "data/cat_parallel_fill_test_results.pt";

  config.useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  config.useSingleBlock = std::nullopt;

  config.isCg = true;
  config.freeIntermediates = true;
  runTest(pt2, results, "cg alloc groups");

  // Both concats are grouped even though 'handed' has nothing to carve: the
  // group owns the result's storage and lays the bands out, which is what the
  // copies write through. 'mixed' carves its three gathers and copies 'tail'.
  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatGroups, 2);
  EXPECT_EQ(stats.numConcatMembers, 0 + 3);
  EXPECT_EQ(stats.numConcatNoMembers, 1);
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 0);

  // With isCg forced there is one grid, and it is the cooperative one: the
  // variant selection builds cgGrid_ only when it is choosing between variants
  // at run time. So the plan to read is the default one, not 'cg'.
  const auto& plan = compilePlansAsConfigured(pt2).multiKernel;
  // Six copies for 'handed' and one for 'tail', each in the step of the concat
  // whose band it fills.
  EXPECT_TRUE(
      plan.sideBySideWith("aten.clone.default", "aten.cat.default", 6 + 1));
  // Neither reader fuses into a concat's kernel, and both run after the last
  // of them -- the bands are written beside the concat, so the whole step has
  // to be over before the result can be read.
  EXPECT_TRUE(
      plan.kernelBoundaryBetween("aten.cat.default", "aten.mul.Scalar"));
  EXPECT_TRUE(plan.runsAfter("aten.mul.Scalar", "aten.cat.default"));
  EXPECT_TRUE(
      plan.kernelBoundaryBetween("aten.cat.default", "aten.add.Scalar"));
  EXPECT_TRUE(plan.runsAfter("aten.add.Scalar", "aten.cat.default"));
}

// A wide concat whose allocation group is laid out in an earlier node than the
// one the concat runs in, at a step past that node's last launch. Its 11
// operands are the prim.ListUnpack outputs of an unbind, which the first node
// writes after its kernel grids; the sort that follows is an eager standalone,
// so the concat, added to a value computed after it, lands in a later node.
// The group is carved at the empty step past the first node's grids, and the
// later node's copies of the 11 operands write into its bands. A step loop that
// stops at its first empty step drops the group, and those copies get empty
// parameters: an illegal address, or garbage in 'joined'.
TEST_F(ExecutorTest, catLateGroupTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  const std::string pt2 = "data/cat_late_group_test.pt2";
  const std::string results = "data/cat_late_group_test_results.pt";

  config.useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  config.useSingleBlock = true;
  runTest(pt2, results, "single-block");
  config.useSingleBlock = std::nullopt;

  config.isCg = true;
  config.freeIntermediates = true;
  runTest(pt2, results, "cg alloc groups");

  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatGroups, 1);
}

// Two wide concats of the same shape over different operands. They deduplicate
// to one project operation that runs twice, so everything the concat works
// with is formal and each instance has to be handed its own actual -- including
// the values the operand copies write, which stand for no node in the graph the
// caller exported.
//
// 'shared' is joined twice by each concat: four regions filled from one buffer,
// across two instances. Each needs a destination of its own. A destination
// reused between the two instances is the failure this is for, and it is not
// visible in the plan -- both instances look right on their own. It shows up
// only in the values, where one concat's region holds the other's data.
TEST_F(ExecutorTest, catDedupTest) {
  auto& config = WaveConfig::get();
  const auto savedFree = config.freeIntermediates;
  auto resetConfig = folly::makeGuard([&, savedFree] {
    config.useSingleBlock = std::nullopt;
    config.isCg = std::nullopt;
    config.freeIntermediates = savedFree;
  });

  const std::string pt2 = "data/cat_dedup_test.pt2";
  const std::string results = "data/cat_dedup_test_results.pt";

  config.useSingleBlock = false;
  runTest(pt2, results, "multi-block");
  config.useSingleBlock = std::nullopt;

  // The two concats reach the same plan, and the copies that fill their bands
  // run off one kernel op with per-launch parameters, so a mix-up between the
  // instances would be a mix-up between two runs of the same code.
  config.isCg = true;
  config.freeIntermediates = true;
  runTest(pt2, results, "cg alloc groups");

  // Both concats are grouped, and each carves the one operand written at the
  // step its result is laid out. The other gather runs a node earlier, so its
  // buffer already exists when the result is allocated and a copy moves it;
  // 'shared' twice and the tail are the graph's, which no launch writes at all.
  //
  // This counted 2 per concat before the layout point came from placement, but
  // none of those four was ever carved: the group was created at a step where
  // its members are not sized, so the collector captured nothing and the run
  // reported "2 of 2 values were not sized by any launch" for each group. The
  // count below is the number the run actually carves.
  auto stats = allocGroupStats(pt2);
  EXPECT_EQ(stats.numConcatGroups, 2);
  EXPECT_EQ(stats.numConcatMembers, 1 + 1);
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

// A concat that promotes one operand: the int32 lengths of an elementwise op
// joined with int64 tensors, so the result is int64. The narrow operand can be
// computed by the concat's own kernel, but it must not write its band there
// with its own element type, nor be copied in raw: either puts pairs of int32
// values into each int64 slot. Without a forced grid there are no allocation
// groups and the concat's own fill moves the operand; with one, the concat
// group places the result and a copy op of the operand's own fills its band.
TEST_F(ExecutorTest, catPromotedOperandTest) {
  auto& config = WaveConfig::get();
  const auto savedCg = config.isCg;
  auto resetConfig = folly::makeGuard([&, savedCg] { config.isCg = savedCg; });

  const char* kGraph = R"(graph(%a, %b, %c):
%d = torch.ops.aten.sub.Tensor(self=%a, other=%a, alpha=-1)
%list[] = prim.ListPack(l0=%b, l1=%d, l2=%c)
%o = torch.ops.aten.cat.default(tensors=%list, dim=0)
return(%o)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["a"] = makeTensorMeta(c10::ScalarType::Int, 1);
  meta["d"] = makeTensorMeta(c10::ScalarType::Int, 1);
  meta["b"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["c"] = makeTensorMeta(c10::ScalarType::Long, 1);
  meta["o"] = makeTensorMeta(c10::ScalarType::Long, 1);

  auto a = (at::arange(10000, at::kInt) % 5) + 1;
  auto b = at::arange(3000, at::kLong) * 3;
  auto c = at::arange(4000, at::kLong) + 11;
  auto reference = at::cat({b, (a + a).to(at::kLong), c});
  auto check = [&](const char* label) {
    SCOPED_TRACE(label);
    auto outputs =
        runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{a, b, c}});
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_TRUE(tensorsMatch(outputs[0], reference))
        << firstDifference(outputs[0], reference);
  };

  config.isCg = std::nullopt;
  check("auto, no allocation groups");
  config.isCg = true;
  check("cg, concat group");
  config.isCg = false;
  check("multi-kernel, concat group");
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

// A step with far more ops than the wave has blocks, whose sizes span three
// orders of magnitude and are settled on device. One thousand independent
// repeat_interleave results -- 90% of them a few tens of thousands of elements,
// a handful in the millions -- each scaled by an elementwise op, plus one
// barrier op (a cooperative-grid cumsum) to make the step a cg grid.
//
// The barrier op is what turns the size spread into the pathology this is
// about. A cg grid is trimmed back to one wave, and with more ops than the wave
// has blocks the trim cannot get there: it strips the widest op a block at a
// time until every op is on one, so the multi-million-element op ends up beside
// the sixteen-thousand-element ones and the step's makespan becomes its alone.
// (Without the barrier op the ordinary pro-rata split handles this graph fine
// -- 1096 blocks over a 432-block wave, skew 1.0.) The size mix comes out of
// the runtime counts tensor, so none of it is known when the grid is compiled.
TEST_F(ExecutorTest, launchSkewTest) {
  constexpr int32_t kNumOps = 1000;
  // Elements in the tensor every op repeats, so an op's size is kSeedLen times
  // its count.
  constexpr int64_t kSeedLen = 32;

  // Millions for a few, hundreds of thousands for a tenth, tens of thousands
  // for the remaining 90%.
  auto countFor = [](int32_t index) -> int64_t {
    if (index < 3) {
      return 62'500;
    }
    return index < 100 ? 1'875 : 500;
  };

  std::string graphStr = "graph(%seed, %counts):\n";
  std::string returns;
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  meta["seed"] = makeTensorMeta(c10::ScalarType::Float, 1);
  meta["counts"] = makeTensorMeta(c10::ScalarType::Long, 1);
  for (int32_t i = 0; i < kNumOps; ++i) {
    graphStr += fmt::format(
        "%c{0} = torch.ops.aten.select.int(self=%counts, dim=0, index={0})\n"
        "%x{0} = torch.ops.aten.repeat_interleave.self_Tensor(self=%seed, repeats=%c{0})\n"
        "%o{0} = torch.ops.aten.mul.Scalar(self=%x{0}, other=2.0)\n",
        i);
    meta[fmt::format("c{}", i)] = makeTensorMeta(c10::ScalarType::Long, 0);
    meta[fmt::format("x{}", i)] = makeTensorMeta(c10::ScalarType::Float, 1);
    meta[fmt::format("o{}", i)] = makeTensorMeta(c10::ScalarType::Float, 1);
    returns += fmt::format("{}%o{}", i == 0 ? "" : ", ", i);
  }
  // One barrier op in the same step as the thousand scales: a cg cumsum over
  // the smallest of them, so it contributes almost nothing to the step's work
  // and only its barrier matters.
  graphStr += fmt::format(
      "%cs = torch.ops.aten.cumsum.default(self=%x{}, dim=0)\n", kNumOps - 1);
  meta["cs"] = makeTensorMeta(c10::ScalarType::Float, 1);
  returns += ", %cs";
  graphStr += "return(" + returns + ")\n";

  auto seed = at::arange(kSeedLen, at::kFloat);
  std::vector<int64_t> countValues;
  countValues.reserve(kNumOps);
  for (int32_t i = 0; i < kNumOps; ++i) {
    countValues.push_back(countFor(i));
  }
  auto counts = at::tensor(countValues, at::kLong);

  // The op count and the size spread are the point of the test; if the graph
  // ever stops producing them the assertions below would pass vacuously.
  ASSERT_EQ(countFor(0) * kSeedLen, 2'000'000);
  ASSERT_EQ(countFor(999) * kSeedLen, 16'000);

  // launchMeta and the per-block clocks, and with them the grid measurement,
  // are only collected under kTiming.
  const int32_t savedTrace = WaveConfig::get().trace;
  auto resetConfig = folly::makeGuard([savedTrace] {
    WaveConfig::get().trace = savedTrace;
    WaveConfig::get().partitionLaunches = false;
    WaveConfig::get().isCg = std::nullopt;
  });
  WaveConfig::get().trace =
      savedTrace | WaveConfig::kGrid | WaveConfig::kTiming;
  WaveConfig::get().isCg = true;

  // How evenly a step's blocks actually ran, from their own clocks: the work
  // done over what the machine could have done in the longest block's time. 1
  // is every block finishing together.
  struct StepBalance {
    LaunchMeta meta;
    double util{0};
    int64_t maxClocks{0};
  };

  // The step of the last run whose grid came out worst balanced, and how its
  // blocks really ran.
  auto worstStep = []() -> StepBalance {
    const auto& info = waveThreadInfo();
    StepBalance worst;
    for (size_t i = 0; i < info.launchMeta.size(); ++i) {
      const auto& step = info.launchMeta[i];
      if (step.gridStats.numOps <= 1 ||
          step.gridStats.skew <= worst.meta.gridStats.skew) {
        continue;
      }
      worst = StepBalance{.meta = step};
      if (i >= info.debugInfo.size() || info.debugInfo[i].empty()) {
        continue;
      }
      int64_t total = 0;
      for (const auto& block : info.debugInfo[i]) {
        total += block.clocks;
        worst.maxClocks = std::max(worst.maxClocks, block.clocks);
      }
      if (worst.maxClocks > 0) {
        worst.util = static_cast<double>(total) /
            (static_cast<double>(worst.maxClocks) *
             static_cast<double>(info.debugInfo[i].size()));
      }
    }
    return worst;
  };

  auto checkOutputs = [&](const std::vector<at::Tensor>& outputs,
                          const char* label) {
    ASSERT_EQ(outputs.size(), static_cast<size_t>(kNumOps) + 1) << label;
    for (int32_t i = 0; i < kNumOps; ++i) {
      auto reference =
          at::repeat_interleave(seed, at::scalar_tensor(countFor(i), at::kLong))
              .mul(2.0);
      ASSERT_TRUE(tensorsMatch(outputs[i], reference))
          << label << " output " << i << ": "
          << firstDifference(outputs[i], reference);
    }
    // The barrier op. Its blocks have to stay co-resident in one cooperative
    // launch however the step is split, and a scan is what notices when they
    // do not.
    auto scan = at::cumsum(
        at::repeat_interleave(
            seed, at::scalar_tensor(countFor(kNumOps - 1), at::kLong)),
        0);
    ASSERT_TRUE(tensorsMatch(outputs[kNumOps], scan))
        << label << " cumsum: " << firstDifference(outputs[kNumOps], scan);
  };

  WaveConfig::get().partitionLaunches = false;
  checkOutputs(
      runWaveProgrammatic(
          nativert::stringToGraph(graphStr), meta, {{seed, counts}}),
      "one launch per step");
  const auto unsplit = worstStep();

  // The premise: more ops than the wave has blocks, so the cg trim leaves
  // almost all of them on one block and the step runs many times longer than a
  // balanced wave of the same work would.
  EXPECT_GT(unsplit.meta.gridStats.numStarved, 0);
  EXPECT_GT(unsplit.meta.gridStats.skew, 5.0f);
  EXPECT_EQ(unsplit.meta.gridStats.numSegments, 1);

  WaveConfig::get().partitionLaunches = true;
  checkOutputs(
      runWaveProgrammatic(
          nativert::stringToGraph(graphStr), meta, {{seed, counts}}),
      "partitioned");
  const auto split = worstStep();

  LOG(INFO) << fmt::format(
      "launchSkew: node {} step {} ops={} target={} skew={:.1f} starved={}; "
      "one launch: util={:.1f}% maxClk={} -> {} launches: util={:.1f}% maxClk={}",
      unsplit.meta.sequenceNumber,
      unsplit.meta.stepIdx,
      unsplit.meta.gridStats.numOps,
      unsplit.meta.gridStats.targetBlocks,
      unsplit.meta.gridStats.skew,
      unsplit.meta.gridStats.numStarved,
      100.0 * unsplit.util,
      unsplit.maxClocks,
      split.meta.gridStats.numSegments,
      100.0 * split.util,
      split.maxClocks);

  // Same step, now spread over several packed waves. An op straddling a launch
  // boundary still computes its whole slice, which is what the matching outputs
  // above establish.
  EXPECT_EQ(split.meta.sequenceNumber, unsplit.meta.sequenceNumber);
  EXPECT_EQ(split.meta.stepIdx, unsplit.meta.stepIdx);
  EXPECT_GT(split.meta.gridStats.numSegments, 1);
  // The point of the split: the longest block, which is the step's makespan,
  // comes down because the op that owned it is no longer confined to one block.
  EXPECT_LT(split.maxClocks, unsplit.maxClocks / 2);
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

// refillWaveFrame must leave a frame an earlier execution ran on holding
// exactly what it held before that execution: the user inputs and nothing
// else. An execution that starts from a frame still carrying the previous
// run's values finds every standalone op's outputs already materialized and
// skips it (nodeOutputsComputed), so the standalone reports the previous run's
// result. Nothing else empties the frame with freeIntermediates off, and a
// repeat over unchanged inputs then still verifies -- the skipped op's stale
// output is the right answer -- so only the frame state shows the defect.
// index_put_chain's a -> b -> c -> d -> e chain returns only 'e', which leaves
// the rest behind as intermediates.
TEST_F(ExecutorTest, frameReuseResetTest) {
  const bool savedFree = WaveConfig::get().freeIntermediates;
  SCOPE_EXIT {
    WaveConfig::get().freeIntermediates = savedFree;
  };
  WaveConfig::get().freeIntermediates = false;

  auto fixture = ModelFixture::load(
      getDataFilePath(dataDir(), "data/index_put_chain_test.pt2"));
  ASSERT_NE(fixture, nullptr);
  setGraphDevice(fixture->model.graph.get(), true);

  WaveGraphExecutor exec(fixture->makeModelContext());
  auto& graph = exec.graph();
  const auto numValues = static_cast<int32_t>(graph.numValues());

  auto inputs = loadSampleInputs(*fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  auto frame = exec.getFrame();
  ASSERT_NE(frame, nullptr);
  fillWaveFrame(graph, *frame, deviceInputs);
  const auto beforeFirstRun = snapshotFrame(*frame, numValues);

  exec.executeWithPrefilledFrame(*frame);
  const auto afterFirstRun = snapshotFrame(*frame, numValues);
  ASSERT_GT(afterFirstRun.size(), beforeFirstRun.size())
      << "the fixture must leave intermediates in the frame, or the reset has "
         "nothing to undo and the test proves nothing";

  refillWaveFrame(exec, graph, *frame, deviceInputs);
  EXPECT_EQ(snapshotFrame(*frame, numValues), beforeFirstRun);

  exec.returnFrame(std::move(frame));
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

// aten.contiguous fused: 2D transposes through shared-memory tiles (int64 and
// many tiles too), and rank-3, sliced and rank-1 inputs copied. torch.export
// drops a contiguous whose input is contiguous when traced, so the elision is
// tested on a programmatic graph below.
TEST_F(ExecutorTest, contiguousTest) {
  runTest("data/contiguous_test.pt2", "data/contiguous_test_results.pt");
}

// A contiguous of an add's fresh result is elided: the graph proves that input
// dense. A contiguous of a transpose stays, as a fused copy rather than an
// eager op. Programmatic graph, no .pt2.
TEST_F(ExecutorTest, contiguousElidedWhenKnownDense) {
  const char* kGraph = R"(graph(%x, %y):
%a = torch.ops.aten.add.Tensor(self=%x, other=%y)
%c = torch.ops.aten.contiguous.default(self=%a)
%o = torch.ops.aten.mul.Tensor(self=%c, other=%c)
%t = torch.ops.aten.t.default(self=%x)
%d = torch.ops.aten.contiguous.default(self=%t)
return(%o, %d)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"x", "y", "a", "c", "o", "t", "d"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
  }

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  int32_t numContiguous = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        numContiguous += kernel.has("aten.contiguous.default");
      }
    }
  }
  EXPECT_EQ(numContiguous, 1) << plan.describe();
  EXPECT_TRUE(plan.fuses({"aten.contiguous.default"}));

  auto x = at::randn({64, 48}, at::kFloat);
  auto y = at::randn({64, 48}, at::kFloat);
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{x, y}});
  ASSERT_EQ(outputs.size(), 2);
  const auto sum = x + y;
  EXPECT_TRUE(tensorsMatch(outputs[0], sum * sum))
      << firstDifference(outputs[0], sum * sum);
  EXPECT_TRUE(tensorsMatch(outputs[1], x.t().contiguous()))
      << firstDifference(outputs[1], x.t().contiguous());
}

// A clone of a 2D transpose takes contiguous's shared-memory tiles; a clone of
// a rank-3 permute keeps the element copy. Both stay fused copies.
TEST_F(ExecutorTest, cloneTransposeTiled) {
  const char* kGraph = R"(graph(%x, %y):
%t = torch.ops.aten.t.default(self=%x)
%c = torch.ops.aten.clone.default(self=%t)
%p = torch.ops.aten.permute.default(self=%y, dims=[2, 0, 1])
%d = torch.ops.aten.clone.default(self=%p)
return(%c, %d)
)";
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const char* name : {"x", "t", "c"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 2);
  }
  for (const char* name : {"y", "p", "d"}) {
    meta[name] = makeTensorMeta(c10::ScalarType::Float, 3);
  }

  auto plan = compilePlanProgrammatic(
      nativert::stringToGraph(kGraph), meta, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(plan.fuses({"aten.clone.default"})) << plan.describe();
  int32_t numTiled = 0;
  int32_t numElementwise = 0;
  for (const auto& node : plan.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        numTiled +=
            kernel.code.find("tw_contiguous<float, true>") != std::string::npos;
        numElementwise += kernel.code.find("tw_contiguous<float, false>") !=
            std::string::npos;
      }
    }
  }
  EXPECT_EQ(numTiled, 1) << plan.describe();
  EXPECT_EQ(numElementwise, 1) << plan.describe();

  auto x = at::randn({70, 45}, at::kFloat);
  auto y = at::randn({6, 5, 4}, at::kFloat);
  auto outputs =
      runWaveProgrammatic(nativert::stringToGraph(kGraph), meta, {{x, y}});
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_TRUE(tensorsMatch(outputs[0], x.t().contiguous()))
      << firstDifference(outputs[0], x.t().contiguous());
  EXPECT_TRUE(tensorsMatch(outputs[1], y.permute({2, 0, 1}).contiguous()))
      << firstDifference(outputs[1], y.permute({2, 0, 1}).contiguous());
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

// (x + 1, None, x, None, None, x * 2): the user outputs hold the None slots,
// the graph's output node only the tensors.  Each None slot must come back
// None; the returned input x, which stays in the frame, must not fill one.
TEST_F(ExecutorTest, noneOutputTest) {
  runTest("data/none_output_test.pt2", "data/none_output_test_results.pt");
}

// The graph returns new_ones(x, [sym_size(y, 0)]), which the optimizer replaces
// with ones_like(y), a new value.
TEST_F(ExecutorTest, replacedOutputTest) {
  runTest(
      "data/replaced_output_test.pt2", "data/replaced_output_test_results.pt");
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

// cumsum, sum and min along the last dim of inputs of rank 2 to 4, as the ads
// preproc graph has them: they run as tw.cumsum_rows, tw.sum_rows and
// tw.min_rows with a row per index into the leading dims, a transposed input
// included.
TEST_F(ExecutorTest, rowsNdTest) {
  runTest("data/rows_nd_test.pt2", "data/rows_nd_test_results.pt");
  WaveConfig::get().isCg = true;
  runTest("data/rows_nd_test.pt2", "data/rows_nd_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;

  auto plans = compilePlans("data/rows_nd_test.pt2");
  for (const char* op : {"tw.cumsum_rows", "tw.sum_rows", "tw.min_rows"}) {
    EXPECT_TRUE(plans.multiKernel.fuses({op})) << op;
  }
  for (const char* op :
       {"aten.cumsum.default", "aten.sum.dim_IntList", "aten.min.dim"}) {
    EXPECT_FALSE(plans.multiKernel.standalone(op)) << op;
  }
}

// A scalar base raised to a tensor of exponents fuses. An empty new_ones, which
// a cumsum and a cat read, runs as a host shortcut with no kernel.
TEST_F(ExecutorTest, powScalarTest) {
  runTest("data/pow_scalar_test.pt2", "data/pow_scalar_test_results.pt");
  WaveConfig::get().isCg = true;
  runTest("data/pow_scalar_test.pt2", "data/pow_scalar_test_results.pt", "cg");
  WaveConfig::get().isCg = std::nullopt;

  auto plans = compilePlans("data/pow_scalar_test.pt2");
  EXPECT_FALSE(plans.multiKernel.standalone("aten.pow.Scalar"));
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
