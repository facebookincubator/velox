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

#include <gtest/gtest.h>

#include <re2/re2.h>
#include <filesystem>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container

#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Weights.h>

#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Pt2Load.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/torchwave/tests/CompiledPlan.h"

namespace torch::wave {
namespace {

std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  auto cwd = std::filesystem::current_path().string();
  if (cwd.size() >= 6 && cwd.compare(cwd.size() - 6, 6, "fbcode") == 0) {
    return cwd + "/" + baseDir + "/" + filePath;
  }
  return cwd + "/" + filePath;
}

torch::_export::ScalarType toExportScalarType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Long:
      return torch::_export::ScalarType::LONG;
    case c10::ScalarType::Float:
      return torch::_export::ScalarType::FLOAT;
    case c10::ScalarType::Bool:
      return torch::_export::ScalarType::BOOL;
    default:
      TORCH_CHECK(false, "unsupported dtype ", static_cast<int>(dtype));
  }
}

torch::_export::TensorMeta makeTensorMeta(c10::ScalarType dtype, int64_t rank) {
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
  std::vector<torch::_export::SymInt> sizes;
  sizes.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    torch::_export::SymInt dim;
    dim.set_as_int(1);
    sizes.push_back(std::move(dim));
  }
  meta.set_sizes(std::move(sizes));
  return meta;
}

class CompileTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    registerBuiltins();
  }

  // Builds a nativert graph from a text description, attaches 'meta', and
  // compiles it into a WaveGraph (host-side compile only, no GPU).
  std::unique_ptr<WaveGraph> compileGraphString(
      const std::string& graphStr,
      const std::unordered_map<std::string, torch::_export::TensorMeta>& meta) {
    auto graph = nativert::stringToGraph(graphStr);
    graph->setTensorValuesMeta(meta);
    setGraphDevice(graph.get(), /*isCuda=*/true);
    auto weights = std::make_shared<nativert::Weights>(graph.get());
    auto modelContext = std::make_unique<ModelContext>();
    modelContext->graph = std::move(graph);
    modelContext->weights = std::move(weights);
    modelContexts_.push_back(std::move(modelContext));
    return std::make_unique<WaveGraph>(modelContexts_.back().get());
  }

  /// Loads a .pt2 model and compiles it into a WaveGraph.
  std::unique_ptr<WaveGraph> loadAndCompile(const std::string& pt2File) {
    const std::string kBaseDir = "velox/experimental/torchwave/tests";
    auto pt2Path = getDataFilePath(kBaseDir, pt2File);

    auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
        std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));

    auto modelNames = getModelNames(*reader);
    EXPECT_FALSE(modelNames.empty()) << "No models found in " << pt2Path;
    if (modelNames.empty()) {
      return nullptr;
    }

    loadedModels_.push_back(loadPt2Model(reader, modelNames[0]));
    auto& graph = *loadedModels_.back().graph;

    const auto& tensorValuesMeta = graph.tensorValuesMeta();
    ValueTypes types;
    types.types.resize(graph.values().size(), nullptr);
    std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
    for (const auto* value : graph.values()) {
      auto it = tensorValuesMeta.find(std::string{value->name()});
      if (it != tensorValuesMeta.end()) {
        auto meta = std::make_unique<nativert::TensorMeta>(it->second);
        types.types[value->id()] = meta.get();
        metaStore.push_back(std::move(meta));
      }
    }

    auto& model = loadedModels_.back();
    auto weights = std::make_shared<nativert::Weights>(
        &graph,
        reader,
        model.tensorPaths,
        torch::_export::archive_spec::WEIGHTS_DIR,
        model.constantPaths,
        torch::_export::archive_spec::CONSTANTS_DIR);

    auto modelContext = std::make_unique<ModelContext>();
    modelContext->graph = std::move(model.graph);
    modelContext->weights = std::move(weights);
    modelContexts_.push_back(std::move(modelContext));
    auto waveGraph = std::make_unique<WaveGraph>(modelContexts_.back().get());
    metaStore_.insert(
        metaStore_.end(),
        std::make_move_iterator(metaStore.begin()),
        std::make_move_iterator(metaStore.end()));
    return waveGraph;
  }

  /// Checks that a Launch in the WaveGraph matches the given regex.
  ///
  /// Iterates over all CompiledNodes (filtered by 'node'), all
  /// ProjectOperations, both grid_ and singleBlockGrid_ (filtered by
  /// 'inSingleBlock'), all levels/steps (filtered by 'level'), and all
  /// launches in each step (filtered by 'exprIdx'). Returns true if any
  /// Launch::toString() matches the regex.
  bool checkGenerated(
      std::string_view regex,
      WaveGraph& graph,
      std::optional<int32_t> node = std::nullopt,
      std::optional<int32_t> level = std::nullopt,
      std::optional<bool> inSingleBlock = std::nullopt,
      std::optional<int32_t> exprIdx = std::nullopt) {
    re2::RE2 re(re2::StringPiece(regex.data(), regex.size()));

    const auto& nodes = graph.nodes();
    for (size_t ni = 0; ni < nodes.size(); ++ni) {
      if (node.has_value() && static_cast<int32_t>(ni) != *node) {
        continue;
      }
      const auto* composite = nodes[ni]->kernels();
      if (!composite) {
        continue;
      }
      for (const auto& op : composite->ops()) {
        auto* projectOp = op.projectOp();

        // Check both grids.
        auto checkGrid = [&](LaunchGrid& grid, bool isSingleBlock) -> bool {
          if (inSingleBlock.has_value() && *inSingleBlock != isSingleBlock) {
            return false;
          }
          for (size_t li = 0; li < grid.size(); ++li) {
            if (level.has_value() && static_cast<int32_t>(li) != *level) {
              continue;
            }
            for (size_t ei = 0; ei < grid[li].size(); ++ei) {
              if (exprIdx.has_value() && static_cast<int32_t>(ei) != *exprIdx) {
                continue;
              }
              auto str = grid[li][ei].toString();
              if (RE2::PartialMatch(str, re)) {
                return true;
              }
            }
          }
          return false;
        };

        if (checkGrid(projectOp->grid(), false)) {
          return true;
        }
        if (checkGrid(projectOp->singleBlockGrid(), true)) {
          return true;
        }
      }
    }
    return false;
  }

  /// Loads a .pt2 model and returns its (un-optimized) nativert graph. The
  /// LoadedModel and its reader are kept alive by this fixture.
  nativert::Graph& loadGraph(const std::string& pt2File) {
    const std::string kBaseDir = "velox/experimental/torchwave/tests";
    auto pt2Path = getDataFilePath(kBaseDir, pt2File);
    auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
        std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));
    auto modelNames = getModelNames(*reader);
    EXPECT_FALSE(modelNames.empty()) << "No models found in " << pt2Path;
    readers_.push_back(reader);
    loadedModels_.push_back(loadPt2Model(reader, modelNames[0]));
    return *loadedModels_.back().graph;
  }

 private:
  // Owns readers so graphs loaded via loadGraph remain valid.
  std::vector<std::shared_ptr<caffe2::serialize::PyTorchStreamReader>> readers_;
  // Owns LoadedModels so the graph (and its nodes) remain valid.
  std::vector<LoadedModel> loadedModels_;
  // Owns ModelContexts so the WaveGraph's borrowed pointer remains valid.
  std::vector<std::unique_ptr<ModelContext>> modelContexts_;
  // Owns TensorMeta objects so pointers in ValueTypes remain valid.
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore_;
};

// toString() takes its defaults from a default-constructed config, so a field
// is reported exactly when it differs from the default the header declares.
TEST(WaveConfigTest, toStringReportsOnlyNonDefaults) {
  WaveConfig config;
  EXPECT_EQ(config.toString(), "defaults");

  config.enableReuse = false;
  EXPECT_EQ(config.toString(), "enableReuse=false");

  config.enableReuse = true;
  config.trace = WaveConfig::kTiming;
  // freeIntermediates defaults to true, so turning it off is what shows up.
  config.freeIntermediates = false;
  EXPECT_EQ(config.toString(), "trace=16, freeIntermediates=false");
}

TEST_F(CompileTest, maskedSelectTest) {
  auto waveGraph = loadAndCompile("data/masked_select_test.pt2");
  ASSERT_NE(waveGraph, nullptr);

  auto str = waveGraph->toString();
  LOG(INFO) << "WaveGraph:\n" << str;
  EXPECT_FALSE(str.empty());

  // Multi-block grid has 3 steps.
  // Step 0: elementwise ops fused with masked_select_head.
  EXPECT_TRUE(checkGenerated("masked_select_head", *waveGraph, 0, 0, false, 0));
  EXPECT_TRUE(
      checkGenerated("aten\\.add\\.Tensor", *waveGraph, 0, 0, false, 0));
  EXPECT_TRUE(
      checkGenerated("aten\\.remainder\\.Scalar", *waveGraph, 0, 0, false, 0));
  EXPECT_TRUE(checkGenerated("aten\\.lt\\.Scalar", *waveGraph, 0, 0, false, 0));

  // Step 1: add_sizes.
  EXPECT_TRUE(checkGenerated("add_sizes", *waveGraph, 0, 1, false, 0));

  // Step 2: masked_select_final.
  EXPECT_TRUE(
      checkGenerated("masked_select_final", *waveGraph, 0, 2, false, 0));

  // No step 3 in multi-block grid.
  EXPECT_FALSE(checkGenerated(".", *waveGraph, 0, 3, false));

  // Multi-block grid should not have the fused single-block variant.
  EXPECT_FALSE(checkGenerated(
      "masked_select\\.default", *waveGraph, 0, std::nullopt, false));

  // Single block grid has 1 step with fused masked_select.
  EXPECT_TRUE(
      checkGenerated("masked_select\\.default", *waveGraph, 0, 0, true, 0));
  EXPECT_TRUE(checkGenerated("aten\\.add\\.Tensor", *waveGraph, 0, 0, true, 0));
  EXPECT_TRUE(checkGenerated("aten\\.lt\\.Scalar", *waveGraph, 0, 0, true, 0));

  // No step 1 in single block grid.
  EXPECT_FALSE(checkGenerated(".", *waveGraph, 0, 1, true));

  // Single block grid should not have the multi-step decomposition.
  EXPECT_FALSE(
      checkGenerated("masked_select_head", *waveGraph, 0, std::nullopt, true));
  EXPECT_FALSE(checkGenerated("add_sizes", *waveGraph, 0, std::nullopt, true));
  EXPECT_FALSE(
      checkGenerated("masked_select_final", *waveGraph, 0, std::nullopt, true));

  // Unregister remainder.Scalar and recompile. remainder becomes standalone.
  auto remainderMeta = Registry::unregister("torch.ops.aten.remainder.Scalar");
  auto noRemGraph = loadAndCompile("data/masked_select_test.pt2");
  ASSERT_NE(noRemGraph, nullptr);
  Registry::restoreRegistry(
      "torch.ops.aten.remainder.Scalar", std::move(remainderMeta));

  auto noRemStr = noRemGraph->toString();
  LOG(INFO) << "WaveGraph without remainder:\n" << noRemStr;

  // Multi-block grid now has 4 steps with standalone remainder at step 0.
  // Step 0: standalone remainder.
  EXPECT_TRUE(checkGenerated(
      "standalone.*remainder\\.Scalar", *noRemGraph, 0, 0, false, 0));

  // Step 1: add + lt fused with masked_select_head (remainder not fused).
  EXPECT_TRUE(
      checkGenerated("masked_select_head", *noRemGraph, 0, 1, false, 0));
  EXPECT_TRUE(
      checkGenerated("aten\\.add\\.Tensor", *noRemGraph, 0, 1, false, 0));
  EXPECT_TRUE(
      checkGenerated("aten\\.lt\\.Scalar", *noRemGraph, 0, 1, false, 0));
  EXPECT_FALSE(checkGenerated("remainder", *noRemGraph, 0, 1, false, 0));

  // Step 2: add_sizes.
  EXPECT_TRUE(checkGenerated("add_sizes", *noRemGraph, 0, 2, false, 0));

  // Step 3: masked_select_final.
  EXPECT_TRUE(
      checkGenerated("masked_select_final", *noRemGraph, 0, 3, false, 0));

  // No step 4.
  EXPECT_FALSE(checkGenerated(".", *noRemGraph, 0, 4, false));

  // Single block grid now has 2 steps with standalone remainder at step 0.
  // Step 0: standalone remainder.
  EXPECT_TRUE(checkGenerated(
      "standalone.*remainder\\.Scalar", *noRemGraph, 0, 0, true, 0));

  // Step 1: fused masked_select with add + lt.
  EXPECT_TRUE(
      checkGenerated("masked_select\\.default", *noRemGraph, 0, 1, true, 0));
  EXPECT_TRUE(
      checkGenerated("aten\\.add\\.Tensor", *noRemGraph, 0, 1, true, 0));
  EXPECT_TRUE(checkGenerated("aten\\.lt\\.Scalar", *noRemGraph, 0, 1, true, 0));

  // No step 2 in single block grid.
  EXPECT_FALSE(checkGenerated(".", *noRemGraph, 0, 2, true));
}

// The same facts as maskedSelectTest, expressed with the relationship matchers:
// what fuses and where the boundaries fall is visible without per-index
// bookkeeping or toString regexes.
TEST_F(CompileTest, planMatchers) {
  auto graph = loadAndCompile("data/masked_select_test.pt2");
  ASSERT_NE(graph, nullptr);

  // Multi-block: the head kernel fuses the elementwise prefix with
  // masked_select_head; the size scan and final compaction run in later steps
  // (kernel boundaries after the head).
  auto multi = CompiledPlan::from(*graph, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(multi.fuses(
      {"tw.masked_select_head",
       "aten.add.Tensor",
       "aten.lt.Scalar",
       "aten.remainder.Scalar"}));
  EXPECT_TRUE(multi.inLaterStep("tw.add_sizes", "tw.masked_select_head"));
  EXPECT_TRUE(multi.inLaterStep("tw.masked_select_final", "tw.add_sizes"));

  // The cg plan for the SAME graph comes out different from the multi-kernel
  // plan: cg fuses every stage into one cooperative kernel ordered by an
  // intra-kernel barrier, whereas multi splits the stages across kernel
  // boundaries with no barrier. Asserting both guards against the mode
  // parameter silently ceasing to change the plan.
  EXPECT_TRUE(multi.kernelBoundaryBetween(
      "tw.masked_select_head", "tw.masked_select_final"));
  auto cg = CompiledPlan::from(*graph, CompiledPlan::Mode::kCG);
  EXPECT_TRUE(cg.fuses({"tw.masked_select_cg", "aten.add.Tensor"}));
  EXPECT_TRUE(cg.barrierBetween("tw.masked_select_cg", "aten.add.Tensor"));

  // Single-block: the whole masked_select fuses into one kernel.
  auto single = CompiledPlan::from(*graph, CompiledPlan::Mode::kSingleBlock);
  EXPECT_TRUE(single.fuses(
      {"aten.masked_select.default", "aten.add.Tensor", "aten.lt.Scalar"}));

  // Drop remainder.Scalar: it can no longer fuse, so it falls to a standalone
  // launch with a boundary before -- and hence a step earlier than -- the
  // elementwise head, which still fuses add + lt.
  auto remainderMeta = Registry::unregister("torch.ops.aten.remainder.Scalar");
  auto noRem = loadAndCompile("data/masked_select_test.pt2");
  Registry::restoreRegistry(
      "torch.ops.aten.remainder.Scalar", std::move(remainderMeta));
  ASSERT_NE(noRem, nullptr);

  auto noRemMulti =
      CompiledPlan::from(*noRem, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(noRemMulti.standalone("aten.remainder.Scalar"));
  EXPECT_TRUE(noRemMulti.kernelBoundaryBetween(
      "aten.remainder.Scalar", "tw.masked_select_head"));
  EXPECT_TRUE(
      noRemMulti.inLaterStep("tw.masked_select_head", "aten.remainder.Scalar"));
  EXPECT_TRUE(noRemMulti.fuses(
      {"tw.masked_select_head", "aten.add.Tensor", "aten.lt.Scalar"}));
}

// tw.index_select reads its source and index as whole tensors (argumentMeta
// wholeTensor). In multi-kernel mode an elementwise producer of a whole-tensor
// input runs in its own earlier kernel: fusing it in would force an
// intra-kernel opBarrier, which promotes the launch to cooperative (whole grid
// resident). So the multi-kernel plan carries no barriers. (A downstream add
// that merely consumes index_select's register output still fuses with it,
// harmlessly and without a barrier -- which is why a name-based boundary
// matcher cannot be used here.)
TEST_F(CompileTest, planWholeTensorBoundary) {
  auto graph = loadAndCompile("data/index_select_test.pt2");
  ASSERT_NE(graph, nullptr);

  auto multi = CompiledPlan::from(*graph, CompiledPlan::Mode::kMultiKernel);
  int32_t numBarriers = 0;
  for (const auto& node : multi.nodes()) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        numBarriers += kernel.numBarriers;
      }
    }
  }
  EXPECT_EQ(numBarriers, 0) << multi.describe();
}

// A standalone op inside a cat that is the tensor repeated by
// repeat_interleave. repeat_interleave lowers to
// repeat_interleave_head(repeats) + repeat_interleave_final(input, prefix,
// total); `total` is inputFromPreviousKernel(2), so placeKernels processes only
// that input and skips the final's input[0] -- the whole cat subtree. The
// standalone op inside the cat's elementwise element is therefore never
// materialized as its own launch, yet the cat's border codegen walks into it
// and calls fusedCode on a standalone op, which aborts.
//
// Regression test for the inputFromPreviousKernel placement gap.
// repeat_interleave lowers to repeat_interleave_head(repeats) +
// repeat_interleave_final(input, prefix, total); total is
// inputFromPreviousKernel(2). placeKernels used to process only that ordering
// input and skip the final's input[0] -- here a cat whose element reads a
// standalone op -- so the standalone was never materialized as its own launch,
// yet extractSubgraph still pulled it in as an interior node and codegen
// aborted (the cat is incidental; the same gap reproduces with the standalone
// directly on input[0]). placeKernels now places the ordering input's stage
// first and then the remaining inputs, so the standalone is materialized.
//
// add.Tensor is re-registered as a schema-less standalone op, standing in for
// an intrinsically-standalone op such as fbgemm.jagged_to_padded_dense_forward
// in the ROO preproc graph; it stays registered, so this is a placement gap,
// not a missing registration. Before the fix this aborted with "forArguments
// requires functionSchema".
TEST_F(CompileTest, standaloneInCatBorderTest) {
  auto f = [] { return makeTensorMeta(c10::ScalarType::Float, 1); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"a", f()},
      {"b", f()},
      {"c", f()},
      {"repeats", makeTensorMeta(c10::ScalarType::Long, 1)},
      {"s", f()},
      {"e", f()},
      {"o", f()},
      {"r", f()}};
  const char* graphStr = R"(graph(%a, %b, %c, %repeats):
%s = torch.ops.aten.add.Tensor(self=%a, other=%b)
%e = torch.ops.aten.mul.Tensor(self=%s, other=%c)
%list[] = prim.ListPack(l0=%e)
%o = torch.ops.aten.cat.default(tensors=%list, dim=0)
%r = torch.ops.aten.repeat_interleave.self_Tensor(self=%o, repeats=%repeats)
return(%r)
)";

  // Re-register add.Tensor as a schema-less standalone op for the test (it
  // stays registered -- the failure is a placement gap, not a missing
  // registration).
  auto addMeta = Registry::unregister("torch.ops.aten.add.Tensor");
  MetadataBuilder("torch.ops.aten.add.Tensor", MetadataBuilder::NoSchema{})
      .isStandalone()
      .argumentMeta({{.isRegister = false}, {.isRegister = false}})
      .returnMeta({{.isRegister = false}})
      .outputConstraints(
          [](NodeCP, const ValueTypes&) -> std::vector<ValueConstraint> {
            return {{.rank = 1, .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();
  auto restore = folly::makeGuard([&] {
    Registry::unregister("torch.ops.aten.add.Tensor");
    Registry::restoreRegistry("torch.ops.aten.add.Tensor", std::move(addMeta));
  });

  auto waveGraph = compileGraphString(graphStr, meta);
  EXPECT_NE(waveGraph, nullptr);
}

// A linear chain of in-place index_puts on a new_ones tensor
//   a = new_ones(...); b = index_put(a, ...); c = index_put(b, ...);
//   d = index_put(c, ...); e = d * 2
// fuses into a single expr. Every tensor buffer created in that expr -- the
// factory result and each in-place index_put result -- is last-read within the
// expr and does not escape, so all of them must be flagged as last uses. Only
// the final result (mul) escapes to the graph output. Each successive clone
// inserted by the index_put rewrite aliases the same buffer as its index_put
// result, so the buffer is represented by exactly one flagged value.
TEST_F(CompileTest, indexPutChainLastUse) {
  auto& graph = loadGraph("data/index_put_chain_test.pt2");

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  // Union of every layer's last-use values.
  std::unordered_set<ValueCP> lastUse;
  for (auto* pn = last; pn != nullptr; pn = pn->input()) {
    lastUse.insert(pn->lastUse.begin(), pn->lastUse.end());
  }

  std::unordered_set<ValueCP> outputs;
  for (const auto& in : graph.outputNode()->inputs()) {
    outputs.insert(in.value);
  }

  // Every live buffer produced inside the fused chain -- the ones factory and
  // each in-place index_put result -- must be flagged as a non-escaping last
  // use.
  int32_t numChainBuffers = 0;
  for (auto* v : graph.values()) {
    if (v == nullptr || v->producer() == nullptr || v->users().empty()) {
      continue;
    }
    const std::string_view target = v->producer()->target();
    const bool isChainBuffer = target == "tw.index_put_elt_one" ||
        target == "torch.ops.aten.ones.default" ||
        target == "torch.ops.aten.new_ones.default";
    if (!isChainBuffer) {
      continue;
    }
    EXPECT_EQ(outputs.count(v), 0u)
        << "chain buffer %" << v->id() << " unexpectedly escapes";
    EXPECT_GT(lastUse.count(v), 0u)
        << "non-escaping chain buffer %" << v->id() << " (" << target
        << ") was not flagged as a last use";
    ++numChainBuffers;
  }
  // ones + three index_put results.
  EXPECT_EQ(numChainBuffers, 4);

  // The final result escapes to the graph output, so it is never a last use.
  for (auto* out : outputs) {
    EXPECT_EQ(lastUse.count(out), 0u)
        << "escaping output %" << out->id() << " must not be a last use";
  }
}

// A shared value c = a + b feeds two independent index_put chains whose ends
// (c2, c4) escape:
//   c1 = index_put(c, ...);  c2 = index_put(c1, ...)
//   c3 = index_put(c, ...);  c4 = index_put(c3, ...)
// c is read by both chains -- two exprs in the same layer -- so its buffer must
// not be reused in place: it must appear as a last use but must NOT be flagged
// reusable. The single-use temporaries (c1, c3) are non-escaping last uses; the
// chain ends escape and are never last uses.
TEST_F(CompileTest, indexPutForkSharedNotReusable) {
  auto& graph = loadGraph("data/index_put_fork_test.pt2");

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  std::unordered_set<ValueCP> outputs;
  for (const auto& in : graph.outputNode()->inputs()) {
    outputs.insert(in.value);
  }
  std::unordered_set<ValueCP> lastUse;
  for (auto* pn = last; pn != nullptr; pn = pn->input()) {
    lastUse.insert(pn->lastUse.begin(), pn->lastUse.end());
  }

  // The shared value c = a + b, read by both chains.
  ValueCP shared = nullptr;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.add.Tensor") {
      shared = v;
      break;
    }
  }
  ASSERT_NE(shared, nullptr);
  EXPECT_GE(shared->users().size(), 2u)
      << "shared value %" << shared->id() << " should feed both chains";

  // c is a last use, but must NOT be reusable in place: two exprs read it, so
  // overwriting its buffer would corrupt the other chain.
  bool sharedIsLastUse = false;
  for (auto* pn = last; pn != nullptr; pn = pn->input()) {
    if (pn->lastUse.count(shared) > 0) {
      sharedIsLastUse = true;
      EXPECT_FALSE(pn->isReusableInput(shared))
          << "shared value %" << shared->id()
          << " is read by two exprs and must not be flagged reusable";
    }
  }
  EXPECT_TRUE(sharedIsLastUse)
      << "shared value %" << shared->id() << " should be a last use";

  // The two chain ends escape to graph outputs, so they are never last uses.
  int32_t numOutputs = 0;
  for (auto* v : graph.values()) {
    if (v == nullptr || outputs.count(v) == 0) {
      continue;
    }
    ++numOutputs;
    EXPECT_EQ(lastUse.count(v), 0u)
        << "escaping output %" << v->id() << " must not be a last use";
  }
  EXPECT_EQ(numOutputs, 2);

  // Each non-escaping index_put temporary (c1, c3) is a single-use last use.
  int32_t numTemps = 0;
  for (auto* v : graph.values()) {
    if (v == nullptr || v->producer() == nullptr || v->users().empty() ||
        outputs.count(v) > 0) {
      continue;
    }
    if (v->producer()->target() != "tw.index_put_elt_one") {
      continue;
    }
    EXPECT_GT(lastUse.count(v), 0u)
        << "non-escaping index_put temp %" << v->id()
        << " must be flagged as a last use";
    ++numTemps;
  }
  EXPECT_EQ(numTemps, 2);
}

// The index_put chain
//   a = base.new_ones(...); b = a.index_put(...); c = b.index_put(...);
//   d = c.index_put(...); e = d * 2; return e
// fuses into one expr. The index_put results write in place, so a, b, c and d
// all share a's storage; only the separate `* 2` result e escapes. a's private
// buffer is therefore produced and consumed entirely within the expr and never
// escapes, so it must be flagged as an expr-local overwritable temp; the
// escaping output e must not be.
TEST_F(CompileTest, exprLocalOverwritableTemps) {
  auto& graph = loadGraph("data/index_put_chain_test.pt2");

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  std::unordered_set<ValueCP> outputs;
  for (const auto& in : graph.outputNode()->inputs()) {
    outputs.insert(in.value);
  }

  auto isOverwritable = [&](ValueCP v) {
    for (auto* pn = last; pn != nullptr; pn = pn->input()) {
      if (pn->isOverwritableTemp(v)) {
        return true;
      }
    }
    return false;
  };

  // The new_ones buffer that backs the in-place chain is an expr-local temp.
  ValueCP ones = nullptr;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        (v->producer()->target() == "torch.ops.aten.new_ones.default" ||
         v->producer()->target() == "torch.ops.aten.ones.default")) {
      ones = v;
      break;
    }
  }
  ASSERT_NE(ones, nullptr);
  EXPECT_TRUE(isOverwritable(ones))
      << "new_ones buffer %" << ones->id()
      << " is a non-escaping expr-local temp and should be overwritable";

  // No graph output may ever be an overwritable temp.
  for (auto* out : outputs) {
    EXPECT_FALSE(isOverwritable(out))
        << "escaping output %" << out->id() << " must not be overwritable";
  }

  // Every flagged temp has a producer and is not a graph output.
  int32_t numOverwritable = 0;
  for (auto* v : graph.values()) {
    if (v == nullptr || !isOverwritable(v)) {
      continue;
    }
    ++numOverwritable;
    EXPECT_NE(v->producer(), nullptr)
        << "overwritable temp %" << v->id() << " must have a producer";
    EXPECT_EQ(outputs.count(v), 0u)
        << "overwritable temp %" << v->id() << " must not escape";
  }
  EXPECT_GE(numOverwritable, 1)
      << "expected at least the new_ones buffer to be an overwritable temp";
}

// The linear index_put chain
//   a = base.new_ones(...); b = a.index_put(...); c = b.index_put(...);
//   d = c.index_put(...); e = d * 2
// is lowered to a defensive clone before each in-place index_put_ (its self).
// Every clone input (a, b, c) is a single-consumer, last-use buffer, so
// rewriteInPlace elides all of them: each in-place writer is rewired to write
// its original buffer, and every clone output is left dead (no users).
TEST_F(CompileTest, cloneElisionRewritesInPlaceChain) {
  auto& graph = loadGraph("data/index_put_chain_test.pt2");

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  // The defensive clones the index_put rewrite inserted, each currently read by
  // its in-place writer.
  std::vector<ValueCP> cloneOutputs;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.clone.default") {
      cloneOutputs.push_back(v);
    }
  }
  ASSERT_FALSE(cloneOutputs.empty())
      << "setup: the index_put chain should contain defensive clones";
  for (ValueCP c : cloneOutputs) {
    ASSERT_FALSE(c->users().empty())
        << "setup: clone %" << c->id() << " should have a consumer pre-pass";
  }

  // Use the optimized constraints (optimizeOnly copies the ValueTypes it is
  // given, so the local 'types' is not updated in place).
  parallelNodes.rewriteInPlace(graph, waveHolder->types());

  for (ValueCP c : cloneOutputs) {
    EXPECT_TRUE(c->users().empty())
        << "clone %" << c->id()
        << " should be elided: its in-place writer is rewired to the original"
        << " last-use buffer";
  }
}

// The fork graph
//   c = a + b;
//   c1 = index_put(c, ...);  c2 = index_put(c1, ...)
//   c3 = index_put(c, ...);  c4 = index_put(c3, ...)
// clones c before each of the two chains. c is read by both chains, so writing
// it in place would corrupt the other chain: rewriteInPlace must KEEP the
// clones whose input is c. The single-use temporaries c1 and c3 are still
// safely elided, so the pass is selective, not a no-op.
TEST_F(CompileTest, cloneElisionKeepsForkedInput) {
  auto& graph = loadGraph("data/index_put_fork_test.pt2");

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  ValueCP shared = nullptr;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.add.Tensor") {
      shared = v;
      break;
    }
  }
  ASSERT_NE(shared, nullptr);
  ASSERT_GE(shared->users().size(), 2u)
      << "shared value %" << shared->id() << " should feed both chains";

  // Clones whose input is the shared value: eliding these would write the
  // shared buffer in place.
  std::vector<ValueCP> sharedClones;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.clone.default" &&
        !v->producer()->inputs().empty() &&
        v->producer()->inputs()[0].value == shared) {
      sharedClones.push_back(v);
    }
  }
  ASSERT_FALSE(sharedClones.empty())
      << "setup: expected clones of the shared value";

  parallelNodes.rewriteInPlace(graph, waveHolder->types());

  for (ValueCP c : sharedClones) {
    EXPECT_FALSE(c->users().empty())
        << "clone %" << c->id()
        << " of the shared value must be kept: the shared buffer has another"
        << " consumer, so in-place would corrupt the other chain";
  }

  // Selectivity: the pass is not a no-op -- the single-use clones (of c1, c3)
  // were elided.
  int32_t elided = 0;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.clone.default" &&
        v->users().empty()) {
      ++elided;
    }
  }
  EXPECT_GT(elided, 0)
      << "expected the pass to elide the single-use clones while keeping the"
      << " shared ones";
}

// The functionalized form of `base[:, 0] = vals`:
//   s1  = slice(base, dim=0);  col = select(s1, dim=1, index=0)
//   cp  = copy(col, vals)                      <- reads base's column 0
//   s2  = slice(base, dim=0);  cl  = clone(s2) <- the defensive clone
//   ss  = select_scatter(cl, cp, dim=1, index=0)
//   out = slice_scatter(base, ss, dim=0)
// The clone's only consumer is the scatter, and base is dead afterwards, so the
// per-expr last-use test alone would elide the clone. That is wrong: cp is
// computed from a read of base at a different index path than the value being
// overwritten, and fusion puts that read in the same kernel as the write, so
// the read would return post-write data. The clone must be kept.
TEST_F(CompileTest, cloneElisionKeepsReadOfWriteTarget) {
  auto rank2 = [] { return makeTensorMeta(c10::ScalarType::Float, 2); };
  auto rank1 = [] { return makeTensorMeta(c10::ScalarType::Float, 1); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"inp", rank2()},
      {"base", rank2()},
      {"vals", rank1()},
      {"s1", rank2()},
      {"col", rank1()},
      {"cp", rank1()},
      {"s2", rank2()},
      {"cl", rank2()},
      {"ss", rank2()},
      {"out", rank2()}};
  // base is produced inside the graph, not a graph input: an externally owned
  // buffer is rejected for in-place reuse before the read/write check is
  // reached, which would make this fixture pass for the wrong reason.
  const char* graphStr =
      R"(graph(%inp, %vals):
%base = torch.ops.aten.ones_like.default(self=%inp)
%s1 = torch.ops.aten.slice.Tensor(self=%base, dim=0, start=0, end=9223372036854775807, step=1)
%col = torch.ops.aten.select.int(self=%s1, dim=1, index=0)
%cp = torch.ops.aten.copy.default(self=%col, src=%vals)
%s2 = torch.ops.aten.slice.Tensor(self=%base, dim=0, start=0, end=9223372036854775807, step=1)
%cl = torch.ops.aten.clone.default(self=%s2)
%ss = torch.ops.aten.select_scatter.default(self=%cl, src=%cp, dim=1, index=0)
%out = torch.ops.aten.slice_scatter.default(self=%base, src=%ss, dim=0, start=0, end=9223372036854775807, step=1)
return(%out)
)";

  auto graphOwner = nativert::stringToGraph(graphStr);
  graphOwner->setTensorValuesMeta(meta);
  setGraphDevice(graphOwner.get(), /*isCuda=*/true);
  auto& graph = *graphOwner;

  ValueTypes types;
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  auto waveHolder = WaveGraph::optimizeOnly(graph, types);

  ParallelNodes parallelNodes;
  auto* last = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(last, nullptr);

  std::vector<ValueCP> cloneOutputs;
  for (auto* v : graph.values()) {
    if (v != nullptr && v->producer() != nullptr &&
        v->producer()->target() == "torch.ops.aten.clone.default") {
      cloneOutputs.push_back(v);
    }
  }
  ASSERT_FALSE(cloneOutputs.empty())
      << "setup: the scatter's defensive clone should be present";
  for (ValueCP c : cloneOutputs) {
    ASSERT_FALSE(c->users().empty())
        << "setup: clone %" << c->id() << " should have a consumer pre-pass";
  }

  parallelNodes.rewriteInPlace(graph, waveHolder->types());

  for (ValueCP c : cloneOutputs) {
    EXPECT_FALSE(c->users().empty())
        << "clone %" << c->id()
        << " must be kept: another operand of the scatter is computed from a"
        << " read of the write target at a different index path";
  }
}

// transpose is registered as a view (viewOfArg + metadataOnly) rather than an
// unconditional standalone, so KernelOperation::setOutputs reaches its
// meta->isView() branch and gives it an OutputDesc with viewNode set: the view
// becomes an output of the enclosing kernel op, materialized as host-side
// metadata at launch, with no step of its own.
//
// transpose(a + transpose(b)) + c over 2-D operands: b is transposed into a's
// layout and the sum is transposed back into c's, so all four ops belong to one
// expression and compile to a single step. Registered standalone, as it used to
// be, each transpose is its own launch and the expression splits into four.
TEST_F(CompileTest, transposeAsFusedView) {
  auto f = [] { return makeTensorMeta(c10::ScalarType::Float, 2); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"a", f()},
      {"b", f()},
      {"c", f()},
      {"tb", f()},
      {"s", f()},
      {"ts", f()},
      {"r", f()}};
  const char* graphStr = R"(graph(%a, %b, %c):
%tb = torch.ops.aten.transpose.int(self=%b, dim0=0, dim1=1)
%s = torch.ops.aten.add.Tensor(self=%a, other=%tb)
%ts = torch.ops.aten.transpose.int(self=%s, dim0=0, dim1=1)
%r = torch.ops.aten.add.Tensor(self=%ts, other=%c)
return(%r)
)";

  auto countSteps = [](WaveGraph& g) {
    auto plan = CompiledPlan::from(g, CompiledPlan::Mode::kMultiKernel);
    int32_t steps = 0;
    for (const auto& node : plan.nodes()) {
      steps += static_cast<int32_t>(node.steps.size());
    }
    return steps;
  };

  // The registered behavior: the whole expression is one step and neither
  // transpose is a launch of its own.
  int32_t fusedSteps = 0;
  {
    auto waveGraph = compileGraphString(graphStr, meta);
    ASSERT_NE(waveGraph, nullptr);
    fusedSteps = countSteps(*waveGraph);
    auto plan =
        CompiledPlan::from(*waveGraph, CompiledPlan::Mode::kMultiKernel);
    EXPECT_FALSE(plan.standalone("aten.transpose.int")) << plan.describe();
    EXPECT_EQ(fusedSteps, 1) << plan.describe();
  }

  // Contrast: re-registered as an unconditional standalone, the same graph
  // splits. Without this the test would still pass if transpose silently
  // stopped appearing in the plan at all.
  auto saved = Registry::unregister("torch.ops.aten.transpose.int");
  auto restore = folly::makeGuard([&] {
    Registry::unregister("torch.ops.aten.transpose.int");
    Registry::restoreRegistry("torch.ops.aten.transpose.int", std::move(saved));
  });
  MetadataBuilder("torch.ops.aten.transpose.int")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(node->inputs()[0].value),
                 .contiguity = Contiguity::kUnknown}};
          })
      .registerOp();

  auto standaloneGraph = compileGraphString(graphStr, meta);
  ASSERT_NE(standaloneGraph, nullptr);
  auto standalonePlan =
      CompiledPlan::from(*standaloneGraph, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(standalonePlan.standalone("aten.transpose.int"))
      << standalonePlan.describe();
  EXPECT_GT(countSteps(*standaloneGraph), fusedSteps)
      << standalonePlan.describe();
}

// An operand that is a transposed view of a value the same kernel just wrote
// is read, at index i, as an element some other block produced. Only a
// grid-wide barrier orders that; the __syncthreads() between two fused
// expressions covers one block. The operand is marked isRegister like every
// elementwise argument, which is why the barrier test cannot go by that flag.
TEST_F(CompileTest, barrierBeforeATransposedLeafOfThisKernel) {
  auto f = [] { return makeTensorMeta(c10::ScalarType::Float, 2); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"a", f()}, {"b", f()}, {"c", f()}, {"s", f()}, {"ts", f()}, {"r", f()}};
  const char* graphStr = R"(graph(%a, %b, %c):
%s = torch.ops.aten.add.Tensor(self=%a, other=%b)
%ts = torch.ops.aten.transpose.int(self=%s, dim0=0, dim1=1)
%r = torch.ops.aten.mul.Tensor(self=%ts, other=%c)
return(%r)
)";

  auto waveGraph = compileGraphString(graphStr, meta);
  ASSERT_NE(waveGraph, nullptr);
  auto plan = CompiledPlan::from(*waveGraph, CompiledPlan::Mode::kMultiKernel);
  EXPECT_TRUE(plan.fuses({"aten.add.Tensor", "aten.mul.Tensor"}))
      << plan.describe();
  EXPECT_TRUE(plan.barrierBetween("aten.mul.Tensor", "aten.add.Tensor"))
      << plan.describe();

  // Contrast: consumed straight through, the intermediate stays in a register
  // and there is nothing to order. Without this the test would still pass if
  // every fused elementwise pair started carrying a barrier.
  std::unordered_map<std::string, torch::_export::TensorMeta> directMeta = {
      {"a", f()}, {"b", f()}, {"c", f()}, {"s", f()}, {"r", f()}};
  const char* directStr = R"(graph(%a, %b, %c):
%s = torch.ops.aten.add.Tensor(self=%a, other=%b)
%r = torch.ops.aten.mul.Tensor(self=%s, other=%c)
return(%r)
)";

  auto directGraph = compileGraphString(directStr, directMeta);
  ASSERT_NE(directGraph, nullptr);
  auto directPlan =
      CompiledPlan::from(*directGraph, CompiledPlan::Mode::kMultiKernel);
  EXPECT_FALSE(directPlan.barrierBetween("aten.mul.Tensor", "aten.add.Tensor"))
      << directPlan.describe();
}

// True when the node producing %<name> reads the same value twice, i.e. its
// two operands were merged. Asking the consumer rather than counting nodes is
// what keeps the answer independent of whether the merged-away node is later
// swept from the graph.
bool operandsMerged(nativert::Graph& graph, std::string_view name) {
  for (const auto& node : graph.nodes()) {
    for (const auto* out : node.outputs()) {
      if (out != nullptr && out->name() == name) {
        EXPECT_EQ(node.inputs().size(), 2u) << name;
        return node.inputs()[0].value == node.inputs()[1].value;
      }
    }
  }
  ADD_FAILURE() << "no node produces %" << name;
  return false;
}

// Common-subexpression elimination, gated separately for compute and views.
//
// The graph pairs one duplicated add (compute) with one duplicated transpose
// (a view), plus a third transpose whose dims differ. Each consumer takes the
// two candidates as its two operands, so "were they merged" reduces to "are
// this node's operands now the same value" -- which, unlike counting nodes,
// does not depend on whether the merged-away node is later swept from the
// graph.
TEST_F(CompileTest, commonSubexpressions) {
  auto f = [] { return makeTensorMeta(c10::ScalarType::Float, 2); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"a", f()},
      {"b", f()},
      {"s1", f()},
      {"s2", f()},
      {"m", f()},
      {"t1", f()},
      {"t2", f()},
      {"t3", f()},
      {"v", f()},
      {"w", f()},
      {"r", f()},
      {"out", f()}};
  const char* graphStr = R"(graph(%a, %b):
%s1 = torch.ops.aten.add.Tensor(self=%a, other=%b)
%s2 = torch.ops.aten.add.Tensor(self=%a, other=%b)
%m = torch.ops.aten.add.Tensor(self=%s1, other=%s2)
%t1 = torch.ops.aten.transpose.int(self=%b, dim0=0, dim1=1)
%t2 = torch.ops.aten.transpose.int(self=%b, dim0=0, dim1=1)
%t3 = torch.ops.aten.transpose.int(self=%b, dim0=1, dim1=0)
%v = torch.ops.aten.add.Tensor(self=%t1, other=%t2)
%w = torch.ops.aten.add.Tensor(self=%t1, other=%t3)
%r = torch.ops.aten.add.Tensor(self=%v, other=%w)
%out = torch.ops.aten.add.Tensor(self=%m, other=%r)
return(%out)
)";

  auto savedCompute = WaveConfig::get().cseCompute;
  auto savedViews = WaveConfig::get().cseViews;
  auto restore = folly::makeGuard([&] {
    WaveConfig::get().cseCompute = savedCompute;
    WaveConfig::get().cseViews = savedViews;
  });

  struct Case {
    bool compute;
    bool views;
  };
  for (const Case c :
       {Case{false, false},
        Case{true, false},
        Case{false, true},
        Case{true, true}}) {
    SCOPED_TRACE(fmt::format("cseCompute={} cseViews={}", c.compute, c.views));
    WaveConfig::get().cseCompute = c.compute;
    WaveConfig::get().cseViews = c.views;

    auto waveGraph = compileGraphString(graphStr, meta);
    ASSERT_NE(waveGraph, nullptr);
    auto& graph = *waveGraph->graph();

    // The duplicated add collapses only under cseCompute, the duplicated
    // transpose only under cseViews: each flag governs its own category.
    EXPECT_EQ(operandsMerged(graph, "m"), c.compute);
    EXPECT_EQ(operandsMerged(graph, "v"), c.views);

    // %t3 transposes the same tensor to the same result but spells its dims the
    // other way round. The key compares attributes verbatim, so it never merges
    // with %t1 -- this pass claims syntactic identity, not equivalence.
    EXPECT_FALSE(operandsMerged(graph, "w"));

    // The survivor must be the earlier of the two in program order. Keeping the
    // later one leaves the earlier one's consumers reading a value produced
    // after them, which makeParallelNodes rejects outright -- and user lists,
    // which is what the pass walks, are not in program order.
    for (const auto& node : graph.nodes()) {
      size_t consumerPos = 0;
      size_t pos = 0;
      for (const auto& other : graph.nodes()) {
        if (&other == &node) {
          consumerPos = pos;
          break;
        }
        ++pos;
      }
      for (const auto& input : node.inputs()) {
        if (input.value == nullptr || input.value->producer() == nullptr) {
          continue;
        }
        size_t producerPos = 0;
        bool found = false;
        pos = 0;
        for (const auto& other : graph.nodes()) {
          if (&other == input.value->producer()) {
            producerPos = pos;
            found = true;
            break;
          }
          ++pos;
        }
        EXPECT_TRUE(!found || producerPos < consumerPos)
            << node.target() << " reads %" << input.value->id()
            << " produced later by " << input.value->producer()->target();
      }
    }
  }
}

// Two nodes that compute the same thing from the same operands are still not
// interchangeable if a buffer either of them touches is written in place: the
// second one reads what the write left behind, and the first does not. The
// pass compares identity, not position, so it refuses on the whole buffer --
// a write anywhere in the graph disqualifies every node that reads or produces
// it, whether or not the write falls between the two candidates.
//
// Three graphs, because the rule has to hold from both ends of a node and the
// negative cases alone would be satisfied by a pass that merged nothing:
//   operand   %c is written, and %c is an operand of both adds
//   output    %s1 is written, and %s1 is what one of the adds produces
//   unrelated %d is written, and neither add goes near it -- the control
TEST_F(CompileTest, commonSubexpressionsMutation) {
  auto savedCompute = WaveConfig::get().cseCompute;
  auto savedViews = WaveConfig::get().cseViews;
  auto restore = folly::makeGuard([&] {
    WaveConfig::get().cseCompute = savedCompute;
    WaveConfig::get().cseViews = savedViews;
  });
  WaveConfig::get().cseCompute = true;
  WaveConfig::get().cseViews = true;

  auto f = [] { return makeTensorMeta(c10::ScalarType::Float, 2); };
  std::unordered_map<std::string, torch::_export::TensorMeta> meta = {
      {"b", f()},
      {"c", f()},
      {"d", f()},
      {"s1", f()},
      {"s2", f()},
      {"mut", f()},
      {"out", f()}};

  // %c is an operand of both adds and is written between them.
  const char* operandStr = R"(graph(%b, %c, %d):
%s1 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%mut = torch.ops.aten.add_.Tensor(self=%c, other=%b)
%s2 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%out = torch.ops.aten.add.Tensor(self=%s1, other=%s2)
return(%out)
)";

  // %s1 is what the first add produces, and it is written before the consumer
  // reads either value. Merging would repoint %s2's reader at a buffer the
  // write has since changed.
  const char* outputStr = R"(graph(%b, %c, %d):
%s1 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%s2 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%mut = torch.ops.aten.add_.Tensor(self=%s1, other=%b)
%out = torch.ops.aten.add.Tensor(self=%s1, other=%s2)
return(%out)
)";

  // The write lands on a buffer neither add reads or produces, so the two are
  // interchangeable and must still merge. Without this case the two above
  // would pass on a pass that had stopped merging altogether.
  const char* unrelatedStr = R"(graph(%b, %c, %d):
%s1 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%mut = torch.ops.aten.add_.Tensor(self=%d, other=%b)
%s2 = torch.ops.aten.add.Tensor(self=%b, other=%c)
%out = torch.ops.aten.add.Tensor(self=%s1, other=%s2)
return(%out)
)";

  struct Case {
    const char* name;
    const char* graphStr;
    bool merges;
  };
  for (const Case c :
       {Case{"operand", operandStr, false},
        Case{"output", outputStr, false},
        Case{"unrelated", unrelatedStr, true}}) {
    SCOPED_TRACE(c.name);
    auto waveGraph = compileGraphString(c.graphStr, meta);
    ASSERT_NE(waveGraph, nullptr);
    EXPECT_EQ(operandsMerged(*waveGraph->graph(), "out"), c.merges);
  }
}
} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  return RUN_ALL_TESTS();
}
