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

#include <folly/init/Init.h>
#include <glog/logging.h>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container

#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Weights.h>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Pt2Load.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/WaveGraph.h"

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

class CompileTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    registerBuiltins();
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

 private:
  // Owns LoadedModels so the graph (and its nodes) remain valid.
  std::vector<LoadedModel> loadedModels_;
  // Owns ModelContexts so the WaveGraph's borrowed pointer remains valid.
  std::vector<std::unique_ptr<ModelContext>> modelContexts_;
  // Owns TensorMeta objects so pointers in ValueTypes remain valid.
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore_;
};

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

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  return RUN_ALL_TESTS();
}
