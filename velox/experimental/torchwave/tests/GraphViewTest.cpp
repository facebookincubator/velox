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

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container

#include "velox/experimental/torchwave/GraphView.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Pt2Load.h"

namespace torch::wave {
namespace {

// Splits a string by newlines and returns only non-empty lines.
std::vector<std::string> nonEmptyLines(const std::string& text) {
  std::vector<std::string> result;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty()) {
      result.push_back(line);
    }
  }
  return result;
}

TEST(GraphViewTest, elementTestProjectNodes) {
  auto pt2Path =
      std::string("velox/experimental/torchwave/tests/data/element_test.pt2");
  auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));
  auto modelNames = getModelNames(*reader);
  ASSERT_FALSE(modelNames.empty());

  auto loaded = loadPt2Model(reader, modelNames[0]);
  auto& graph = *loaded.graph;

  // Build ProjectNodes using ParallelExpr, same as printGraphView.
  ParallelNodes parallelNodes;
  auto* lastProjectNode = parallelNodes.makeParallelNodes(graph);
  ASSERT_NE(lastProjectNode, nullptr);

  // Collect ProjectNodes in order (first to last).
  std::vector<const ProjectNode*> projectNodeList;
  for (auto* pn = lastProjectNode; pn != nullptr; pn = pn->input()) {
    projectNodeList.push_back(pn);
  }
  std::reverse(projectNodeList.begin(), projectNodeList.end());
  ASSERT_GE(projectNodeList.size(), 2);

  // Produce the full toString output.
  std::string output;
  PlanObjectSet border;
  for (const auto* pn : projectNodeList) {
    output += pn->toString(graph, border);
    border.insert(pn->nodes().begin(), pn->nodes().end());
  }
  std::cout << output;

  // Expected lines from the element_test model. Each non-empty line must
  // appear in the output.
  const std::vector<std::string> expectedLines = {
      "ProjectNode 0:",
      "0.0: [%0 - %5] = prim.Input",
      "ProjectNode 1:",
      "1.0: %7 = torch.ops.aten.sub.Tensor(torch.ops.aten.add.Tensor(%0, %1), %2)",
      "1.1: %9 = torch.ops.aten.sub.Tensor(torch.ops.aten.add.Tensor(%2, %1, alpha=3), %0)",
      "1.2: %10 = torch.ops.aten.sub.Tensor(%2, %0)",
      "1.3: %11 = torch.ops.aten.add.Tensor(%3, %4, alpha=2)",
      "1.4: %13 = torch.ops.aten.add.Tensor(torch.ops.aten.sub.Tensor(%5, %4), %3)",
      "input: ProjectNode 0",
  };

  auto outputLines = nonEmptyLines(output);
  for (const auto& expected : expectedLines) {
    bool found = false;
    for (const auto& outputLine : outputLines) {
      if (outputLine.find(expected) != std::string::npos) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Expected line not found in output: " << expected;
  }
}

} // namespace
} // namespace torch::wave
