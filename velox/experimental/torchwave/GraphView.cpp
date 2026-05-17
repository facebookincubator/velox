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

#include "velox/experimental/torchwave/GraphView.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/TensorMeta.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

void printTensorMeta(
    std::string_view name,
    const torch::nativert::TensorMeta& tm) {
  std::cout << "  path: " << name << "  shape: [";
  if (!tm.hasSymbolicShape()) {
    const auto sizes = tm.sizes();
    for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << sizes[i];
    }
    std::cout << "]  dtype: " << c10::toString(tm.dtype())
              << "  numel: " << tm.numel() << "\n";
  } else {
    std::cout << "symbolic]  dtype: " << c10::toString(tm.dtype()) << "\n";
  }
}

void printModelParams(const torch::nativert::Graph& graph) {
  const auto& sig = graph.signature();
  const auto& weightsMeta = graph.weightsMeta();
  const auto& tensorValuesMeta = graph.tensorValuesMeta();

  std::cout << "\nParameters:\n";
  for (const auto& [inputName, fqn] : sig.inputsToParameters()) {
    const std::string fqnStr(fqn);
    if (auto it = weightsMeta.find(fqnStr); it != weightsMeta.end()) {
      printTensorMeta(fqn, it->second);
    }
  }

  std::cout << "\nBuffers:\n";
  for (const auto& [inputName, fqn] : sig.inputsToBuffers()) {
    const std::string fqnStr(fqn);
    if (auto it = weightsMeta.find(fqnStr); it != weightsMeta.end()) {
      printTensorMeta(fqn, it->second);
    }
  }

  std::cout << "\nSample Inputs:\n";
  for (const auto& name : sig.userInputs()) {
    if (auto it = tensorValuesMeta.find(name); it != tensorValuesMeta.end()) {
      auto* value = graph.tryGetValue(name);
      if (value) {
        std::cout << "%" << value->id() << " ";
      }
      printTensorMeta(name, it->second);
    }
  }
}

void printGraphView(
    torch::nativert::Graph& graph,
    bool optimize,
    bool valueMeta) {
  initialize();
  std::cout << "\nGraph Signature:\n";
  std::cout << graph.signature() << "\n";
  std::cout << "\nOutput Node:\n";
  std::cout << graph.outputNode()->toString() << "\n";

  std::cout << "\nAll Nodes:\n";
  int nodeIdx = 0;
  for (const auto& node : graph.nodes()) {
    std::cout << nodeIdx++ << ": " << node.toString() << "\n";
  }

  ValueTypes types;
  std::vector<std::unique_ptr<torch::nativert::TensorMeta>> metaStore;
  std::unique_ptr<WaveGraph> waveGraphHolder;
  if (optimize || valueMeta) {
    initValueTypes(graph, types, metaStore);
    waveGraphHolder = WaveGraph::optimizeOnly(graph, std::move(types));
  }

  const ValueTypes* typesPtr =
      (valueMeta && waveGraphHolder) ? &waveGraphHolder->types() : nullptr;
  std::cout << "\nProject Nodes:\n";
  torch::wave::ParallelNodes parallelNodes;
  auto* lastProjectNode = parallelNodes.makeParallelNodes(graph);
  std::vector<const torch::wave::ProjectNode*> projectNodeList;
  for (auto* pn = lastProjectNode; pn != nullptr; pn = pn->input()) {
    projectNodeList.push_back(pn);
  }
  std::reverse(projectNodeList.begin(), projectNodeList.end());
  torch::wave::PlanObjectSet border;
  for (const auto* pn : projectNodeList) {
    std::cout << pn->toString(graph, border, typesPtr);
    border.insert(pn->nodes().begin(), pn->nodes().end());
  }

  std::unordered_map<std::string, int32_t> functionCounts;
  for (const auto* pn : projectNodeList) {
    pn->distinctFunctions(functionCounts);
  }
  std::vector<std::pair<std::string, int32_t>> sorted(
      functionCounts.begin(), functionCounts.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  std::cout << "\nDistinct Functions:\n";
  for (const auto& [name, count] : sorted) {
    std::cout << count << ": " << name << "\n";
  }

  std::unordered_map<std::string, int32_t> nameCounts;
  for (const auto& [key, count] : functionCounts) {
    auto pos = key.find(' ');
    auto name = (pos != std::string::npos) ? key.substr(0, pos) : key;
    nameCounts[name] += count;
  }
  std::vector<std::pair<std::string, int32_t>> sortedNames(
      nameCounts.begin(), nameCounts.end());
  std::sort(
      sortedNames.begin(), sortedNames.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
      });
  std::cout << "\nDistinct Function Names:\n";
  for (const auto& [name, count] : sortedNames) {
    std::cout << count << ": " << name << "\n";
  }

  const auto& tensorValuesMeta = graph.tensorValuesMeta();
  std::cout << "\nValue Tensor Metadata:\n";
  for (const auto* value : graph.values()) {
    auto it = tensorValuesMeta.find(std::string{value->name()});
    if (it == tensorValuesMeta.end()) {
      continue;
    }
    const auto& tm = it->second;
    std::cout << value->id() << " : ";
    if (!tm.hasSymbolicShape()) {
      const auto sizes = tm.sizes();
      std::cout << "dtype=" << c10::toString(tm.dtype()) << " shape=[";
      for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << sizes[i];
      }
      std::cout << "] device=" << tm.device() << " numel=" << tm.numel()
                << "\n";
    } else {
      std::cout << "dtype=" << c10::toString(tm.dtype())
                << " shape=[symbolic] device=" << tm.device() << "\n";
    }
  }
}

void compileGraph(torch::nativert::Graph& graph) {
  ValueTypes types;
  std::vector<std::unique_ptr<torch::nativert::TensorMeta>> metaStore;
  initValueTypes(graph, types, metaStore);
  WaveGraph waveGraph(graph, types);
  std::cout << waveGraph.toString() << "\n";
}

} // namespace torch::wave
