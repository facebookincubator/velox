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

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <folly/init/Init.h>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/DescribePt.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GraphView.h"
#include "velox/experimental/torchwave/Pt2Load.h"

DEFINE_string(pt2, "", "Path to a .pt2 file (open source torch.export format)");
DEFINE_string(
    model_name,
    "",
    "Substring filter for model names (empty matches all)");
DEFINE_bool(
    print_params,
    false,
    "Print model parameters and sample input metadata");
DEFINE_bool(print_graph, false, "Print graph signature and output node");
DEFINE_bool(
    list_models,
    false,
    "List model names found in the package and exit");
DEFINE_bool(compile, false, "Compile the graph");
DEFINE_bool(optimize, false, "Optimize graph before printing");
DEFINE_bool(value_meta, false, "Show value type and rank annotations");
DEFINE_string(
    describe_pt,
    "",
    "Path to a .pt file to describe tensor contents");

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  torch::wave::initialize();

  if (!FLAGS_describe_pt.empty()) {
    torch::wave::describePt(FLAGS_describe_pt);
    return 0;
  }

  if (FLAGS_pt2.empty()) {
    std::cerr << "Error: --pt2 is required\n";
    return 1;
  }

  std::cout << "Loading pt2: " << FLAGS_pt2 << "\n";

  auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(FLAGS_pt2));

  const auto allModelNames = torch::wave::getModelNames(*reader);

  if (FLAGS_list_models) {
    for (const auto& name : allModelNames) {
      std::cout << name << "\n";
    }
    return 0;
  }

  for (const auto& modelName : allModelNames) {
    if (!FLAGS_model_name.empty() &&
        modelName.find(FLAGS_model_name) == std::string::npos) {
      continue;
    }

    std::cout << "\n=== Model: " << modelName << " ===\n";

    auto loaded = torch::wave::loadPt2Model(reader, modelName);
    auto& graph = *loaded.graph;

    std::cout << "Number of nodes: " << graph.nodes().size() << "\n";

    if (FLAGS_print_params) {
      torch::wave::printModelParams(graph);
    }

    if (FLAGS_print_graph) {
      torch::wave::printGraphView(graph, FLAGS_optimize, FLAGS_value_meta);
    }

    if (FLAGS_compile) {
      const auto& tensorValuesMeta = graph.tensorValuesMeta();
      torch::wave::ValueTypes types;
      types.types.resize(graph.values().size(), nullptr);
      std::vector<std::unique_ptr<torch::nativert::TensorMeta>> metaStore;
      for (const auto* value : graph.values()) {
        auto it = tensorValuesMeta.find(std::string{value->name()});
        if (it != tensorValuesMeta.end()) {
          auto meta = std::make_unique<torch::nativert::TensorMeta>(it->second);
          types.types[value->id()] = meta.get();
          metaStore.push_back(std::move(meta));
        }
      }
      torch::wave::WaveGraph waveGraph(graph, types);
      std::cout << waveGraph.toString() << "\n";
      return 0;
    }
  }

  return 0;
}
