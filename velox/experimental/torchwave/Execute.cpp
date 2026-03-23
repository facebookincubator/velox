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
#include <memory>
#include <string>

#include <ATen/ATen.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/kernels/KernelFactory.h>

#include "velox/experimental/torchwave/Execute.h"

namespace torch::wave {

std::unique_ptr<torch::nativert::ExecutionFrame> executeGraph(
    const LoadedModel& model) {
  const auto& graph = *model.graph;
  const auto& reader = model.reader;
  const auto& modelName = model.modelName;

  // Create Weights from the archive.
  auto weights = std::make_shared<torch::nativert::Weights>(
      &graph,
      reader,
      model.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      model.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  std::cout << "Weights loaded\n";

  // Initialize kernels for every node in the graph.
  torch::nativert::ExecutorConfig config;
  torch::nativert::KernelFactory factory;
  auto execKernels =
      factory.initializeNodeKernels(graph, weights, config, reader);
  auto& nodeKernels = execKernels.nodeKernels;

  std::cout << "Initialized " << nodeKernels.size() << " kernels\n";

  // Create the execution frame (pre-populates weight/constant values).
  auto frame = std::make_unique<torch::nativert::ExecutionFrame>(
      graph, *weights, config);

  // Try to load sample inputs from the package.
  const auto& userInputs = graph.signature().userInputs();
  std::string sampleInputsPath = fmt::format(
      torch::_export::archive_spec::SAMPLE_INPUTS_FILENAME_FORMAT, modelName);
  bool loadedFromPackage = false;
  if (reader->hasRecord(sampleInputsPath)) {
    auto size = reader->getRecordSize(sampleInputsPath);
    std::vector<char> buffers(size);
    reader->getRecord(sampleInputsPath, buffers.data(), size);
    auto value = torch::jit::pickle_load(buffers);
    if (value.isTuple() && value.toTupleRef().elements().size() == 2) {
      const auto& argsVal = value.toTupleRef().elements().at(0);
      const auto& kwargsVal = value.toTupleRef().elements().at(1);
      // Collect flat inputs: args then kwargs values.
      std::vector<c10::IValue> flatInputs;
      if (argsVal.isTuple()) {
        for (const auto& arg : argsVal.toTupleRef().elements()) {
          flatInputs.push_back(arg);
        }
      }
      if (kwargsVal.isTuple()) {
        for (const auto& kwarg : kwargsVal.toTupleRef().elements()) {
          flatInputs.push_back(kwarg);
        }
      } else if (kwargsVal.isGenericDict()) {
        for (const auto& entry : kwargsVal.toGenericDict()) {
          flatInputs.push_back(entry.value());
        }
      }
      auto count = std::min(flatInputs.size(), userInputs.size());
      for (size_t i = 0; i < count; ++i) {
        auto* val = graph.tryGetValue(userInputs[i]);
        if (val) {
          frame->setIValue(val->id(), flatInputs[i]);
        }
      }
      std::cout << "Loaded " << count << " sample inputs from package\n";
      loadedFromPackage = true;
    }
  }

  // Fall back to zero tensors for any inputs not loaded.
  if (!loadedFromPackage) {
    const auto& tensorValuesMeta = graph.tensorValuesMeta();
    for (const auto& name : userInputs) {
      auto it = tensorValuesMeta.find(name);
      if (it != tensorValuesMeta.end()) {
        const auto& tm = it->second;
        auto* value = graph.tryGetValue(name);
        if (value && !tm.hasSymbolicShape()) {
          auto tensor =
              at::zeros(tm.sizes(), at::TensorOptions().dtype(tm.dtype()));
          frame->setIValue(value->id(), std::move(tensor));
        }
      }
    }
    std::cout
        << "User inputs populated with zeros (no sample inputs in package)\n";
  }

  // Execute all nodes except prim.Input (0) and prim.Output (last).
  for (size_t i = 1; i + 1 < nodeKernels.size(); ++i) {
    const auto* node = nodeKernels[i]->node();
    std::cout << "Executing node " << i << ": " << node->target() << "\n";
    nodeKernels[i]->compute(*frame);
  }

  std::cout << "Execution complete\n";
  return frame;
}

} // namespace torch::wave
