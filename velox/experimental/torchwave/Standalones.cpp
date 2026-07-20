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

#include "velox/experimental/torchwave/Standalones.h"

#include <ATen/ATen.h>

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

void runStandaloneShortcut(
    const LaunchData& data,
    nativert::ExecutionFrame& frame) {
  const auto& args = data.args;
  const auto& intArgs = data.intArgs;

  // Reads operand 'i' as a tensor from the frame.
  auto tensorAt = [&](size_t i) -> at::Tensor {
    const auto& iv = frame.getIValue(args[i]->id());
    TORCH_CHECK(
        iv.isTensor(),
        "runStandaloneShortcut: shortcut ",
        static_cast<int>(data.launch->standaloneShortcut),
        " operand ",
        i,
        " value %",
        args[i]->id(),
        " is not a tensor (tag=",
        static_cast<int>(iv.tag),
        ")");
    return iv.toTensor();
  };
  // Reads operand 'i' as an integer: a dynamic value (args[i] set) is read from
  // the frame; a constant comes from intArgs.
  auto intAt = [&](size_t i) -> int64_t {
    return args[i] != nullptr ? frame.getIValue(args[i]->id()).toInt()
                              : static_cast<int64_t>(intArgs[i]);
  };
  auto setOutput = [&](c10::IValue value) {
    frame.setIValue(data.actualOutputs[0], std::move(value));
  };

  switch (data.launch->standaloneShortcut) {
    case StandaloneShortcut::kView: {
      auto self = tensorAt(0);
      // All-constant dims pass through directly; otherwise the dims come from a
      // value operand (an int list in the frame).
      if (!data.intList.empty()) {
        setOutput(self.view(data.intList));
      } else {
        auto list = frame.getIValue(args[1]->id()).toIntList();
        setOutput(self.view(std::vector<int64_t>(list.begin(), list.end())));
      }
      break;
    }
    case StandaloneShortcut::kSlice: {
      // (Tensor self, int dim, int? start, int? end, int step).
      auto self = tensorAt(0);
      setOutput(at::slice(self, intAt(1), intAt(2), intAt(3), intAt(4)));
      break;
    }
    case StandaloneShortcut::kSelectInt: {
      // (Tensor self, int dim, int index).
      auto self = tensorAt(0);
      setOutput(at::select(self, intAt(1), intAt(2)));
      break;
    }
    case StandaloneShortcut::kUnsqueeze: {
      // (Tensor self, int dim).
      auto self = tensorAt(0);
      setOutput(at::unsqueeze(self, intAt(1)));
      break;
    }
    case StandaloneShortcut::kTranspose: {
      // (Tensor self, int dim0, int dim1).
      auto self = tensorAt(0);
      setOutput(at::transpose(self, intAt(1), intAt(2)));
      break;
    }
    case StandaloneShortcut::kNarrow: {
      // (Tensor self, int dim, int start, int length).
      auto self = tensorAt(0);
      setOutput(at::narrow(self, intAt(1), intAt(2), intAt(3)));
      break;
    }
    case StandaloneShortcut::kListPack: {
      c10::List<at::Tensor> list;
      list.reserve(args.size());
      for (auto* value : args) {
        const auto& iv = frame.getIValue(value->id());
        TORCH_CHECK(
            iv.isTensor(),
            "runStandaloneShortcut: kListPack element %",
            value->id(),
            " is not a tensor (tag=",
            static_cast<int>(iv.tag),
            ")");
        list.push_back(iv.toTensor());
      }
      setOutput(c10::IValue(std::move(list)));
      break;
    }
    case StandaloneShortcut::kNone:
      break;
  }
}

} // namespace torch::wave
