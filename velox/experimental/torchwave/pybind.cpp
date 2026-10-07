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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "velox/experimental/torchwave/Model.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace py = pybind11;

namespace {

// Runs the model on 'inputs', the graph's user inputs in order. An exported
// graph may carry non-tensor user inputs: a None or str argument stays as a
// dead placeholder (torch.export lowers the value to a literal at its use
// site), and a dynamic-shape graph takes its symbolic sizes as int inputs. Map
// each item to an IValue accordingly - None or str to an empty tensor (the
// executor cannot bind such a value; the slot is dead), Python int/float to a
// scalar, everything else to a tensor. Returns the output tensors.
std::vector<at::Tensor> runOptionalTensors(
    torch::wave::TorchWaveModel& self,
    const py::sequence& inputs) {
  std::vector<c10::IValue> ivalues;
  ivalues.reserve(inputs.size());
  for (const auto& item : inputs) {
    if (item.is_none() || py::isinstance<py::str>(item)) {
      ivalues.emplace_back(at::zeros({}));
    } else if (py::isinstance<py::bool_>(item)) {
      // Before the int branch: bool is a subclass of int in Python, so a
      // True/False would otherwise arrive as an integer scalar.
      ivalues.emplace_back(item.cast<bool>());
    } else if (py::isinstance<py::int_>(item)) {
      ivalues.emplace_back(item.cast<int64_t>());
    } else if (py::isinstance<py::float_>(item)) {
      ivalues.emplace_back(item.cast<double>());
    } else {
      ivalues.emplace_back(py::cast<at::Tensor>(item));
    }
  }
  return torch::wave::outputsToTensors(self.run(std::move(ivalues)));
}

} // namespace

PYBIND11_MODULE(_torchwave, m) {
  m.doc() = "TorchWave C++ bindings";

  py::class_<torch::wave::WaveConfig>(m, "WaveConfig")
      .def_readwrite("block_size", &torch::wave::WaveConfig::blockSize)
      .def_readwrite("all_standalone", &torch::wave::WaveConfig::allStandalone);

  // Whole-graph alternative to an AOTInductor model container: load() compiles
  // a packaged .pt2 and run() executes it on the GPU.
  py::class_<torch::wave::TorchWaveModel>(m, "TorchWaveModel")
      .def("run", &runOptionalTensors, py::arg("inputs"))
      .def("__call__", &runOptionalTensors, py::arg("inputs"));

  m.def(
      "load",
      [](std::string_view path, std::string_view model_name) {
        return torch::wave::TorchWaveModel::load(path, model_name, {});
      },
      py::arg("path"),
      py::arg("model_name") = "");

  m.def(
      "wave_config",
      []() -> torch::wave::WaveConfig& {
        return torch::wave::WaveConfig::get();
      },
      py::return_value_policy::reference);

  m.def(
      "register_elementwise_op",
      [](const std::string& qualifiedName,
         const std::string& elementwiseFuncName,
         bool isStandalone) {
        torch::wave::Registry::registerElementwiseOp(
            qualifiedName, elementwiseFuncName, isStandalone);
      },
      py::arg("qualified_name"),
      py::arg("elementwise_func_name"),
      py::arg("is_standalone"));
}
