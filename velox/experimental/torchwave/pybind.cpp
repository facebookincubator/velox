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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "velox/experimental/torchwave/DelegateExecutor.h"
#include "velox/experimental/torchwave/KernelHandlerRegistration.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace py = pybind11;

PYBIND11_MODULE(_torchwave, m) {
  m.doc() = "TorchWave C++ bindings";

  // Register torchwave as a nativert delegate backend so that
  // PyModelRunner can dispatch executorch_call_delegate nodes to
  // the torchwave executor.
  torch::wave::registerTorchwaveHandler();

  py::class_<torch::wave::DelegateExecutor>(m, "DelegateExecutor")
      .def(py::init<const std::string&>(), py::arg("pt2_path"))
      .def("run", &torch::wave::DelegateExecutor::run, py::arg("inputs"))
      .def("__repr__", &torch::wave::DelegateExecutor::toString);

  py::class_<torch::wave::WaveConfig>(m, "WaveConfig")
      .def_readwrite("block_size", &torch::wave::WaveConfig::blockSize)
      .def_readwrite(
          "single_block_path_block_size",
          &torch::wave::WaveConfig::singleBlockPathBlockSize)
      .def_readwrite("all_standalone", &torch::wave::WaveConfig::allStandalone)
      .def_readwrite(
          "num_standalone_threads",
          &torch::wave::WaveConfig::numStandaloneThreads);

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
