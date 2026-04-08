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
}
