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

#include <unordered_map>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Model.h"
#include "velox/experimental/torchwave/PackedInput.h"
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
std::vector<c10::IValue> toInputValues(const py::sequence& inputs) {
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
  return ivalues;
}

// 'stream' is the cudaStream_t (an int, as torch.cuda.Stream.cuda_stream gives
// it) the caller's PyTorch work runs on, 0 for the legacy default stream. The
// run waits for it, dispatches standalones under it and synchronizes it, so the
// caller must have made it the current stream. The GIL is released for the run,
// so other Python threads proceed while it blocks.
std::vector<at::Tensor> runOptionalTensors(
    torch::wave::TorchWaveModel& self,
    const py::sequence& inputs,
    bool reuse,
    intptr_t stream) {
  auto ivalues = toInputValues(inputs);
  std::vector<at::Tensor> tensors;
  {
    py::gil_scoped_release noGil;
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    torch::wave::TorchStreamGuard streamGuard(reinterpret_cast<void*>(stream));
    auto outputs = reuse ? self.runReuse(std::move(ivalues))
                         : self.run(std::move(ivalues));
    tensors = torch::wave::outputsToTensors(std::move(outputs));
  }
  return tensors;
}

// Encodes the packed inputs among 'inputs' for the model's recipes. Returns
// (input index, value, stats) for every input the encoding replaces: each
// packed input with its encoded host tensor and stats, and each lengths input
// of a windowed slot with its trimmed lengths (a tensor, or an int for a scalar
// length) and empty stats. The caller puts each value at its index before
// run().
py::list encodePackedInputs(
    torch::wave::TorchWaveModel& self,
    const py::sequence& inputs,
    int32_t numThreads,
    bool pinned) {
  auto ivalues = toInputValues(inputs);
  const auto before = ivalues;
  std::vector<torch::wave::PackStats> stats;
  torch::wave::encodePackedInputs(
      self.packRecipes(),
      ivalues,
      {.numThreads = numThreads, .pinned = pinned},
      &stats);
  const auto& recipes = self.packRecipes();
  std::unordered_map<int32_t, size_t> recipeOf;
  for (size_t i = 0; i < recipes.size(); ++i) {
    recipeOf[recipes[i].inputIndex] = i;
  }
  py::list result;
  for (size_t index = 0; index < ivalues.size(); ++index) {
    const auto& value = ivalues[index];
    const auto& original = before[index];
    const bool replaced = value.isTensor() ? !original.isTensor() ||
            value.toTensor().unsafeGetTensorImpl() !=
                original.toTensor().unsafeGetTensorImpl()
                                           : value.isInt() &&
            (!original.isInt() || value.toInt() != original.toInt());
    if (!replaced) {
      continue;
    }
    py::dict statsDict;
    auto it = recipeOf.find(static_cast<int32_t>(index));
    if (it != recipeOf.end()) {
      const auto& recipeStats = stats.at(it->second);
      statsDict["input_bytes"] = recipeStats.inputBytes;
      statsDict["packed_bytes"] = recipeStats.packedBytes;
      statsDict["range_us"] = recipeStats.rangeUs;
      statsDict["layout_us"] = recipeStats.layoutUs;
      statsDict["write_us"] = recipeStats.writeUs;
      statsDict["total_us"] = recipeStats.totalUs;
    }
    if (value.isTensor()) {
      result.append(py::make_tuple(index, value.toTensor(), statsDict));
    } else {
      result.append(py::make_tuple(index, value.toInt(), statsDict));
    }
  }
  return result;
}

} // namespace

PYBIND11_MODULE(_torchwave, m) {
  m.doc() = "TorchWave C++ bindings";

  py::class_<torch::wave::WaveConfig>(m, "WaveConfig")
      .def_readwrite("block_size", &torch::wave::WaveConfig::blockSize)
      .def_readwrite("all_standalone", &torch::wave::WaveConfig::allStandalone)
      // Read-only: set these through set_is_cg, set_trace, set_enable_reuse
      // and set_kernel_cache_dir, which write the instance the executor reads.
      .def_readonly("is_cg", &torch::wave::WaveConfig::isCg)
      .def_readonly("trace", &torch::wave::WaveConfig::trace)
      .def_readonly("enable_reuse", &torch::wave::WaveConfig::enableReuse)
      .def_readonly(
          "kernel_cache_dir", &torch::wave::WaveConfig::kernelCacheDir);

  // Whole-graph alternative to an AOTInductor model container: load() compiles
  // a packaged .pt2 and run() executes it on the GPU.
  py::class_<torch::wave::TorchWaveModel>(m, "TorchWaveModel")
      .def(
          "run",
          [](torch::wave::TorchWaveModel& self,
             const py::sequence& inputs,
             intptr_t stream) {
            return runOptionalTensors(self, inputs, /*reuse=*/false, stream);
          },
          py::arg("inputs"),
          py::arg("stream") = 0)
      .def(
          "run_reuse",
          [](torch::wave::TorchWaveModel& self,
             const py::sequence& inputs,
             intptr_t stream) {
            return runOptionalTensors(self, inputs, /*reuse=*/true, stream);
          },
          py::arg("inputs"),
          py::arg("stream") = 0)
      .def(
          "__call__",
          [](torch::wave::TorchWaveModel& self,
             const py::sequence& inputs,
             intptr_t stream) {
            return runOptionalTensors(self, inputs, /*reuse=*/false, stream);
          },
          py::arg("inputs"),
          py::arg("stream") = 0)
      // The packed inputs load() rewrote (set_pack_inputs), one JSON recipe
      // each.
      .def(
          "pack_recipes",
          [](const torch::wave::TorchWaveModel& self) {
            std::vector<std::string> recipes;
            for (const auto& recipe : self.packRecipes()) {
              recipes.push_back(recipe.toJson());
            }
            return recipes;
          })
      .def(
          "encode_packed_inputs",
          &encodePackedInputs,
          py::arg("inputs"),
          py::arg("num_threads") = 1,
          py::arg("pinned") = true);

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

  // Per-op timing/perf report from the most recent run on this thread,
  // populated when the executor's trace has the kTiming (16) bit set.
  m.def("last_perf_report", []() {
    return torch::wave::waveThreadInfo().perfReport;
  });

  // Set/get the trace bits on the exact WaveConfig instance the executor reads
  // (setWaveTrace/getWaveTrace live in the executor's TU, avoiding a duplicated
  // inline-static WaveConfig::get() instance that a wave_config().trace write
  // from this TU could hit instead).
  m.def("set_trace", &torch::wave::setWaveTrace, py::arg("trace"));
  m.def("set_num_sms", &torch::wave::setNumSms, py::arg("num_sms"));
  m.def("set_pack_inputs", &torch::wave::setPackInputs, py::arg("on"));
  m.def("get_trace", &torch::wave::getWaveTrace);
  m.def("set_is_cg", &torch::wave::setIsCg, py::arg("on"));
  m.def("set_enable_reuse", &torch::wave::setEnableReuse, py::arg("on"));
  m.def(
      "set_kernel_cache_dir", &torch::wave::setKernelCacheDir, py::arg("dir"));

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
