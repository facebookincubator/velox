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

#include "velox/experimental/torchwave/tests/ExecutorTestBase.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h> // @manual
#include <glog/logging.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>

#include <caffe2/caffe2/serialize/file_adapter.h>

#include <torch/nativert/graph/GraphPasses.h>
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

// debug_single_ops is now WaveConfig::debugSingleOps

DEFINE_bool(print_timing, false, "Print timing for wave graph execution");
DEFINE_int32(num_repeats, 1, "Number of timed repetitions for each run type");
DEFINE_bool(
    standalone_kernels,
    false,
    "Treat all operations as standalone kernels");
DEFINE_int32(block_dim, 256, "CUDA thread block size");
DEFINE_int32(
    single_block,
    -1,
    "Force single block grid: -1=auto, 0=multi, 1=single");
DEFINE_int32(cg, -1, "Force cooperative grid: -1=auto, 0=off, 1=on");
DEFINE_bool(wave_only, false, "Skip serial CPU and serial GPU execution");
DEFINE_int32(
    single_ops,
    -1,
    "Debug single ops mode: -1=run normal and single-ops, 0=normal only, 1=single-ops only");
DEFINE_bool(
    standalone_timing,
    false,
    "Print standalone execution times after wave run");
DEFINE_bool(
    print_full_mismatch,
    false,
    "Print full tensor contents for mismatched outputs");
DEFINE_int32(
    list,
    0,
    "List WaveGraph before execution: 0=off, 1=kExprs, 2=kGrids");
DEFINE_int32(trace, 0, "Trace bit mask: 1=nodes, 2=launches");
DEFINE_string(
    print_options,
    "",
    "Comma-separated NodePrinter options: D<n>=maxDepth, L<n>=maxLength, "
    "S=shortNames, V=per-line values, NA=no attributes, VN=value names");
DEFINE_string(
    save_reference_frame,
    "",
    "Path to save execution frame after nativert GPU run");
DEFINE_string(
    reference_frame,
    "",
    "Path to load reference frame for verifying wave intermediates");
DEFINE_bool(
    reverify,
    false,
    "Re-verify all previously passed reference values each step to detect corruption");
DEFINE_string(
    kernel_cache_dir,
    "",
    "If non-empty, cache compiled CUDA kernels (cubin) in this directory");
DEFINE_int32(
    max_elementwise_vars,
    7,
    "Max pointer variables in elementwise codegen; beyond this, storage expressions are inlined");
DEFINE_int32(
    out_of_line_expr_size,
    10'000,
    "Character threshold for extracting elementwise subtrees into noinline helpers");
DEFINE_string(
    trace_values,
    "",
    "Comma-separated list of value ids to trace during wave execution");
DEFINE_int32(
    tensor_print_limit,
    100,
    "Max elements printed per tensor when tracing values; 0 for no limit");
DEFINE_bool(
    throw_on_error,
    true,
    "Throw on kernel error; if false, print error after verify instead");
DEFINE_bool(
    no_elementwise_fast_path,
    false,
    "Skip elementwise fast path; always use slow path with complexIdx");
DEFINE_bool(
    continue_after_mismatch,
    false,
    "Log reference mismatches but continue instead of throwing");
DEFINE_bool(
    kernel_debug_output,
    false,
    "Enable device-side debug printfs. Emergency use only");
DEFINE_bool(
    debug_single_ops,
    false,
    "Launch kernel once per block for debugging, waiting after each launch");
DEFINE_bool(
    auto_adjust_cost,
    false,
    "Adjust per-op cost multipliers after each execution based on actual thread block clocks");

namespace torch::wave {

using Clock = std::chrono::high_resolution_clock;

std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  auto cwd = std::filesystem::current_path().string();
  if (cwd.size() >= 6 && cwd.compare(cwd.size() - 6, 6, "fbcode") == 0) {
    return cwd + "/" + baseDir + "/" + filePath;
  }
  return cwd + "/" + filePath;
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    throw std::runtime_error("Cannot open " + path);
  }
  auto size = file.tellg();
  file.seekg(0);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

namespace {

void flattenIValue(const c10::IValue& value, std::vector<c10::IValue>& out) {
  if (value.isObject()) {
    auto obj = value.toObject();
    for (size_t i = 0; i < obj->slots().size(); ++i) {
      out.push_back(obj->getSlot(i));
    }
  } else if (value.isTuple()) {
    for (const auto& elem : value.toTupleRef().elements()) {
      out.push_back(elem);
    }
  } else {
    out.push_back(value);
  }
}

void deepFlattenIValue(
    const c10::IValue& value,
    std::vector<c10::IValue>& out) {
  if (value.isNone()) {
    return;
  }
  if (value.isObject()) {
    auto obj = value.toObject();
    for (size_t i = 0; i < obj->slots().size(); ++i) {
      deepFlattenIValue(obj->getSlot(i), out);
    }
  } else if (value.isTuple()) {
    for (const auto& elem : value.toTupleRef().elements()) {
      deepFlattenIValue(elem, out);
    }
  } else {
    out.push_back(value);
  }
}

void fillFrameInputs(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    std::vector<c10::IValue> inputs) {
  const auto& userInputNames = graph.signature().userInputs();
  // Flatten one level to handle Objects/Tuples wrapping the actual inputs.
  std::vector<c10::IValue> flat;
  for (auto& v : inputs) {
    flattenIValue(v, flat);
  }
  // If still too many, flatten one more level.
  while (flat.size() > userInputNames.size()) {
    std::vector<c10::IValue> next;
    bool changed = false;
    for (auto& v : flat) {
      auto before = next.size();
      flattenIValue(v, next);
      if (next.size() - before != 1) {
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    flat = std::move(next);
  }
  size_t flatIdx = 0;
  for (const auto& name : userInputNames) {
    auto* value = graph.tryGetValue(name);
    if (value && flatIdx < flat.size()) {
      frame.setIValue(value->id(), std::move(flat[flatIdx++]));
    }
  }
}

} // namespace

std::vector<c10::IValue> loadReferenceValues(const std::string& path) {
  auto data = readFile(path);
  auto ivalue = pickleLoadWithTypes(data);
  std::vector<c10::IValue> values;
  if (ivalue.isList()) {
    for (const auto& item : ivalue.toListRef()) {
      flattenIValue(item, values);
    }
  } else {
    flattenIValue(ivalue, values);
  }
  return values;
}

std::pair<std::vector<c10::IValue>, int64_t> inputsToDevice(
    std::vector<c10::IValue>& inputs) {
  std::vector<at::Tensor> hostTensors;
  std::vector<size_t> tensorIndices;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].isTensor()) {
      hostTensors.push_back(inputs[i].toTensor());
      tensorIndices.push_back(i);
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> deviceTensors;

  auto start = Clock::now();
  tensorsToDevice(hostTensors, deviceTensors, stream);
  stream.wait();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start)
                .count();

  std::vector<c10::IValue> result(inputs.size());
  for (size_t i = 0, ti = 0; i < inputs.size(); ++i) {
    if (ti < tensorIndices.size() && tensorIndices[ti] == i) {
      result[i] = c10::IValue(deviceTensors[ti]);
      ++ti;
    } else {
      result[i] = inputs[i];
    }
  }

  return {std::move(result), us};
}

std::vector<c10::IValue> outputsToHost(
    const std::vector<c10::IValue>& outputs,
    const std::string& /*label*/) {
  std::vector<at::Tensor> deviceTensors;
  std::vector<size_t> tensorIndices;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      deviceTensors.push_back(outputs[i].toTensor());
      tensorIndices.push_back(i);
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> hostTensors;
  tensorsToHost(deviceTensors, hostTensors, stream);
  stream.wait();

  std::vector<c10::IValue> result(outputs.size());
  for (size_t i = 0, ti = 0; i < outputs.size(); ++i) {
    if (ti < tensorIndices.size() && tensorIndices[ti] == i) {
      result[i] = hostTensors[ti];
      ++ti;
    } else {
      result[i] = outputs[i];
    }
  }

  return result;
}

std::unordered_map<int32_t, std::string> snapshotFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues) {
  std::unordered_map<int32_t, std::string> result;
  for (int32_t id = 0; id < numValues; ++id) {
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    if (iv.isTensor()) {
      auto sizes = iv.toTensor().sizes();
      std::string shape;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) {
          shape += ",";
        }
        shape += std::to_string(sizes[i]);
      }
      result[id] = "[" + shape + "]";
    } else {
      result[id] = "scalar";
    }
  }
  return result;
}

void compareFrameSnapshots(
    const std::unordered_map<int32_t, std::string>& serial,
    const std::unordered_map<int32_t, std::string>& wave,
    int32_t numValues) {
  int32_t mismatches = 0;
  for (int32_t id = 0; id < numValues; ++id) {
    auto serialIt = serial.find(id);
    auto waveIt = wave.find(id);
    bool serialSet = serialIt != serial.end();
    bool waveSet = waveIt != wave.end();
    if (serialSet != waveSet) {
      LOG(WARNING) << "Frame mismatch at id " << id
                   << ": serial=" << (serialSet ? serialIt->second : "none")
                   << " wave=" << (waveSet ? waveIt->second : "none");
      ++mismatches;
    } else if (serialSet && serialIt->second != waveIt->second) {
      LOG(WARNING) << "Frame shape mismatch at id " << id
                   << ": serial=" << serialIt->second
                   << " wave=" << waveIt->second;
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    LOG(WARNING) << "Total frame mismatches: " << mismatches << " (serial has "
                 << serial.size() << " values, wave has " << wave.size()
                 << " values)";
  }
}

void standaloneReport(WaveGraphExecutor& executor) {
  auto stats = executor.getStandaloneStats();
  if (stats.empty()) {
    return;
  }
  std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  LOG(INFO) << "Standalone timing:";
  for (const auto& [name, micros] : stats) {
    LOG(INFO) << "  " << micros << " us  " << name;
  }
}

// --- ModelFixture ---

std::unique_ptr<ModelFixture> ModelFixture::load(const std::string& pt2Path) {
  auto fixture = std::make_unique<ModelFixture>();
  fixture->pt2Path = pt2Path;

  fixture->reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));

  auto modelNames = getModelNames(*fixture->reader);
  EXPECT_FALSE(modelNames.empty()) << "No models found in " << pt2Path;
  if (modelNames.empty()) {
    return nullptr;
  }

  fixture->model = loadPt2Model(fixture->reader, modelNames[0]);
  prepareGraph(fixture->model.graph.get());
  const auto& graph = *fixture->model.graph;

  fixture->weights = std::make_shared<nativert::Weights>(
      &graph,
      fixture->reader,
      fixture->model.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      fixture->model.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  return fixture;
}

void ModelFixture::prepareGraph(nativert::Graph* graph) {
  nativert::selectScalarOverload(graph);
}

std::vector<std::unique_ptr<nativert::OpKernel>> ModelFixture::makeKernels()
    const {
  nativert::KernelFactory factory;
  auto execKernels =
      factory.initializeNodeKernels(*model.graph, weights, config, reader);
  return std::move(execKernels.nodeKernels);
}

std::unique_ptr<ModelContext> ModelFixture::makeModelContext() {
  TORCH_CHECK(
      model.graph != nullptr,
      "ModelFixture::makeModelContext: graph already moved");
  auto ctx = std::make_unique<ModelContext>();
  ctx->graph = std::move(model.graph);
  ctx->weights = weights;
  ctx->config = config;
  return ctx;
}

// --- ExecutorTestBase ---

void ExecutorTestBase::SetUpTestSuite() {
  // Diagnostic: print device properties and test basic CUDA ops.
  LOG(INFO) << "CUDA device count: " << at::cuda::device_count();
  auto* props = at::cuda::getDeviceProperties(0);
  LOG(INFO) << "Device 0: " << props->name << " sm_" << props->major
            << props->minor;

  // Test empty (no kernel launch) vs zeros (requires fill kernel).
  LOG(INFO) << "Testing at::empty on CUDA...";
  try {
    auto e = at::empty(
        {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    LOG(INFO) << "at::empty OK, data_ptr=" << e.data_ptr();
  } catch (const std::exception& e) {
    LOG(ERROR) << "at::empty FAILED: " << e.what();
  }

  LOG(INFO) << "Testing at::ones on CUDA (fill kernel)...";
  try {
    auto o = at::ones(
        {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    LOG(INFO) << "at::ones OK: " << o.cpu().sum().item<float>();
  } catch (const std::exception& e) {
    LOG(ERROR) << "at::ones FAILED: " << e.what();
  }

  WaveConfig::get().printTiming = FLAGS_print_timing;
  WaveConfig::get().allStandalone = FLAGS_standalone_kernels;
  WaveConfig::get().blockSize = FLAGS_block_dim;
  WaveConfig::get().trace = FLAGS_trace;
  if (FLAGS_single_block >= 0) {
    WaveConfig::get().useSingleBlock = FLAGS_single_block != 0;
  }
  if (FLAGS_cg >= 0) {
    WaveConfig::get().isCg = FLAGS_cg != 0;
  }
  WaveConfig::get().kernelCacheDir = FLAGS_kernel_cache_dir;
  WaveConfig::get().maxElementwiseVars = FLAGS_max_elementwise_vars;
  WaveConfig::get().outOfLineExprSize = FLAGS_out_of_line_expr_size;
  WaveConfig::get().traceValues = FLAGS_trace_values;
  WaveConfig::get().tensorPrintElementLimit = FLAGS_tensor_print_limit;
  WaveConfig::get().reverify = FLAGS_reverify;
  WaveConfig::get().throwOnError = FLAGS_throw_on_error;
  WaveConfig::get().noElementwiseFastPath = FLAGS_no_elementwise_fast_path;
  WaveConfig::get().continueAfterMismatch = FLAGS_continue_after_mismatch;
  WaveConfig::get().kernelDebugOutput = FLAGS_kernel_debug_output;
  WaveConfig::get().debugSingleOps = FLAGS_debug_single_ops;
  WaveConfig::get().autoAdjustCost = FLAGS_auto_adjust_cost;
  if (!FLAGS_print_options.empty()) {
    NodePrinter::setDefaults(
        NodePrinter::parsePrintOptions(FLAGS_print_options));
  }

  initialize();
}

std::vector<c10::IValue> ExecutorTestBase::executeSerialWithTrace(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels) {
  auto traceState = parseTraceValues(FLAGS_trace_values);

  // Trace user inputs already in the frame.
  for (const auto* value : graph.userInputs()) {
    if (value) {
      std::vector<nativert::ValueId> ids = {value->id()};
      traceFrameValues("serial input", ids, frame, traceState);
    }
  }

  for (size_t nodeIdx = 1; nodeIdx + 1 < nodeKernels.size(); ++nodeIdx) {
    auto* node = nodeKernels[nodeIdx]->node();
    nodeKernels[nodeIdx]->compute(frame);
    for (const auto& input : node->inputs()) {
      std::vector<nativert::ValueId> ids = {input.value->id()};
      traceFrameValues("serial in", ids, frame, traceState);
    }
    for (auto* output : node->outputs()) {
      if (output) {
        std::vector<nativert::ValueId> ids = {output->id()};
        traceFrameValues("serial out", ids, frame, traceState);
      }
    }
  }
  return frame.tryMoveUserOutputs();
}

RunTiming ExecutorTestBase::runSerial(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected) {
  const auto& graph = *fixture.model.graph;

  auto serialConfig = fixture.config;
  if (!FLAGS_save_reference_frame.empty()) {
    serialConfig.tryFreeUnmanagedValuesAfterUse = false;
  }
  auto kernels = fixture.makeKernels();
  nativert::SerialGraphExecutor executor(
      graph, std::move(kernels), serialConfig);

  auto frame = std::make_unique<nativert::ExecutionFrame>(
      graph, *fixture.weights, serialConfig);

  auto inputs = loadSampleInputs(fixture);
  LOG(INFO) << "Serial: loaded " << inputs.size() << " inputs, graph expects "
            << graph.signature().userInputs().size();
  for (size_t i = 0; i < std::min<size_t>(inputs.size(), 5); ++i) {
    LOG(INFO) << "  input[" << i << "]: tag=" << inputs[i].tagKind()
              << " isTensor=" << inputs[i].isTensor()
              << " isObject=" << inputs[i].isObject()
              << " isTuple=" << inputs[i].isTuple()
              << " isList=" << inputs[i].isList();
  }

  // Snapshot frame after construction (weights filled, before user inputs).
  serialFrameSnapshot_ =
      snapshotFrame(*frame, static_cast<int32_t>(graph.numValues()));
  LOG(INFO) << "Serial frame before execute: " << serialFrameSnapshot_.size()
            << " non-none slots of " << graph.numValues();

  auto traceState = parseTraceValues(FLAGS_trace_values);
  bool traceSerial = !traceState.empty();
  if (traceSerial) {
    serialConfig.tryFreeUnmanagedValuesAfterUse = false;
  }

  fillFrameInputs(graph, *frame, std::move(inputs));

  auto start = Clock::now();
  std::vector<c10::IValue> outputs;
  if (traceSerial) {
    auto traceKernels = fixture.makeKernels();
    nativert::SerialGraphExecutor tempExecutor(
        graph, std::move(traceKernels), serialConfig);
    outputs =
        executeSerialWithTrace(graph, *frame, tempExecutor.stealKernels());
  } else {
    outputs = executor.executeWithPrefilledFrame(*frame);
  }
  auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       Clock::now() - start)
                       .count();

  // Re-snapshot after execute to capture user inputs filled by the executor.
  serialFrameSnapshot_ =
      snapshotFrame(*frame, static_cast<int32_t>(graph.numValues()));
  LOG(INFO) << "Serial frame after execute: " << serialFrameSnapshot_.size()
            << " non-none slots of " << graph.numValues();

  if (!FLAGS_save_reference_frame.empty()) {
    saveReferenceFrame(*frame, graph, FLAGS_save_reference_frame);
    LOG(INFO) << "Saved reference frame to " << FLAGS_save_reference_frame;
  }

  verifyOutputs(outputs, expected, "serial-pass1");

  // Run a second time with the same frame to test frame reuse.
  auto inputs2 = loadSampleInputs(fixture);
  fillFrameInputs(graph, *frame, std::move(inputs2));
  auto outputs2 = executor.executeWithPrefilledFrame(*frame);
  verifyOutputs(outputs2, expected, "serial-pass2");

  return {0, executeUs, {}};
}

RunTiming ExecutorTestBase::runSerialOnDevice(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected) {
  auto& graph = *fixture.model.graph;

  auto kernels = fixture.makeKernels();
  nativert::SerialGraphExecutor executor(
      graph, std::move(kernels), fixture.config);

  auto frame = std::make_unique<nativert::ExecutionFrame>(
      graph, *fixture.weights, fixture.config);

  auto inputs = loadSampleInputs(fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  // Set CUDA device so that at::kCUDA allocations target the right GPU.
  auto* waveDevice = facebook::velox::wave::currentDevice();
  if (!waveDevice) {
    waveDevice = facebook::velox::wave::getDevice();
  }
  at::cuda::set_device(static_cast<c10::DeviceIndex>(waveDevice->deviceId));

  fillFrameInputs(graph, *frame, std::move(deviceInputs));

  auto start = Clock::now();
  auto outputs = executor.executeWithPrefilledFrame(*frame);
  auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       Clock::now() - start)
                       .count();

  if (!FLAGS_save_reference_frame.empty()) {
    saveReferenceFrame(
        *frame,
        static_cast<int32_t>(graph.numValues()),
        FLAGS_save_reference_frame);
    LOG(INFO) << "Saved GPU reference frame (" << graph.numValues()
              << " slots) to " << FLAGS_save_reference_frame;
  }

  auto hostOutputs = outputsToHost(outputs, "serial-gpu");
  verifyOutputs(hostOutputs, expected, "serial-gpu");
  return {dataMovUs, executeUs, {}};
}

void ExecutorTestBase::fillWaveFrame(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    const std::vector<c10::IValue>& deviceInputs) {
  const auto& userInputNames = graph.signature().userInputs();
  // Flatten to match user input count.
  std::vector<c10::IValue> flat;
  for (const auto& v : deviceInputs) {
    flattenIValue(v, flat);
  }
  while (flat.size() > userInputNames.size()) {
    std::vector<c10::IValue> next;
    bool changed = false;
    for (auto& v : flat) {
      auto before = next.size();
      flattenIValue(v, next);
      if (next.size() - before != 1) {
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    flat = std::move(next);
  }
  size_t flatIdx = 0;
  for (const auto& name : userInputNames) {
    auto* value = graph.tryGetValue(name);
    EXPECT_NE(value, nullptr) << "No value for input " << name;
    if (value && flatIdx < flat.size()) {
      const auto& iv = flat[flatIdx++];
      if (!iv.isNone()) {
        if (iv.isTensor()) {
          frame.setIValue(value->id(), c10::IValue(iv.toTensor().clone()));
        } else {
          frame.setIValue(value->id(), iv);
        }
      }
    }
  }
}

RunTiming ExecutorTestBase::runWave(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected,
    const std::function<void(nativert::ExecutionFrame&)>& alterInputs) {
  std::unordered_map<int32_t, c10::IValue> refFrame;
  if (!FLAGS_reference_frame.empty()) {
    refFrame = loadReferenceFrame(FLAGS_reference_frame);
    WaveConfig::get().referenceFrame = &refFrame;
    LOG(INFO) << "Loaded reference frame with " << refFrame.size()
              << " values from " << FLAGS_reference_frame;
  }

  WaveGraphExecutor waveExec(fixture.makeModelContext());
  auto& graph = waveExec.graph();

  auto pooledFrame = waveExec.getFrame();
  EXPECT_NE(pooledFrame, nullptr);

  auto inputs = loadSampleInputs(fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  fillWaveFrame(graph, *pooledFrame, deviceInputs);

  if (alterInputs) {
    alterInputs(*pooledFrame);
  }

  if (!serialFrameSnapshot_.empty()) {
    auto waveSnapshot =
        snapshotFrame(*pooledFrame, static_cast<int32_t>(graph.numValues()));
    LOG(INFO) << "Wave frame after fillWaveFrame: " << waveSnapshot.size()
              << " non-none slots of " << graph.numValues();
    compareFrameSnapshots(
        serialFrameSnapshot_,
        waveSnapshot,
        static_cast<int32_t>(graph.numValues()));
  }

  auto waveStart = Clock::now();
  auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
  auto waveUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - waveStart)
                    .count();

  waveExec.returnFrame(std::move(pooledFrame));

  lastRefTensorsChecked_ = waveExec.numRefTensorsChecked();
  lastRefNodesChecked_ = waveExec.numRefNodesChecked();

  WaveConfig::get().referenceFrame = nullptr;

  auto hostOutputs = outputsToHost(waveOutputs, "wave");
  verifyOutputs(hostOutputs, expected, "wave");
  return {dataMovUs, waveUs, waveThreadInfo().debugInfo};
}

void ExecutorTestBase::runTest(
    const std::string& pt2File,
    const std::string& resultsFile,
    const std::string& label) {
  auto baseDir = dataDir();
  auto pt2Path =
      pt2File[0] == '/' ? pt2File : getDataFilePath(baseDir, pt2File);
  auto resultsPath = resultsFile[0] == '/'
      ? resultsFile
      : getDataFilePath(baseDir, resultsFile);
  auto inputsPath = pt2Path.substr(0, pt2Path.size() - 4) + "_inputs.pt";
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);

  // Check if an external inputs file exists.
  if (std::ifstream(inputsPath).good()) {
    LOG(INFO) << "Loading external inputs from " << inputsPath;
    fixture->externalInputsPath = inputsPath;
  }

  std::vector<c10::IValue> expected;
  try {
    expected = loadReferenceValues(resultsPath);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to load reference values from " << resultsPath
                 << ": " << e.what() << " — skipping result verification";
  }

  runTestWithFixture(
      std::move(fixture), expected, label.empty() ? pt2File : label);
}

void ExecutorTestBase::runTestWithFixture(
    std::unique_ptr<ModelFixture> fixture,
    const std::vector<c10::IValue>& expected,
    const std::string& label) {
  displayName_ = label;
  auto displayName = label;

  const int repeats = FLAGS_num_repeats;

  if (!FLAGS_wave_only) {
    // Serial CPU — runs before device placement so graph has CPU attributes.
    auto serialCold = runSerial(*fixture, expected);
    int64_t serialMin = INT64_MAX, serialMax = 0, serialSum = 0;
    std::vector<int64_t> serialTimes(repeats);
    for (int i = 0; i < repeats; ++i) {
      auto t = runSerial(*fixture, expected);
      serialTimes[i] = t.executeUs;
      serialMin = std::min(serialMin, t.executeUs);
      serialMax = std::max(serialMax, t.executeUs);
      serialSum += t.executeUs;
    }
    std::sort(serialTimes.begin(), serialTimes.end());
    auto serialP90 = serialTimes[repeats * 90 / 100];
    LOG(INFO) << displayName << " serial CPU (" << repeats
              << " repeats): min=" << serialMin
              << " avg=" << serialSum / repeats << " p90=" << serialP90
              << " max=" << serialMax << " us (cold=" << serialCold.executeUs
              << " us)";

    // Set graph device to CUDA once — used by serial GPU below and wave.
    setGraphDevice(fixture->model.graph.get(), true);

    // Serial GPU.
    auto deviceCold = runSerialOnDevice(*fixture, expected);
    int64_t deviceMin = INT64_MAX, deviceMax = 0, deviceSum = 0;
    std::vector<int64_t> deviceTimes(repeats);
    for (int i = 0; i < repeats; ++i) {
      auto t = runSerialOnDevice(*fixture, expected);
      deviceTimes[i] = t.executeUs;
      deviceMin = std::min(deviceMin, t.executeUs);
      deviceMax = std::max(deviceMax, t.executeUs);
      deviceSum += t.executeUs;
    }
    std::sort(deviceTimes.begin(), deviceTimes.end());
    auto deviceP90 = deviceTimes[repeats * 90 / 100];
    LOG(INFO) << displayName
              << " serial GPU H2D: cold=" << deviceCold.dataTransferUs << " us";
    LOG(INFO) << displayName << " serial GPU (" << repeats
              << " repeats): min=" << deviceMin
              << " avg=" << deviceSum / repeats << " p90=" << deviceP90
              << " max=" << deviceMax << " us (cold=" << deviceCold.executeUs
              << " us)";
  } else {
    // wave_only: apply placement here since the serial GPU path was skipped.
    setGraphDevice(fixture->model.graph.get(), true);
  }

  // Wave.
  WaveGraphExecutor waveExec(fixture->makeModelContext());
  auto& graph = waveExec.graph();

  if (FLAGS_list > 0) {
    auto mode = static_cast<Listing>(FLAGS_list - 1);
    std::cout << waveExec.waveGraph()->toString(mode) << std::endl;
  }

  auto inputs = loadSampleInputs(*fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  if (FLAGS_single_ops != 1) {
    auto pooledFrame = waveExec.getFrame();
    ASSERT_NE(pooledFrame, nullptr);

    // Initial run: fill frame, execute, verify.
    fillWaveFrame(graph, *pooledFrame, deviceInputs);

    auto waveStart = Clock::now();
    auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
    auto waveColdUs = std::chrono::duration_cast<std::chrono::microseconds>(
                          Clock::now() - waveStart)
                          .count();

    auto hostOutputs = outputsToHost(waveOutputs, "wave");
    verifyOutputs(hostOutputs, expected, "wave");
    if (!FLAGS_throw_on_error) {
      auto& errors = waveThreadInfo().errors;
      if (!errors.empty()) {
        LOG(ERROR) << "Wave kernel errors:\n" << errors;
      }
    }

    // Repeated runs: refill inputs and re-execute only.
    int64_t waveMin = INT64_MAX, waveMax = 0, waveSum = 0;
    std::vector<int64_t> waveTimes(repeats);
    std::vector<std::vector<DebugInfo>> lastDebugInfo;
    for (int i = 0; i < repeats; ++i) {
      fillWaveFrame(graph, *pooledFrame, deviceInputs);

      auto start = Clock::now();
      waveExec.executeWithPrefilledFrame(*pooledFrame);
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - start)
                    .count();
      waveTimes[i] = us;
      waveMin = std::min(waveMin, us);
      waveMax = std::max(waveMax, us);
      waveSum += us;
      if (i == repeats - 1) {
        lastDebugInfo = waveThreadInfo().debugInfo;
      }
      if (WaveConfig::get().trace & WaveConfig::kTiming) {
        const auto& report = waveThreadInfo().perfReport;
        if (!report.empty()) {
          std::cout << "--- repeat " << i << " ---\n" << report << std::endl;
        }
      }
    }

    waveExec.returnFrame(std::move(pooledFrame));

    std::sort(waveTimes.begin(), waveTimes.end());
    auto waveP90 = waveTimes[repeats * 90 / 100];
    LOG(INFO) << displayName << " wave H2D: cold=" << dataMovUs << " us";
    LOG(INFO) << displayName << " wave (" << repeats
              << " repeats): min=" << waveMin << " avg=" << waveSum / repeats
              << " p90=" << waveP90 << " max=" << waveMax
              << " us (cold=" << waveColdUs << " us)";

    for (size_t li = 0; li < lastDebugInfo.size(); ++li) {
      const auto& blocks = lastDebugInfo[li];
      struct OpStats {
        int64_t count{0};
        int64_t minClocks{std::numeric_limits<int64_t>::max()};
        int64_t maxClocks{0};
        int64_t sumClocks{0};
      };
      std::map<int32_t, OpStats> opMap;
      for (const auto& b : blocks) {
        auto& s = opMap[b.op];
        s.count++;
        s.minClocks = std::min(s.minClocks, b.clocks);
        s.maxClocks = std::max(s.maxClocks, b.clocks);
        s.sumClocks += b.clocks;
      }
      std::vector<std::pair<int32_t, OpStats>> sorted(
          opMap.begin(), opMap.end());
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return (a.second.sumClocks / a.second.count) <
            (b.second.sumClocks / b.second.count);
      });
      std::stringstream ss;
      ss << displayName << " launch " << li << ": ";
      bool first = true;
      for (const auto& [op, s] : sorted) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << "op " << op << " count: " << s.count << " " << s.minClocks << "/"
           << (s.sumClocks / s.count) << "/" << s.maxClocks;
      }
      LOG(INFO) << ss.str();
    }

    if (FLAGS_standalone_timing) {
      standaloneReport(waveExec);
    }
  }

  if (FLAGS_single_ops != 0) {
    WaveConfig::get().debugSingleOps = true;
    auto debugLabel = displayName + " debug_single_ops";
    auto debugFrame = waveExec.getFrame();
    ASSERT_NE(debugFrame, nullptr);
    for (int run = 0; run < 2; ++run) {
      fillWaveFrame(graph, *debugFrame, deviceInputs);
      auto outputs = waveExec.executeWithPrefilledFrame(*debugFrame);
      if (run == 1) {
        auto hostOutputs = outputsToHost(outputs, debugLabel);
        verifyOutputs(hostOutputs, expected, debugLabel);
        if (!FLAGS_throw_on_error) {
          auto& errors = waveThreadInfo().errors;
          if (!errors.empty()) {
            LOG(ERROR) << "Wave kernel errors (debug_single_ops):\n" << errors;
          }
        }
      }
    }
    waveExec.returnFrame(std::move(debugFrame));
    WaveConfig::get().debugSingleOps = false;
  }
}

std::vector<c10::IValue> ExecutorTestBase::loadSampleInputs(
    ModelFixture& fixture) {
  // If an external inputs file is available, load from it.
  if (!fixture.externalInputsPath.empty()) {
    auto data = readFile(fixture.externalInputsPath);
    auto ivalue = pickleLoadWithTypes(data);
    std::vector<c10::IValue> inputs;
    if (ivalue.isList()) {
      for (const auto& item : ivalue.toListRef()) {
        flattenIValue(item, inputs);
      }
    } else {
      flattenIValue(ivalue, inputs);
    }
    return inputs;
  }

  auto modelName = getModelNames(*fixture.reader)[0];
  std::string sampleInputsPath = fmt::format(
      torch::_export::archive_spec::SAMPLE_INPUTS_FILENAME_FORMAT, modelName);

  std::vector<c10::IValue> inputs;
  if (fixture.reader->hasRecord(sampleInputsPath)) {
    auto size = fixture.reader->getRecordSize(sampleInputsPath);
    std::vector<char> buffers(size);
    fixture.reader->getRecord(sampleInputsPath, buffers.data(), size);
    auto value = pickleLoadWithTypes(buffers);
    if (value.isTuple() && value.toTupleRef().elements().size() == 2) {
      const auto& argsVal = value.toTupleRef().elements().at(0);
      if (argsVal.isTuple()) {
        for (const auto& arg : argsVal.toTupleRef().elements()) {
          flattenIValue(arg, inputs);
        }
      }
      const auto& kwargsVal = value.toTupleRef().elements().at(1);
      if (kwargsVal.isTuple()) {
        for (const auto& kwarg : kwargsVal.toTupleRef().elements()) {
          flattenIValue(kwarg, inputs);
        }
      } else if (kwargsVal.isGenericDict()) {
        auto dict = kwargsVal.toGenericDict();
        for (const auto& entry : dict) {
          flattenIValue(entry.value(), inputs);
        }
      }
    }
  }
  EXPECT_FALSE(inputs.empty()) << "No sample inputs found in package";
  return inputs;
}

void ExecutorTestBase::verifyOutputs(
    const std::vector<c10::IValue>& outputs,
    const std::vector<c10::IValue>& expected,
    const std::string& label) {
  if (expected.empty()) {
    return;
  }
  auto fullLabel = displayName_.empty() ? label : displayName_ + " " + label;
  ASSERT_EQ(outputs.size(), expected.size())
      << fullLabel << ": output count mismatch";
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& actual = outputs[i];
    const auto& exp = expected[i];
    if (exp.isTensor()) {
      ASSERT_TRUE(actual.isTensor())
          << fullLabel << " output " << i << " expected tensor, got non-tensor";
      auto match = tensorsMatch(actual.toTensor(), exp.toTensor());
      auto limit = WaveConfig::get().tensorPrintElementLimit;
      EXPECT_TRUE(match) << fullLabel << " output " << i
                         << " differs from expected"
                         << "\n  "
                         << firstDifference(actual.toTensor(), exp.toTensor())
                         << "\n  expected: "
                         << tensorDebugString(exp.toTensor(), limit)
                         << "\n  actual:   "
                         << tensorDebugString(actual.toTensor(), limit);
      if (!match && FLAGS_print_full_mismatch) {
        LOG(INFO) << fullLabel << " output " << i << " expected:\n"
                  << tensorToString(exp.toTensor());
        LOG(INFO) << fullLabel << " output " << i << " actual:\n"
                  << tensorToString(actual.toTensor());
      }
    } else if (exp.isDouble()) {
      ASSERT_TRUE(actual.isDouble())
          << fullLabel << " output " << i << " expected double, got "
          << actual.tagKind();
      EXPECT_NEAR(actual.toDouble(), exp.toDouble(), 1e-5)
          << fullLabel << " output " << i << " scalar mismatch";
    } else if (exp.isInt()) {
      ASSERT_TRUE(actual.isInt()) << fullLabel << " output " << i
                                  << " expected int, got " << actual.tagKind();
      EXPECT_EQ(actual.toInt(), exp.toInt())
          << fullLabel << " output " << i << " scalar mismatch";
    } else if (exp.isBool()) {
      ASSERT_TRUE(actual.isBool())
          << fullLabel << " output " << i << " expected bool, got "
          << actual.tagKind();
      EXPECT_EQ(actual.toBool(), exp.toBool())
          << fullLabel << " output " << i << " scalar mismatch";
    } else {
      ADD_FAILURE() << fullLabel << " output " << i
                    << " unsupported expected type: " << exp.tagKind();
    }
  }
}

} // namespace torch::wave
