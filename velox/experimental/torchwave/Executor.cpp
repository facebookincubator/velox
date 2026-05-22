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

#include "velox/experimental/torchwave/Executor.h"

#include <ATen/ATen.h>
#include <folly/ScopeGuard.h>
#include <gflags/gflags.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <torch/nativert/kernels/C10Kernel.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/GpuArena.h"

namespace torch::wave {

namespace {

struct GlobalResources {
  std::unique_ptr<facebook::velox::wave::GpuArena> deviceArena;
  std::unique_ptr<facebook::velox::wave::GpuArena> pinnedArena;
  std::unique_ptr<facebook::velox::wave::GpuArena> managedArena;
  std::unique_ptr<StreamPool> streamPool;
  std::unique_ptr<EventPool> eventPool;
};

GlobalResources* globals() {
  static GlobalResources instance;
  return &instance;
}

std::atomic<bool>& initialized() {
  static std::atomic<bool> instance{false};
  return instance;
}

int64_t storageExtentBytes(const at::Tensor& t) {
  if (t.numel() == 0) {
    return 0;
  }
  int64_t maxOffset = 0;
  for (int64_t d = 0; d < t.dim(); d++) {
    if (t.size(d) > 1) {
      maxOffset += (t.size(d) - 1) * t.stride(d);
    }
  }
  return (maxOffset + 1) * t.element_size();
}

} // namespace

void initialize() {
  if (initialized().exchange(true)) {
    return;
  }
  registerBuiltins();
  facebook::velox::wave::Device* device = nullptr;
  try {
    device = facebook::velox::wave::getDevice();
  } catch (...) {
    return;
  }
  if (!device) {
    return;
  }
  facebook::velox::wave::setDevice(device);
  auto* g = globals();
  // Unit allocation size for host-device communication buffers. Tensors are
  // allocated separately from the PyTorch caching allocator.
  g->deviceArena = std::make_unique<facebook::velox::wave::GpuArena>(
      10'000'000,
      facebook::velox::wave::getDeviceAllocator(device),
      40'000'000);
  g->pinnedArena = std::make_unique<facebook::velox::wave::GpuArena>(
      10'000'000, facebook::velox::wave::getHostAllocator(device));
  g->managedArena = std::make_unique<facebook::velox::wave::GpuArena>(
      100'000'000, facebook::velox::wave::getAllocator(device));
  g->streamPool = std::make_unique<StreamPool>(
      []() { return std::make_unique<facebook::velox::wave::Stream>(); });
  g->eventPool = std::make_unique<EventPool>(
      []() { return std::make_unique<facebook::velox::wave::Event>(); });
}

void tensorsToDevice(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream) {
  auto deviceId = facebook::velox::wave::currentDevice()->deviceId;
  auto device =
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(deviceId));

  // Compute total storage bytes needed. For non-contiguous tensors, the storage
  // extent (first to last element) can exceed numel * element_size.
  int64_t totalBytes = 0;
  std::vector<int64_t> sizes(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    sizes[i] = storageExtentBytes(in[i]);
    totalBytes += sizes[i];
  }

  // Allocate contiguous pinned host buffer and copy tensor data into it.
  auto pinned = at::empty(
      {totalBytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  auto* pinnedBase = pinned.data_ptr<uint8_t>();
  int64_t offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    TORCH_CHECK(i < sizes.size());
    memcpy(pinnedBase + offset, in.at(i).data_ptr(), sizes.at(i));
    offset += sizes[i];
  }

  // Allocate device storage and async copy, preserving original strides.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    TORCH_CHECK(i < sizes.size());
    int64_t numElements = sizes.at(i) / in.at(i).element_size();
    auto deviceFlat = at::empty({numElements}, in[i].options().device(device));
    stream.hostToDeviceAsync(
        deviceFlat.data_ptr(), pinnedBase + offset, sizes[i]);
    out[i] = deviceFlat.as_strided(in[i].sizes(), in[i].strides());
    offset += sizes[i];
  }

  // Wait for copies to complete before the pinned buffer goes out of scope.
  stream.wait();
}

void tensorsToHost(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream) {
  int64_t totalBytes = 0;
  std::vector<int64_t> sizes(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    sizes[i] = storageExtentBytes(in[i]);
    totalBytes += sizes[i];
  }

  auto* g = globals();

  // Allocate contiguous pinned host buffer.
  auto pinnedBuffer = g->pinnedArena->allocateBytes(totalBytes);
  auto* pinnedBase = pinnedBuffer->as<uint8_t>();

  // Gather device pointers for a single D2H copy.
  // All input tensors are assumed contiguous in device memory only if they came
  // from tensorsToDevice. In general, copy each tensor individually.
  int64_t offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    TORCH_CHECK(i < sizes.size());
    stream.deviceToHostAsync(
        pinnedBase + offset, in.at(i).data_ptr(), sizes.at(i));
    offset += sizes[i];
  }

  // Build output tensors backed by the pinned buffer.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    TORCH_CHECK(i < sizes.size());
    auto ref = new facebook::velox::wave::WaveBufferPtr(pinnedBuffer);
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        sizes.at(i),
        at::DataPtr(
            pinnedBase + offset,
            ref,
            [](void* ctx) {
              delete static_cast<facebook::velox::wave::WaveBufferPtr*>(ctx);
            },
            c10::Device(c10::kCPU)));
    out[i] = at::empty({0}, in[i].options().device(c10::kCPU))
                 .set_(std::move(storage), 0, in[i].sizes(), in[i].strides());
    offset += sizes[i];
  }
  stream.wait();
}

namespace {

using SavedValues = std::vector<std::pair<nativert::ValueId, c10::IValue>>;

SavedValues replaceCpuOnlyArgs(
    const Launch& launch,
    nativert::ExecutionFrame& frame) {
  SavedValues saved;
  for (size_t i = 0; i < launch.argOnCpu.size(); ++i) {
    auto deviceId = launch.argOnDevice[i]->id();
    auto& deviceIv = frame.getIValue(deviceId);
    if (deviceIv.isTensor()) {
      saved.emplace_back(deviceId, deviceIv);
      frame.setIValue(deviceId, c10::IValue(deviceIv.toTensor().cpu()));
    }
  }
  return saved;
}

void restoreCpuOnlyArgs(SavedValues& saved, nativert::ExecutionFrame& frame) {
  for (auto& [id, iv] : saved) {
    frame.setIValue(id, std::move(iv));
  }
}

} // namespace

void executeNode(
    NodeCP node,
    nativert::OpKernel* kernel,
    nativert::ExecutionFrame& frame) {
  auto trace = WaveConfig::get().trace;
  if (trace) {
    std::cout << "  node " << standaloneToString(node);
    const auto* kernelNode = kernel->node();
    for (size_t argIdx = 0; argIdx < kernelNode->inputs().size(); ++argIdx) {
      auto inputId = kernelNode->inputs()[argIdx].value->id();
      const auto& iv = frame.getIValue(inputId);
      if (iv.isNone()) {
        LOG(WARNING) << node->target() << " arg " << argIdx << " '"
                     << kernelNode->inputs()[argIdx].name << "' value %"
                     << inputId << " is None in frame";
      } else if (iv.isTensorList()) {
        auto tl = iv.toTensorList();
        for (size_t ti = 0; ti < tl.size(); ++ti) {
          if (!at::Tensor(tl[ti]).defined()) {
            LOG(WARNING) << node->target() << " arg " << argIdx
                         << " tensorList[" << ti << "] is undefined, value %"
                         << inputId;
          }
        }
      }
    }
  }
  try {
    kernel->compute(frame);
  } catch (const std::exception& ex) {
    std::stringstream diag;
    diag << "Error in node: " << standaloneToString(node) << ": " << ex.what()
         << "\n  inputs:";
    for (size_t argIdx = 0; argIdx < node->inputs().size(); ++argIdx) {
      auto inputId = node->inputs()[argIdx].value->id();
      const auto& iv = frame.getIValue(inputId);
      diag << "\n    arg " << argIdx << " '" << node->inputs()[argIdx].name
           << "' %" << inputId << ": " << iv.tagKind();
      if (iv.isTensor() && iv.toTensor().defined()) {
        diag << " " << iv.toTensor().sizes() << " " << iv.toTensor().device();
      } else if (iv.isTensorList()) {
        diag << " size=" << iv.toTensorList().size();
      }
    }
    LOG(ERROR) << diag.str();
    throw;
  } catch (...) {
    LOG(ERROR) << "Error in node: " << standaloneToString(node);
    throw;
  }
  if (trace & WaveConfig::kTensors) {
    for (auto* output : node->outputs()) {
      auto outputId = output->id();
      const auto& iv = frame.getIValue(outputId);
      if (iv.isTensor()) {
        std::cout << "    %" << outputId << " " << traceIValue(iv) << std::endl;
      }
    }
  }
}

void runStandalones(
    const std::vector<LaunchData>& standalones,
    ExecutionState& state,
    const folly::F14FastMap<NodeCP, nativert::OpKernel*>& kernelMap,
    const folly::F14FastMap<NodeCP, int32_t>& standaloneIndices,
    std::vector<StandaloneStats>& standaloneStats) {
  using Clock = std::chrono::high_resolution_clock;
  for (const auto& data : standalones) {
    auto* actualNode = data.standalone;

    auto kernelIt = kernelMap.find(actualNode);
    TORCH_CHECK(
        kernelIt != kernelMap.end(),
        "No kernel for node ",
        actualNode->target());

    SavedValues savedDeviceValues;
    if (data.launch && !data.launch->argOnCpu.empty()) {
      savedDeviceValues = replaceCpuOnlyArgs(*data.launch, *state.frame);
    }
    traceFrameValues(
        "input", data.actualInputs, *state.frame, state.traceState);
    auto start = Clock::now();
    executeNode(actualNode, kernelIt->second, *state.frame);
    restoreCpuOnlyArgs(savedDeviceValues, *state.frame);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now() - start)
                  .count();
    if (WaveConfig::get().trace & WaveConfig::kFrame) {
      for (auto outputId : data.actualOutputs) {
        const auto& iv = state.frame->getIValue(outputId);
        std::cout << "    %" << outputId << " = " << traceIValue(iv)
                  << std::endl;
      }
    }
    traceFrameValues(
        "output", data.actualOutputs, *state.frame, state.traceState);

    auto idxIt = standaloneIndices.find(actualNode);
    if (idxIt != standaloneIndices.end()) {
      TORCH_CHECK(
          idxIt->second >= 0 &&
          static_cast<size_t>(idxIt->second) < standaloneStats.size());
      standaloneStats.at(idxIt->second).micros += us;
    }
  }
}

WaveGraphExecutor::WaveGraphExecutor(std::shared_ptr<ModelContext> modelContext)
    : GraphExecutorBase(*modelContext->graph, {}, modelContext->config),
      modelContext_(std::move(modelContext)) {
  waveGraph_ = std::make_unique<WaveGraph>(modelContext_);

  // Create OpKernels only for standalone nodes, after WaveGraph construction
  // so that the kernels' captured Value* pointers reflect the post-mutation
  // graph.
  for (const auto& [node, idx] : waveGraph_->standaloneIndices()) {
    std::string target(node->target());
    std::unique_ptr<nativert::OpKernel> kernel;
    if (nativert::PrimKernelRegistry()->Has(target)) {
      kernel = nativert::PrimKernelRegistry()->Create(target, node);
    } else if (c10::starts_with(target, "torch.ops")) {
      kernel = std::make_unique<nativert::C10Kernel>(node);
    }
    if (kernel) {
      kernelMap_[node] = kernel.get();
      nodeKernels_.push_back(std::move(kernel));
    }
  }
  framePool_ = std::make_unique<Pool<nativert::ExecutionFrame>>(
      [this]() { return makeDeviceFrame(); });
}

std::unique_ptr<nativert::ExecutionFrame> WaveGraphExecutor::makeFrame() {
  return std::make_unique<nativert::ExecutionFrame>(
      graph_, *modelContext_->weights, executorConfig_);
}

std::unique_ptr<nativert::ExecutionFrame> WaveGraphExecutor::makeDeviceFrame() {
  auto frame = makeFrame();

  // Collect all persistent tensor values and their ids.
  auto persistentValues = nativert::ExecutionFrame::getPersistentValues(
      graph_, modelContext_->weights.get());
  std::vector<nativert::ValueId> tensorIds;
  std::vector<at::Tensor> hostTensors;
  for (auto& [id, iv] : persistentValues) {
    if (iv.isTensor()) {
      tensorIds.push_back(id);
      hostTensors.push_back(iv.toTensor());
    }
  }

  if (!hostTensors.empty()) {
    auto stream = globals()->streamPool->get();
    std::vector<at::Tensor> deviceTensors;
    tensorsToDevice(hostTensors, deviceTensors, *stream);
    stream->wait();
    globals()->streamPool->put(std::move(stream));

    for (size_t i = 0; i < tensorIds.size(); ++i) {
      TORCH_CHECK(i < deviceTensors.size());
      frame->setIValue(
          tensorIds.at(i), c10::IValue(std::move(deviceTensors.at(i))));
    }
  }

  return frame;
}

std::unique_ptr<nativert::ExecutionFrame> WaveGraphExecutor::getFrame() {
  return framePool_->get();
}

void WaveGraphExecutor::returnFrame(
    std::unique_ptr<nativert::ExecutionFrame> frame) {
  frame->clearNonPersistentValues();
  framePool_->put(std::move(frame));
}

std::vector<c10::IValue> WaveGraphExecutor::execute(
    nativert::ExecutionFrame& /*frame*/,
    std::vector<c10::IValue> inputs) {
  auto pooledFrame = getFrame();
  fillUserInputs(*pooledFrame, std::move(inputs));
  auto outputs = executeWithPrefilledFrame(*pooledFrame);
  returnFrame(std::move(pooledFrame));
  return outputs;
}

std::vector<c10::IValue> WaveGraphExecutor::executeWithPrefilledFrame(
    nativert::ExecutionFrame& frame) {
  // Move any user input tensors that are not on device to device.
  const auto& userInputs = graph_.signature().userInputs();
  std::vector<nativert::ValueId> tensorIds;
  std::vector<at::Tensor> hostTensors;
  for (const auto& name : userInputs) {
    auto* value = graph_.tryGetValue(name);
    if (!value) {
      continue;
    }
    const auto& ivalue = frame.getIValue(value->id());
    if (ivalue.isTensor() && !ivalue.toTensor().is_cuda()) {
      tensorIds.push_back(value->id());
      hostTensors.push_back(ivalue.toTensor());
    }
  }

  if (!hostTensors.empty()) {
    auto stream = globals()->streamPool->get();
    std::vector<at::Tensor> deviceTensors;
    tensorsToDevice(hostTensors, deviceTensors, *stream);
    stream->wait();
    globals()->streamPool->put(std::move(stream));

    for (size_t i = 0; i < tensorIds.size(); ++i) {
      TORCH_CHECK(i < deviceTensors.size());
      frame.setIValue(
          tensorIds.at(i), c10::IValue(std::move(deviceTensors.at(i))));
    }
  }

  executeWave(frame, *waveGraph_);
  if (WaveConfig::get().trace & WaveConfig::kTensors) {
    for (auto* value : graph_.outputs()) {
      const auto& iv = frame.getIValue(value->id());
      std::cout << "  output %" << value->id() << " " << traceIValue(iv)
                << std::endl;
    }
  }
  auto& savePath = WaveConfig::get().saveReferenceFramePath;
  if (!savePath.empty()) {
    saveReferenceFrame(
        frame, static_cast<int32_t>(graph_.numValues()), savePath);
    LOG(INFO) << "Saved wave reference frame to " << savePath;
    savePath.clear();
  }

  // tryMoveUserOutputs moves the output IValues out of the frame, decoupling
  // them so the frame can be safely returned to the pool. However, it does
  // not move outputs whose graph-level default is Constant(None) — in that
  // case the result slot stays None even though the frame has a computed
  // tensor (e.g. dynamic-shape outputs computed on device). The loop below
  // copies these non-moved outputs from the frame into the results.
  auto results = frame.tryMoveUserOutputs();
  auto outputValues = graph_.outputs();
  for (size_t i = 0; i < results.size() && i < outputValues.size(); ++i) {
    TORCH_CHECK(i < results.size() && i < outputValues.size());
    if (results.at(i).isNone()) {
      const auto& iv = frame.getIValue(outputValues.at(i)->id());
      if (!iv.isNone()) {
        results.at(i) = iv;
      }
    }
  }
  return results;
}

void WaveGraphExecutor::executeWave(
    nativert::ExecutionFrame& frame,
    WaveGraph& waveGraph) {
  // Ensure the thread's CUDA device is set for tensor allocation.
  auto* waveDevice = facebook::velox::wave::currentDevice();
  if (!waveDevice) {
    waveDevice = facebook::velox::wave::getDevice();
  }
  facebook::velox::wave::setDevice(waveDevice);

  auto*& threadWaveGraph = torch::wave::waveGraph();
  auto* prevWaveGraph = threadWaveGraph;
  if (!prevWaveGraph) {
    threadWaveGraph = &waveGraph;
  }
  SCOPE_EXIT {
    threadWaveGraph = prevWaveGraph;
  };

  Timer w("top exec", WaveConfig::get().printTiming);
  auto* g = globals();
  launchDebugInfos_.clear();

  // Get a reusable ExecutionState from the pool.
  auto statePtr = waveGraph.getState();
  auto& state = *statePtr;
  SCOPE_EXIT {
    if (statePtr->stream) {
      statePtr->streamPool->put(std::move(statePtr->stream));
    }
    waveGraph.returnState(std::move(statePtr));
  };

  // Fill per-call fields.
  state.frame = &frame;
  state.valueTypes = &waveGraph.types();
  state.deviceArena = g->deviceArena.get();
  state.pinnedArena = g->pinnedArena.get();
  state.streamPool = g->streamPool.get();
  state.stream = g->streamPool->get();
  state.kernelMap = &kernelMap_;
  state.waveGraph = &waveGraph;
  state.standaloneIndices = &waveGraph.standaloneIndices();
  state.standaloneStats = &waveGraph.standaloneStats();
  state.launchDebugInfos = &launchDebugInfos_;
  state.numRefTensorsChecked = &numRefTensorsChecked_;
  state.numRefNodesChecked = &numRefNodesChecked_;
  state.traceState = parseTraceValues(WaveConfig::get().traceValues);
  state.traceState.traced.clear();
  state.verifiedIds.clear();

  for (const auto& node : waveGraph.nodes()) {
    node->execute(state);
  }
  state.stream->wait();
}

std::vector<std::vector<DebugInfo>> WaveGraphExecutor::getDebugInfo() {
  auto* g = globals();
  auto stream = g->streamPool->get();

  // Queue D2H transfers for all launches.
  for (auto& info : launchDebugInfos_) {
    stream->deviceToHostAsync(
        info.pinnedInfo, info.deviceInfo, info.numBlocks * sizeof(DebugInfo));
  }
  stream->wait();
  g->streamPool->put(std::move(stream));

  // Copy from pinned memory into the result.
  std::vector<std::vector<DebugInfo>> result;
  result.reserve(launchDebugInfos_.size());
  for (auto& info : launchDebugInfos_) {
    result.emplace_back(info.pinnedInfo, info.pinnedInfo + info.numBlocks);
  }
  return result;
}

std::vector<std::pair<std::string, int64_t>>
WaveGraphExecutor::getStandaloneStats() const {
  const auto& indices = waveGraph_->standaloneIndices();
  const auto& stats = waveGraph_->standaloneStats();
  std::vector<std::pair<std::string, int64_t>> result;
  result.reserve(indices.size());
  for (const auto& [node, idx] : indices) {
    Subgraph sg;
    sg.root = node;
    sg.inputs = inputValues(node);
    result.emplace_back(
        "standalone " + sg.toString(),
        idx < static_cast<int32_t>(stats.size()) ? stats[idx].micros : 0);
  }
  return result;
}

} // namespace torch::wave
