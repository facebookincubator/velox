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

thread_local WaveThreadInfo threadInfo;

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
  if (trace & WaveConfig::kLaunches) {
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

WaveGraphExecutor::WaveGraphExecutor(std::unique_ptr<ModelContext> modelContext)
    : GraphExecutorBase(*modelContext->graph, {}, modelContext->config),
      modelContext_(std::move(modelContext)) {
  waveGraph_ = std::make_unique<WaveGraph>(modelContext_.get());

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
  ++frameGeneration_;
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

  // Get a reusable ExecutionState from the pool.
  auto statePtr = waveGraph.getState();
  auto& state = *statePtr;
  state.launchDebugInfos.clear();
  SCOPE_EXIT {
    if (statePtr->stream) {
      statePtr->streamPool->put(std::move(statePtr->stream));
    }
    waveGraph.returnState(std::move(statePtr));
  };

  // Invalidate launch caches when the frame was returned and re-obtained
  // (clearNonPersistentValues clears intermediates, making cached value
  // IDs stale).
  if (state.lastFrameGeneration != frameGeneration_) {
    for (auto& steps : state.stepVectors) {
      for (auto& sv : steps) {
        sv.hasLaunchCache = false;
        for (auto& data : sv.kernels) {
          data.tensorsInFrame.clear();
          data.tensorOffsets.clear();
          data.scalarsInFrame.clear();
          data.scalarOffsets.clear();
        }
      }
    }
    state.lastFrameGeneration = frameGeneration_;
  }
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
  for (auto& s : *state.standaloneStats) {
    s.micros = 0;
  }
  state.numRefTensorsChecked = &numRefTensorsChecked_;
  state.numRefNodesChecked = &numRefNodesChecked_;
  state.traceState = parseTraceValues(WaveConfig::get().traceValues);
  state.traceState.traced.clear();
  state.verifiedIds.clear();

  auto wallStart = std::chrono::high_resolution_clock::now();
  for (const auto& node : waveGraph.nodes()) {
    node->execute(state);
  }
  state.stream->wait();
  auto wallUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - wallStart)
                    .count();
  if (WaveConfig::get().keepStatsOnThread) {
    collectDebugInfo(state);
    if (WaveConfig::get().autoAdjustCost) {
      adjustCosts(state);
    }
    threadInfo.errors = errorString();
    if (WaveConfig::get().trace & WaveConfig::kTiming) {
      threadInfo.perfReport = makePerfReport(state, wallUs);
    }
    if (WaveConfig::get().throwOnError && !threadInfo.errors.empty()) {
      TORCH_CHECK(false, "Wave kernel error:\n", threadInfo.errors);
    }
  }
}

const WaveThreadInfo& waveThreadInfo() {
  return threadInfo;
}

void WaveGraphExecutor::collectDebugInfo(ExecutionState& state) {
  auto& infos = state.launchDebugInfos;
  threadInfo.debugInfo.clear();
  threadInfo.launchMeta.clear();
  if (infos.empty()) {
    return;
  }
  auto stream = state.streamPool->get();
  for (auto& info : infos) {
    if (info.pinnedInfo && info.numBlocks > 0) {
      stream->deviceToHostAsync(
          info.pinnedInfo, info.deviceInfo, info.numBlocks * sizeof(DebugInfo));
    }
  }
  stream->wait();
  state.streamPool->put(std::move(stream));

  threadInfo.debugInfo.reserve(infos.size());
  threadInfo.launchMeta.reserve(infos.size());
  for (auto& info : infos) {
    if (info.pinnedInfo && info.numBlocks > 0) {
      threadInfo.debugInfo.emplace_back(
          info.pinnedInfo, info.pinnedInfo + info.numBlocks);
    } else {
      threadInfo.debugInfo.emplace_back();
    }
    LaunchMeta meta;
    meta.sequenceNumber = info.sequenceNumber;
    meta.stepIdx = info.stepIdx;
    meta.numBlocks = info.numBlocks;
    if (WaveConfig::get().trace & WaveConfig::kTiming) {
      auto seq = info.sequenceNumber;
      auto step = info.stepIdx;
      if (seq < static_cast<int32_t>(state.stepVectors.size()) &&
          step < static_cast<int32_t>(state.stepVectors[seq].size())) {
        auto& sv = state.stepVectors[seq][step];
        meta.gatherUs = sv.gatherUs;
        meta.gridUs = sv.gridUs;
        meta.allocUs = sv.allocUs;
        meta.fillUs = sv.fillUs;
        meta.kernelUs = sv.kernelUs;
        meta.standaloneUs = sv.standaloneUs;
        meta.standaloneBound = sv.standaloneBound;
        meta.noDtoH = sv.noDtoH;
        meta.inputBytes = sv.inputBytes;
        meta.outputBytes = sv.outputBytes;
      }
    }
    threadInfo.launchMeta.push_back(std::move(meta));
  }

  // Copy standalone timing sorted by time descending.
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    threadInfo.standaloneTimes.clear();
    threadInfo.standaloneLabels.clear();
    if (state.standaloneStats && state.standaloneIndices) {
      // Build index→node map by inverting node→index.
      std::unordered_map<int32_t, NodeCP> idxToNode;
      for (auto& [node, idx] : *state.standaloneIndices) {
        idxToNode[idx] = node;
      }
      std::vector<std::pair<int64_t, std::string>> sorted;
      for (size_t i = 0; i < state.standaloneStats->size(); ++i) {
        auto us = (*state.standaloneStats)[i].micros;
        if (us > 0) {
          auto it = idxToNode.find(static_cast<int32_t>(i));
          std::string label = it != idxToNode.end()
              ? standaloneToString(it->second)
              : "standalone[" + std::to_string(i) + "]";
          sorted.emplace_back(us, std::move(label));
        }
      }
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
      });
      for (auto& [us, label] : sorted) {
        threadInfo.standaloneTimes.push_back(us);
        threadInfo.standaloneLabels.push_back(std::move(label));
      }
    }
  }
}

void WaveGraphExecutor::adjustCosts(ExecutionState& state) {
  const auto& info = waveThreadInfo();
  auto trace = WaveConfig::get().trace;
  bool doTrace = (trace & (WaveConfig::kNodes | WaveConfig::kLaunches)) != 0;

  for (size_t mi = 0; mi < info.launchMeta.size(); ++mi) {
    const auto& m = info.launchMeta[mi];
    auto seq = m.sequenceNumber;
    auto step = m.stepIdx;
    if (seq >= static_cast<int32_t>(state.stepVectors.size()) ||
        step >= static_cast<int32_t>(state.stepVectors[seq].size())) {
      continue;
    }
    auto& sv = state.stepVectors[seq][step];
    if (sv.kernels.empty() || mi >= info.debugInfo.size() ||
        info.debugInfo[mi].empty()) {
      continue;
    }
    const auto& debugBlocks = info.debugInfo[mi];

    // Sum clocks per launch using block position, not opcode matching.
    // Block 0..numBlocksPerLaunch[0]-1 belong to launch 0, etc.
    std::vector<int64_t> launchClocks(sv.kernels.size(), 0);
    int32_t blockStart = 0;
    for (size_t li = 0; li < sv.kernels.size(); ++li) {
      int32_t nBlocks =
          li < sv.numBlocksPerLaunch.size() ? sv.numBlocksPerLaunch[li] : 0;
      for (int32_t b = 0; b < nBlocks; ++b) {
        auto idx = blockStart + b;
        if (idx < static_cast<int32_t>(debugBlocks.size())) {
          launchClocks[li] += debugBlocks[idx].clocks;
        }
      }
      blockStart += nBlocks;
    }

    int64_t totalActualClocks = 0;
    for (auto c : launchClocks) {
      totalActualClocks += c;
    }
    if (totalActualClocks == 0) {
      continue;
    }

    float totalExpectedCost = 0;
    for (size_t i = 0; i < sv.kernels.size(); ++i) {
      if (i < sv.costs.size()) {
        totalExpectedCost += sv.costs[i];
      }
    }
    if (totalExpectedCost <= 0) {
      continue;
    }

    bool redoGrid = false;
    for (size_t i = 0; i < sv.kernels.size(); ++i) {
      float actualFraction = static_cast<float>(launchClocks[i]) /
          static_cast<float>(totalActualClocks);
      float expectedFraction =
          i < sv.costs.size() ? sv.costs[i] / totalExpectedCost : 0;
      if (expectedFraction <= 0) {
        continue;
      }

      float ratio = actualFraction / expectedFraction;
      float oldAdjust = sv.kernels[i].costAdjustFactor > 0
          ? sv.kernels[i].costAdjustFactor
          : 1.0f;

      bool atMaxBlocks = i < sv.numBlocksPerLaunch.size() &&
          i < sv.maxBlocks.size() &&
          sv.numBlocksPerLaunch[i] >= sv.maxBlocks[i];

      float newAdjust;
      if (atMaxBlocks || (ratio > 0.9f && ratio < 1.1f)) {
        newAdjust = oldAdjust;
      } else {
        newAdjust = oldAdjust * ratio;
      }
      sv.kernels[i].costAdjustFactor = newAdjust;

      bool needRedo =
          newAdjust != oldAdjust && (ratio > 1.1f || ratio < (1.0f / 1.1f));
      if (needRedo) {
        redoGrid = true;
      }
      if (doTrace && newAdjust != oldAdjust) {
        auto opCode = sv.kernels[i].launch && sv.kernels[i].launch->op
            ? sv.kernels[i].launch->op->opCode()
            : -1;
        std::cout
            << fmt::format(
                   "  op {} expected={:.4f} actual={:.4f} ratio={:.4f} adjust={:.4f}{}",
                   opCode,
                   expectedFraction,
                   actualFraction,
                   ratio,
                   newAdjust,
                   needRedo ? " redo grid" : "")
            << std::endl;
      }
    }
    if (redoGrid) {
      sv.hasGridCache = false;
    }
  }
}

std::string WaveGraphExecutor::makePerfReport(
    ExecutionState& state,
    int64_t wallUs) const {
  const auto& info = waveThreadInfo();
  std::stringstream ss;
  double wallSec = wallUs / 1e6;

  // Compute total input size from user inputs.
  int64_t totalInputBytes = 0;
  int64_t totalDataBytes = 0;
  for (const auto& meta : info.launchMeta) {
    totalDataBytes += meta.inputBytes + meta.outputBytes;
  }
  // User input size from frame inputs.
  auto& frame = *state.frame;
  auto numValues = static_cast<int32_t>(graph().numValues());
  if (waveGraph_) {
    auto* inputNode = waveGraph_->graph()->inputNode();
    if (inputNode) {
      for (auto* output : inputNode->outputs()) {
        if (!output) {
          continue;
        }
        auto id = output->id();
        if (id < 0 || id >= numValues) {
          continue;
        }
        auto& iv = frame.getIValue(id);
        if (iv.isTensor() && iv.toTensor().defined()) {
          totalInputBytes +=
              iv.toTensor().numel() * iv.toTensor().element_size();
        }
      }
    }
  }

  ss << "=== Performance Report ===\n";
  ss << fmt::format("E2E wall time: {} us ({:.3f} s)\n", wallUs, wallSec);
  if (wallUs > 0 && totalInputBytes > 0) {
    double inputGBs = totalInputBytes / (wallSec * 1e9);
    ss << fmt::format(
        "Input throughput: {:.2f} GB/s ({:.1f} MB input)\n",
        inputGBs,
        totalInputBytes / 1e6);
  }
  if (wallUs > 0 && totalDataBytes > 0) {
    double dataGBs = totalDataBytes / (wallSec * 1e9);
    ss << fmt::format(
        "Internal throughput: {:.2f} GB/s ({:.1f} MB total data)\n",
        dataGBs,
        totalDataBytes / 1e6);
  }

  // Per-node, per-step report.
  // Group launches by sequenceNumber.
  std::map<int32_t, std::vector<size_t>> nodeToLaunches;
  for (size_t i = 0; i < info.launchMeta.size(); ++i) {
    nodeToLaunches[info.launchMeta[i].sequenceNumber].push_back(i);
  }

  // Compute per-node wall times including standalone time.
  std::vector<std::pair<int32_t, int64_t>> nodeWallTimes;
  // Track which sequence numbers have kernel launches.
  std::set<int32_t> nodesWithLaunches;
  for (auto& [seq, indices] : nodeToLaunches) {
    nodesWithLaunches.insert(seq);
    int64_t nodeUs = 0;
    for (auto idx : indices) {
      const auto& m = info.launchMeta[idx];
      nodeUs += m.gatherUs + m.gridUs + m.allocUs + m.fillUs + m.kernelUs +
          m.standaloneUs;
    }
    nodeWallTimes.emplace_back(seq, nodeUs);
  }

  // Sort nodes by wall time descending.
  std::sort(
      nodeWallTimes.begin(),
      nodeWallTimes.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  // Collect all referenced op codes for the legend.
  std::set<int32_t> referencedOps;

  for (auto& [seq, nodeUs] : nodeWallTimes) {
    ss << fmt::format("\nNode {}: {} us\n", seq, nodeUs);
    auto it = nodeToLaunches.find(seq);
    if (it == nodeToLaunches.end()) {
      continue;
    }
    for (auto idx : it->second) {
      const auto& m = info.launchMeta[idx];
      if (m.numBlocks == 0) {
        // Standalone-only step.
        int32_t numStandalones = 0;
        auto seq = m.sequenceNumber;
        auto step = m.stepIdx;
        if (seq < static_cast<int32_t>(state.stepVectors.size()) &&
            step < static_cast<int32_t>(state.stepVectors[seq].size())) {
          numStandalones = static_cast<int32_t>(
              state.stepVectors[seq][step].standalones.size());
        }
        ss << fmt::format(
            "  step {}: {} standalones  {} us\n",
            m.stepIdx,
            numStandalones,
            m.standaloneUs);
        continue;
      }
      auto stepUs = m.gatherUs + m.gridUs + m.allocUs + m.fillUs + m.kernelUs;
      double bytesTotal = m.inputBytes + m.outputBytes;
      double gbps = m.kernelUs > 0 ? bytesTotal / (m.kernelUs * 1e3) : 0.0;
      ss << fmt::format(
          "  step {}: {} us  blocks={}  in={:.1f}KB out={:.1f}KB  {:.1f} GB/s",
          m.stepIdx,
          stepUs,
          m.numBlocks,
          m.inputBytes / 1024.0,
          m.outputBytes / 1024.0,
          gbps);
      ss << fmt::format(
          "  [gather={} grid={} alloc={} fill={} kernel={}]",
          m.gatherUs,
          m.gridUs,
          m.allocUs,
          m.fillUs,
          m.kernelUs);
      if (m.standaloneUs > 0) {
        ss << fmt::format(
            " standalone={}{}", m.standaloneUs, m.standaloneBound ? "*" : "");
      }
      if (m.noDtoH) {
        ss << " noDtoH";
      }
      ss << "\n";

      // Thread block balance for this step.
      if (idx < info.debugInfo.size() && !info.debugInfo[idx].empty()) {
        const auto& blocks = info.debugInfo[idx];
        int64_t maxClocks = 0;
        int64_t totalClocks = 0;
        int64_t totalBarrier = 0;
        for (const auto& b : blocks) {
          maxClocks = std::max(maxClocks, b.clocks);
          totalClocks += b.clocks;
          totalBarrier += b.barrierClocks;
        }
        double util = maxClocks > 0 ? 100.0 * totalClocks /
                (maxClocks * static_cast<int64_t>(blocks.size()))
                                    : 0.0;
        double syncPct =
            totalClocks > 0 ? 100.0 * totalBarrier / totalClocks : 0.0;
        ss << fmt::format(
            "    balance: util={:.1f}% sync={:.1f}% maxClk={} blocks={}\n",
            util,
            syncPct,
            maxClocks,
            blocks.size());

        // Per-op breakdown sorted by max clocks descending.
        struct OpStats {
          int32_t op;
          int64_t opMin{std::numeric_limits<int64_t>::max()};
          int64_t opMax{0};
          int64_t opSum{0};
          int64_t opBarrier{0};
          int64_t count{0};
          int64_t numElements{0};
        };
        std::map<int32_t, OpStats> opMap;
        for (const auto& b : blocks) {
          auto& s = opMap[b.op];
          s.op = b.op;
          s.opMin = std::min(s.opMin, b.clocks);
          s.opMax = std::max(s.opMax, b.clocks);
          s.opSum += b.clocks;
          s.opBarrier += b.barrierClocks;
          s.count++;
          referencedOps.insert(b.op);
        }
        // Get per-op element counts from step vectors.
        auto seq = m.sequenceNumber;
        auto step = m.stepIdx;
        if (seq < static_cast<int32_t>(state.stepVectors.size()) &&
            step < static_cast<int32_t>(state.stepVectors[seq].size())) {
          for (const auto& kern : state.stepVectors[seq][step].kernels) {
            if (kern.launch && kern.launch->op) {
              auto opCode = kern.launch->op->opCode();
              auto it = opMap.find(opCode);
              if (it != opMap.end()) {
                it->second.numElements += kern.numElements;
              }
            }
          }
        }
        auto fmtSize = [](int64_t n) -> std::string {
          if (n >= 1000000) {
            return fmt::format("{:.1f}M", n / 1e6);
          }
          if (n >= 1000) {
            return fmt::format("{:.1f}K", n / 1e3);
          }
          return std::to_string(n);
        };
        std::vector<OpStats> sortedOps;
        sortedOps.reserve(opMap.size());
        for (auto& [op, s] : opMap) {
          sortedOps.push_back(s);
        }
        std::sort(sortedOps.begin(), sortedOps.end(), [](auto& a, auto& b) {
          return a.opMax > b.opMax;
        });
        for (auto& s : sortedOps) {
          auto opAvg = s.opSum / s.count;
          ss << fmt::format(
              "      op {} ({} blocks, {}): clk max/avg/min={}/{}/{} barrier={}\n",
              s.op,
              s.count,
              fmtSize(s.numElements),
              s.opMax,
              opAvg,
              s.opMin,
              s.opBarrier);
        }
      }
    }
  }

  // Top consumers.
  ss << "\n=== Top Consumers ===\n";
  int64_t totalNodeUs = 0;
  for (auto& [seq, us] : nodeWallTimes) {
    totalNodeUs += us;
  }
  for (size_t i = 0; i < std::min(nodeWallTimes.size(), size_t(10)); ++i) {
    auto& [seq, us] = nodeWallTimes[i];
    double pct = totalNodeUs > 0 ? 100.0 * us / totalNodeUs : 0.0;
    ss << fmt::format("  Node {}: {} us ({:.1f}%)\n", seq, us, pct);
  }

  // Top standalones.
  if (!info.standaloneTimes.empty()) {
    ss << "\nTop standalones (% wall time):\n";
    for (size_t i = 0; i < std::min(info.standaloneTimes.size(), size_t(10));
         ++i) {
      double pct = wallUs > 0 ? 100.0 * info.standaloneTimes[i] / wallUs : 0.0;
      ss << fmt::format(
          "  {} us ({:.1f}%): {}\n",
          info.standaloneTimes[i],
          pct,
          i < info.standaloneLabels.size() ? info.standaloneLabels[i] : "?");
    }
  }

  // Op legend: map op codes to their kernel operation expressions.
  if (!referencedOps.empty() && waveGraph_) {
    ss << "\n=== Op Legend ===\n";
    struct OpLegendEntry {
      float cost;
      std::string label;
    };
    std::map<int32_t, OpLegendEntry> opLabels;
    for (const auto& compiledNode : waveGraph_->nodes()) {
      auto* inv = compiledNode->kernels();
      if (!inv || !inv->kernel()) {
        continue;
      }
      WithPrintOptions guard("D2,L4,S");
      for (const auto& kop : inv->kernel()->kernelOps()) {
        if (referencedOps.count(kop->opCode())) {
          opLabels[kop->opCode()] = {kop->unitCost(), kop->toString()};
        }
      }
    }
    for (auto& [opCode, entry] : opLabels) {
      ss << fmt::format(
          "  Op {} cost={:.1f} {}\n", opCode, entry.cost, entry.label);
    }
  }

  return ss.str();
}

std::string WaveGraphExecutor::errorString() const {
  const auto& info = waveThreadInfo();
  if (info.debugInfo.empty()) {
    return {};
  }

  struct ErrorEntry {
    int32_t sequenceNumber{0};
    int32_t stepIdx{0};
    int32_t blockIdx{0};
    int32_t opCode{0};
    int32_t line{0};
    int64_t extra0{0};
    int64_t extra1{0};
    std::string message;
  };

  // Collect all errors grouped by (sequenceNumber, opCode).
  using Key = std::pair<int32_t, int32_t>;
  std::map<Key, std::vector<ErrorEntry>> errorsByOp;

  for (size_t li = 0; li < info.debugInfo.size(); ++li) {
    const auto& meta = info.launchMeta[li];
    const auto& blocks = info.debugInfo[li];
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
      const auto& dbg = blocks[bi];
      if (dbg.line != 0) {
        ErrorEntry entry;
        entry.sequenceNumber = meta.sequenceNumber;
        entry.stepIdx = meta.stepIdx;
        entry.blockIdx = static_cast<int32_t>(bi);
        entry.opCode = dbg.op;
        entry.line = dbg.line;
        entry.extra0 = dbg.extra[0];
        entry.extra1 = dbg.extra[1];
        entry.message =
            std::string(dbg.message, strnlen(dbg.message, sizeof(dbg.message)));
        errorsByOp[{meta.sequenceNumber, dbg.op}].push_back(entry);
      }
    }
  }

  if (errorsByOp.empty()) {
    return {};
  }

  // Build opCode → Launch text map from the WaveGraph structure.
  std::map<Key, std::string> opText;
  const auto& nodes = waveGraph_->nodes();
  for (const auto& node : nodes) {
    const auto* composite = node->kernels();
    if (!composite) {
      continue;
    }
    // Walk ops to find matching opcodes.
    for (size_t oi = 0; oi < composite->ops().size(); ++oi) {
      const auto& op = composite->ops()[oi];
      auto* projectOp = op.projectOp();
      auto scanGrid = [&](const LaunchGrid& grid) {
        for (const auto& step : grid) {
          for (const auto& launch : step) {
            if (launch.op) {
              int32_t code = launch.op->opCode();
              // Check all sequence numbers that have errors with this opcode.
              for (const auto& [key, _] : errorsByOp) {
                if (key.second == code) {
                  Key k = {key.first, code};
                  if (opText.find(k) == opText.end()) {
                    opText[k] = fmt::format(
                        "Seq {} Op {} {}", key.first, oi, launch.toString());
                  }
                }
              }
            }
          }
        }
      };
      scanGrid(projectOp->grid());
      scanGrid(projectOp->singleBlockGrid());
      scanGrid(projectOp->cgGrid());
    }
  }

  std::stringstream ss;
  for (const auto& [key, entries] : errorsByOp) {
    auto it = opText.find(key);
    if (it != opText.end()) {
      ss << it->second << "\n";
    }
    for (const auto& entry : entries) {
      ss << "  Seq " << entry.sequenceNumber << " step " << entry.stepIdx
         << " TB " << entry.blockIdx << " L " << entry.line << " "
         << entry.extra0 << " " << entry.extra1;
      if (!entry.message.empty()) {
        ss << " " << entry.message;
      }
      ss << "\n";
    }
  }
  return ss.str();
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
