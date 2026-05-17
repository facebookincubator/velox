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
#include <gflags/gflags.h>
#include <chrono>
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/GpuArena.h"

DEFINE_bool(print_timing, false, "Print timing for wave graph execution");

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

bool initialized = false;

} // namespace

void initialize() {
  if (initialized) {
    return;
  }
  initialized = true;
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
  g->deviceArena = std::make_unique<facebook::velox::wave::GpuArena>(
      100'000'000,
      facebook::velox::wave::getDeviceAllocator(device),
      400'000'000);
  g->pinnedArena = std::make_unique<facebook::velox::wave::GpuArena>(
      100'000'000, facebook::velox::wave::getHostAllocator(device));
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
  auto device = c10::Device(c10::kCUDA, deviceId);

  // Compute total bytes needed across all tensors.
  int64_t totalBytes = 0;
  std::vector<int64_t> sizes(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    sizes[i] = in[i].nbytes();
    totalBytes += sizes[i];
  }

  // Allocate contiguous pinned host buffer and copy tensor data into it.
  auto pinned = at::empty(
      {totalBytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  auto* pinnedBase = pinned.data_ptr<uint8_t>();
  int64_t offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    memcpy(pinnedBase + offset, in[i].data_ptr(), sizes[i]);
    offset += sizes[i];
  }

  // Allocate device tensors via PyTorch and async copy from pinned memory.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = at::empty(in[i].sizes(), in[i].options().device(device));
    stream.hostToDeviceAsync(out[i].data_ptr(), pinnedBase + offset, sizes[i]);
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
    sizes[i] = in[i].nbytes();
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
    stream.deviceToHostAsync(pinnedBase + offset, in[i].data_ptr(), sizes[i]);
    offset += sizes[i];
  }

  // Build output tensors backed by the pinned buffer.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    auto ref = new facebook::velox::wave::WaveBufferPtr(pinnedBuffer);
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        sizes[i],
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
}

void runStandalones(
    const std::vector<LaunchData>& standalones,
    ExecutionState& state,
    const std::unordered_map<NodeCP, nativert::OpKernel*>& kernelMap,
    const std::unordered_map<NodeCP, int32_t>& standaloneIndices,
    std::vector<StandaloneStats>& standaloneStats) {
  using Clock = std::chrono::high_resolution_clock;
  for (const auto& data : standalones) {
    auto* actualNode = data.standalone;

    auto kernelIt = kernelMap.find(actualNode);
    TORCH_CHECK(
        kernelIt != kernelMap.end(),
        "No kernel for node ",
        actualNode->target());

    auto start = Clock::now();
    kernelIt->second->compute(*state.frame);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now() - start)
                  .count();

    auto idxIt = standaloneIndices.find(actualNode);
    if (idxIt != standaloneIndices.end()) {
      standaloneStats[idxIt->second].micros += us;
    }
  }
}

WaveGraphExecutor::WaveGraphExecutor(
    nativert::Graph& graph,
    std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels,
    const nativert::ExecutorConfig& executorConfig,
    std::shared_ptr<nativert::Weights> weights)
    : GraphExecutorBase(graph, std::move(nodeKernels), executorConfig),
      weights_(std::move(weights)) {
  for (auto& k : nodeKernels_) {
    kernelMap_[k->node()] = k.get();
  }
  ValueTypes valueTypes;
  initValueTypes(graph_, valueTypes, metaStore_);
  waveGraph_ = std::make_unique<WaveGraph>(graph, std::move(valueTypes));
  framePool_ = std::make_unique<Pool<nativert::ExecutionFrame>>(
      [this]() { return makeDeviceFrame(); });
}


std::unique_ptr<nativert::ExecutionFrame> WaveGraphExecutor::makeFrame() {
  return std::make_unique<nativert::ExecutionFrame>(
      graph_, *weights_, executorConfig_);
}

std::unique_ptr<nativert::ExecutionFrame> WaveGraphExecutor::makeDeviceFrame() {
  auto frame = makeFrame();

  // Collect all persistent tensor values and their ids.
  auto persistentValues =
      nativert::ExecutionFrame::getPersistentValues(graph_, weights_.get());
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
      frame->setIValue(tensorIds[i], c10::IValue(std::move(deviceTensors[i])));
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
      frame.setIValue(tensorIds[i], c10::IValue(std::move(deviceTensors[i])));
    }
  }

  executeWave(frame, *waveGraph_);
  // tryMoveUserOutputs moves the output IValues out of the frame, decoupling
  // them so the frame can be safely returned to the pool.
  return frame.tryMoveUserOutputs();
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

  Timer w("top exec", FLAGS_print_timing);
  auto* g = globals();
  launchDebugInfos_.clear();

  // Get a reusable ExecutionState from the pool.
  auto statePtr = waveGraph.getState();
  auto& state = *statePtr;

  // RAII guard returns the stream and state to the pool on scope exit.
  struct StateGuard {
    WaveGraph& graph;
    std::unique_ptr<ExecutionState> state;
    ~StateGuard() {
      if (state->stream) {
        state->streamPool->put(std::move(state->stream));
      }
      graph.returnState(std::move(state));
    }
  } guard{waveGraph, std::move(statePtr)};

  // Fill per-call fields.
  state.frame = &frame;
  state.valueTypes = &waveGraph.types();
  state.deviceArena = g->deviceArena.get();
  state.pinnedArena = g->pinnedArena.get();
  state.managedArena = g->managedArena.get();
  state.streamPool = g->streamPool.get();
  state.eventPool = g->eventPool.get();
  state.stream = g->streamPool->get();
  state.kernelMap = &kernelMap_;
  state.waveGraph = &waveGraph;
  state.standaloneIndices = &waveGraph.standaloneIndices();
  state.standaloneStats = &waveGraph.standaloneStats();
  state.launchDebugInfos = &launchDebugInfos_;

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
    for (const auto& input : node->inputs()) {
      sg.inputs.push_back(input.value);
    }
    result.emplace_back(
        "standalone " + sg.toString(),
        idx < static_cast<int32_t>(stats.size()) ? stats[idx].micros : 0);
  }
  return result;
}

} // namespace torch::wave
