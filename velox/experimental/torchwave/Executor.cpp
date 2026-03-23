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

facebook::velox::wave::WaveBufferPtr tensorsToDevice(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream) {
  // Compute total bytes needed across all tensors.
  int64_t totalBytes = 0;
  std::vector<int64_t> sizes(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    sizes[i] = in[i].nbytes();
    totalBytes += sizes[i];
  }

  auto* g = globals();

  // Allocate contiguous pinned host buffer and copy tensor data into it.
  auto pinnedBuffer = g->pinnedArena->allocateBytes(totalBytes);
  auto* pinnedBase = pinnedBuffer->as<uint8_t>();
  int64_t offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    memcpy(pinnedBase + offset, in[i].data_ptr(), sizes[i]);
    offset += sizes[i];
  }

  // Allocate contiguous device buffer.
  auto deviceBuffer = g->deviceArena->allocateBytes(totalBytes);
  auto* deviceBase = deviceBuffer->as<uint8_t>();

  // Async H2D copy.
  stream.hostToDeviceAsync(deviceBase, pinnedBase, totalBytes);

  // Build output tensors pointing into the device buffer. Each tensor's
  // storage holds a copy of the WaveBufferPtr to keep the device memory alive.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    auto ref = new facebook::velox::wave::WaveBufferPtr(deviceBuffer);
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        sizes[i],
        at::DataPtr(
            deviceBase + offset,
            ref,
            [](void* ctx) {
              delete static_cast<facebook::velox::wave::WaveBufferPtr*>(ctx);
            },
            c10::Device(c10::kCUDA)));
    out[i] = at::empty({0}, in[i].options())
                 .set_(std::move(storage), 0, in[i].sizes(), in[i].strides());
    offset += sizes[i];
  }

  return deviceBuffer;
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

WaveGraphExecutor::WaveGraphExecutor(
    const nativert::Graph& graph,
    std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels,
    const nativert::ExecutorConfig& executorConfig,
    std::shared_ptr<nativert::Weights> weights)
    : GraphExecutorBase(graph, std::move(nodeKernels), executorConfig),
      weights_(std::move(weights)) {
  initValueTypes();
  waveGraph_ = std::make_unique<WaveGraph>(graph_, valueTypes_);
  framePool_ = std::make_unique<Pool<nativert::ExecutionFrame>>(
      [this]() { return makeDeviceFrame(); });
}

void WaveGraphExecutor::initValueTypes() {
  const auto& tensorValuesMeta = graph_.tensorValuesMeta();
  valueTypes_.types.resize(graph_.values().size(), nullptr);
  for (const auto* value : graph_.values()) {
    auto it = tensorValuesMeta.find(std::string{value->name()});
    if (it != tensorValuesMeta.end()) {
      auto meta = std::make_unique<nativert::TensorMeta>(it->second);
      valueTypes_.types[value->id()] = meta.get();
      metaStore_.push_back(std::move(meta));
    }
  }
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
    const WaveGraph& waveGraph) {
  auto* g = globals();
  ExecutionState state{
      .frame = frame,
      .valueTypes = &valueTypes_,
      .deviceArena = g->deviceArena.get(),
      .pinnedArena = g->pinnedArena.get(),
      .managedArena = g->managedArena.get(),
      .streamPool = g->streamPool.get(),
      .eventPool = g->eventPool.get()};
  for (const auto& node : waveGraph.nodes()) {
    node->execute(state);
  }
}

void makeGrid(
    const std::vector<OpInvocation>& ops,
    const std::vector<int64_t>& numElements,
    std::vector<BlockInfo>& blocks,
    std::vector<int32_t>& opIndices,
    std::vector<OpInvocation*>& nextOps) {
  const int32_t blockSize = 256;

  // Compute cost per op: numElements * unitCost.
  std::vector<float> costs(ops.size());
  float totalCost = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    costs[i] = static_cast<float>(numElements[i]) * ops[i].op()->unitCost();
    totalCost += costs[i];
  }

  // Max blocks each op could use.
  std::vector<int32_t> maxBlocks(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    maxBlocks[i] =
        static_cast<int32_t>((numElements[i] + blockSize - 1) / blockSize);
    if (maxBlocks[i] < 1) {
      maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count.
  int32_t numSMs = 100;
  auto* device = facebook::velox::wave::currentDevice();
  if (device) {
    numSMs = device->numSM;
  }
  int32_t targetBlocks = numSMs * 4;

  // Assign blocks pro rata by cost, at least 1 per op, capped by maxBlocks.
  std::vector<int32_t> numBlocksPerOp(ops.size());
  int32_t totalAssigned = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    float fraction = (totalCost > 0) ? costs[i] / totalCost : 1.0f / ops.size();
    int32_t assigned =
        std::max(1, static_cast<int32_t>(fraction * targetBlocks + 0.5f));
    assigned = std::min(assigned, maxBlocks[i]);
    numBlocksPerOp[i] = assigned;
    totalAssigned += assigned;
  }

  // Fill blocks and opIndices.
  blocks.resize(totalAssigned);
  opIndices.resize(totalAssigned);
  int32_t blockIdx = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto opCode = ops[i].op()->opCodes()[0];
    auto nBlocks = numBlocksPerOp[i];
    auto elements = numElements[i];
    for (int32_t b = 0; b < nBlocks; ++b) {
      auto& info = blocks[blockIdx];
      info.op = opCode;
      info.blockInOp = b;
      info.numBlocksInOp = nBlocks;
      // Distribute elements across blocks.
      auto elemsPerBlock = (elements + nBlocks - 1) / nBlocks;
      auto startRow = static_cast<int64_t>(b) * elemsPerBlock;
      auto endRow = std::min(startRow + elemsPerBlock, elements);
      info.rowsForBlock = static_cast<int32_t>(endRow - startRow);
      info.rowIdx = 0;
      info.params = nullptr;
      info.returnData = nullptr;
      info.debug = nullptr;
      opIndices[blockIdx] = static_cast<int32_t>(i);
      ++blockIdx;
    }
  }

  nextOps.clear();
}

namespace {

void fillTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Tensors with more than 3 dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
}

} // namespace

void CompositeInvocation::execute(ExecutionState& state) {
  auto& frame = state.frame;
  const auto& types = *state.valueTypes;

  // 1. Allocate outputs and collect numElements per op.
  std::vector<int64_t> numElements;
  numElements.reserve(ops.size());
  for (auto& op : ops) {
    numElements.push_back(op.allocateOutput(frame, types));
  }

  // 2. Build the grid.
  std::vector<BlockInfo> blocks;
  std::vector<int32_t> opIndices;
  std::vector<OpInvocation*> nextOps;
  makeGrid(ops, numElements, blocks, opIndices, nextOps);

  // 3. Compute total param bytes and track per-op param offsets relative to
  //    the start of the pinned buffer.
  auto numBlocks = blocks.size();
  auto blockInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(BlockInfo);
  std::vector<int64_t> opParamOffsets(ops.size());
  int64_t paramCursor = blockInfoBytes;
  for (size_t i = 0; i < ops.size(); ++i) {
    opParamOffsets[i] = paramCursor;
    paramCursor += ops[i].paramSize();
  }
  auto totalPinnedBytes = paramCursor;

  // 4. Allocate pinned host buffer: BlockInfos + params.
  auto pinnedBuffer = state.pinnedArena->allocateBytes(totalPinnedBytes);
  auto* pinnedBase = pinnedBuffer->as<uint8_t>();

  // 5. Copy BlockInfos to beginning of pinned buffer.
  if (!blocks.empty()) {
    memcpy(pinnedBase, blocks.data(), blockInfoBytes);
  }

  // 6. Fill params from frame values after the BlockInfos.
  for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx) {
    auto& op = ops[opIdx];
    auto* paramBase = pinnedBase + opParamOffsets[opIdx];
    const auto& values = op.values();
    const auto& offsets = op.offsets();
    const auto& constants = op.constants();
    auto numInputValues = values.size();

    const auto& orderedInputs = op.op()->orderedInputs();
    auto numInputs = op.op()->numInputs();

    // Fill input value params.
    for (size_t i = 0; i < numInputValues; ++i) {
      auto* value = values[i];
      auto offset = offsets[i];
      auto* dest = paramBase + offset;
      auto it = op.bindings().find(value->id());
      auto actualId = it != op.bindings().end() ? it->second : value->id();
      const auto& ivalue = frame.getIValue(actualId);
      if (ivalue.isTensor()) {
        fillTensorParam(ivalue.toTensor(), dest);
      } else if (ivalue.isInt()) {
        *reinterpret_cast<int64_t*>(dest) = ivalue.toInt();
      } else if (ivalue.isDouble()) {
        *reinterpret_cast<double*>(dest) = ivalue.toDouble();
      } else if (ivalue.isBool()) {
        *reinterpret_cast<int64_t*>(dest) = ivalue.toBool() ? 1 : 0;
      } else {
        TORCH_CHECK(false, "Unsupported IValue type for kernel param");
      }
    }

    // Fill output params via bindings.
    for (size_t i = numInputs; i < orderedInputs.size() &&
         (numInputValues + (i - numInputs)) < offsets.size();
         ++i) {
      auto formalId = orderedInputs[i]->id();
      auto bindIt = op.bindings().find(formalId);
      if (bindIt == op.bindings().end()) {
        continue;
      }
      auto actualId = bindIt->second;
      auto offsetIdx = numInputValues + (i - numInputs);
      auto offset = offsets[offsetIdx];
      auto* dest = paramBase + offset;
      const auto& ivalue = frame.getIValue(actualId);
      if (ivalue.isTensor()) {
        fillTensorParam(ivalue.toTensor(), dest);
      } else {
        TORCH_CHECK(false, "Expected tensor for output param");
      }
    }

    // Fill constant params.
    for (size_t i = 0; i < constants.size(); ++i) {
      auto offsetIdx = numInputValues + i;
      if (offsetIdx >= offsets.size()) {
        break;
      }
      auto offset = offsets[offsetIdx];
      auto* dest = paramBase + offset;
      const auto* c = constants[i];
      if (c->isInt()) {
        *reinterpret_cast<int64_t*>(dest) = c->toInt();
      } else if (c->isDouble()) {
        *reinterpret_cast<double*>(dest) = c->toDouble();
      } else if (c->isBool()) {
        *reinterpret_cast<int64_t*>(dest) = c->toBool() ? 1 : 0;
      } else {
        TORCH_CHECK(false, "Unsupported constant type for kernel param");
      }
    }
  }

  // 7. Allocate device memory.
  auto deviceBuffer = state.deviceArena->allocateBytes(totalPinnedBytes);
  auto* deviceBase = deviceBuffer->as<uint8_t>();

  // 8. Patch BlockInfo params pointers to point to the device-side param
  //    area for the corresponding op (will be valid after H2D copy).
  auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);
  for (size_t b = 0; b < numBlocks; ++b) {
    auto opIdx = opIndices[b];
    pinnedBlocks[b].params = deviceBase + opParamOffsets[opIdx];
  }

  // 9. Get a stream and enqueue H2D transfer.
  auto stream = state.streamPool->get();
  stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);

  // 10. Launch the composite kernel on the same stream.
  const int32_t blockSize = 256;
  TorchWaveParams params;
  params.info = reinterpret_cast<BlockInfo*>(deviceBase);
  void* args[] = {&params};
  kernel->launch(
      static_cast<int32_t>(numBlocks), blockSize, 0, stream.get(), args);

  // 11. Wait for the kernel to complete and return the stream to the pool.
  stream->wait();
  state.streamPool->put(std::move(stream));
}

void CompiledNode::execute(ExecutionState& state) {
  for (auto& group : kernels_) {
    for (auto& invocation : group) {
      invocation->execute(state);
    }
  }
}

std::vector<std::vector<Dim>> elementwiseOutputShape(
    const std::vector<const nativert::Value*>& inputs,
    const nativert::Node* /*node*/,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
  int64_t maxNumel = -1;
  std::vector<Dim> bestShape;
  for (const auto* input : inputs) {
    auto it = map.find(input->id());
    TORCH_CHECK(
        it != map.end(),
        "Input value %v",
        input->id(),
        " not found in FormalToActual map");
    auto actualId = it->second;
    auto tensor = frame.getTensor(actualId);
    auto numel = tensor.numel();
    if (numel > maxNumel) {
      maxNumel = numel;
      auto sizes = tensor.sizes();
      bestShape.assign(sizes.begin(), sizes.end());
    }
  }
  return {bestShape};
}

} // namespace torch::wave
