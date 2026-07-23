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
#include <c10/core/CachingDeviceAllocator.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Standalones.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <torch/nativert/kernels/C10Kernel.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/GpuArena.h"

// Forward declaration of the CUDA runtime call used to synchronize the default
// stream. This translation unit is built in a CPU-configured target without the
// CUDA headers; the symbol resolves from the CUDA runtime linked into the final
// binary. PyTorch dispatches eager standalone ops to the default stream.
extern "C" int cudaStreamSynchronize(void* stream);

// current_device() is a non-inline C10_CUDA_API symbol resolved at final link
// (same rationale as cudaStreamSynchronize: this TU has no CUDA headers).
// Allocator peak stats are read/reset through the CPU-safe, device-agnostic
// c10::getDeviceAllocator(CUDA) interface
// (<c10/core/CachingDeviceAllocator.h>).
namespace c10::cuda {
c10::DeviceIndex current_device();
} // namespace c10::cuda

namespace torch::wave {

namespace {

// nativert's KernelFactory routes the _operator.* scalar ops by operator, not
// by the node's output type: the scalar arithmetic ops (add/sub/mul/pow) use
// ScalarBinaryOpKernel and neg/truediv/sqrt/trunc use SymFloatOpKernel.
// SymIntOpKernel only implements floordiv/mod/sym_max/sym_min, so choosing a
// kernel from a SymInt/SymBool output type alone gives the wrong kernel for
// these (e.g. _operator.sub on a SymInt output -> SymIntOpKernel ->
// "unsupported operator for SymInt"). Mirror nativert's classification.
bool isScalarBinaryOp(std::string_view target) {
  return target == "_operator.add" || target == "_operator.sub" ||
      target == "_operator.mul" || target == "_operator.pow";
}

bool isSymFloatOp(std::string_view target) {
  return target == "_operator.neg" || target == "_operator.truediv" ||
      target == "torch._sym_sqrt" || target == "math.trunc";
}

thread_local WaveThreadInfo threadInfo;

// Synchronizes the CUDA default stream (stream 0), where eager ATen standalone
// ops are dispatched, so they are complete before executeWave returns.
void syncTorchDefaultStream() {
  cudaStreamSynchronize(nullptr);
}

// Resets the torch CUDA caching allocator's peak stats on the active device so
// the peak read back after a run reflects only that run.
void resetPeakAllocatedBytes() {
  c10::getDeviceAllocator(c10::DeviceType::CUDA)
      ->resetPeakStats(c10::cuda::current_device());
}

// Peak bytes allocated by the torch CUDA caching allocator since the last
// resetPeakStats, on the active device. Captures the transient intra-run
// high-water mark, not just per-step samples.
int64_t peakAllocatedBytes() {
  auto* allocator = c10::getDeviceAllocator(c10::DeviceType::CUDA);
  auto stats = allocator->getDeviceStats(c10::cuda::current_device());
  return stats
      .allocated_bytes[static_cast<size_t>(
          c10::CachingAllocator::StatType::AGGREGATE)]
      .peak;
}

// TSC ticks per microsecond, calibrated once. folly::hardware_timestamp()
// (rdtsc) costs ~ns, whereas std::chrono here is backed by kvm-clock at ~tens
// of us/call -- too expensive for per-standalone timing of many cheap ops.
double tscTicksPerMicro() {
  static const double ticksPerMicro = [] {
    auto t0 = std::chrono::steady_clock::now();
    auto c0 = folly::hardware_timestamp();
    while (std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0)
               .count() < 2000) {
    }
    auto c1 = folly::hardware_timestamp();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    return static_cast<double>(c1 - c0) /
        static_cast<double>(std::max<int64_t>(1, us));
  }();
  return ticksPerMicro;
}

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
  // Run the one-time NVRTC/system-header initialization here, on the
  // (main) thread that sets up the executor, unless it was already done
  // elsewhere. ensureInit() touches the filesystem and publishes the shared
  // /tmp/wavesystemheaders.txt; doing it lazily on a Wave compile-pool thread
  // inside a heavyweight host (NCCL/Thrift/folly) is what hangs warmup()
  // (T275179010). initialize() is idempotent, so this is a cheap no-op if the
  // header init has already happened.
  facebook::velox::wave::CompiledKernel::initialize();
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

  // Contiguify and force standard CPU allocation.  Pickle-deserialized
  // tensors may sit on CUDA managed memory pages that SIGSEGV on memcpy
  // into pinned memory.
  std::vector<at::Tensor> contig(in.size());
  int64_t totalBytes = 0;
  std::vector<int64_t> sizes(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    contig[i] = in[i].contiguous().cpu().clone();
    sizes[i] = static_cast<int64_t>(contig[i].nbytes());
    totalBytes += sizes[i];
  }

  // Allocate contiguous pinned host buffer and copy tensor data into it.
  auto pinned = at::empty(
      {totalBytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  auto* pinnedBase = pinned.data_ptr<uint8_t>();
  int64_t offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    memcpy(pinnedBase + offset, contig[i].data_ptr(), sizes[i]);
    offset += sizes[i];
  }

  // Allocate device storage and async copy.
  out.resize(in.size());
  offset = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    auto deviceFlat =
        at::empty({contig[i].numel()}, contig[i].options().device(device));
    stream.hostToDeviceAsync(
        deviceFlat.data_ptr(), pinnedBase + offset, sizes[i]);
    out[i] = deviceFlat.reshape(contig[i].sizes());
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

void executeNode(
    NodeCP node,
    nativert::OpKernel* kernel,
    nativert::ExecutionFrame& frame,
    TraceState* traceState) {
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
  // Trace requested input values (--trace_values) before the op runs. Done
  // here so every executeNode caller -- generic standalones, pre-pass,
  // deferred, and ready-graph nodes -- traces consistently.
  if (traceState != nullptr && !traceState->empty()) {
    std::vector<nativert::ValueId> ids;
    for (const auto& input : node->inputs()) {
      ids.push_back(input.value->id());
    }
    traceFrameValues("input", ids, frame, *traceState);
  }
  // Move cpuOnly-flagged tensor args (e.g. tensor_split indices) to CPU before
  // the op and restore after, for every executeNode caller -- the ready-graph,
  // deferred, and pre-pass paths call executeNode directly and would otherwise
  // leave the arg on GPU.
  std::vector<std::pair<nativert::ValueId, c10::IValue>> savedCpuOnly;
  if (const auto* meta = Registry::metadata(node->target())) {
    const auto& nodeInputs = node->inputs();
    for (size_t i = 0; i < nodeInputs.size() && i < meta->argumentMeta.size();
         ++i) {
      if (!meta->argumentMeta[i].cpuOnly) {
        continue;
      }
      auto id = nodeInputs[i].value->id();
      const auto& iv = frame.getIValue(id);
      if (iv.isTensor() && iv.toTensor().is_cuda()) {
        savedCpuOnly.emplace_back(id, iv);
        frame.setIValue(id, c10::IValue(iv.toTensor().cpu()));
      }
    }
  }
  SCOPE_EXIT {
    for (auto& [id, iv] : savedCpuOnly) {
      frame.setIValue(id, std::move(iv));
    }
  };
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
  // Trace requested output (produced) values after the op runs.
  if (traceState != nullptr && !traceState->empty()) {
    std::vector<nativert::ValueId> ids;
    for (auto* output : node->outputs()) {
      if (output) {
        ids.push_back(output->id());
      }
    }
    traceFrameValues("output", ids, frame, *traceState);
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
    std::vector<StandaloneStats>& standaloneStats,
    bool timing) {
  for (const auto& data : standalones) {
    auto* actualNode = data.standalone;
    const bool isShortcut = data.launch != nullptr &&
        data.launch->standaloneShortcut != StandaloneShortcut::kNone;
    // Metadata-only ops (shortcut or generic, e.g. unsqueeze / a SymInt-list
    // prim.ListPack) do no device work, so they need no timing sync below.
    const bool metadataOnly =
        data.launch != nullptr && data.launch->metadataOnly;

    // Skip if this node's output is already materialized.  See
    // nodeOutputsComputed: re-executing would re-read recycled input buffers.
    if (nodeOutputsComputed(actualNode, *state.frame)) {
      continue;
    }

    // Skip standalone ops with None inputs — they depend on values
    // from later PNs.  The grid-standalone retry loop in executeWave will
    // retry them after all PNs execute.
    bool hasNoneInput = false;
    for (const auto& input : actualNode->inputs()) {
      if (isUnreadyNoneDependency(input.value, *state.frame)) {
        hasNoneInput = true;
        break;
      }
    }
    if (hasNoneInput) {
      if (state.deferredStandalones) {
        state.deferredStandalones->push_back(actualNode);
      }
      continue;
    }

    // Shortcut ops run via runStandaloneShortcut (not executeNode), so trace
    // their inputs here; executeNode traces inputs/outputs for the generic
    // path.
    if (isShortcut) {
      traceFrameValues(
          "input", data.actualInputs, *state.frame, state.traceState);
    }

    // Per-op timing uses the TSC (folly::hardware_timestamp, ~ns) rather than
    // std::chrono, which is backed by a slow kvm-clock here (~tens of us/call)
    // and would dwarf the metadata-only ops it is meant to measure.
    uint64_t startTicks = timing ? folly::hardware_timestamp() : 0;
    if (isShortcut) {
      // Metadata-only op: call the typed ATen primitive directly, bypassing the
      // boxed nativert dispatch.
      ++state.numShortcutsRun;
      runStandaloneShortcut(data, *state.frame);
    } else {
      auto kernelIt = kernelMap.find(actualNode);
      TORCH_CHECK(
          kernelIt != kernelMap.end(),
          "No kernel for node ",
          actualNode->target());
      // executeNode moves cpuOnly-flagged args (e.g. tensor_split indices) to
      // CPU and restores them via a SCOPE_EXIT, so no outer swap is needed
      // here.
      ++state.numStandalonesRun;
      executeNode(
          actualNode, kernelIt->second, *state.frame, &state.traceState);
    }
    if (timing) {
      // A metadata-only op only manipulates host-side tensor metadata and
      // enqueues nothing on the device, so it needs no sync. Syncing here
      // would, via the legacy default stream, drain the wave stream and
      // massively over-attribute its time. Only real eager ops need a sync to
      // capture their GPU time.
      if (!metadataOnly) {
        syncTorchDefaultStream();
      }
      auto us = static_cast<int64_t>(
          (folly::hardware_timestamp() - startTicks) / tscTicksPerMicro());
      auto idxIt = standaloneIndices.find(actualNode);
      if (idxIt != standaloneIndices.end()) {
        TORCH_CHECK(
            idxIt->second >= 0 &&
            static_cast<size_t>(idxIt->second) < standaloneStats.size());
        standaloneStats.at(idxIt->second).micros += us;
      }
    }

    if (WaveConfig::get().trace & WaveConfig::kFrame) {
      for (auto outputId : data.actualOutputs) {
        const auto& iv = state.frame->getIValue(outputId);
        std::cout << "    %" << outputId << " = " << traceIValue(iv)
                  << std::endl;
      }
    }
    if (isShortcut) {
      traceFrameValues(
          "output", data.actualOutputs, *state.frame, state.traceState);
    }
  }
}

void runShortcutStandalones(
    const std::vector<LaunchData>& shortcuts,
    ExecutionState& state,
    bool timing,
    int64_t& outUs) {
  // These ops only build host-side tensor metadata; they enqueue nothing on the
  // device and read no wave-stream data, so no per-op timing and no stream
  // sync. Time the whole batch once with the TSC (~ns) rather than the per-op
  // microsecond clock, which on kvm-clock would dwarf the work it measures.
  uint64_t startTicks = timing ? folly::hardware_timestamp() : 0;
  const bool tracing = !state.traceState.empty() ||
      (WaveConfig::get().trace & WaveConfig::kFrame);
  for (const auto& data : shortcuts) {
    if (tracing) {
      traceFrameValues(
          "input", data.actualInputs, *state.frame, state.traceState);
    }
    ++state.numShortcutsRun;
    runStandaloneShortcut(data, *state.frame);
    if (WaveConfig::get().trace & WaveConfig::kFrame) {
      for (auto outputId : data.actualOutputs) {
        const auto& iv = state.frame->getIValue(outputId);
        std::cout << "    %" << outputId << " = " << traceIValue(iv)
                  << std::endl;
      }
    }
    if (tracing) {
      traceFrameValues(
          "output", data.actualOutputs, *state.frame, state.traceState);
    }
  }
  if (timing) {
    outUs = static_cast<int64_t>(
        (folly::hardware_timestamp() - startTicks) / tscTicksPerMicro());
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
    } else if (isScalarBinaryOp(target)) {
      kernel = std::make_unique<nativert::ScalarBinaryOpKernel>(node);
    } else if (isSymFloatOp(target)) {
      kernel = std::make_unique<nativert::SymFloatOpKernel>(node);
    } else {
      bool hasSymIntOutput = false;
      bool hasSymBoolOutput = false;
      for (auto* output : node->outputs()) {
        if (output->type().kind() == nativert::Type::Kind::SymInt) {
          hasSymIntOutput = true;
        } else if (output->type().kind() == nativert::Type::Kind::SymBool) {
          hasSymBoolOutput = true;
        }
      }
      if (hasSymIntOutput) {
        kernel = std::make_unique<nativert::SymIntOpKernel>(node);
      } else if (hasSymBoolOutput) {
        kernel = std::make_unique<nativert::SymBoolOpKernel>(node);
      }
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
    } else if (!iv.isNone()) {
      frame->setIValue(id, iv);
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
  // them so the frame can be safely returned to the pool. However, it does not
  // move outputs whose graph-level default is Constant(None) -- in that case
  // the result slot stays None even though the frame has a computed tensor
  // (e.g. dynamic-shape outputs computed on device). The loop below copies
  // these non-moved outputs from the frame into the results.
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

  // A user output can still be None here when it is an elided no-op /
  // metadata-only view (e.g. view(x, [-1]) of an already-contiguous tensor):
  // wave aliases such a view to its input and never writes the view's own
  // value, so a graph-output view stays None even though its input tensor is
  // present. The loop above cannot reach these when userOutputs() is longer
  // than the output-node operand list (graph_.outputs()), so recover them here
  // directly from the flattened user-output list, materializing the view from
  // its input.
  const auto& userOutputs = graph_.userOutputs();
  for (size_t i = 0; i < results.size() && i < userOutputs.size(); ++i) {
    if (!results.at(i).isNone()) {
      continue;
    }
    const auto* valuePtr = std::get_if<nativert::Value*>(&userOutputs[i]);
    if (valuePtr == nullptr || *valuePtr == nullptr) {
      continue;
    }
    const nativert::Value* v = *valuePtr;
    const auto& iv = frame.getIValue(v->id());
    if (!iv.isNone()) {
      results.at(i) = iv;
      continue;
    }
    const auto* prod = v->producer();
    if (prod == nullptr || prod->inputs().empty()) {
      continue;
    }
    const auto tgt = prod->target();
    bool viewLike = tgt.find("view") != std::string_view::npos ||
        tgt.find("reshape") != std::string_view::npos ||
        tgt.find("flatten") != std::string_view::npos;
    if (!viewLike) {
      continue;
    }
    const auto* inputVal = prod->inputs()[0].value;
    if (inputVal == nullptr) {
      continue;
    }
    const auto& inIv = frame.getIValue(inputVal->id());
    if (inIv.isTensor() && inIv.toTensor().defined()) {
      // Observed elided views are all view(x, [-1]) (flatten); reshape(-1)
      // reproduces them and is a no-op when the input is already 1-D.
      results.at(i) = inIv.toTensor().reshape(-1);
      LOG(WARNING) << "Recovered elided view output %" << v->id()
                   << " (producer " << tgt << ") from input %"
                   << inputVal->id();
    }
  }
  return results;
}

// Releases a frame value. Under debug_single_ops, if this frame slot is the
// sole owner of the tensor's storage (no live view/alias references it), the
// whole storage -- not just this tensor's possibly-partial view -- is filled
// with 0xdd before it is dropped, so any use-after-free of a released buffer
// reads an obvious poison pattern instead of stale-but-valid data.
void freeFrameValue(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    facebook::velox::wave::Stream* stream) {
  if (WaveConfig::get().debugSingleOps) {
    const auto& iv = frame.getIValue(id);
    if (iv.isTensor()) {
      const at::Tensor& t = iv.toTensor();
      // Poison only if this frame slot is the sole owner: no other frame slot
      // holds the same TensorImpl (use_count==1 -- in-place/aliasing ops like
      // index_put/masked_put put the same tensor in the result slot too), and
      // no other tensor references the storage (a view would keep
      // storage.use_count > 1 while TensorImpl.use_count stays 1). Both must
      // hold, else a live value still sees this storage.
      if (t.defined() && t.has_storage() && t.use_count() == 1 &&
          t.storage().use_count() == 1) {
        const auto& storage = t.storage();
        void* ptr = storage.mutable_data();
        auto nbytes = static_cast<size_t>(storage.nbytes());
        if (ptr != nullptr && nbytes > 0) {
          if (t.is_cuda() && stream != nullptr) {
            // Enqueue the poison on the wave stream: it is ordered after the
            // kernels that used this buffer and before any later wave op that
            // reuses it, so it can't race buffer reuse (a default-stream memset
            // would). A genuine use-after-free on the wave stream still reads
            // 0xdd; a legitimate reuse overwrites the poison first.
            stream->memset(ptr, 0xdd, nbytes);
          } else {
            std::memset(ptr, 0xdd, nbytes);
          }
        }
      }
    }
  }
  frame.setIValue(id, c10::IValue());
}

// If a reference frame is loaded, compares the current contents of frame value
// 'id' against the recorded reference just before it is freed. A mismatch means
// the value was already corrupted (a stray write, or an aliased premature free
// of an overlapping buffer) BEFORE this free -- pinpointing which value went
// bad and at which free point, to compare against its intended last use.
void checkValueBeforeFree(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id) {
  auto* ref = WaveConfig::get().referenceFrame;
  if (ref == nullptr) {
    return;
  }
  auto it = ref->find(id);
  if (it == ref->end() || !it->second.isTensor()) {
    return;
  }
  const auto& iv = frame.getIValue(id);
  std::optional<at::Tensor> actual = iv.isTensor()
      ? std::optional<at::Tensor>(iv.toTensor())
      : scalarLikeToTensor(iv);
  if (!actual || !actual->defined() || actual->numel() == 0) {
    return;
  }
  const auto& refTensor = it->second.toTensor();
  if (!tensorsMatch(*actual, refTensor)) {
    auto limit = WaveConfig::get().tensorPrintElementLimit;
    LOG(ERROR) << "REF-BEFORE-FREE mismatch value %" << id << "\n  "
               << firstDifference(*actual, refTensor)
               << "\n  expected: " << tensorDebugString(refTensor, limit)
               << "\n  actual:   " << tensorDebugString(*actual, limit);
  }
}

void advanceSyncedStages(ExecutionState& state) {
  // Stage tracking exists only to bundle intermediate freeing with syncs, so
  // there is nothing to do (and no cost to pay) when freeing is off.
  if (!WaveConfig::get().freeIntermediates) {
    return;
  }
  auto& frame = *state.frame;

  // In debug_single_ops mode (or when a reference frame is loaded for the
  // before-free check), sync BOTH the wave stream and the default stream (eager
  // standalones run there) before freeing, so all kernels that could read a
  // buffer have finished. Any later access to a freed buffer is then a genuine
  // use-after-free, the poison memset cannot race an in-flight kernel, and the
  // before-free reference check reads settled data.
  if (WaveConfig::get().debugSingleOps ||
      WaveConfig::get().referenceFrame != nullptr) {
    if (state.stream != nullptr) {
      state.stream->wait();
    }
    cudaStreamSynchronize(nullptr);
  }

  // kFrame trace: collect the graph-output value ids (outputNode's inputs), so
  // the free trace can flag if any of them is ever freed -- they must not be.
  const bool traceFrame = (WaveConfig::get().trace & WaveConfig::kFrame) != 0;
  std::unordered_set<nativert::ValueId> graphOutputs;
  if (traceFrame && state.waveGraph != nullptr) {
    if (auto* outputNode = state.waveGraph->graph()->outputNode()) {
      for (const auto& input : outputNode->inputs()) {
        if (input.value != nullptr) {
          graphOutputs.insert(input.value->id());
        }
      }
    }
  }
  auto traceFree =
      [&](nativert::ValueId id, const char* kind, size_t seq, size_t step) {
        if (!traceFrame) {
          return;
        }
        const auto& iv = frame.getIValue(id);
        int64_t useCount = -1;
        uintptr_t storagePtr = 0;
        int64_t storageBytes = 0;
        if (iv.isTensor() && iv.toTensor().defined() &&
            iv.toTensor().has_storage()) {
          const auto& st = iv.toTensor().storage();
          useCount = st.use_count();
          storagePtr = reinterpret_cast<uintptr_t>(st.data());
          storageBytes = static_cast<int64_t>(st.nbytes());
        }
        LOG(INFO) << "TWFREE " << kind << " %" << id << " storage=0x"
                  << std::hex << storagePtr << std::dec
                  << " size=" << storageBytes << " node=" << seq
                  << " step=" << step << " use_count=" << useCount
                  << (graphOutputs.count(id) != 0 ? "  <-- GRAPH OUTPUT" : "");
      };

  for (size_t seq = 0; seq < state.stepVectors.size(); ++seq) {
    auto& steps = state.stepVectors[seq];
    for (size_t step = 0; step < steps.size(); ++step) {
      auto& sv = steps[step];
      if (sv.executionStage != ExecutionStage::kAllocated) {
        continue;
      }
      // The wave stream was just waited on, so this step's kernels are done and
      // its freeable buffers can be released.
      sv.executionStage = ExecutionStage::kSynced;
      // The node's last-use tensors were stamped onto its last step; they go
      // free in this same sync (no dedicated sync of their own).
      for (auto id : sv.lastUseIds) {
        traceFree(id, "lastUse", seq, step);
        checkValueBeforeFree(frame, id);
        freeFrameValue(frame, id, state.stream.get());
      }
    }
  }
}

void syncWaveStream(ExecutionState& state) {
  state.stream->wait();
  advanceSyncedStages(state);
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

  // Make the graph's value types available to node printers (e.g. the
  // execution-trace prints via standaloneToString) for the duration of this
  // wave execution. The guard restores the previous thread-local print options
  // on exit.
  PrintOptions printOptions = NodePrinter::defaults();
  printOptions.valueTypes = &waveGraph.types();
  WithPrintOptions printOptionsGuard(printOptions);
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

  std::vector<NodeCP> deferredStandalones;
  state.deferredStandalones = &deferredStandalones;
  state.numStandalonesRun = 0;
  state.numShortcutsRun = 0;

  // Reset the allocator's peak so the peak read back after the run reflects
  // only this run's transient high-water mark.
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    resetPeakAllocatedBytes();
  }

  // Reset per-step lifecycle stages (step vectors are pooled across runs) so
  // freeIntermediates freeing only ever fires for this run's steps.
  if (WaveConfig::get().freeIntermediates) {
    for (auto& steps : state.stepVectors) {
      for (auto& sv : steps) {
        sv.executionStage = ExecutionStage::kNotStarted;
        sv.lastUseIds.clear();
      }
    }
  }
  for (const auto& node : waveGraph.nodes()) {
    node->execute(state);
  }

  // Sanity check (replaces the former deferred-standalone retry pass): every
  // standalone must have executed in place during the composite passes above.
  // A standalone skipped for an unready-None input -- a cross-ProjectNode
  // back-edge whose input is produced by a later composite -- is left with a
  // None output here.  Rather than silently retrying to a fixpoint, fail
  // loudly: such a leftover is a real scheduling gap to fix at the partitioner.
  for (auto* deferredNode : deferredStandalones) {
    const auto& output = frame.getIValue(deferredNode->outputs()[0]->id());
    TORCH_CHECK(
        !output.isNone(),
        "wave: standalone '",
        deferredNode->target(),
        "' (output id ",
        deferredNode->outputs()[0]->id(),
        ") was deferred on a cross-ProjectNode None input and left unexecuted; "
        "fix the ordering at scheduling time instead of relying on a runtime retry");
  }
  state.deferredStandalones = nullptr;
  // Fusion-coverage summary: how much of the graph wave covered as composite
  // (fused) / standalone / shortcut, vs. left uncovered.  The eager C10
  // fallback has been removed, so an uncovered node (output still None after
  // execution) is a real coverage gap to fix at the source, not a
  // silently-absorbed leftover.  Logged once, under any --trace bit.  Placed
  // after all deferred/grid standalones have run so the standalone and shortcut
  // counts (incremented at their execution sites) are complete.
  if (WaveConfig::get().trace != 0) {
    static std::atomic<bool> fusionLogged{false};
    if (!fusionLogged.exchange(true)) {
      auto& graph = *waveGraph.graph();
      int64_t uncovered = 0;
      for (auto& gnode : graph.nodes()) {
        if (gnode.target() == "prim.Input" || gnode.target() == "prim.Output") {
          continue;
        }
        for (auto* output : gnode.outputs()) {
          if (frame.getIValue(output->id()).isNone()) {
            ++uncovered;
            break;
          }
        }
      }
      auto totalNodes = static_cast<int64_t>(graph.nodes().size());
      auto numComposites = waveGraph.nodes().size();
      int64_t numStandalones = state.numStandalonesRun;
      int64_t numShortcuts = state.numShortcutsRun;
      int64_t fusedNodes =
          totalNodes - uncovered - numStandalones - numShortcuts;
      std::cout << "FUSION: nativert_graph_nodes=" << totalNodes
                << " wave_composite_kernels=" << numComposites
                << " fused_nodes=" << fusedNodes
                << " standalone_ops=" << numStandalones
                << " shortcut_ops=" << numShortcuts
                << " uncovered_ops=" << uncovered << " (~"
                << (100.0 * fusedNodes / totalNodes) << "% fused, ~"
                << (100.0 * uncovered / totalNodes) << "% uncovered)"
                << std::endl;
    }
  }
  // Sync the wave stream and the PyTorch default stream: eager standalone ops
  // run on the default stream while fused kernels run on the wave stream, and
  // the two are otherwise unordered. Both must complete before executeWave
  // returns so all results this invocation produced are visible to the
  // caller.
  syncWaveStream(state);
  syncTorchDefaultStream();
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
      // Peak allocated GPU memory over this run, from the allocator's own
      // high-water mark (reset at the start of the run above), so it captures
      // transient intra-step peaks rather than just the per-step samples.
      threadInfo.peakBytes = peakAllocatedBytes();
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
        meta.shortcutUs = sv.shortcutUs;
        meta.standaloneBound = sv.standaloneBound;
        meta.noDtoH = sv.noDtoH;
        meta.inputBytes = sv.inputBytes;
        meta.outputBytes = sv.outputBytes;
        meta.currentBytes = sv.currentBytes;
        meta.refCheckUs = sv.refCheckUs;
      }
    }
    threadInfo.launchMeta.push_back(std::move(meta));
  }

  // Copy standalone timing sorted by time descending.
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    threadInfo.standaloneTimes.clear();
    threadInfo.standaloneLabels.clear();
    threadInfo.standaloneTargets.clear();
    if (state.standaloneStats && state.standaloneIndices) {
      // Build index→node map by inverting node→index.
      std::unordered_map<int32_t, NodeCP> idxToNode;
      for (auto& [node, idx] : *state.standaloneIndices) {
        idxToNode[idx] = node;
      }
      std::vector<std::tuple<int64_t, std::string, std::string>> sorted;
      for (size_t i = 0; i < state.standaloneStats->size(); ++i) {
        auto us = (*state.standaloneStats)[i].micros;
        if (us > 0) {
          auto it = idxToNode.find(static_cast<int32_t>(i));
          std::string label = it != idxToNode.end()
              ? standaloneToString(it->second)
              : "standalone[" + std::to_string(i) + "]";
          std::string target = it != idxToNode.end()
              ? std::string(it->second->target())
              : "standalone[" + std::to_string(i) + "]";
          sorted.emplace_back(us, std::move(label), std::move(target));
        }
      }
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) > std::get<0>(b);
      });
      for (auto& [us, label, target] : sorted) {
        threadInfo.standaloneTimes.push_back(us);
        threadInfo.standaloneLabels.push_back(std::move(label));
        threadInfo.standaloneTargets.push_back(std::move(target));
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

  // Compute total input size from user inputs.
  int64_t totalInputBytes = 0;
  int64_t totalDataBytes = 0;
  // Total reference-frame checking time (device-to-host copy + comparison).
  // This is debug-only overhead included in the measured wall time, so
  // subtract it to report the real e2e time.
  int64_t totalRefCheckUs = 0;
  for (const auto& meta : info.launchMeta) {
    totalDataBytes += meta.inputBytes + meta.outputBytes;
    totalRefCheckUs += meta.refCheckUs;
  }
  bool refChecking = WaveConfig::get().referenceFrame != nullptr;
  wallUs -= totalRefCheckUs;
  double wallSec = static_cast<double>(wallUs) / 1e6;
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
  if (refChecking) {
    ss << "WARNING reference frame checking is on.\n";
  }
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
  ss << fmt::format(
      "Peak GPU RAM: {}\n", facebook::velox::succinctBytes(info.peakBytes));

  // Time split across all steps: kernel (GPU), standalone (eager ATen on the
  // default stream), and interpretation (gather + grid + alloc + fill = the
  // host-side scheduling overhead). For a standalone-bound step the measured
  // kernelUs reflects waiting on the standalone work rather than the GPU
  // kernel, so estimate the kernel time from that step's max thread-block
  // clocks (1 clock ~= 0.7 ns).
  {
    double kernelUs = 0.0;
    int64_t standaloneUs = 0;
    int64_t interpUs = 0;
    for (size_t i = 0; i < info.launchMeta.size(); ++i) {
      const auto& m = info.launchMeta[i];
      interpUs += m.gatherUs + m.gridUs + m.allocUs + m.fillUs;
      standaloneUs += m.standaloneUs + m.shortcutUs;
      if (m.standaloneBound) {
        int64_t maxClocks = 0;
        if (i < info.debugInfo.size()) {
          for (const auto& b : info.debugInfo[i]) {
            maxClocks = std::max(maxClocks, b.clocks);
          }
        }
        kernelUs += static_cast<double>(maxClocks) * 0.7 / 1000.0;
      } else {
        kernelUs += static_cast<double>(m.kernelUs);
      }
    }
    ss << fmt::format(
        "Kernel time: {:.0f} us  Standalone time: {} us  Interpretation time: {} us\n",
        kernelUs,
        standaloneUs,
        interpUs);
  }

  // Per-node, per-step report.
  // Group launches by sequenceNumber.
  std::map<int32_t, std::vector<size_t>> nodeToLaunches;
  for (size_t i = 0; i < info.launchMeta.size(); ++i) {
    nodeToLaunches[info.launchMeta[i].sequenceNumber].push_back(i);
  }

  // Compute per-node wall times. Within a step the fused kernel (wave stream)
  // and the eager standalones (default stream) run concurrently, so the
  // step's wall is interpretation plus the larger of the two, not their sum.
  std::vector<std::pair<int32_t, int64_t>> nodeWallTimes;
  // Track which sequence numbers have kernel launches.
  std::set<int32_t> nodesWithLaunches;
  for (auto& [seq, indices] : nodeToLaunches) {
    nodesWithLaunches.insert(seq);
    int64_t nodeUs = 0;
    for (auto idx : indices) {
      const auto& m = info.launchMeta[idx];
      int64_t interp = m.gatherUs + m.gridUs + m.allocUs + m.fillUs;
      int64_t hostStandalone = m.standaloneUs + m.shortcutUs;
      nodeUs += interp + std::max(m.kernelUs, hostStandalone);
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
  // Total GPU thread-block clocks spent in each op across all blocks/steps.
  std::map<int32_t, int64_t> opTotalClocks;

  // Distinct standalone targets and their counts for a step, as
  // " [target xN, target xM, ...]" (empty if the step has no standalones).
  auto standaloneBreakdown = [&](int32_t seq, int32_t step) -> std::string {
    if (seq >= static_cast<int32_t>(state.stepVectors.size()) ||
        step >= static_cast<int32_t>(state.stepVectors[seq].size())) {
      return "";
    }
    std::map<std::string, int32_t> counts;
    const auto& sv = state.stepVectors[seq][step];
    for (const auto* list : {&sv.standalones, &sv.shortcutStandalones}) {
      for (const auto& data : *list) {
        if (data.launch && data.launch->standalone) {
          counts[std::string(data.launch->standalone->target())]++;
        }
      }
    }
    if (counts.empty()) {
      return "";
    }
    std::string s = " [";
    bool first = true;
    for (const auto& [name, c] : counts) {
      if (!first) {
        s += ", ";
      }
      s += name + " x" + std::to_string(c);
      first = false;
    }
    s += "]";
    return s;
  };

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
              state.stepVectors[seq][step].standalones.size() +
              state.stepVectors[seq][step].shortcutStandalones.size());
        }
        ss << fmt::format(
            "  step {}: {} standalones  GPU RAM={}  {} us",
            m.stepIdx,
            numStandalones,
            facebook::velox::succinctBytes(m.currentBytes),
            m.standaloneUs);
        ss << standaloneBreakdown(m.sequenceNumber, m.stepIdx) << "\n";
        continue;
      }
      auto stepUs = m.gatherUs + m.gridUs + m.allocUs + m.fillUs + m.kernelUs;
      double bytesTotal = m.inputBytes + m.outputBytes;
      double gbps = m.kernelUs > 0 ? bytesTotal / (m.kernelUs * 1e3) : 0.0;
      ss << fmt::format(
          "  step {}: {} us  blocks={}  GPU RAM={}  in={:.1f}KB out={:.1f}KB  {:.1f} GB/s",
          m.stepIdx,
          stepUs,
          m.numBlocks,
          facebook::velox::succinctBytes(m.currentBytes),
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
      if (m.shortcutUs > 0) {
        ss << fmt::format(" shortcut={}", m.shortcutUs);
      }
      if (m.standaloneUs > 0) {
        ss << fmt::format(
            " standalone={}{}", m.standaloneUs, m.standaloneBound ? "*" : "");
      }
      // Op-target breakdown covers both standalone and shortcut lists; print
      // it whenever either ran.
      if (m.standaloneUs > 0 || m.shortcutUs > 0) {
        ss << standaloneBreakdown(m.sequenceNumber, m.stepIdx);
      }
      if (m.noDtoH) {
        ss << " noDtoH";
      }
      if (m.refCheckUs > 0) {
        ss << fmt::format(" refcheck={}", m.refCheckUs);
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
          opTotalClocks[b.op] += b.clocks;
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

  // Standalones grouped by op target: total time and occurrence count, sorted
  // by total time descending (all targets, no cutoff).
  if (!info.standaloneTimes.empty()) {
    ss << "\nStandalones by target (% wall time):\n";
    std::unordered_map<std::string, std::pair<int64_t, int32_t>> byTarget;
    static const std::string kUnknownTarget = "?";
    for (size_t i = 0; i < info.standaloneTimes.size(); ++i) {
      // Both branches are lvalues so the const ref binds without copying the
      // label (a temporary in the false branch would force a copy).
      const std::string& target = i < info.standaloneTargets.size()
          ? info.standaloneTargets[i]
          : kUnknownTarget;
      auto& entry = byTarget[target];
      entry.first += info.standaloneTimes[i];
      entry.second += 1;
    }
    std::vector<std::pair<std::string, std::pair<int64_t, int32_t>>> sorted(
        byTarget.begin(), byTarget.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
      return a.second.first > b.second.first;
    });
    for (const auto& [target, agg] : sorted) {
      double pct = wallUs > 0
          ? 100.0 * static_cast<double>(agg.first) / static_cast<double>(wallUs)
          : 0.0;
      ss << fmt::format(
          "  {} us ({:.1f}%) x{}: {}\n", agg.first, pct, agg.second, target);
    }
  }

  // Op legend: map op codes to their kernel operation expressions, annotated
  // with each op's share of total GPU thread-block clocks.
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

    // Grand total thread-block clocks over all kernel ops.
    int64_t grandTotalClocks = 0;
    for (const auto& [op, clk] : opTotalClocks) {
      grandTotalClocks += clk;
    }
    auto pctOf = [&](int64_t clk) -> double {
      return grandTotalClocks > 0 ? 100.0 * static_cast<double>(clk) /
              static_cast<double>(grandTotalClocks)
                                  : 0.0;
    };

    // Top 10 ops by thread-block clocks, on one line at the head.
    std::vector<std::pair<int32_t, int64_t>> byClocks(
        opTotalClocks.begin(), opTotalClocks.end());
    std::sort(byClocks.begin(), byClocks.end(), [](auto& a, auto& b) {
      return a.second > b.second;
    });
    ss << "  Top ops by clocks: ";
    int32_t shown = 0;
    for (const auto& [op, clk] : byClocks) {
      if (shown >= 10) {
        break;
      }
      if (shown > 0) {
        ss << ", ";
      }
      ss << fmt::format("op {} {:.1f}%", op, pctOf(clk));
      ++shown;
    }
    ss << "\n";

    for (auto& [opCode, entry] : opLabels) {
      auto clk = opTotalClocks.count(opCode) ? opTotalClocks[opCode] : 0;
      ss << fmt::format(
          "  Op {} cost={:.1f} clocks={} ({:.1f}%) {}\n",
          opCode,
          entry.cost,
          clk,
          pctOf(clk),
          entry.label);
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
