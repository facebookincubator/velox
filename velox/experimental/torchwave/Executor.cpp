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

#include "velox/experimental/torchwave/AllocGroup.h"

#include <ATen/ATen.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <folly/CppAttributes.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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

// Owned by velox/experimental/wave/common/Compile.cu; see initialize().
DECLARE_bool(cuda_lineinfo);

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

struct GlobalResources {
  std::unique_ptr<facebook::velox::wave::GpuArena> deviceArena;
  std::unique_ptr<facebook::velox::wave::GpuArena> pinnedArena;
  std::unique_ptr<facebook::velox::wave::GpuArena> managedArena;
  std::unique_ptr<StreamPool> streamPool;
  /// Ordering-only events (cudaEventDisableTiming), used on every run.
  std::unique_ptr<EventPool> eventPool;
  /// Timing-enabled events, used only under the kTiming trace bit. Kept
  /// separate because elapsedTime() throws on a disable-timing event, so the
  /// two kinds must never be mixed.
  std::unique_ptr<EventPool> timingEventPool;
  /// Non-owning wrapper on the CUDA stream eager standalones run on, so events
  /// can be recorded on it from this CPU-configured translation unit.
  std::unique_ptr<facebook::velox::wave::Stream> torchStreamWrapper;
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

namespace {
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local int64_t tAllocCallUs = 0;
} // namespace

int64_t threadAllocCallUs() {
  return tAllocCallUs;
}

ScopedAllocCall::ScopedAllocCall()
    : timing_(
          WaveConfig::get().printTiming ||
          (WaveConfig::get().trace & WaveConfig::kTiming)),
      start_(timing_ ? folly::hardware_timestamp() : 0) {}

ScopedAllocCall::~ScopedAllocCall() {
  if (timing_) {
    tAllocCallUs += static_cast<int64_t>(
        (folly::hardware_timestamp() - start_) / tscTicksPerMicro());
  }
}

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
  // Wave takes its NVRTC options from gflags, not from an API, and freezes
  // them in ensureInit() on the first compile. So a WaveConfig knob that
  // affects codegen has to be pushed into the gflag before that point, which
  // is here -- CompiledKernel::initialize() below is what triggers ensureInit.
  if (WaveConfig::get().kernelLineInfo) {
    FLAGS_cuda_lineinfo = true;
  }
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
  // Non-blocking: wave kernels and the eager standalones on the torch stream
  // are ordered explicitly with events (see newStepEvents and its callers), so
  // the driver's implicit legacy-default-stream serialization is not wanted --
  // it is exactly what prevents the two from overlapping. The step chain only
  // covers what happens inside a run; waitForTorchStream covers entering one.
  g->streamPool = std::make_unique<StreamPool>([]() {
    return std::make_unique<facebook::velox::wave::Stream>(
        /*nonBlocking=*/true);
  });
  g->eventPool = std::make_unique<EventPool>([]() {
    return std::make_unique<facebook::velox::wave::Event>(/*withTime=*/false);
  });
  g->timingEventPool = std::make_unique<EventPool>([]() {
    return std::make_unique<facebook::velox::wave::Event>(/*withTime=*/true);
  });
  // TODO: eager standalones are dispatched to the legacy default stream, which
  // is what every sync in this file already assumes (see
  // syncTorchDefaultStream). If PyTorch is ever built with per-thread default
  // streams, or standalones move onto a pool stream, this has to become a real
  // c10::cuda::getCurrentCUDAStream().stream() query, which needs a small
  // CUDA-configured shim because torch_wave cannot include CUDAStream.h.
  g->torchStreamWrapper = facebook::velox::wave::Stream::external(nullptr);
}

facebook::velox::wave::Stream& torchStream() {
  return *globals()->torchStreamWrapper;
}

void waitForTorchStream(facebook::velox::wave::Stream& stream) {
  auto* g = globals();
  // initialize() bails before creating either when there is no device.
  if (g->eventPool == nullptr || g->torchStreamWrapper == nullptr) {
    return;
  }
  auto event = g->eventPool->get();
  event->record(torchStream());
  event->wait(stream);
  // Safe to recycle right away: cudaStreamWaitEvent captured the event's state
  // at the call above, so a later re-record cannot undo the edge just made.
  event->reset();
  g->eventPool->put(std::move(event));
}

StepEvents& newStepEvents(ExecutionState& state, int32_t seq, int32_t stepIdx) {
  const bool doEventTiming =
      (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto* pool = doEventTiming ? globals()->timingEventPool.get()
                             : globals()->eventPool.get();
  StepEvents events;
  events.sequenceNumber = seq;
  events.stepIdx = stepIdx;
  events.waveDone = pool->get();
  events.standaloneDone = pool->get();
  if (doEventTiming) {
    events.waveBegin = pool->get();
    events.standaloneBegin = pool->get();
  }
  state.stepEvents.push_back(std::move(events));
  return state.stepEvents.back();
}

void releaseStepEvents(ExecutionState& state) {
  const bool doEventTiming =
      (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto* pool = doEventTiming ? globals()->timingEventPool.get()
                             : globals()->eventPool.get();
  auto give = [&](EventP& event) {
    if (event) {
      event->reset();
      pool->put(std::move(event));
    }
  };
  for (auto& events : state.stepEvents) {
    give(events.waveBegin);
    give(events.waveDone);
    give(events.standaloneBegin);
    give(events.standaloneDone);
  }
  state.stepEvents.clear();
  give(state.timelineBase);
  state.lastWaveDone = nullptr;
  state.lastStandaloneDone = nullptr;
  state.syncedCursor = 0;
  state.returnedAtStep.clear();
  state.executedSteps = 0;
}

void tensorsToDevice(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream) {
  // The device tensors below come from the caching allocator, which can hand
  // back a block an eager op on the torch stream is still reading; the copies
  // go out on a non-blocking stream that no longer waits for it implicitly.
  waitForTorchStream(stream);
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
  // 'in' can hold tensors an eager op on the torch stream is still writing.
  waitForTorchStream(stream);
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
      // Per-op GPU attribution needs a sync after each op, which serializes the
      // standalones against each other and against the wave stream. That is the
      // largest single perturbation in the measurement, so it is opt-in; the
      // per-step standaloneBegin/standaloneDone pair measures the same device
      // time without it. Without the knob the recorded number is host dispatch
      // time only.
      //
      // A metadata-only op only manipulates host-side tensor metadata and
      // enqueues nothing on the device, so it never needs the sync. Syncing
      // there would, via the legacy default stream, drain the wave stream and
      // massively over-attribute its time.
      if (!metadataOnly && WaveConfig::get().perOpStandaloneTiming) {
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
  return runInputs(std::move(inputs));
}

std::vector<c10::IValue> WaveGraphExecutor::runInputs(
    std::vector<c10::IValue> inputs) {
  auto pooledFrame = getFrame();
  // Returned on the throwing paths too: a frame that never comes back is gone
  // from the pool for the process's life, so a caller that retries after an
  // error drains it one frame per attempt.
  SCOPE_EXIT {
    returnFrame(std::move(pooledFrame));
  };
  fillUserInputs(*pooledFrame, std::move(inputs));
  return executeWithPrefilledFrame(*pooledFrame);
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
    facebook::velox::wave::Stream* stream,
    ExecutionState* state) {
  // A released value leaves an empty frame slot, which is indistinguishable
  // from one that was never produced. Record it so the coverage report can
  // tell the two apart.
  if (state != nullptr && WaveConfig::get().trace != 0) {
    state->freedValueIds.insert(id);
  }
  if (allocTraceEnabled()) {
    const auto& iv = frame.getIValue(id);
    // Only a solely-owned CUDA storage is donatable: another frame slot or a
    // view holding the same storage means the buffer does not actually become
    // free here. Same ownership test the debugSingleOps poison uses below.
    if (iv.isTensor()) {
      const at::Tensor& t = iv.toTensor();
      if (t.defined() && t.has_storage() && t.is_cuda() && t.use_count() == 1 &&
          t.storage().use_count() == 1) {
        std::cout << "ALLOCEV free " << id << ' '
                  << static_cast<int64_t>(t.storage().nbytes()) << ' '
                  << recordedSizeKey(id) << '\n';
      }
    }
  }
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
  // Offer the buffer to the donation pool before clearing the slot: the slot
  // goes either way, and the pool only accepts storage nothing else references.
  if (state != nullptr) {
    const auto& iv = frame.getIValue(id);
    if (iv.isTensor()) {
      donateFreedTensor(*state, iv.toTensor());
    }
  }
  frame.setIValue(id, c10::IValue());
}

// Pool key: byte size in the high bits, dtype in the low bits. Buffers only
// swap between values of the same width, which is what makes the hit path a
// non-reallocating resize_.
int64_t donationKey(int64_t bytes, c10::ScalarType dtype) {
  return (bytes << 8) | static_cast<int64_t>(dtype);
}

// Donation hands a freed buffer to the next same-size request, which assumes
// one allocation per output. Allocation groups carve many outputs out of one
// buffer, so a group's slots are never solely-owned storages to begin with and
// the pool would only add bookkeeping to a path built to avoid it.
bool donationActive() {
  return WaveConfig::get().donateBuffers && !allocGroupEnabled();
}

bool donateFreedTensor(ExecutionState& state, const at::Tensor& tensor) {
  if (!donationActive()) {
    return false;
  }
  // Sole ownership is the whole safety argument on the donor side: another
  // frame slot or a view over the same storage means someone still reads it.
  if (!tensor.defined() || !tensor.has_storage() || !tensor.is_cuda() ||
      tensor.use_count() != 1 || tensor.storage().use_count() != 1) {
    return false;
  }
  const auto bytes = static_cast<int64_t>(tensor.storage().nbytes());
  if (bytes <= 0) {
    return false;
  }
  const uint64_t start = folly::hardware_timestamp();
  state.donatable[donationKey(bytes, tensor.scalar_type())].push_back(tensor);
  state.donatableBytes += bytes;
  state.donationUs += static_cast<int64_t>(
      static_cast<double>(folly::hardware_timestamp() - start) /
      tscTicksPerMicro());
  return true;
}

at::Tensor takeDonatedTensor(
    ExecutionState& state,
    int64_t bytes,
    c10::ScalarType dtype,
    c10::IntArrayRef dims) {
  // An empty pool is the common case on the first steps of a run; keep that
  // path down to one branch.
  if (!donationActive() || bytes <= 0 || state.donatable.empty()) {
    return {};
  }
  const uint64_t start = folly::hardware_timestamp();
  at::Tensor result;
  // Keyed by byte size AND dtype so a hit is a single resize_ that cannot
  // reallocate (same element count, same width), rather than rebuilding a
  // tensor over the storage -- two dispatcher calls would cost more than the
  // allocation being avoided.
  auto it = state.donatable.find(donationKey(bytes, dtype));
  if (it != state.donatable.end() && !it->second.empty()) {
    result = std::move(it->second.back());
    it->second.pop_back();
    if (it->second.empty()) {
      state.donatable.erase(it);
    }
    state.donatableBytes -= bytes;
    if (result.sizes() != dims) {
      result.resize_(dims);
    }
    ++state.donationHits;
  } else {
    ++state.donationMisses;
  }
  state.donationUs += static_cast<int64_t>(
      static_cast<double>(folly::hardware_timestamp() - start) /
      tscTicksPerMicro());
  return result;
}

void evictDonatable(ExecutionState& state, int64_t limitBytes) {
  if (state.donatableBytes <= limitBytes || state.donatable.empty()) {
    return;
  }
  const uint64_t start = folly::hardware_timestamp();
  // Shed the big buffers first: they are what the ceiling is about, and one
  // pass is enough. Take the largest size present and drop everything at least
  // half that big, rather than sorting the whole table on every miss.
  while (state.donatableBytes > limitBytes && !state.donatable.empty()) {
    int64_t largest = 0;
    for (const auto& [key, buffers] : state.donatable) {
      largest = std::max(largest, key >> 8);
    }
    const int64_t threshold = largest / 2;
    for (auto it = state.donatable.begin(); it != state.donatable.end();) {
      const int64_t sizeBytes = it->first >> 8;
      if (sizeBytes < threshold) {
        ++it;
        continue;
      }
      state.donatableBytes -=
          sizeBytes * static_cast<int64_t>(it->second.size());
      state.donationEvictions += static_cast<int64_t>(it->second.size());
      it = state.donatable.erase(it);
    }
    // A threshold of 0 would leave the table untouched and spin.
    if (threshold == 0) {
      break;
    }
  }
  state.donationUs += static_cast<int64_t>(
      static_cast<double>(folly::hardware_timestamp() - start) /
      tscTicksPerMicro());
}

void clearDonationPool(ExecutionState& state) {
  state.donatable.clear();
  state.donatableBytes = 0;
}

// If a reference frame is loaded, compares the current contents of frame value
// 'id' against the recorded reference just before it is freed. A mismatch means
// the value was already corrupted (a stray write, or an aliased premature free
// of an overlapping buffer) BEFORE this free -- pinpointing which value went
// bad and at which free point, to compare against its intended last use.
void checkValueBeforeFree(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    const WaveGraph* waveGraph) {
  auto* ref = WaveConfig::get().referenceFrame;
  if (ref == nullptr) {
    return;
  }
  // The input of an elided clone is overwritten in place on purpose, so it no
  // longer holds what the reference recorded; comparing it reports a bug that
  // is not there.
  if (waveGraph != nullptr && waveGraph->isElidedCloneInput(id)) {
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
  // A shape-only (meta) tensor carries no data; comparing it would force a
  // .cpu() copy that throws "Cannot copy out of meta tensor". Skip it.
  if (!actual || !actual->defined() || actual->numel() == 0 ||
      actual->is_meta()) {
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

void resolvePendingReturns(ExecutionState& state, int32_t throughStep) {
  while (!state.pendingReturns.empty() &&
         state.pendingReturns.front().executedStep <= throughStep) {
    const auto pending = state.pendingReturns.front();
    state.pendingReturns.pop_front();
    if (pending.waveDone != nullptr) {
      pending.waveDone->wait();
    }
    auto& buffer =
        state.pinnedBuffers.at(pending.sequenceNumber).at(pending.stepIdx);
    TORCH_CHECK(
        buffer != nullptr,
        "The pinned buffer of node ",
        pending.sequenceNumber,
        " step ",
        pending.stepIdx,
        " was released before its deferred return data was read");
    processReturnData(
        state.stepVectors.at(pending.sequenceNumber).at(pending.stepIdx),
        *state.frame,
        buffer->as<uint8_t>());
    ++state.numDeferredReturns;
    state.deferredStepSpan += state.executedSteps - pending.executedStep;
  }
}

void resolveAllPendingReturns(ExecutionState& state) {
  resolvePendingReturns(state, std::numeric_limits<int32_t>::max());
}

namespace {

// Bytes a frame value would give back, from the view's own extent rather than
// its whole storage: a view and its base are separate ids over one buffer, so
// charging each the full storage over-states the total by more than an order of
// magnitude. numel/element_size are inline accessors -- no storage access, no
// hash lookup.
int64_t frameValueBytes(nativert::ExecutionFrame& frame, nativert::ValueId id) {
  const auto& ivalue = frame.getIValue(id);
  if (!ivalue.isTensor()) {
    return 0;
  }
  const auto& tensor = ivalue.toTensor();
  return tensor.defined() ? tensor.numel() * tensor.element_size() : 0;
}

// Takes a swept step's stamped bytes back off the running total. A plain
// subtraction of what the step accumulated -- the per-id sizes are already
// summed, so nothing is looked up again.
void releaseDelayedFreeBytes(ExecutionState& state, StepVectors& sv) {
  state.delayedFreeBytes -= sv.lastUseBytes;
  sv.lastUseBytes = 0;
  if (state.delayedFreeBytes < 0) {
    state.delayedFreeBytes = 0;
  }
}

} // namespace

// Charges the enclosing scope to ExecutionState::freeUs. Releasing a frame
// value can hand memory back to the caching allocator, which is the same kind
// of cost as allocating it, so the report keeps the two together. On the TSC:
// this fires once per swept step, and chrono here is kvm-clock at tens of
// microseconds a call.
namespace {
class ScopedFreeTimer {
 public:
  explicit ScopedFreeTimer(ExecutionState& state)
      : state_(state),
        timing_(
            WaveConfig::get().printTiming ||
            (WaveConfig::get().trace & WaveConfig::kTiming)),
        start_(timing_ ? folly::hardware_timestamp() : 0) {}

  // Scope guard: the reference and const members already make it
  // non-assignable, so say so rather than leave it to the reader.
  ScopedFreeTimer(const ScopedFreeTimer&) = delete;
  ScopedFreeTimer(ScopedFreeTimer&&) = delete;
  ScopedFreeTimer& operator=(const ScopedFreeTimer&) = delete;
  ScopedFreeTimer& operator=(ScopedFreeTimer&&) = delete;

  ~ScopedFreeTimer() {
    if (timing_) {
      state_.freeUs += static_cast<int64_t>(
          static_cast<double>(folly::hardware_timestamp() - start_) /
          tscTicksPerMicro());
    }
  }

 private:
  ExecutionState& state_;
  const bool timing_;
  const uint64_t start_;
};

// Returns the step's vectors, or nullptr if either index is out of range.
// Launch metadata carries indices that can outlive the vectors they refer to,
// so every reader has to bounds-check; doing it here also covers a negative
// index, which the open-coded checks did not.
StepVectors* FOLLY_NULLABLE
findStepVectors(ExecutionState& state, int32_t seq, int32_t step) {
  if (seq < 0 || seq >= static_cast<int32_t>(state.stepVectors.size())) {
    return nullptr;
  }
  auto& steps = state.stepVectors[seq];
  if (step < 0 || step >= static_cast<int32_t>(steps.size())) {
    return nullptr;
  }
  return &steps[step];
}
} // namespace

void addLastUseId(
    ExecutionState& state,
    StepVectors& sv,
    nativert::ValueId id) {
  sv.lastUseIds.push_back(id);
  if (WaveConfig::get().maxDelayedFree <= 0) {
    return;
  }
  const auto bytes = frameValueBytes(*state.frame, id);
  sv.lastUseBytes += bytes;
  state.delayedFreeBytes += bytes;
  if (state.delayedFreeBytes > state.maxDelayedFreeSeen) {
    state.maxDelayedFreeSeen = state.delayedFreeBytes;
  }
}

bool enforceDelayedFreeLimit(ExecutionState& state) {
  const auto limit = WaveConfig::get().maxDelayedFree;
  if (limit <= 0 || state.delayedFreeBytes <= limit) {
    return false;
  }
  syncTorchDefaultStream();
  syncWaveStream(state);
  ++state.numMemoryStalls;
  return true;
}

void sampleRunAhead(ExecutionState& state) {
  if (!(WaveConfig::get().trace & WaveConfig::kTiming)) {
    return;
  }
  // Walk back from the newest step and stop at the first one the device has
  // finished: steps complete in order, so the run of incomplete ones at the end
  // is the depth. query() does not block.
  auto inFlight = [](const EventP& event) {
    return event && event->recorded() && !event->query();
  };
  int32_t depth = 0;
  for (auto it = state.stepEvents.rbegin(); it != state.stepEvents.rend();
       ++it) {
    if (!inFlight(it->waveDone) && !inFlight(it->standaloneDone)) {
      break;
    }
    ++depth;
  }
  state.runAheadSum += depth;
  ++state.runAheadSamples;
  state.runAheadMax = std::max(state.runAheadMax, depth);
  if (depth == 0) {
    ++state.numDrainedStarts;
  }
}

namespace {

// True if a still-pending transfer produced at or before 'throughStep' brings
// back a value in 'ids'. Those are the only ones a sweep about to free 'ids'
// has to resolve first: parsing into a frame slot that has already been cleared
// reads a None. Every other transfer can stay in flight past the sweep, which
// is the point -- the sweep runs after every launch, so resolving
// unconditionally would end the deferral about two steps after it started
// however far away the real consumer is.
bool pendingReturnFreedBy(
    const ExecutionState& state,
    int32_t throughStep,
    const std::vector<nativert::ValueId>& ids) {
  for (const auto& pending : state.pendingReturns) {
    if (pending.executedStep > throughStep) {
      return false;
    }
    const auto& producer =
        state.stepVectors.at(pending.sequenceNumber).at(pending.stepIdx);
    for (const auto& data : producer.kernels) {
      for (auto id : data.returnValues) {
        if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

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
          useCount = static_cast<int64_t>(st.use_count());
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
      // A value this step sent back can also be one it last-uses, and parsing
      // the pinned buffer into a frame slot that has just been cleared would
      // read a None. The wave stream was waited on above, so the transfer has
      // landed and this only does the copy.
      if (pendingReturnFreedBy(state, sv.executedStep, sv.lastUseIds)) {
        resolvePendingReturns(state, sv.executedStep);
      }
      releaseDelayedFreeBytes(state, sv);
      // The node's last-use tensors were stamped onto its last step; they go
      // free in this same sync (no dedicated sync of their own).
      ScopedFreeTimer freeTimer(state);
      for (auto id : sv.lastUseIds) {
        traceFree(id, "lastUse", seq, step);
        checkValueBeforeFree(frame, id, state.waveGraph);
        freeFrameValue(frame, id, state.stream.get(), &state);
      }
    }
  }
}

void syncWaveStream(ExecutionState& state) {
  state.stream->wait();
  advanceSyncedStages(state);
}

void freeLastUseNow(
    ExecutionState& state,
    const std::vector<nativert::ValueId>& ids) {
  auto& frame = *state.frame;
  ScopedFreeTimer freeTimer(state);
  for (auto id : ids) {
    checkValueBeforeFree(frame, id, state.waveGraph);
    freeFrameValue(frame, id, state.stream.get(), &state);
  }
}

void advanceCompletedStages(ExecutionState& state) {
  if (!WaveConfig::get().freeIntermediates) {
    return;
  }
  // The debug paths in advanceSyncedStages force full syncs on both streams
  // before freeing; keep using that stricter version there rather than the
  // query sweep, so debug modes stay maximally serialized.
  if (WaveConfig::get().debugSingleOps ||
      WaveConfig::get().referenceFrame != nullptr) {
    return;
  }
  // A step can be declared synced only once the host knows its work is done.
  // The event edges give the GPU its ordering but tell the host nothing, so ask
  // the step's own events; query() does not block.
  //
  // Resume at the oldest step not yet synced and stop at the first that is not
  // done. Steps complete in order: each stream records its events in step
  // order, and the cross-stream edges make a step's work wait on the previous
  // step's work on the other stream, so the later of a step's two completion
  // events is monotone in the step index. Without the early exit this rescans
  // every step of the run on every launch, which is quadratic in the step
  // count -- a few thousand cudaEventQuery calls per run on the ROO graph.
  auto& frame = *state.frame;
  auto pending = [](const EventP& event) {
    return event && event->recorded() && !event->query();
  };
  while (state.syncedCursor < state.stepEvents.size()) {
    // The loop condition above bounds syncedCursor.
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    const auto& events = state.stepEvents[state.syncedCursor];
    if (pending(events.waveDone) || pending(events.standaloneDone)) {
      return;
    }
    auto* svPtr = findStepVectors(state, events.sequenceNumber, events.stepIdx);
    if (svPtr == nullptr) {
      ++state.syncedCursor;
      continue;
    }
    auto& sv = *svPtr;
    // A blocking sync (syncWaveStream, and so every syncEachStep step) already
    // ran advanceSyncedStages over this step. Nothing is left to free, so skip
    // it rather than parking the cursor on it for the rest of the run.
    if (sv.executionStage == ExecutionStage::kSynced) {
      ++state.syncedCursor;
      continue;
    }
    // The step being issued right now already has its events but is not marked
    // kAllocated until its outputs exist. Stop rather than step over it, or its
    // lastUseIds would never be freed.
    if (sv.executionStage != ExecutionStage::kAllocated) {
      return;
    }
    ++state.syncedCursor;
    sv.executionStage = ExecutionStage::kSynced;
    // Same reason as in advanceSyncedStages: parse before freeing. This step's
    // events have just been observed complete, so its transfer has landed and
    // resolving up to it does not block.
    if (pendingReturnFreedBy(state, sv.executedStep, sv.lastUseIds)) {
      resolvePendingReturns(state, sv.executedStep);
    }
    releaseDelayedFreeBytes(state, sv);
    {
      ScopedFreeTimer freeTimer(state);
      for (auto id : sv.lastUseIds) {
        checkValueBeforeFree(frame, id, state.waveGraph);
        freeFrameValue(frame, id, state.stream.get(), &state);
      }
    }
  }
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
  auto*& threadConfigOverride = torch::wave::waveConfigOverride();
  auto* prevConfigOverride = threadConfigOverride;
  threadConfigOverride = waveGraph.configOverride();
  SCOPE_EXIT {
    threadWaveGraph = prevWaveGraph;
    threadConfigOverride = prevConfigOverride;
  };

  // When the caller asserts all model inputs, weights, and constants are
  // contiguous (WaveConfig::inputContiguous), the optimizer marked those
  // producer-less values contiguous and downstream passes may rely on it.
  // Verify the assumption here and fail loudly rather than produce silently
  // wrong results.
  if (WaveConfig::get().inputContiguous) {
    for (const auto* value : graph_.values()) {
      if (value == nullptr || value->producer() != nullptr) {
        continue;
      }
      const auto& iv = frame.getIValue(value->id());
      if (iv.isTensor()) {
        const auto& tensor = iv.toTensor();
        TORCH_CHECK(
            tensor.is_contiguous(),
            "input_contiguous is set but producer-less value %",
            value->id(),
            " (",
            value->name(),
            ") is not contiguous");
      }
    }
  }

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
  // The per-step event chain below starts empty, so nothing orders this run
  // against what the caller left on the torch stream: the eager ops that
  // produced the inputs, and the buffers they still hold, which the caching
  // allocator is free to hand this run's first outputs. A blocking wave stream
  // used to cover both implicitly.
  waitForTorchStream(*state.stream);
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

  // Per-step GPU timeline events. releaseStepEvents pools them on the normal
  // path, after the final stream syncs, so none is recycled while still
  // pending. This guard is the exception path: those syncs may not have run, so
  // just destroy the events -- cudaEventDestroy on a pending event is defined
  // (the resources are freed once the device reaches it), only re-recording one
  // would be wrong.
  state.stepEvents.clear();
  state.timelineBase.reset();
  state.lastWaveDone = nullptr;
  state.lastStandaloneDone = nullptr;
  state.syncedCursor = 0;
  state.returnedAtStep.clear();
  state.executedSteps = 0;
  state.pendingReturns.clear();
  SCOPE_EXIT {
    state.stepEvents.clear();
    state.timelineBase.reset();
    state.lastWaveDone = nullptr;
    state.lastStandaloneDone = nullptr;
    state.syncedCursor = 0;
    state.returnedAtStep.clear();
    state.executedSteps = 0;
    // Exception path only: on the normal path everything has been parsed
    // below. Dropping an entry here is safe -- the frame is abandoned with the
    // exception -- whereas parsing one whose transfer may still be in flight
    // is not.
    state.pendingReturns.clear();
  };
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    state.timelineBase = globals()->timingEventPool->get();
    state.timelineBase->record(*state.stream);
  }

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
  state.numLastUseEarly = 0;
  state.numLastUseAtNodeEnd = 0;
  state.numDeferredReturns = 0;
  state.deferredStepSpan = 0;
  state.runAheadSum = 0;
  state.runAheadSamples = 0;
  state.runAheadMax = 0;
  state.numDrainedStarts = 0;
  state.delayedFreeBytes = 0;
  state.numMemoryStalls = 0;
  state.maxDelayedFreeSeen = 0;
  state.numDeferredOps = 0;
  state.numDeferredSteps = 0;
  state.numGridRedos = 0;
  state.freeUs = 0;
  // donatableBytes is deliberately not reset: the pool outlives the execution,
  // so the byte accounting has to stay attached to what it actually holds.
  state.donationHits = 0;
  state.donationMisses = 0;
  state.donationEvictions = 0;
  state.donationUs = 0;
  state.freedValueIds.clear();
  state.releasedAtStep.clear();
  // Runs after the closing stream syncs on the normal path, and on any throw.
  // The pool is kept across executions -- the next run of this graph allocates
  // the same shapes, so a warm pool is where the hit rate comes from -- but it
  // is trimmed to the ceiling here so the memory it holds between runs stays
  // bounded. Whatever survives is released when the ExecutionState dies.
  SCOPE_EXIT {
    const auto cap = WaveConfig::get().donationCarryBytes;
    if (cap >= 0) {
      evictDonatable(state, cap);
    }
  };
  if (WaveConfig::get().freeIntermediates) {
    for (auto& steps : state.stepVectors) {
      for (auto& sv : steps) {
        sv.executionStage = ExecutionStage::kNotStarted;
        sv.lastUseIds.clear();
        sv.lastUseBytes = 0;
      }
    }
  }
  for (const auto& node : waveGraph.nodes()) {
    node->execute(state);
  }

  // The last steps' transfers may still be outstanding under
  // WaveConfig::deferD2h. Everything below reads the frame, so parse them now.
  resolveAllPendingReturns(state);

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
      // Coverage is counted from the compiled structure, not from what is
      // left in the frame. An empty frame slot proves nothing: an op fused
      // into a kernel produces an internal intermediate that never gets a
      // frame value at all, and freeIntermediates reclaims most of the ones
      // that do. Inferring coverage from isNone() reported 3720 of 4697 ROO
      // nodes as uncovered when nearly all of them had run.
      auto& graph = *waveGraph.graph();
      folly::F14FastSet<NodeCP> fused;
      for (const auto& composite : waveGraph.nodes()) {
        for (const auto& op : composite->kernels()->ops()) {
          const auto& nodeMap = op.nodeMap();
          // Union over the grid variants: which one runs is a runtime choice,
          // but every variant covers the same nodes.
          auto* projectOp = op.projectOp();
          for (auto* grid :
               {&projectOp->grid(),
                &projectOp->singleBlockGrid(),
                &projectOp->cgGrid()}) {
            for (const auto& step : *grid) {
              for (const auto& launch : step) {
                if (launch.op == nullptr) {
                  continue;
                }
                for (auto* formal : launch.op->allNodes()) {
                  auto it = nodeMap.find(formal);
                  fused.insert(it != nodeMap.end() ? it->second : formal);
                }
              }
            }
          }
        }
      }
      auto totalNodes = static_cast<int64_t>(graph.nodes().size());
      auto numComposites = waveGraph.nodes().size();
      int64_t numStandalones = state.numStandalonesRun;
      int64_t numShortcuts = state.numShortcutsRun;
      auto fusedNodes = static_cast<int64_t>(fused.size());
      // prim.Input / prim.Output are graph plumbing, not work to cover.
      int64_t plumbing = 0;
      for (auto& gnode : graph.nodes()) {
        if (gnode.target() == "prim.Input" || gnode.target() == "prim.Output") {
          ++plumbing;
        }
      }
      const int64_t coverable = totalNodes - plumbing;
      int64_t uncovered =
          coverable - fusedNodes - numStandalones - numShortcuts;
      std::cout << "FUSION: nativert_graph_nodes=" << totalNodes
                << " coverable=" << coverable
                << " wave_composite_kernels=" << numComposites
                << " fused_nodes=" << fusedNodes
                << " standalone_ops=" << numStandalones
                << " shortcut_ops=" << numShortcuts
                << " uncovered_ops=" << uncovered << " (~"
                << (100.0 * static_cast<double>(fusedNodes) /
                    static_cast<double>(coverable))
                << "% fused, ~"
                << (100.0 * static_cast<double>(uncovered) /
                    static_cast<double>(coverable))
                << "% uncovered)" << std::endl;
    }
  }
  // Sync the wave stream and the PyTorch default stream: eager standalone ops
  // run on the default stream while fused kernels run on the wave stream, and
  // the two are otherwise unordered. Both must complete before executeWave
  // returns so all results this invocation produced are visible to the
  // caller. Default stream first: syncWaveStream releases frame values off a
  // wave-stream wait alone, and an eager standalone still in flight can be
  // reading one of them (same order as enforceDelayedFreeLimit).
  syncTorchDefaultStream();
  syncWaveStream(state);
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
  // Both streams have been waited on and collectDebugInfo has read the elapsed
  // times, so the events can go back to their pools for the next run.
  releaseStepEvents(state);
}

const WaveThreadInfo& waveThreadInfo() {
  return threadInfo;
}

bool allocTraceEnabled() {
  static const bool enabled = std::getenv("TW_ALLOC_TRACE") != nullptr;
  return enabled;
}

void logAllocEvent(const char* kind, int32_t valueId, int64_t bytes) {
  std::cout << "ALLOCEV " << kind << ' ' << valueId << ' ' << bytes << " ?\n";
}

namespace {
folly::F14FastMap<int32_t, std::string>& sizeKeys() {
  static thread_local folly::F14FastMap<int32_t, std::string> keys;
  return keys;
}
} // namespace

void logKeyedAllocEvent(
    const char* kind,
    int32_t valueId,
    int64_t bytes,
    const std::string& sizeKey) {
  sizeKeys()[valueId] = sizeKey;
  std::cout << "ALLOCEV " << kind << ' ' << valueId << ' ' << bytes << ' '
            << sizeKey << '\n';
}

const std::string& recordedSizeKey(int32_t valueId) {
  static const std::string kUnknown = "?";
  auto it = sizeKeys().find(valueId);
  return it != sizeKeys().end() ? it->second : kUnknown;
}

namespace {

// Walks the run's step events in execution order and fills each step's
// device-measured spans and the idle that preceded it. All timestamps are read
// as offsets from state.timelineBase, which puts the wave stream and the torch
// stream on one comparable timeline (cudaEventElapsedTime across streams is
// valid -- the timestamps are device-global).
void computeGpuTimeline(ExecutionState& state) {
  if (!state.timelineBase || !state.timelineBase->recorded()) {
    return;
  }
  auto us = [&](const EventP& event) -> std::optional<double> {
    if (!event || !event->recorded()) {
      return std::nullopt;
    }
    return event->elapsedTime(*state.timelineBase) * 1000.0;
  };
  // End of the previous step's GPU work. A step with no device work at all
  // (shortcut-only) has neither bound, so it is skipped and this carries
  // forward rather than resetting.
  std::optional<double> prevEnd;
  for (const auto& events : state.stepEvents) {
    auto waveBegin = us(events.waveBegin);
    auto waveDone = us(events.waveDone);
    auto standaloneBegin = us(events.standaloneBegin);
    auto standaloneDone = us(events.standaloneDone);

    std::optional<double> start;
    std::optional<double> end;
    for (const auto& candidate : {waveBegin, standaloneBegin}) {
      if (candidate && (!start || *candidate < *start)) {
        start = candidate;
      }
    }
    for (const auto& candidate : {waveDone, standaloneDone}) {
      if (candidate && (!end || *candidate > *end)) {
        end = candidate;
      }
    }
    if (!start || !end) {
      continue;
    }
    auto* svPtr = findStepVectors(state, events.sequenceNumber, events.stepIdx);
    if (svPtr == nullptr) {
      continue;
    }
    auto& sv = *svPtr;
    if (waveBegin && waveDone) {
      sv.kernelGpuUs = static_cast<int64_t>(*waveDone - *waveBegin);
    }
    if (standaloneBegin && standaloneDone) {
      sv.standaloneGpuUs =
          static_cast<int64_t>(*standaloneDone - *standaloneBegin);
    }
    sv.gpuIdleUs =
        prevEnd ? static_cast<int64_t>(std::max(0.0, *start - *prevEnd)) : 0;
    prevEnd = end;
  }
}

} // namespace

namespace {

// Records how many thread-block clocks one unit of the step's modelled cost
// bought, so the next execution of the same step can express a minimum block
// size in microseconds rather than in cost units. Total block time over total
// modelled work: both scale with the step's size, so the ratio survives a
// change of batch.
void recordStepClockRate(ExecutionState& state, const LaunchDebugInfo& info) {
  if (info.pinnedInfo == nullptr || info.numBlocks <= 0 ||
      info.sequenceNumber < 0 ||
      info.sequenceNumber >= static_cast<int32_t>(state.stepVectors.size())) {
    return;
  }
  auto& steps = state.stepVectors.at(info.sequenceNumber);
  if (info.stepIdx < 0 || info.stepIdx >= static_cast<int32_t>(steps.size())) {
    return;
  }
  auto& sv = steps.at(info.stepIdx);
  double totalCost = 0;
  for (auto cost : sv.costs) {
    totalCost += cost;
  }
  if (totalCost <= 0) {
    return;
  }
  int64_t totalClocks = 0;
  int64_t maxClocks = 0;
  for (int32_t block = 0; block < info.numBlocks; ++block) {
    const auto clocks = info.pinnedInfo[block].clocks;
    totalClocks += clocks;
    maxClocks = std::max(maxClocks, clocks);
  }
  if (totalClocks > 0) {
    sv.clocksPerCost = static_cast<double>(totalClocks) / totalCost;
  }
  // Same figure the balance line prints: mean block clocks over the slowest.
  if (maxClocks > 0) {
    sv.measuredUtil = static_cast<double>(totalClocks) /
        (static_cast<double>(maxClocks) * info.numBlocks);
    sv.measuredBlocks = info.numBlocks;
  }
}

} // namespace

void WaveGraphExecutor::collectDebugInfo(ExecutionState& state) {
  auto& infos = state.launchDebugInfos;
  threadInfo.debugInfo.clear();
  threadInfo.launchMeta.clear();
  threadInfo.gpuIdleUs = 0;
  threadInfo.numLastUseEarly = state.numLastUseEarly;
  threadInfo.numLastUseAtNodeEnd = state.numLastUseAtNodeEnd;
  threadInfo.numDeferredReturns = state.numDeferredReturns;
  threadInfo.deferredStepSpan = state.deferredStepSpan;
  threadInfo.runAheadSum = state.runAheadSum;
  threadInfo.runAheadSamples = state.runAheadSamples;
  threadInfo.runAheadMax = state.runAheadMax;
  threadInfo.numDrainedStarts = state.numDrainedStarts;
  threadInfo.numMemoryStalls = state.numMemoryStalls;
  threadInfo.maxDelayedFreeSeen = state.maxDelayedFreeSeen;
  threadInfo.numDeferredOps = state.numDeferredOps;
  threadInfo.numDeferredSteps = state.numDeferredSteps;
  threadInfo.numGridRedos = state.numGridRedos;
  threadInfo.freeUs = state.freeUs;
  threadInfo.donationHits = state.donationHits;
  threadInfo.donationMisses = state.donationMisses;
  threadInfo.donationEvictions = state.donationEvictions;
  threadInfo.donationUs = state.donationUs;
  if (WaveConfig::get().trace & WaveConfig::kTiming) {
    computeGpuTimeline(state);
  }
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
    recordStepClockRate(state, info);
    LaunchMeta meta;
    meta.sequenceNumber = info.sequenceNumber;
    meta.stepIdx = info.stepIdx;
    meta.numBlocks = info.numBlocks;
    if (WaveConfig::get().trace & WaveConfig::kTiming) {
      auto* svPtr = findStepVectors(state, info.sequenceNumber, info.stepIdx);
      if (svPtr != nullptr) {
        auto& sv = *svPtr;
        meta.gatherUs = sv.gatherUs;
        meta.gridUs = sv.gridUs;
        meta.allocUs = sv.allocUs;
        meta.allocCallUs = sv.allocCallUs;
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
        meta.elidedCloneBytes = sv.elidedCloneBytes;
        meta.allocGroups = sv.allocGroups;
        meta.allocGroupTensors = sv.allocGroupTensors;
        meta.kernelGpuUs = sv.kernelGpuUs;
        meta.standaloneGpuUs = sv.standaloneGpuUs;
        meta.gpuIdleUs = sv.gpuIdleUs;
        threadInfo.gpuIdleUs += sv.gpuIdleUs;
        meta.d2hDepFused = sv.d2hDepFused;
        meta.d2hDepStandalone = sv.d2hDepStandalone;
        meta.d2hDepShortcut = sv.d2hDepShortcut;
        meta.d2hDepOnPrevStep = sv.d2hDepOnPrevStep;
        meta.d2hNearestProducer = sv.d2hNearestProducer;
        meta.viewNodeDescs = sv.viewNodeDescs;
        meta.numFused = static_cast<int32_t>(sv.kernels.size());
        meta.numStandalone = static_cast<int32_t>(sv.standalones.size());
        meta.numShortcut = static_cast<int32_t>(sv.shortcutStandalones.size());
        meta.gridStats = sv.gridStats;
        meta.segments = sv.segments;
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

    // Sum clocks per launch through launchIndices rather than opcode matching.
    // The blocks of one launch are not necessarily a contiguous run: the emit
    // order follows projected latency, and a partitioned step splits an op
    // across launches.
    std::vector<int64_t> launchClocks(sv.kernels.size(), 0);
    for (size_t b = 0; b < debugBlocks.size() && b < sv.launchIndices.size();
         ++b) {
      const auto launchIdx = sv.launchIndices[b];
      if (launchIdx >= 0 &&
          launchIdx < static_cast<int32_t>(launchClocks.size())) {
        launchClocks[launchIdx] += debugBlocks[b].clocks;
      }
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
    double inputGBs = static_cast<double>(totalInputBytes) / (wallSec * 1e9);
    ss << fmt::format(
        "Input throughput: {:.2f} GB/s ({:.1f} MB input)\n",
        inputGBs,
        static_cast<double>(totalInputBytes) / 1e6);
  }
  if (wallUs > 0 && totalDataBytes > 0) {
    double dataGBs = static_cast<double>(totalDataBytes) / (wallSec * 1e9);
    ss << fmt::format(
        "Internal throughput: {:.2f} GB/s ({:.1f} MB total data)\n",
        dataGBs,
        static_cast<double>(totalDataBytes) / 1e6);
  }
  ss << fmt::format(
      "Peak GPU RAM: {}\n", facebook::velox::succinctBytes(info.peakBytes));
  if (WaveConfig::get().freeIntermediates) {
    ss << fmt::format(
        "Last-use release: {} values, {} before their node's last step\n",
        info.numLastUseEarly + info.numLastUseAtNodeEnd,
        info.numLastUseEarly);
  }
  if (info.runAheadSamples > 0) {
    // Steps that start with the queue already drained have their whole
    // interpretation exposed as idle, so this is the direct measure of how much
    // room deeper run-ahead has.
    ss << fmt::format(
        "Host run-ahead: mean {:.2f} steps in flight, max {}, {} of {} steps "
        "started with the queue drained\n",
        static_cast<double>(info.runAheadSum) / info.runAheadSamples,
        info.runAheadMax,
        info.numDrainedStarts,
        info.runAheadSamples);
  }
  if (info.numGridRedos > 0) {
    ss << fmt::format(
        "Grid redos: {} steps changed variant and redid their setup pass\n",
        info.numGridRedos);
  }
  {
    // What the grouping actually bought: every group is one allocator call in
    // place of one per tensor it covers, so the saving is the difference.
    int64_t groups = 0;
    int64_t tensors = 0;
    for (const auto& m : info.launchMeta) {
      groups += m.allocGroups;
      tensors += m.allocGroupTensors;
    }
    if (groups > 0) {
      ss << fmt::format(
          "Alloc groups: {} groups over {} tensors, {} fewer allocator calls\n",
          groups,
          tensors,
          tensors - groups);
    }
  }
  if (info.numDeferredSteps > 0) {
    ss << fmt::format(
        "Deferred setup: {} ops over {} steps needed a second pass\n",
        info.numDeferredOps,
        info.numDeferredSteps);
  }
  if (info.maxDelayedFreeSeen > 0) {
    // The high-water mark is what the run-ahead adds to the peak; the stall
    // count is how often the ceiling had to claw it back.
    ss << fmt::format(
        "Delayed frees: peak {} held by in-flight steps, {} drains forced by "
        "the {} ceiling\n",
        facebook::velox::succinctBytes(info.maxDelayedFreeSeen),
        info.numMemoryStalls,
        facebook::velox::succinctBytes(WaveConfig::get().maxDelayedFree));
  }
  if (info.numDeferredReturns > 0) {
    // A span of 1 is a transfer the very next step consumed, which is what
    // waiting at the producer already gave; the excess over 1 is the run-ahead
    // WaveConfig::deferD2h actually bought.
    ss << fmt::format(
        "Deferred D2H: {} transfers, {:.1f} steps in flight on average\n",
        info.numDeferredReturns,
        static_cast<double>(info.deferredStepSpan) / info.numDeferredReturns);
  }

  // Time split across all steps. Kernel and standalone are device-measured from
  // each step's event pair; GPU idle is the summed gap between one step's GPU
  // work ending and the next step's starting, i.e. time the device had nothing
  // to do because the host was still interpreting. Interpretation is the host
  // side of that (gather + grid + alloc + fill).
  //
  // A step with no events falls back to the old estimate: for a standalone-
  // bound step the host-measured kernelUs reflects waiting on the standalone
  // work rather than on the GPU, so approximate from the step's max
  // thread-block clocks (1 clock ~= 0.7 ns).
  {
    double kernelUs = 0.0;
    int64_t standaloneUs = 0;
    int64_t interpUs = 0;
    int64_t allocUs = 0;
    int64_t allocCallUs = 0;
    int64_t gpuIdleUs = 0;
    for (size_t i = 0; i < info.launchMeta.size(); ++i) {
      const auto& m = info.launchMeta[i];
      interpUs += m.gatherUs + m.gridUs + m.allocUs + m.fillUs;
      allocUs += m.allocUs;
      allocCallUs += m.allocCallUs;
      gpuIdleUs += m.gpuIdleUs;
      if (m.standaloneGpuUs > 0) {
        standaloneUs += m.standaloneGpuUs;
      } else {
        standaloneUs += m.standaloneUs + m.shortcutUs;
      }
      if (m.kernelGpuUs > 0) {
        kernelUs += static_cast<double>(m.kernelGpuUs);
      } else if (m.standaloneBound) {
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
        "Kernel time: {:.0f} us  Standalone time: {} us  GPU idle: {} us  Interpretation time: {} us  Free time: {} us\n",
        kernelUs,
        standaloneUs,
        gpuIdleUs,
        interpUs,
        info.freeUs);
    // The allocation phase split in two, because the halves are fixed by
    // different things: the calls by allocating fewer and larger buffers (or a
    // bigger arena), the setup by cheaper shape arithmetic and view building.
    ss << fmt::format(
        "  of which allocation: {} us total = {} us in allocator calls + {} us computing sizes and building views\n",
        allocUs,
        allocCallUs,
        allocUs - allocCallUs);
    if (info.donationHits + info.donationMisses > 0) {
      ss << fmt::format(
          "Buffer donation: {} hits, {} misses ({:.0f}% of kernel allocations), {} evicted, {} us in the pool\n",
          info.donationHits,
          info.donationMisses,
          100.0 * static_cast<double>(info.donationHits) /
              static_cast<double>(info.donationHits + info.donationMisses),
          info.donationEvictions,
          info.donationUs);
    }
    if (WaveConfig::get().perOpStandaloneTiming) {
      ss << "WARNING per-op standalone timing is on: it syncs after every eager "
            "op, which serializes the streams and inflates idle.\n";
    }
  }

  // How much of each step is genuinely blocked on an earlier step's D2H. The
  // producing step waits for the transfer today; these counts say how much of
  // the work that follows would really have had to wait for it.
  {
    int32_t stepsWithDep = 0;
    int32_t stepsDepOnPrev = 0;
    int32_t depFused = 0, depStandalone = 0, depShortcut = 0;
    int32_t totFused = 0, totStandalone = 0, totShortcut = 0;
    for (const auto& m : info.launchMeta) {
      totFused += m.numFused;
      totStandalone += m.numStandalone;
      totShortcut += m.numShortcut;
      depFused += m.d2hDepFused;
      depStandalone += m.d2hDepStandalone;
      depShortcut += m.d2hDepShortcut;
      if (m.d2hDepFused + m.d2hDepStandalone + m.d2hDepShortcut > 0) {
        ++stepsWithDep;
        if (m.d2hDepOnPrevStep > 0) {
          ++stepsDepOnPrev;
        }
      }
    }
    ss << fmt::format(
        "D2H dependencies: {} of {} steps read a value an earlier step "
        "returned; {} of those from the immediately preceding step\n",
        stepsWithDep,
        info.launchMeta.size(),
        stepsDepOnPrev);
    ss << fmt::format(
        "  dependent ops: fused {}/{}  standalone {}/{}  shortcut {}/{}\n",
        depFused,
        totFused,
        depStandalone,
        totStandalone,
        depShortcut,
        totShortcut);
    int32_t viewDescs = 0;
    for (const auto& m : info.launchMeta) {
      viewDescs += m.viewNodeDescs;
    }
    ss << fmt::format(
        "  fused outputs built by a host-side view: {}\n", viewDescs);
  }
  // How far each step's grid is from a balanced, fully occupied wave, and
  // which of the two pathologies it suffers from. They need different fixes:
  // starvation is ops stuck on one block once a step has as many ops as the
  // wave has blocks, poisoning is one shared-memory-hungry op cutting the
  // occupancy of a step whose work sits elsewhere.
  {
    const std::array<float, 4> kSkewBuckets{1.1f, 1.5f, 2.0f, 4.0f};
    std::array<int32_t, 5> skewHistogram{};
    int32_t measuredSteps = 0;
    int32_t starvedSteps = 0;
    int64_t starvedOps = 0;
    int32_t poisonedSteps = 0;
    int32_t splitSteps = 0;
    float worstSkew = 0;
    int32_t worstSeq = -1;
    int32_t worstStep = -1;
    for (const auto& m : info.launchMeta) {
      const auto& stats = m.gridStats;
      if (stats.numOps == 0) {
        continue;
      }
      ++measuredSteps;
      size_t bucket = 0;
      while (bucket < kSkewBuckets.size() &&
             stats.skew >= kSkewBuckets[bucket]) {
        ++bucket;
      }
      ++skewHistogram[bucket];
      if (stats.numStarved > 0) {
        ++starvedSteps;
        starvedOps += stats.numStarved;
      }
      if (stats.occupancy < stats.bestOccupancy) {
        ++poisonedSteps;
      }
      if (stats.numSegments > 1) {
        ++splitSteps;
      }
      if (stats.skew > worstSkew) {
        worstSkew = stats.skew;
        worstSeq = m.sequenceNumber;
        worstStep = m.stepIdx;
      }
    }
    if (measuredSteps > 0) {
      ss << fmt::format(
          "Grid balance: {} steps; skew <1.1 {}, <1.5 {}, <2 {}, <4 {}, >=4 {}; worst {:.1f} at node {} step {}\n",
          measuredSteps,
          skewHistogram[0],
          skewHistogram[1],
          skewHistogram[2],
          skewHistogram[3],
          skewHistogram[4],
          worstSkew,
          worstSeq,
          worstStep);
      ss << fmt::format(
          "  block starvation: {} steps, {} ops left on one block that could use more; shared-memory poisoning: {} steps; split into several launches: {}\n",
          starvedSteps,
          starvedOps,
          poisonedSteps,
          splitSteps);
    }
  }
  ss << "WaveConfig: " << WaveConfig::get().toString() << "\n";

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
      if (m.kernelGpuUs > 0 || m.standaloneGpuUs > 0) {
        // Device-measured: the step's wall is the host interpretation plus the
        // longer of the two device spans plus the idle that preceded it.
        nodeUs +=
            interp + std::max(m.kernelGpuUs, m.standaloneGpuUs) + m.gpuIdleUs;
      } else {
        int64_t hostStandalone = m.standaloneUs + m.shortcutUs;
        nodeUs += interp + std::max(m.kernelUs, hostStandalone);
      }
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
        if (const auto* stepVectors =
                findStepVectors(state, m.sequenceNumber, m.stepIdx)) {
          numStandalones = static_cast<int32_t>(
              stepVectors->standalones.size() +
              stepVectors->shortcutStandalones.size());
        }
        ss << fmt::format(
            "  step {}: {} standalones  GPU RAM={}  {} us",
            m.stepIdx,
            numStandalones,
            facebook::velox::succinctBytes(m.currentBytes),
            m.standaloneUs);
        // Standalone-only steps are exactly where the device tends to go idle.
        if (m.standaloneGpuUs > 0 || m.gpuIdleUs > 0) {
          ss << fmt::format(" gpu={} idle={}", m.standaloneGpuUs, m.gpuIdleUs);
        }
        if (m.d2hDepStandalone + m.d2hDepShortcut > 0) {
          ss << fmt::format(
              " d2hDep=[standalone {}/{} shortcut {}/{} back={}]",
              m.d2hDepStandalone,
              m.numStandalone,
              m.d2hDepShortcut,
              m.numShortcut,
              m.d2hNearestProducer);
        }
        if (m.elidedCloneBytes > 0) {
          ss << fmt::format(
              "  elided copies={}",
              facebook::velox::succinctBytes(m.elidedCloneBytes));
        }
        if (m.allocGroups > 0) {
          ss << fmt::format(
              "  allocGroups={}/{}", m.allocGroups, m.allocGroupTensors);
        }
        ss << standaloneBreakdown(m.sequenceNumber, m.stepIdx) << "\n";
        continue;
      }
      auto stepUs = m.gatherUs + m.gridUs + m.allocUs + m.fillUs + m.kernelUs;
      double bytesTotal = static_cast<double>(m.inputBytes + m.outputBytes);
      double gbps = m.kernelUs > 0
          ? bytesTotal / (static_cast<double>(m.kernelUs) * 1e3)
          : 0.0;
      ss << fmt::format(
          "  step {}: {} us  blocks={}  GPU RAM={}  in={:.1f}KB out={:.1f}KB  {:.1f} GB/s",
          m.stepIdx,
          stepUs,
          m.numBlocks,
          facebook::velox::succinctBytes(m.currentBytes),
          m.inputBytes / 1024.0,
          m.outputBytes / 1024.0,
          gbps);
      if (m.elidedCloneBytes > 0) {
        ss << fmt::format(
            "  elided copies={}",
            facebook::velox::succinctBytes(m.elidedCloneBytes));
      }
      // groups/tensors: the allocator calls this step made for its grouped
      // outputs, over the calls it would have made without the grouping.
      if (m.allocGroups > 0) {
        ss << fmt::format(
            "  allocGroups={}/{}", m.allocGroups, m.allocGroupTensors);
      }
      // kernel= is the host cost of issuing the step, gpu= what the device
      // actually spent on it. They diverge once the two streams overlap, and
      // that divergence is the interesting signal.
      ss << fmt::format(
          "  [gather={} grid={} alloc={}(call={} setup={}) fill={} kernel={} "
          "gpu={} idle={}]",
          m.gatherUs,
          m.gridUs,
          m.allocUs,
          m.allocCallUs,
          m.allocUs - m.allocCallUs,
          m.fillUs,
          m.kernelUs,
          m.kernelGpuUs,
          m.gpuIdleUs);
      if (m.shortcutUs > 0) {
        ss << fmt::format(" shortcut={}", m.shortcutUs);
      }
      if (m.standaloneUs > 0) {
        ss << fmt::format(
            " standalone={}{}", m.standaloneUs, m.standaloneBound ? "*" : "");
      }
      if (m.standaloneGpuUs > 0) {
        ss << fmt::format(" standaloneGpu={}", m.standaloneGpuUs);
      }
      // Ops here that must wait for an earlier step's transfer, and how far
      // back the nearest producing step is (1 = the step just before).
      if (m.d2hDepFused + m.d2hDepStandalone + m.d2hDepShortcut > 0) {
        ss << fmt::format(
            " d2hDep=[fused {}/{} standalone {}/{} shortcut {}/{} back={}]",
            m.d2hDepFused,
            m.numFused,
            m.d2hDepStandalone,
            m.numStandalone,
            m.d2hDepShortcut,
            m.numShortcut,
            m.d2hNearestProducer);
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
        const auto numBlocks = static_cast<int32_t>(blocks.size());
        int64_t maxClocks = 0;
        int64_t totalClocks = 0;
        int64_t totalBarrier = 0;
        for (const auto& b : blocks) {
          maxClocks = std::max(maxClocks, b.clocks);
          totalClocks += b.clocks;
          totalBarrier += b.barrierClocks;
        }
        double syncPct =
            totalClocks > 0 ? 100.0 * totalBarrier / totalClocks : 0.0;

        // The launches of a split step run back to back on one stream, so the
        // step's makespan is the sum of their maxima rather than the largest
        // block anywhere in it, and a block is only idle relative to the
        // launch it actually ran in. Scoring the whole step against one global
        // max reads a short launch queued behind a long one as imbalance: on
        // the ROO graph node 36 step 3 reported 70% while both of its launches
        // were above 75%. Each launch owns a contiguous range of this block
        // array, so score them one at a time and add the rectangles up.
        struct Span {
          int32_t first;
          int32_t count;
          int64_t max{0};
          int64_t sum{0};
        };
        std::vector<Span> spans;
        for (const auto& s : m.segments) {
          if (s.firstBlock >= 0 && s.numBlocks > 0 &&
              s.firstBlock + s.numBlocks <= numBlocks) {
            spans.push_back({s.firstBlock, s.numBlocks});
          }
        }
        // A step the packer left whole, or one whose segments do not describe
        // this array, is one launch over all of it -- which makes the numbers
        // below identical to scoring the step as a single rectangle.
        int32_t covered = 0;
        for (const auto& s : spans) {
          covered += s.count;
        }
        if (spans.empty() || covered != numBlocks) {
          spans.assign(1, Span{0, numBlocks});
        }
        int64_t makespan = 0;
        int64_t rectangles = 0;
        for (auto& s : spans) {
          for (int32_t b = s.first; b < s.first + s.count; ++b) {
            s.max = std::max(s.max, blocks[b].clocks);
            s.sum += blocks[b].clocks;
          }
          makespan += s.max;
          rectangles += s.max * s.count;
        }
        double util = rectangles > 0
            ? 100.0 * static_cast<double>(totalClocks) /
                static_cast<double>(rectangles)
            : 0.0;

        if (spans.size() == 1) {
          ss << fmt::format(
              "    balance: util={:.1f}% sync={:.1f}% maxClk={} blocks={}\n",
              util,
              syncPct,
              maxClocks,
              numBlocks);
        } else {
          ss << fmt::format(
              "    balance: util={:.1f}% sync={:.1f}% makespan={} maxClk={} "
              "blocks={} launches={}\n",
              util,
              syncPct,
              makespan,
              maxClocks,
              numBlocks,
              spans.size());
          // targetBlocks is one wave at the occupancy the step launches with,
          // so this says whether a launch is also several hardware waves --
          // the same effect one level down, and invisible in the block count.
          const int32_t wave = m.gridStats.targetBlocks;
          for (size_t li = 0; li < spans.size(); ++li) {
            const auto& s = spans[li];
            ss << fmt::format(
                "      launch {}: util={:.1f}% max={} blocks={} first={}",
                li,
                s.max > 0 ? 100.0 * static_cast<double>(s.sum) /
                        static_cast<double>(s.max * s.count)
                          : 0.0,
                s.max,
                s.count,
                s.first);
            if (wave > 0) {
              ss << fmt::format(" waves={}", (s.count + wave - 1) / wave);
            }
            ss << "\n";
          }
        }

        // Per-op breakdown sorted by max clocks descending.
        struct OpStats {
          int32_t op;
          int64_t opMin{std::numeric_limits<int64_t>::max()};
          int64_t opMax{0};
          int64_t opSum{0};
          int64_t opBarrier{0};
          int64_t count{0};
          int64_t numElements{0};
          /// Blocks this op contributed to each launch, parallel to 'spans'.
          /// Which launch an op landed in is the thing that decides whether it
          /// is holding one up: an op is only measured against the others it
          /// actually ran beside.
          std::vector<int32_t> perLaunch;
        };
        // Block index -> launch, from the contiguous ranges above.
        std::vector<int32_t> blockLaunch(numBlocks, 0);
        for (size_t li = 0; li < spans.size(); ++li) {
          for (int32_t b = spans[li].first;
               b < spans[li].first + spans[li].count;
               ++b) {
            blockLaunch[b] = static_cast<int32_t>(li);
          }
        }
        std::map<int32_t, OpStats> opMap;
        for (int32_t i = 0; i < numBlocks; ++i) {
          const auto& b = blocks[i];
          auto& s = opMap[b.op];
          s.op = b.op;
          s.opMin = std::min(s.opMin, b.clocks);
          s.opMax = std::max(s.opMax, b.clocks);
          s.opSum += b.clocks;
          s.opBarrier += b.barrierClocks;
          s.count++;
          s.perLaunch.resize(spans.size(), 0);
          ++s.perLaunch[blockLaunch[i]];
          referencedOps.insert(b.op);
          opTotalClocks[b.op] += b.clocks;
        }
        // Get per-op element counts from step vectors.
        if (auto* sv = findStepVectors(state, m.sequenceNumber, m.stepIdx)) {
          for (const auto& kern : sv->kernels) {
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
              "      op {} ({} blocks, {}): clk max/avg/min={}/{}/{} barrier={}",
              s.op,
              s.count,
              fmtSize(s.numElements),
              s.opMax,
              opAvg,
              s.opMin,
              s.opBarrier);
          if (spans.size() > 1) {
            ss << " in";
            for (size_t li = 0; li < s.perLaunch.size(); ++li) {
              if (s.perLaunch[li] > 0) {
                ss << fmt::format(" L{}={}", li, s.perLaunch[li]);
              }
            }
          }
          ss << "\n";
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
    std::array<int64_t, 6> extra{};
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
        std::copy(
            std::begin(dbg.extra), std::end(dbg.extra), entry.extra.begin());
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
         << " TB " << entry.blockIdx << " L " << entry.line;
      // Trailing zeroes are unset fields, not measurements: a site that reports
      // two values would otherwise read as one reporting six.
      size_t numExtra = entry.extra.size();
      while (numExtra > 2 && entry.extra[numExtra - 1] == 0) {
        --numExtra;
      }
      for (size_t i = 0; i < numExtra; ++i) {
        ss << " " << entry.extra[i];
      }
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
