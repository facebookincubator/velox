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
#include "velox/experimental/cudf/exec/CudfOperator.h"

namespace facebook::velox::cudf_velox {

CudfOperatorBase::CudfOperatorBase(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    const core::PlanNodeId& planNodeId,
    const std::string& operatorName,
    std::optional<nvtx3::color> color,
    NvtxMethodFlag nvtxMethods,
    std::optional<common::SpillConfig> spillConfig,
    std::optional<std::shared_ptr<const core::PlanNode>> planNode)
    : Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          operatorName,
          spillConfig),
      NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
      className_(operatorName),
      nvtxMethods_(nvtxMethods),
      gpuTimingEnabled_(CudfConfig::getInstance().gpuTimingEnabled) {
  if (gpuTimingEnabled_) {
    initEvents();
  }
}

CudfOperatorBase::~CudfOperatorBase() {
  if (gpuTimingEnabled_) {
    if (readIdx_ < writeIdx_) {
      cudaDeviceSynchronize();
    }
    destroyEvents();
  }
}

void CudfOperatorBase::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
      nvtxMethods_ & NvtxMethodFlag::kAddInput, className_);
  if (gpuTimingEnabled_) {
    auto stream = cudf::get_default_stream(cudf::allow_default_stream);
    recordTimingStart(stream);
    doAddInput(std::move(input));
    recordTimingStopAndEnqueue(stream);
  } else {
    doAddInput(std::move(input));
  }
  checkCudaErrorInDebug();
}

RowVectorPtr CudfOperatorBase::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
      nvtxMethods_ & NvtxMethodFlag::kGetOutput, className_);
  RowVectorPtr result;
  if (gpuTimingEnabled_) {
    auto stream = cudf::get_default_stream(cudf::allow_default_stream);
    recordTimingStart(stream);
    result = doGetOutput();
    recordTimingStopAndEnqueue(stream);
    resolveCompletedSlots();
  } else {
    result = doGetOutput();
  }
  checkCudaErrorInDebug();
  return result;
}

void CudfOperatorBase::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
      nvtxMethods_ & NvtxMethodFlag::kNoMoreInput, className_);
  doNoMoreInput();
  checkCudaErrorInDebug();
}

void CudfOperatorBase::close() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
      nvtxMethods_ & NvtxMethodFlag::kClose, className_);
  if (gpuTimingEnabled_) {
    cudaDeviceSynchronize();
    resolveCompletedSlots();
  }
  doClose();
  checkCudaErrorInDebug();
}

void CudfOperatorBase::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfOperatorBase::doClose() {
  Operator::close();
}

void CudfOperatorBase::initEvents() {
  for (uint32_t i = 0; i < kTimingSlots; i++) {
    VELOX_CHECK_EQ(
        static_cast<int>(cudaEventCreate(&timingSlots_[i].start)),
        static_cast<int>(cudaSuccess),
        "Failed to create CUDA start event for timing slot {}",
        i);
    VELOX_CHECK_EQ(
        static_cast<int>(cudaEventCreate(&timingSlots_[i].stop)),
        static_cast<int>(cudaSuccess),
        "Failed to create CUDA stop event for timing slot {}",
        i);
  }
}

void CudfOperatorBase::destroyEvents() {
  for (uint32_t i = 0; i < kTimingSlots; i++) {
    cudaEventDestroy(timingSlots_[i].start);
    cudaEventDestroy(timingSlots_[i].stop);
  }
}

void CudfOperatorBase::gpuTimingCallback(void* userData) {
  auto* slot = static_cast<TimingSlot*>(userData);
  slot->ready.store(true, std::memory_order_release);
}

void CudfOperatorBase::recordTimingStart(rmm::cuda_stream_view stream) {
  if (writeIdx_ - readIdx_ >= kTimingSlots) {
    LOG(WARNING) << className_
                 << ": GPU timing ring buffer overflow, forcing sync";
    auto& oldest = timingSlots_[readIdx_ & (kTimingSlots - 1)];
    cudaEventSynchronize(oldest.stop);
    float ms = 0;
    if (cudaEventElapsedTime(&ms, oldest.start, oldest.stop) == cudaSuccess) {
      addRuntimeStat(
          kGpuWallTime,
          RuntimeCounter(
              static_cast<int64_t>(ms * 1e6), RuntimeCounter::Unit::kNanos));
    }
    ++readIdx_;
  }
  auto& slot = timingSlots_[writeIdx_ & (kTimingSlots - 1)];
  slot.ready.store(false, std::memory_order_relaxed);
  VELOX_CHECK_EQ(
      static_cast<int>(cudaEventRecord(slot.start, stream.value())),
      static_cast<int>(cudaSuccess),
      "Failed to record CUDA start event for timing slot {}",
      writeIdx_);
}

void CudfOperatorBase::recordTimingStopAndEnqueue(
    rmm::cuda_stream_view stream) {
  auto& slot = timingSlots_[writeIdx_ & (kTimingSlots - 1)];
  VELOX_CHECK_EQ(
      static_cast<int>(cudaEventRecord(slot.stop, stream.value())),
      static_cast<int>(cudaSuccess),
      "Failed to record CUDA stop event for timing slot {}",
      writeIdx_);
  VELOX_CHECK_EQ(
      static_cast<int>(
          cudaLaunchHostFunc(stream.value(), gpuTimingCallback, &slot)),
      static_cast<int>(cudaSuccess),
      "Failed to launch host callback for timing slot {}",
      writeIdx_);
  ++writeIdx_;
}

void CudfOperatorBase::resolveCompletedSlots() {
  while (readIdx_ < writeIdx_) {
    auto& slot = timingSlots_[readIdx_ & (kTimingSlots - 1)];
    if (!slot.ready.load(std::memory_order_acquire)) {
      break;
    }
    float ms = 0;
    if (cudaEventElapsedTime(&ms, slot.start, slot.stop) != cudaSuccess) {
      LOG(WARNING) << className_
                   << ": cudaEventElapsedTime failed for timing slot "
                   << readIdx_ << ", skipping";
      ++readIdx_;
      continue;
    }
    addRuntimeStat(
        kGpuWallTime,
        RuntimeCounter(
            static_cast<int64_t>(ms * 1e6), RuntimeCounter::Unit::kNanos));
    ++readIdx_;
  }
}

} // namespace facebook::velox::cudf_velox
