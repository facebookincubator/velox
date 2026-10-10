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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/NvtxTracing.h"

#include "velox/exec/Task.h"

#include <fmt/format.h>

#include <mutex>
#include <unordered_map>

namespace facebook::velox::cudf_velox {
namespace {

nvtxDomainHandle_t threadDomain() {
  static nvtxDomainHandle_t const domain =
      nvtx3::domain::get<VeloxThreadDomain>();
  return domain;
}

nvtxDomainHandle_t veloxDomain() {
  static nvtxDomainHandle_t const domain = nvtx3::domain::get<VeloxDomain>();
  return domain;
}

// Returns the part of a Presto task id that distinguishes tasks within one
// query: stageId.stageExecutionId.id.attemptNumber. Task::shortId() is no use
// here, as it hashes only the query id and so is identical for every task of a
// query.
std::string taskSuffix(const std::string& taskId) {
  const auto dot = taskId.find('.');
  if (dot == std::string::npos) {
    return taskId;
  }
  return taskId.substr(dot + 1);
}

// What this thread is currently attributed to. A Driver holds a pool thread
// from enqueue until it blocks, so this changes only on a swap.
struct ThreadOccupancy {
  uint64_t key{0};
  nvtxRangeId_t range{0};
};
thread_local ThreadOccupancy tlsOccupancy;

} // namespace

void noteDriverOnThread(uint64_t key, const char* label) {
  auto& current = tlsOccupancy;
  if (current.key == key) {
    return; // same Driver still on this thread
  }
  if (current.range != 0) {
    nvtxDomainRangeEnd(threadDomain(), current.range);
    current.range = 0;
  }
  current.key = key;
  if (key == 0) {
    return;
  }
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.colorType = NVTX_COLOR_ARGB;
  // Colour by driver id so neighbouring bands on one thread differ.
  static constexpr uint32_t kPalette[] = {
      0xFF4E79A7,
      0xFFF28E2B,
      0xFF59A14F,
      0xFFB07AA1,
      0xFF76B7B2,
      0xFFEDC948,
      0xFFFF9DA7,
      0xFF9C755F};
  attr.color = kPalette[key % (sizeof(kPalette) / sizeof(kPalette[0]))];
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = label;
  // Start/end rather than push/pop: the band must survive the operator call
  // that opened it, and is closed by a different call later.
  current.range = nvtxDomainRangeStartEx(threadDomain(), &attr);
}

uint32_t nvtxRegisterTask(const exec::Task* task) {
  if (task == nullptr) {
    return 0;
  }
  static std::mutex mutex;
  static std::unordered_map<std::string, uint32_t> categories;
  // Category 0 means "no category" to NVTX, so ids start at 1.
  static uint32_t nextCategory{1};
  uint32_t category{0};
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto [it, inserted] =
        categories.emplace(task->taskId(), nextCategory);
    if (!inserted) {
      return it->second;
    }
    category = nextCategory++;
  }

  nvtxDomainNameCategoryA(veloxDomain(), category, task->taskId().c_str());

  const auto text = fmt::format("velox.task {}", task->taskId());
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.category = category;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = text.c_str();
  nvtxDomainMarkEx(veloxDomain(), &attr);
  return category;
}

std::string
nvtxDriverLabel(const exec::Task* task, int32_t pipelineId, int32_t driverId) {
  if (task == nullptr) {
    return fmt::format("p{}d{}", pipelineId, driverId);
  }
  return fmt::format(
      "t{} p{}d{}", taskSuffix(task->taskId()), pipelineId, driverId);
}

} // namespace facebook::velox::cudf_velox
