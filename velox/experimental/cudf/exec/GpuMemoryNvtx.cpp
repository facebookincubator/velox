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

#include "velox/experimental/cudf/exec/GpuMemoryNvtx.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

namespace {

/// Counter names must be unique within a capture: the SQLite export records the
/// scope tree in NVTX_SCOPES and the samples in GENERIC_EVENTS, but nothing
/// ties a counter to the scope it was registered against, so same-named
/// counters are indistinguishable afterwards. This ordinal supplies the
/// uniqueness, and a registration marker carrying it supplies the full
/// identity.
uint64_t nextCounterOrdinal() {
  static std::atomic<uint64_t> ordinal{0};
  return ordinal.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// The default domain, not the "velox" domain that carries the operator ranges.
/// Nsight renders a named domain as an extra level between a scope and its
/// counters, which buries the plan nesting. The cost is that
/// --nvtx-domain-include=velox keeps the ranges but drops these counters.
nvtxDomainHandle_t counterDomain() {
  return nullptr;
}

const nvtxSemanticsCounter_t& byteCounterSemantics() {
  static const auto semantics = [] {
    nvtxSemanticsCounter_t result{};
    result.header.structSize = sizeof(nvtxSemanticsCounter_t);
    result.header.semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1;
    result.header.version = NVTX_COUNTER_SEMANTIC_VERSION;
    result.header.next = nullptr;
    result.flags = NVTX_COUNTER_FLAG_LIMIT_MIN |
        NVTX_COUNTER_FLAG_VALUETYPE_ABSOLUTE |
        NVTX_COUNTER_FLAG_INTERPOLATION_UNTIL_NEXT;
    result.unit = "bytes";
    result.unitScaleNumerator = 1;
    result.unitScaleDenominator = 1;
    result.limitType = NVTX_COUNTER_LIMIT_I64;
    result.min.i64 = 0;
    return result;
  }();
  return semantics;
}

uint64_t registerScope(const std::string& path, uint64_t parentScope) {
  nvtxScopeAttr_t attributes{};
  attributes.structSize = sizeof(nvtxScopeAttr_t);
  attributes.path = path.c_str();
  attributes.parentScope = parentScope;
  attributes.scopeId = NVTX_SCOPE_NONE;
  return nvtxScopeRegister(counterDomain(), &attributes);
}

uint64_t registerByteCounter(const std::string& name, uint64_t scopeId) {
  nvtxCounterAttr_t attributes{};
  attributes.structSize = sizeof(nvtxCounterAttr_t);
  attributes.schemaId = NVTX_PAYLOAD_ENTRY_TYPE_INT64;
  attributes.name = name.c_str();
  attributes.description =
      "Logical requested GPU allocation bytes attributed by Velox-cuDF.";
  attributes.scopeId = scopeId;
  attributes.semantics = &byteCounterSemantics().header;
  attributes.counterId = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(counterDomain(), &attributes);
}

int64_t clampToInt64(uint64_t bytes) {
  return static_cast<int64_t>(std::min(
      bytes, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

/// NVTX timestamps the sample here, which is when the ledger recorded the
/// transition.
void sampleCounter(uint64_t counterId, uint64_t bytes) {
  // The unattributed owner has no ancestor levels, so part of its set is
  // legitimately unregistered.
  if (counterId == NVTX_COUNTER_ID_NONE) {
    return;
  }
  nvtxCounterSampleInt64(counterDomain(), counterId, clampToInt64(bytes));
}

void emitMark(const std::string& text) {
  nvtxEventAttributes_t attributes{};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = text.c_str();
  nvtxDomainMarkEx(counterDomain(), &attributes);
}

std::string displayField(std::string_view value) {
  return value.empty() ? std::string{"<none>"} : std::string{value};
}

/// Escapes the NVTX scope path separators so a value cannot forge a level.
std::string escapeScopeField(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const auto character : value) {
    if (character == '\\' || character == '/' || character == '[' ||
        character == ']') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return result;
}

// One label per hierarchy level. The registered counter name is the label plus
// " #<ordinal>". The "logical requested live bytes" qualifier lives in the
// counter description rather than the label, to keep rows readable in a narrow
// column without losing the distinction from physical memory.

std::string queryScopeLabel(const GpuMemoryOwner& owner) {
  return "query " + escapeScopeField(displayField(owner.queryId));
}

std::string fragmentScopeLabel(const GpuMemoryOwner& owner) {
  // A Presto task executes exactly one plan fragment, so the task identifier
  // names the fragment instance.
  return "fragment " + escapeScopeField(displayField(owner.taskId));
}

/// Zero-padded pre-order index plus one dash per level of depth, so Nsight's
/// lexical row sort reproduces EXPLAIN order and nesting stays legible.
std::string planNodeScopeLabel(const GpuMemoryPlanLocation& location) {
  const auto* node = location.node();
  if (node == nullptr) {
    return "unmapped";
  }
  return fmt::format(
      "{:03d} {}{} [{}]",
      location.order,
      std::string(location.depth(), '-') + (location.depth() > 0 ? " " : ""),
      escapeScopeField(node->planNodeType),
      escapeScopeField(node->planNodeId));
}

std::string operatorCounterLabel(const GpuMemoryOwner& owner) {
  return fmt::format(
      "op{} {} d{}",
      owner.operatorId,
      escapeScopeField(displayField(owner.operatorType)),
      owner.driverId);
}

struct ScopeCounter {
  uint64_t scopeId{NVTX_SCOPE_NONE};
  uint64_t counterId{NVTX_COUNTER_ID_NONE};
  /// Only queries publish a high-water mark, as a flat reference against their
  /// live row. Deeper levels would repeat information their own live row's
  /// maximum already gives.
  uint64_t peakCounterId{NVTX_COUNTER_ID_NONE};
};

/// Registration cycle of the ids currently handed out.
///
/// A reset retires every id, so a counter set cached by the ledger before it
/// must not be sampled afterwards. Read once per transition instead of taking
/// the registration lock.
std::atomic<uint64_t> counterEpoch{1};

struct CounterState {
  std::mutex mutex;
  bool initialized{false};
  uint64_t rootScopeId{NVTX_SCOPE_NONE};
  uint64_t globalCounterId{NVTX_COUNTER_ID_NONE};
  /// Keyed by queryId.
  std::unordered_map<std::string, ScopeCounter> queries;
  /// Keyed by taskUuid.
  std::unordered_map<std::string, ScopeCounter> tasks;
  /// Keyed by taskUuid and plan node identifier, so that the same plan node
  /// identifier in concurrent tasks stays separate.
  std::unordered_map<std::string, uint64_t> planNodeScopes;
  /// Keyed by taskUuid, plan node identifier and pipeline.
  std::unordered_map<std::string, uint64_t> pipelineScopes;
  /// Keyed by the ledger's plan node handle.
  std::unordered_map<uint64_t, uint64_t> planNodeCounters;
  std::unordered_map<uint64_t, GpuMemoryNvtxCounters> owners;
  /// Every registration marker emitted so far, replayed when a new query
  /// appears.
  ///
  /// Replay is quadratic in the number of queries a worker runs, so the list is
  /// bounded. Past the bound later registrations are still emitted once, but no
  /// longer restated into subsequent captures.
  std::vector<std::string> registrationMarkers;
  /// Whether the bound above has been reached, so a capture can say so rather
  /// than appear to be missing counters.
  bool markersTruncated{false};
};

CounterState& counterState() {
  // Intentionally leaked: driver threads may emit during static destruction.
  static auto* state = new CounterState;
  return *state;
}

/// Roughly six markers per operator instance and a few hundred instances per
/// query, so this spans dozens of queries before it engages.
constexpr size_t kMaxRegistrationMarkers = 20'000;

/// Emits the marker mapping this counter's ordinal to the full owner identity,
/// which is what makes a capture self-describing.
uint64_t registerLevelCounter(
    std::string_view label,
    uint64_t scopeId,
    std::string_view identity,
    CounterState& state) {
  const auto ordinal = nextCounterOrdinal();
  const auto name = fmt::format("{} #{}", label, ordinal);
  const auto counterId = registerByteCounter(name, scopeId);
  const auto marker = fmt::format(
      "velox-gpu-memory-counter #{} name=[{}] {}", ordinal, name, identity);
  emitMark(marker);
  if (state.registrationMarkers.size() < kMaxRegistrationMarkers) {
    state.registrationMarkers.push_back(marker);
  } else {
    state.markersTruncated = true;
  }
  sampleCounter(counterId, 0);
  return counterId;
}

std::string planNodeScopeKey(
    std::string_view taskUuid,
    std::string_view planNodeId) {
  return std::string{taskUuid} + '\x1f' + std::string{planNodeId};
}

std::string ownerIdentity(const GpuMemoryOwner& owner, std::string_view level) {
  return fmt::format(
      "level={} query=[{}] taskId=[{}] taskUuid=[{}] plan=[{}] planType=[{}] "
      "operator={} operatorType=[{}] pipeline={} driver={}",
      level,
      displayField(owner.queryId),
      displayField(owner.taskId),
      displayField(owner.taskUuid),
      displayField(owner.planNodeId),
      displayField(owner.planNodeType),
      owner.operatorId,
      displayField(owner.operatorType),
      owner.pipelineId,
      owner.driverId);
}

/// Opens a counter session: the root scope and the process-wide counter.
void initializeLocked(CounterState& state) {
  if (state.initialized) {
    return;
  }
  state.rootScopeId = registerScope("Velox GPU memory", NVTX_SCOPE_ROOT);
  state.globalCounterId = registerLevelCounter(
      "overall live bytes", state.rootScopeId, "level=global", state);
  state.initialized = true;
}

/// Registers the query, task and plan node scopes on the owner's path, then
/// returns the scope the operator's own counter belongs to.
uint64_t resolveOwnerScopeLocked(
    CounterState& state,
    const GpuMemoryOwner& owner,
    const GpuMemoryPlanLocation& planLocation,
    uint64_t planNodeKey,
    GpuMemoryNvtxCounters& counters) {
  auto query = state.queries.find(std::string(gpuMemoryQueryKey(owner)));
  if (query == state.queries.end()) {
    // A new query means a new capture window is plausible. Nsight replays scope
    // registrations into each session but not markers, so restate the identity
    // of everything registered earlier or a capture that did not span a
    // counter's registration cannot resolve it.
    for (const auto& marker : state.registrationMarkers) {
      emitMark(marker);
    }
    if (state.markersTruncated) {
      emitMark(
          "velox-gpu-memory-counter-registrations-truncated "
          "reason=[replay bound reached]");
    }
    const auto label = queryScopeLabel(owner);
    ScopeCounter entry;
    entry.scopeId = registerScope(label, state.rootScopeId);
    entry.counterId = registerLevelCounter(
        label, entry.scopeId, ownerIdentity(owner, "query"), state);
    entry.peakCounterId = registerLevelCounter(
        label + " peak",
        entry.scopeId,
        ownerIdentity(owner, "query-peak"),
        state);
    query = state.queries.emplace(std::string(gpuMemoryQueryKey(owner)), entry)
                .first;
  }
  counters.queryCounterId = query->second.counterId;
  counters.queryPeakCounterId = query->second.peakCounterId;

  auto task = state.tasks.find(owner.taskUuid);
  if (task == state.tasks.end()) {
    const auto label = fragmentScopeLabel(owner);
    ScopeCounter entry;
    entry.scopeId = registerScope(label, query->second.scopeId);
    entry.counterId = registerLevelCounter(
        label, entry.scopeId, ownerIdentity(owner, "fragment"), state);
    task = state.tasks.emplace(owner.taskUuid, entry).first;
  }
  counters.taskCounterId = task->second.counterId;

  // Plan nodes are siblings under the task rather than nested by the plan.
  // Nesting them mirrors the plan faithfully, but a TPC-H plan is over twenty
  // levels deep, which indents every counter off the right of the timeline and
  // makes a node's own aggregate render below its children's subtree. A flat
  // list keeps each plan node and its operator instances readable together.
  const auto* node = planLocation.node();
  const auto planNodeId =
      node == nullptr ? std::string{"<unmapped>"} : node->planNodeId;
  const auto label = planNodeScopeLabel(planLocation);

  const auto key = planNodeScopeKey(owner.taskUuid, planNodeId);
  auto scope = state.planNodeScopes.find(key);
  if (scope == state.planNodeScopes.end()) {
    const auto scopeId = registerScope(label, task->second.scopeId);
    scope = state.planNodeScopes.emplace(key, scopeId).first;
  }
  const auto planNodeScopeId = scope->second;

  // Velox executes a task as pipelines of drivers, and a single plan node can
  // span two of them: a hash join's build and probe run in different pipelines.
  // Giving the pipeline its own level makes that split visible instead of
  // hiding it in the operator label.
  const auto pipelineKey = planNodeScopeKey(
      owner.taskUuid, planNodeId + '\x1f' + std::to_string(owner.pipelineId));
  auto pipeline = state.pipelineScopes.find(pipelineKey);
  if (pipeline == state.pipelineScopes.end()) {
    const auto scopeId = registerScope(
        fmt::format("pipeline {}", owner.pipelineId), planNodeScopeId);
    pipeline = state.pipelineScopes.emplace(pipelineKey, scopeId).first;
  }

  auto planNodeCounter = state.planNodeCounters.find(planNodeKey);
  if (planNodeCounter == state.planNodeCounters.end()) {
    const auto counterId = registerLevelCounter(
        label, planNodeScopeId, ownerIdentity(owner, "planNode"), state);
    planNodeCounter =
        state.planNodeCounters.emplace(planNodeKey, counterId).first;
  }
  counters.planNodeCounterId = planNodeCounter->second;

  return pipeline->second;
}

} // namespace

GpuMemoryNvtxCounters registerGpuMemoryNvtxOwner(
    uint64_t ownerId,
    uint64_t planNodeKey,
    const GpuMemoryOwner& owner,
    const GpuMemoryPlanLocation& planLocation) noexcept {
  try {
    auto& state = counterState();
    std::lock_guard<std::mutex> lock(state.mutex);
    initializeLocked(state);
    if (const auto known = state.owners.find(ownerId);
        known != state.owners.end()) {
      return known->second;
    }

    GpuMemoryNvtxCounters counters;
    counters.epoch = counterEpoch.load(std::memory_order_relaxed);
    counters.globalCounterId = state.globalCounterId;

    if (ownerId == 0) {
      counters.ownerCounterId = registerLevelCounter(
          "unattributed", state.rootScopeId, "level=unattributed", state);
    } else {
      const auto ownerScopeId = resolveOwnerScopeLocked(
          state, owner, planLocation, planNodeKey, counters);
      counters.ownerCounterId = registerLevelCounter(
          operatorCounterLabel(owner),
          ownerScopeId,
          ownerIdentity(owner, "operator"),
          state);
    }
    state.owners.emplace(ownerId, counters);
    return counters;
  } catch (...) {
    // Profiler diagnostics must never alter query execution.
    return {};
  }
}

GpuMemoryNvtxCounters registerGpuMemoryNvtxUnattributedOwner() noexcept {
  return registerGpuMemoryNvtxOwner(
      0, 0, GpuMemoryOwner{}, GpuMemoryPlanLocation{});
}

void sampleGpuMemoryNvtxCounters(
    const GpuMemoryNvtxCounters& counters,
    const GpuMemoryUpdate& update) noexcept {
  if (counters.epoch != counterEpoch.load(std::memory_order_acquire)) {
    return;
  }
  try {
    sampleCounter(counters.globalCounterId, update.globalCurrentBytes);
    sampleCounter(counters.queryCounterId, update.queryCurrentBytes);
    sampleCounter(counters.queryPeakCounterId, update.queryPeakBytes);
    sampleCounter(counters.taskCounterId, update.taskCurrentBytes);
    sampleCounter(counters.planNodeCounterId, update.planNodeCurrentBytes);
    sampleCounter(counters.ownerCounterId, update.ownerCurrentBytes);
  } catch (...) {
    // Profiler diagnostics must never alter query execution.
  }
}

void markGpuMemoryAllocationFailure(
    uint64_t ownerId,
    std::size_t requestedBytes,
    const GpuMemoryUpdate& state,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept {
  try {
    emitMark(
        fmt::format(
            "velox-gpu-memory-allocation-failure owner={} requestedBytes={} "
            "globalCurrentBytes={} globalPeakBytes={} planNodeCurrentBytes={} "
            "ownerCurrentBytes={} cudaFreeBytes={} cudaTotalBytes={} cudaStatus=[{}]",
            ownerId,
            requestedBytes,
            state.globalCurrentBytes,
            state.globalPeakBytes,
            state.planNodeCurrentBytes,
            state.ownerCurrentBytes,
            cudaFreeBytes,
            cudaTotalBytes,
            cudaStatus));
  } catch (...) {
    // Failure markers must not mask the original allocation failure.
  }
}

void markGpuMemoryDataLoss(std::string_view reason) noexcept {
  try {
    emitMark(fmt::format("velox-gpu-memory-data-loss reason=[{}]", reason));
  } catch (...) {
    // Diagnostics only.
  }
}

/// Closes the current counter session.
///
/// The epoch is retired before the zeroes are written. That stops every counter
/// set the ledger still holds, including the sets held by tracked resources
/// that deliberately outlive unregistration, so a late allocation cannot leave
/// a counter high after this has finished.
void resetGpuMemoryNvtxCounters() noexcept {
  try {
    auto& state = counterState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
      return;
    }
    counterEpoch.fetch_add(1, std::memory_order_release);

    for (const auto& [ownerId, counters] : state.owners) {
      sampleCounter(counters.ownerCounterId, 0);
    }
    for (const auto& [planNodeKey, counterId] : state.planNodeCounters) {
      sampleCounter(counterId, 0);
    }
    for (const auto& [taskUuid, entry] : state.tasks) {
      sampleCounter(entry.counterId, 0);
    }
    for (const auto& [queryId, entry] : state.queries) {
      sampleCounter(entry.peakCounterId, 0);
      sampleCounter(entry.counterId, 0);
    }
    sampleCounter(state.globalCounterId, 0);

    state.owners.clear();
    state.planNodeCounters.clear();
    state.planNodeScopes.clear();
    state.pipelineScopes.clear();
    state.tasks.clear();
    state.queries.clear();
    state.registrationMarkers.clear();
    // Every counter and scope id above has just been retired, so the next
    // session rebuilds the tree from a fresh root rather than sampling ids that
    // no longer have an owner.
    state.globalCounterId = NVTX_COUNTER_ID_NONE;
    state.rootScopeId = NVTX_SCOPE_NONE;
    state.initialized = false;
  } catch (...) {
    // Profiler diagnostics must never alter query execution.
  }
}

} // namespace facebook::velox::cudf_velox
