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

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <ATen/core/ivalue.h>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Executor.h"

// Step-level helpers shared by the legacy step loop and the bound one.

namespace torch::wave {

/// The StepVectors of (sequenceNumber, stepIdx), growing 'allSteps' as needed.
StepVectors& getStepVectors(
    std::vector<std::vector<StepVectors>>& allSteps,
    int32_t sequenceNumber,
    int32_t stepIdx);

/// The pinned or device buffer of (sequenceNumber, stepIdx), (re)allocated
/// from 'arena' when absent or smaller than 'requiredBytes'. 'initFunc' runs on
/// a newly allocated buffer.
facebook::velox::wave::WaveBufferPtr& getOrAllocateBuffer(
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>& buffers,
    int32_t sequenceNumber,
    int32_t stepIdx,
    int64_t requiredBytes,
    facebook::velox::wave::GpuArena* arena,
    const std::function<void(void*, int64_t)>& initFunc = nullptr);

/// Latest executed step among the pending transfers whose returned values
/// 'readIds' contains, or -1 when it reads none of them.
int32_t neededPendingStep(
    const ExecutionState& state,
    const folly::F14FastSet<nativert::ValueId>& readIds);

/// The same over the stepKey()s of the steps that send what the reader reads,
/// sorted.
int32_t neededPendingStep(
    const ExecutionState& state,
    const int64_t* producers,
    int32_t numProducers);

/// Under WaveConfig::syncEachStep, drains both streams before a step
/// allocates.
/// Sets 'data.viewReadIds'.
void setViewReadIds(LaunchData& data);

void drainBeforeStepIfRequested(ExecutionState& state);

/// Makes the frame hold a CUDA tensor of 'dims' for 'actualId': keeps or
/// resizes the one there, captures it for an allocation group being collected,
/// or allocates it.
void ensureCudaTensor(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId actualId,
    c10::IntArrayRef dims,
    const std::string& sizeKey = std::string("dyn"));

/// Runs a fused output's host-side view as a metadata shortcut. Returns false
/// when the view has no recipe.
bool runViewRecipe(const OutputDesc& desc, nativert::ExecutionFrame& frame);

/// The metadata shortcut that computes the host-side view 'node', or null when
/// no shortcut covers it.
std::shared_ptr<const ViewRecipe> makeViewRecipe(NodeCP node);

/// Writes a scalar kernel parameter from 'ivalue' into 'dest'.
void fillScalarParam(
    const c10::IValue& ivalue,
    void* dest,
    nativert::ValueId valueId);

/// Writes the descriptor of an absent tensor into 'dest'.
void fillEmptyTensorParam(void* dest);

/// Bound path: decides which values live in the table alone
/// (ExecutionState::tableOnly) from what host code reads, and turns the
/// metadata shortcuts whose input is not host-read into DeriveOps. Runs after
/// every node's programs are built.
void bindTableOnly(ExecutionState& state, WaveGraph& waveGraph);

} // namespace torch::wave
