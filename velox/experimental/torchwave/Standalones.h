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

#include <torch/nativert/executor/ExecutionFrame.h>

#include "velox/experimental/torchwave/CompiledOp.h"

namespace torch::wave {

/// Executes a metadata-only standalone op (data.launch->standaloneShortcut !=
/// kNone) by calling the typed ATen primitive directly, bypassing nativert's
/// boxed dispatch. Reads operands from 'frame' via data.args / data.intArgs /
/// data.intList and writes the result to data.actualOutputs[0].
void runStandaloneShortcut(
    const LaunchData& data,
    nativert::ExecutionFrame& frame);

} // namespace torch::wave
