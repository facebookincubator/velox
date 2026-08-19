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

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

/// Registers the setOutputs-based metadata for aten.cat.default and
/// aten.stack.default, replacing the standalone-only registrations in
/// Builtins.cpp. Both ops share one fused implementation: a stack is a cat that
/// gives each operand a single position along a new dimension.
void registerConcatMetadata();

/// True if 'node' is an aten.cat / aten.stack that will run fused with a result
/// of rank > 1. Such a concat allocates its output on the host and hands each
/// operand a (generally strided) view of the region it writes, so no operand's
/// extent may be computed on device inside the concat's own kernel.
bool concatNeedsHostShapes(NodeCP node, const ValueTypes& types);

} // namespace torch::wave
