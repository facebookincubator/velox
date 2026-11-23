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
#include <optional>
#include <string>
#include <vector>

#include "velox/experimental/torchwave/PackedInput.h"

namespace torch::nativert {
class Graph;
class Value;
} // namespace torch::nativert

namespace torch::wave {

/// The graph side of packed inputs (PackedInput.h): finding them, rewriting
/// their splits into tw.unpack_slot, and the further rewrites that give slots
/// windows. Kept out of the packer library so that it depends on ATen only.

/// Finds the packed inputs of 'graph' and rewrites each one's split into
/// per-slot tw.unpack_slot nodes, which keep the names (and so the tensor
/// metadata) of the split's outputs. An input is packed only if
/// - it is a user input, rank 1, int64, read by nothing but the split;
/// - the split is split_with_sizes along dim 0, unpacked into its outputs;
/// - every split size is a distinct user-input SymInt, so that each length is
///   one feature's, or a constant.
/// Any other input is left as it was. Returns one recipe per rewritten input,
/// in user-input order. Mutates 'graph'.
std::vector<PackRecipe> rewritePackedInputs(nativert::Graph& graph);

/// A further rewrite of a graph whose packed inputs rewritePackedInputs() has
/// just rewritten, typically one that sets slot windows (setSlotWindow()) and
/// removes the ops they replace. Runs on the graph before it is compiled, so
/// it sees ops in their exported spelling.
using PackRewrite =
    std::function<void(nativert::Graph&, std::vector<PackRecipe>&)>;

/// Adds 'rewrite' to the ones rewritePackedInputs() runs, in registration
/// order, after its own.
void registerPackRewrite(PackRewrite rewrite);

/// The slot whose values 'value' is: which recipe, and which slot of it.
struct SlotRef {
  int32_t recipe{-1};
  int32_t slot{-1};
};

/// The slot 'value' is the tw.unpack_slot output of, or nullopt.
std::optional<SlotRef> packedSlotOf(
    const std::vector<PackRecipe>& recipes,
    const nativert::Value& value);

/// A row of a per-row lengths input: element 'row' of unbind(view(input,
/// [numRows, -1])), where 'input' is a user input.
struct LengthsRow {
  int32_t input{-1};
  int32_t row{-1};
  int64_t numRows{0};
};

/// The lengths row 'value' is, or nullopt.
std::optional<LengthsRow> lengthsRowOf(
    const nativert::Graph& graph,
    const nativert::Value& value);

/// Gives 'slot' of 'recipe' the window 'window' (its guardGroup, maxLength
/// and keepTail), its per-row lengths being 'lengths'. Fails, leaving the
/// recipe unchanged, if the host could not apply it: the slot's length is a
/// constant or is read by more than its tw.unpack_slot, the slot already has a
/// window, or the recipe's windowed slots take their lengths from another
/// input. The caller is responsible for the values and lengths reaching
/// nothing that needs them untrimmed. Returns why it failed, or nullopt.
std::optional<std::string> setSlotWindow(
    const nativert::Graph& graph,
    PackRecipe& recipe,
    int32_t slot,
    const LengthsRow& lengths,
    const PackSlot& window);

} // namespace torch::wave
