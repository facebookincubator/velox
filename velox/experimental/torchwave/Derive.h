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

#include <cstdint>
#include <vector>

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Reserve.h"
#include "velox/experimental/torchwave/ValueTable.h"

// The Derive op of the bound execution path: an entry computed from another
// entry with no memory of its own -- the metadata arithmetic of the standalone
// shortcuts, over table entries.

namespace torch::wave {

/// One metadata-only op over table entries.
struct DeriveOp {
  StandaloneShortcut kind{StandaloneShortcut::kNone};
  /// The operands in schema order: a value id, or -1 where 'constants' holds
  /// an integer constant at the same position.
  std::vector<int32_t> operands;
  std::vector<int64_t> constants;
  /// An all-constant integer list operand (view and expand sizes).
  std::vector<int64_t> intList;
  int32_t output{-1};
  /// The outputs of a multi-output op: the elements an unbind or split makes,
  /// or what a list unpack yields. Empty for a single-output op.
  std::vector<int32_t> outputs;
  /// For a list unpack: the element values of the unpacked list.
  std::vector<int32_t> elements;
  /// Host code reads the output, so the frame gets a tensor for it.
  bool materialize{false};
  /// Per entry of 'outputs': host code reads it.
  std::vector<bool> materializeOutputs;
  /// Index of the shortcut launch this stands for, for the fallback, and of
  /// the list unpack an unbind or split absorbed.
  int32_t shortcut{-1};
  int32_t secondShortcut{-1};
};

/// A DeriveOp packed into a StepCode: its lists are spans into the code's ids
/// and constants.
struct DeriveRec {
  StandaloneShortcut kind{StandaloneShortcut::kNone};
  bool materialize{false};
  int32_t output{-1};
  /// Into StepCode::ids.
  Span operands;
  Span outputs;
  Span elements;
  /// Per output, 1 when host code reads it. Into StepCode::ids.
  Span materializeOutputs;
  /// Into StepCode::consts.
  Span constants;
  Span intList;
  int32_t shortcut{-1};
  int32_t secondShortcut{-1};
};

/// What runDerive reads of one op.
struct DeriveView {
  StandaloneShortcut kind{StandaloneShortcut::kNone};
  int32_t output{-1};
  ArrayView<int32_t> operands;
  ArrayView<int64_t> constants;
  ArrayView<int64_t> intList;
  ArrayView<int32_t> outputs;
  ArrayView<int32_t> elements;
};

/// Whether 'kind' can run as a DeriveOp.
bool derivable(StandaloneShortcut kind);

/// Computes the output entry of 'op' from its operands' entries. Returns false
/// when an operand is not in the table this execution or the op needs what
/// the table cannot express (a view of a non-contiguous tensor), which sends
/// it to the frame-based shortcut.
bool runDerive(const DeriveView& op, const ValueTable& table);

} // namespace torch::wave
