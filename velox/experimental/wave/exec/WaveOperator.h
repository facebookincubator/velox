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

#include "velox/experimental/wave/exec/Wave.h"

namespace facebook::velox::wave {

class CompileState;

class WaveOperator {
 public:
  WaveOperator(CompileState& state, const TypePtr& outputType);

  /// True if may reduce cardinality without duplicating input rows.
  bool isFilter() {
    return isFilter_;
  }

  /// True if a single input can produce zero to multiple outputs.
  bool isExpanding() const {
    return isExpanding_;
  }

  // If 'this' is a cardinality change (filter, join, unnest...),
  // returns the instruction where the projected through columns get
  // wrapped. Columns that need to be accessed through the change are
  // added here.
  virtual AbstractWrap* findWrap() const {
    return nullptr;
  }

  virtual std::string toString() const = 0;

  void definesSubfields(
      CompileState& state,
      const TypePtr& type,
      const std::string& parentPath = "");

  /// Returns the operand if this is defined by 'this'.
  AbstractOperand* defines(Value value) {
    auto it = defines_.find(value);
    if (it == defines_.end()) {
      return nullptr;
    }
    return it->second;
  }

 protected:
  bool isFilter_{false};

  bool isExpanding_{false};

  TypePtr outputType_;

  // The operands that are first defined here.
  folly::F14FastMap<Value, AbstractOperand*, ValueHasher, ValueComparer>
      defines_;

  // The operand for values that are projected through 'this'.
  folly::F14FastMap<Value, AbstractOperand*, ValueHasher, ValueComparer>
      projects_;

  std::vector<std::shared_ptr<Program>> programs_;

  // Executable instances of 'this'. A Driver may instantiate multiple
  // executable instances to processs consecutive input batches in parallel.
  std::vector<ThreadBlockProgram*> executables_;

  // Buffers containing unified memory for 'executables_' and all instructions,
  // operands etc. referenced from these.  This does not include buffers for
  // intermediate results.
  std::vector<WaveBufferPtr> executableMemory_;

  /// The wave that produces each subfield. More than  one subfield can be
  /// produced by the same wave.
  folly::F14FastMap<common::Subfield*, std::shared_ptr<Wave>> fieldToWave_;
};

} // namespace facebook::velox::wave
