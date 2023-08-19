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

#include "velox/experimental/wave/exec/Instruction.h"
#include "velox/expression/Expr.h"
#include "velox/type/Subfield.h"

#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/exec/ExprKernel.h"

#include <folly/hash/Hash.h>

namespace facebook::velox::wave {

// A value a kernel can depend on. Either a dedupped exec::Expr or a dedupped
// subfield. Subfield between operators, Expr inside  an Expr.
struct Value {
  Value() = default;
  Value(const exec::Expr* expr) : expr(expr), subfield(nullptr) {}

  Value(const common::Subfield* subfield) : expr(nullptr), subfield(subfield) {}
  ~Value() = default;

  bool operator==(const Value& other) {
    return expr == other.expr && subfield == other.subfield;
  }

  const exec::Expr* expr;
  const common::Subfield* subfield;
};

struct ValueHasher {
  size_t operator()(const Value& value) const {
    return folly::hasher<uint64_t>()(
               reinterpret_cast<uintptr_t>(value.subfield)) ^
        folly::hasher<uint64_t>()(reinterpret_cast<uintptr_t>(value.expr));
  }
};

struct ValueComparer {
  bool operator()(const Value& left, const Value& right) const {
    return left.expr == right.expr && left.subfield == right.subfield;
  }
};

} // namespace facebook::velox::wave
namespace folly {
template <>
struct hasher<::facebook::velox::wave::Value> {
  size_t operator()(const ::facebook::velox::wave::Value value) const {
    return folly::hasher<uint64_t>()(
               reinterpret_cast<uintptr_t>(value.subfield)) ^
        folly::hasher<uint64_t>()(reinterpret_cast<uintptr_t>(value.expr));
  }
};
} // namespace folly

namespace facebook::velox::wave {

class Wave;

struct BufferReference {
  // Ordinal of the instruction that assigns a value to the Operand.
  int32_t insruction;
  // Offset of Operand struct in the executable image.
  int32_t offset;
};

class Program {
 public:
  void add(std::unique_ptr<AbstractInstruction> instruction) {
    instructions_.push_back(std::move(instruction));
  }

  // Initialized executableImage and relocation infromation and places for
  // parameters.
  void prepareForDevice(GpuArena& arena);

  ThreadBlockProgram* instantiate(GpuArena& arena);

  // Patches device side 'instance' to reference newly allocated buffers for up
  // to 'numRows' of result data starting at instruction at 'continuePoint'.
  void setBuffers(
      ThreadBlockProgram* instance,
      int32_t continuePoint,
      int32_t numRows);

  const std::vector<Value>& dependsOn() const {
    return dependsOn_;
  }

  AbstractOperand* findOperand(const Value& value) {
    return nullptr;
  }

  std::vector<Value> dependsOn_;
  folly::F14FastMap<Value, AbstractOperand*, ValueHasher, ValueComparer>
      produces_;
  std::vector<std::unique_ptr<AbstractInstruction>> instructions_;

  // Relocation info.The first int is the offset of a pointer in the executable
  // representation .The secon is the offset it points to inside the
  // representation.
  std::vector<std::pair<int32_t, int32_t>> relocation_;

  // FDescribes the places in the executable image that need a WaveBuffer's
  // address to be patched in before execution.
  std::vector<BufferReference> buffers;
  // Bytes to copy to device. The relocations and buffer reference patches given
  // in 'relocations_' and 'buffers_' must be applied to the image before
  // starting a kernel interpreting the image.
  std::vector<uint64_t> executableImage_;

  // The size of the device side contiguous memory for 'this'.
  int32_t sizeOnDevice_{0};
};

class Wave {
 public:
  Wave(GpuArena& arena);

  /// Adds program and reserves space for maxRows of output.
  void addProgram(Program*, int32_t maxRows);

  // Returns Event for syncing with the arrival of 'this'.
  Event* event() {}

  void start(Stream* stream);

 private:
  GpuArena& arena_;
  Stream* stream_;

  // Event recorded on 'stream_' right after kernel launch.
  std::unique_ptr<Event> event_;

  // At start, errors are clear and row counts are input row counts. On return,
  // errors are set and the output row count of each block may be set if this
  // has cardinality change.
  BlockStatus* statuses;
  ThreadBlockProgram** programs_;
};

} // namespace facebook::velox::wave
