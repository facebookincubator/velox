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
#include <string>
#include <vector>

#include "velox/experimental/torchwave/ValueTable.h"

namespace torch::wave {

using Dim = uint32_t;

/// The shape a ReserveFn produces for one output tensor. Fixed capacity, so a
/// reserve allocates nothing.
struct Shape {
  int8_t rank{0};
  Dim dims[kMaxDims]{};

  int64_t numElements() const {
    int64_t n = 1;
    for (int32_t i = 0; i < rank; ++i) {
      n *= dims[i];
    }
    return n;
  }

  /// A 1-d shape of 'size' elements.
  void setVector(Dim size) {
    rank = 1;
    dims[0] = size;
  }

  /// The shape of 't'.
  void setLike(const TableTensor& t) {
    rank = t.rank;
    for (int32_t i = 0; i < t.rank; ++i) {
      dims[i] = static_cast<Dim>(t.dims[i]);
    }
  }

  bool operator==(const Shape& other) const {
    if (rank != other.rank) {
      return false;
    }
    for (int32_t i = 0; i < rank; ++i) {
      if (dims[i] != other.dims[i]) {
        return false;
      }
    }
    return true;
  }
};

/// An integer operand of a ReserveFn: a constant (an attribute of the invoked
/// node) or a scalar value in the table.
struct ScalarRef {
  /// kAbsent: an optional operand the invoked node does not have.
  enum Kind : uint8_t { kConstant, kValue, kAbsent };

  Kind kind{kConstant};
  int32_t valueId{-1};
  int64_t constant{0};
};

/// A run of elements in one of a StepCode's arrays.
struct Span {
  int32_t begin{0};
  int32_t size{0};
};

/// A read-only run of elements, as a ReserveFn sees an integer list.
template <typename T>
struct ArrayView {
  const T* data{nullptr};
  int32_t count{0};

  size_t size() const {
    return static_cast<size_t>(count);
  }
  bool empty() const {
    return count == 0;
  }
  const T& operator[](size_t i) const {
    return data[i];
  }
  const T* begin() const {
    return data;
  }
  const T* end() const {
    return data + count;
  }
};

/// Names an operand of the formal node: an input ordinal or an input or
/// attribute name.
struct OperandName {
  /* implicit */ OperandName(int32_t ordinalIn) : ordinal(ordinalIn) {}
  /* implicit */ OperandName(const char* nameIn) : name(nameIn) {}

  int32_t ordinal{-1};
  const char* name{nullptr};
};

/// What a ReserveFn reads, so the bind pass can resolve it once per launch.
struct ReserveInputs {
  /// Tensor or TensorList inputs; a TensorList becomes one group per list.
  std::vector<OperandName> tensors;
  /// Integer inputs or attributes.
  std::vector<OperandName> scalars;
  /// Integer list inputs (a prim.ListPack of scalars) or attributes.
  std::vector<OperandName> intLists;
  /// Integer inputs or attributes the node may lack; they follow 'scalars' in
  /// ReserveArgs::scalars, as kAbsent when missing.
  std::vector<OperandName> optionalScalars;
  /// Integer tensor inputs whose elements the function reads. Each gets a host
  /// copy, made right after the step that produces it, and the function runs
  /// after that copy has landed.
  std::vector<OperandName> hostTensors;
  /// When set, one more tensor group after 'tensors': the values input
  /// 'elementwiseLeaves' of the node traces back to through the fused
  /// subgraph through elementwise producers.
  int32_t elementwiseLeaves{-1};
};

/// The operands of a ReserveFn as bind resolves them, by value id. Packed into
/// the step's flat code (StepCode), which ReserveArgs reads.
struct ReserveBinding {
  /// tensors[i] is tensor group i: one id for a tensor input, one per element
  /// for a TensorList.
  std::vector<std::vector<int32_t>> tensors;
  std::vector<ScalarRef> scalars;
  std::vector<std::vector<ScalarRef>> intLists;
  /// The ids of ReserveInputs::hostTensors, in order.
  std::vector<int32_t> hostTensors;
};

/// The operands of one ReserveFn call: views into the step's flat code, read
/// over the table. Made on the stack for each call; it allocates nothing.
struct ReserveArgs {
  /// The whole table, for the validity of the entries below.
  const ValueTable* table{nullptr};
  /// Tensor group i is the ids ids[groups[i].begin...].
  const Span* groups{nullptr};
  int32_t numGroups{0};
  const int32_t* ids{nullptr};
  const ScalarRef* scalars{nullptr};
  int32_t numScalars{0};
  /// Integer list i is scalarPool[intLists[i].begin...].
  const Span* intLists{nullptr};
  int32_t numLists{0};
  const ScalarRef* scalarPool{nullptr};
  /// The ids of ReserveInputs::hostTensors, in order.
  const int32_t* hostIds{nullptr};
  int32_t numHost{0};
  /// Threads per block of the launch, for outputs with one entry per block or
  /// per tile.
  int32_t blockSize{0};

  size_t groupSize(size_t group) const {
    return static_cast<size_t>(groups[group].size);
  }

  int32_t id(size_t group, size_t element = 0) const {
    return ids[groups[group].begin + element];
  }

  const ValueEntry& entry(size_t group, size_t element = 0) const {
    return table->entries[id(group, element)];
  }

  const TableTensor& tensor(size_t group, size_t element = 0) const {
    return entry(group, element).t;
  }

  int64_t scalar(size_t i) const {
    return scalarValue(scalars[i]);
  }

  int64_t scalarValue(const ScalarRef& ref) const {
    return ref.kind == ScalarRef::kValue ? table->entries[ref.valueId].scalar
                                         : ref.constant;
  }

  bool present(size_t i) const {
    return scalars[i].kind != ScalarRef::kAbsent;
  }

  int32_t numIntLists() const {
    return numLists;
  }

  ArrayView<ScalarRef> intList(size_t i) const {
    return {scalarPool + intLists[i].begin, intLists[i].size};
  }

  /// Element 'k' of host tensor 'i'. Its copy has landed: the step program
  /// orders the call after the transfer that brings it.
  int64_t hostInt(size_t i, int64_t k) const {
    const auto id = hostIds[i];
    const auto* data = table->cold[id].hostData;
    return table->entries[id].t.elementSize == sizeof(int32_t)
        ? static_cast<const int32_t*>(data)[k]
        : static_cast<const int64_t*>(data)[k];
  }
};

/// Computes the shapes of an op's outputs from pre-resolved operands. Writes
/// one Shape per output tensor into 'out' (several for a TensorList output);
/// 'numOut' is how many the caller provided. Returns false only on an internal
/// error; the step program runs it after everything it reads.
using ReserveFn = bool (*)(const ReserveArgs& args, Shape* out, int32_t numOut);

/// Whether every operand 'args' reads is in its table this execution. Builds
/// the fields of tensor entries published from the frame.
bool reserveOperandsReady(const ReserveArgs& args);

/// Built-in ReserveFns standing for a SizeExpr: group 0 of 'args' holds its
/// values and 'intLists' its constant shapes. A value absent from the table
/// counts as None, as it does in the frame.
///
/// The element count of kMax, kSum and the broadcast form, as a 1-d Shape.
/// kMax counts an integer value as that many elements.
bool maxCountReserve(const ReserveArgs& args, Shape* out, int32_t numOut);
bool sumCountReserve(const ReserveArgs& args, Shape* out, int32_t numOut);
bool broadcastCountReserve(const ReserveArgs& args, Shape* out, int32_t numOut);

/// The dims of kMax: the shape of the largest tensor value, the first one when
/// several are equal; rank -1 when no value is a tensor.
bool maxShapeReserve(const ReserveArgs& args, Shape* out, int32_t numOut);

/// The shape of the input the byLargestInput shortcut sizes outputs by, as
/// SizeExpr::numElements picks it: the tensor that was largest when the running
/// maximum over tensors and integers last moved to a tensor. Rank -1 when there
/// is none, which is also the case for the broadcast form.
bool largestInputReserve(const ReserveArgs& args, Shape* out, int32_t numOut);

/// The dims of the broadcast form: the values' and constant shapes'
/// right-aligned broadcast.
bool broadcastShapeReserve(const ReserveArgs& args, Shape* out, int32_t numOut);

} // namespace torch::wave
