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

#include "velox/experimental/torchwave/Standalones.h"

#include <algorithm>
#include <limits>
#include <optional>

#include <ATen/ATen.h>
#include <c10/core/WrapDimMinimal.h>

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

namespace {

// Shape and strides of 'self', in buffers the caller can edit before handing
// them to aliasTensor. Rank is bounded by kMaxDims in practice, but a shortcut
// can also run on a graph value the kernels never see, so this is not fixed.
using DimVector = c10::SmallVector<int64_t, 5>;

struct Geometry {
  DimVector sizes;
  DimVector strides;
};

Geometry geometryOf(const at::Tensor& self) {
  const auto sizes = self.sizes();
  const auto strides = self.strides();
  return {
      DimVector(sizes.begin(), sizes.end()),
      DimVector(strides.begin(), strides.end())};
}

} // namespace

void runStandaloneShortcut(
    const LaunchData& data,
    nativert::ExecutionFrame& frame) {
  const auto& args = data.args;
  const auto& intArgs = data.intArgs;

  // Reads operand 'i' as a tensor from the frame.
  auto tensorAt = [&](size_t i) -> at::Tensor {
    const auto& iv = frame.getIValue(args[i]->id());
    TORCH_CHECK(
        iv.isTensor(),
        "runStandaloneShortcut: shortcut ",
        static_cast<int>(data.launch->standaloneShortcut),
        " operand ",
        i,
        " value %",
        args[i]->id(),
        " is not a tensor (tag=",
        static_cast<int>(iv.tag),
        ")");
    return iv.toTensor();
  };
  // Reads operand 'i' as an integer: a dynamic value (args[i] set) is read from
  // the frame; a constant comes from intArgs. A dynamic bound may be a SymInt
  // (a data-dependent size from item()/sym_size), which is not a plain Int
  // IValue, so read it through the SymInt.
  auto intAt = [&](size_t i) -> int64_t {
    if (args[i] == nullptr) {
      return static_cast<int64_t>(intArgs[i]);
    }
    const auto& iv = frame.getIValue(args[i]->id());
    return iv.isSymInt() ? iv.toSymInt().guard_int(__FILE__, __LINE__)
                         : iv.toInt();
  };
  // Reads an optional int operand for slice start/end: a None value (an omitted
  // bound, e.g. from t[:end]) becomes nullopt; otherwise as intAt.
  auto optIntAt = [&](size_t i) -> std::optional<int64_t> {
    if (args[i] != nullptr && frame.getIValue(args[i]->id()).isNone()) {
      return std::nullopt;
    }
    return intAt(i);
  };
  auto setOutput = [&](c10::IValue value) {
    frame.setIValue(data.actualOutputs[0], std::move(value));
  };

  switch (data.launch->standaloneShortcut) {
    case StandaloneShortcut::kView: {
      auto self = tensorAt(0);
      // All-constant dims pass through directly; otherwise the dims come from a
      // value operand (an int list in the frame).
      if (!data.intList.empty()) {
        setOutput(self.view(data.intList));
      } else {
        auto list = frame.getIValue(args[1]->id()).toIntList();
        setOutput(self.view(std::vector<int64_t>(list.begin(), list.end())));
      }
      break;
    }
    case StandaloneShortcut::kSlice: {
      // (Tensor self, int dim, int? start, int? end, int step). start/end are
      // optional (None) and may be data-dependent SymInts (e.g. t[:end] where
      // end is a runtime value), so the clamping below is load-bearing, not a
      // formality -- it is aten's slice() semantics reproduced verbatim.
      auto self = tensorAt(0);
      const auto rank = self.dim();
      TORCH_CHECK_INDEX(
          rank > 0, "slice() cannot be applied to a 0-dim tensor.");
      const auto dim = c10::maybe_wrap_dim(intAt(1), rank);
      const auto step = intAt(4);
      TORCH_CHECK(step > 0, "slice step must be positive");
      auto geometry = geometryOf(self);
      const auto extent = geometry.sizes[dim];
      // An omitted end is aten's open-ended INT64_MAX, which the clamp below
      // turns into 'extent'; it must not be confused with a negative bound.
      auto start = optIntAt(2).value_or(0);
      auto end = optIntAt(3).value_or(std::numeric_limits<int64_t>::max());
      if (start < 0) {
        start += extent;
      }
      if (end < 0) {
        end += extent;
      }
      start = std::clamp<int64_t>(start, 0, extent);
      end = std::clamp<int64_t>(end, start, extent);
      const auto storageOffset =
          self.storage_offset() + start * geometry.strides[dim];
      geometry.sizes[dim] = (end - start + step - 1) / step; // round up
      geometry.strides[dim] *= step;
      setOutput(
          aliasTensor(self, geometry.sizes, geometry.strides, storageOffset));
      break;
    }
    case StandaloneShortcut::kSelectInt: {
      // (Tensor self, int dim, int index). Selecting drops 'dim'.
      auto self = tensorAt(0);
      const auto rank = self.dim();
      TORCH_CHECK_INDEX(
          rank > 0, "select() cannot be applied to a 0-dim tensor.");
      const auto dim = c10::maybe_wrap_dim(intAt(1), rank);
      auto geometry = geometryOf(self);
      const auto extent = geometry.sizes[dim];
      auto index = intAt(2);
      if (index < 0) {
        index += extent;
      }
      TORCH_CHECK_INDEX(
          index >= 0 && index < extent,
          "select(): index ",
          intAt(2),
          " out of range for tensor of size ",
          self.sizes(),
          " at dimension ",
          dim);
      const auto storageOffset =
          self.storage_offset() + index * geometry.strides[dim];
      geometry.sizes.erase(geometry.sizes.begin() + dim);
      geometry.strides.erase(geometry.strides.begin() + dim);
      setOutput(
          aliasTensor(self, geometry.sizes, geometry.strides, storageOffset));
      break;
    }
    case StandaloneShortcut::kUnsqueeze: {
      // (Tensor self, int dim).
      auto self = tensorAt(0);
      setOutput(at::unsqueeze(self, intAt(1)));
      break;
    }
    case StandaloneShortcut::kTranspose: {
      // (Tensor self, int dim0, int dim1).
      auto self = tensorAt(0);
      setOutput(at::transpose(self, intAt(1), intAt(2)));
      break;
    }
    case StandaloneShortcut::kNarrow: {
      // (Tensor self, int dim, int start, int length). Unlike slice, narrow
      // rejects an out-of-range range instead of clamping it.
      auto self = tensorAt(0);
      const auto rank = self.dim();
      TORCH_CHECK(rank > 0, "narrow() cannot be applied to a 0-dim tensor.");
      const auto dim = c10::maybe_wrap_dim(intAt(1), rank);
      auto geometry = geometryOf(self);
      const auto extent = geometry.sizes[dim];
      const auto length = intAt(3);
      auto start = intAt(2);
      TORCH_CHECK(length >= 0, "narrow(): length must be non-negative.");
      TORCH_CHECK_INDEX(
          -extent <= start && start <= extent,
          "start out of range (expected to be in range of [",
          -extent,
          ", ",
          extent,
          "], but got ",
          start,
          ")");
      if (start < 0) {
        start += extent;
      }
      TORCH_CHECK(
          start <= extent - length,
          "start (",
          start,
          ") + length (",
          length,
          ") exceeds dimension size (",
          extent,
          ").");
      const auto storageOffset =
          self.storage_offset() + start * geometry.strides[dim];
      geometry.sizes[dim] = length;
      setOutput(
          aliasTensor(self, geometry.sizes, geometry.strides, storageOffset));
      break;
    }
    case StandaloneShortcut::kListPack: {
      c10::List<at::Tensor> list;
      list.reserve(args.size());
      for (auto* value : args) {
        const auto& iv = frame.getIValue(value->id());
        TORCH_CHECK(
            iv.isTensor(),
            "runStandaloneShortcut: kListPack element %",
            value->id(),
            " is not a tensor (tag=",
            static_cast<int>(iv.tag),
            ")");
        list.push_back(iv.toTensor());
      }
      setOutput(c10::IValue(std::move(list)));
      break;
    }
    case StandaloneShortcut::kNone:
      break;
  }
}

} // namespace torch::wave
