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

#include <string>
#include <string_view>
#include <unordered_map>

#include <aten/src/ATen/core/function_schema.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

/// Describes element-wise operations like binary and unary arithmetic.
struct ElementwiseOp {
  std::string functionName;

  /// addition can have alpha
  int32_t numArgs{2};

  /// Attribute names that map to extra args in the CUDA function, e.g.
  /// {"alpha"} for add/sub, {"min", "max"} for clamp.
  std::vector<std::string> attributeArgs;
};

using Dim = uint32_t;

/// A map from value ids in a lambda matching OutputReserveFunc and
/// the value ids in the frame passed to OutputReserveFunc. The same
/// kernel operation can be invoked with many different sets of
/// inputs to produce sizes for each.
using FormalToActual = std::unordered_map<int32_t, int32_t>;

// Returns shapes to reserve for outputs given inputs. Can return multiple
// shapes for output params that are tuples of tensors.
using OutputReserveFunc = std::function<std::vector<std::vector<Dim>>(
    const nativert::Node* node,
    nativert::ExecutionFrame&,
    FormalToActual map)>;

struct ArgumentMeta {
  // If can be passed / returned in a register. Caller must read from / assign
  // to tensor if needed.
  bool isRegister{false};

  /// For outputs that are materialized tensors, function that determines the
  /// size to allocate based on inputs and execution state.
  OutputReserveFunc reserveShape{nullptr};

  /// True if actual size is determined on device, e.g. stream compaction.
  bool shapeSetOnDevice{false};
};

struct Metadata {
  enum Kind {
    kMetadata,
    kMapKernel,
    kMapAndBarrier,
  };

  Kind kind;

  const c10::FunctionSchema* functionSchema;

  /// The ordinal in arguments for the argument that determines the number of
  /// elements the kernel runs on. If many, the number of lanes to process y the
  /// kernel is the sum of the sizes of the args. If arg is a tensor, we get the
  /// number of elements. If it is a number or zero dim tensor, we use the
  /// number.
  std::vector<int32_t> sizeArgs;

  std::vector<ArgumentMeta> argumentMeta;

  std::vector<ArgumentMeta> returnMeta;

  /// True if requires a D to H transfer for return status, for example a length
  /// after stream compaction.
  bool hasReturn{false};

  /// If inputs are small, has a single block variant that is
  /// embeddable between __synthreads() and needs no kernel
  /// boundary. For example, stream compaction for under 10K elements
  /// is better as a single block than as 3 consecutive multiblock
  /// kernels.
  Metadata* singleBlockVariant{nullptr};

  /// True if all values to be computed by a block must be ready before calling.
  /// True for example for single block stream compaction or first stage of
  /// multi-kernel stream compaction.
  bool hasBarrier{false};

  /// If this represents a multiple kernel launch operation, e.g. a multiblock
  /// stream compaction, then each launch has its own Metadata Ops with dynamic
  /// number of launches will be standalone.
  std::unique_ptr<Metadata> nextKernel;

  /// The input can be overwritten and used as output if there are no concurrent
  /// or subsequent uses of input. True for example of elementwise arithmetic.
  bool inPlaceIfLastUse{false};

  /// True if must be launched as its own kernel sequence  with no fusion.
  bool isStandalone{false};

  /// Unit cost for scheduling, e.g. proportional block assignment.
  float cost{1.0f};

  std::unique_ptr<ElementwiseOp> elementWise;
};

class Registry {
 public:
  static void registerMetadata(std::string_view op, Metadata metadata);
  static const Metadata* metadata(std::string_view op);

  /// Registers an elementwise op by its qualified aten name (e.g.
  /// "torch.ops.aten.add.Tensor"). Looks up the FunctionSchema from the
  /// dispatcher, then creates a Metadata entry with sizeArgs={0},
  /// inPlaceIfLastUse=true, and an ElementwiseOp whose functionName is "--"
  /// followed by the op name part (e.g. "--add").
  static void registerElementwise(
      std::string_view qualifiedName,
      std::vector<std::string> attributeArgs = {});

 private:
  static std::unordered_map<std::string, Metadata>& registry();
};

void registerBuiltins();

} // namespace torch::wave
