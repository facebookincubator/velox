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
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <folly/container/F14Map.h>
#include <torch/csrc/utils/generated_serialization_types.h> // @manual=//caffe2:torch-cpp

namespace torch::wave {

class WaveGraph;

/// A size or stride expression over the export's symbols: s<N> for an input
/// dim, u<N> for a value only the data decides (an item(), the count of a
/// masked_select). Parsed from the sympy srepr strings an exported program
/// carries for every dim it could not fix, which nativert's TensorMeta turns
/// into -1.
struct SymExpr {
  enum class Kind : uint8_t {
    kConst,
    kSymbol,
    kAdd,
    kMul,
    kMax,
    kMin,
    kPow,
    kFloorDiv,
    kMod,
  };

  Kind kind{Kind::kConst};
  int64_t value{0};
  /// The symbol's name, for kSymbol.
  std::string symbol;
  std::vector<SymExpr> args;

  static SymExpr constant(int64_t value) {
    return SymExpr{.kind = Kind::kConst, .value = value};
  }

  bool isConstant() const {
    return kind == Kind::kConst;
  }

  /// The value with each symbol taken from 'symbols', or nullopt when one is
  /// missing.
  std::optional<int64_t> evaluate(
      const folly::F14FastMap<std::string, int64_t>& symbols) const;

  /// Adds the names of the symbols this refers to.
  void collectSymbols(std::vector<std::string>& out) const;

  /// Infix form, e.g. "1000*u1".
  std::string toString() const;
};

/// Parses one sympy srepr string, e.g. "Mul(Integer(799), Symbol('u0',
/// integer=True))". Returns nullopt for a form it does not know.
std::optional<SymExpr> parseSympy(std::string_view text);

/// The symbolic sizes and strides of one tensor value. A dim the export fixed
/// is a constant.
struct SymShape {
  std::vector<SymExpr> dims;
  std::vector<SymExpr> strides;

  /// True when some dim or stride depends on a symbol whose name starts with
  /// 'prefix' ('u' for data-dependent, 's' for input dims).
  bool dependsOn(char prefix) const;
};

/// The shapes of an exported graph's tensors by value name, fixed dims as
/// constants, and the names of the values whose expressions would not parse.
struct SymShapes {
  folly::F14FastMap<std::string, SymShape> byName;
  /// Integer values (an item(), a sym_size) by name.
  folly::F14FastMap<std::string, SymExpr> scalarByName;
  std::vector<std::string> unparsed;
};

/// One dim or stride as the serialized program has it: a fixed value or a
/// sympy srepr string.
using SymDimPart = std::variant<int64_t, std::string>;

/// Records the shape of tensor 'name' in 'shapes'. Front ends that hold the
/// program in another serialization (the thrift one) convert to parts and call
/// this.
void addSymShape(
    SymShapes& shapes,
    const std::string& name,
    const std::vector<SymDimPart>& dims,
    const std::vector<SymDimPart>& strides);

/// Records the integer value 'name' in 'shapes'.
void addSymScalar(
    SymShapes& shapes,
    const std::string& name,
    const SymDimPart& value);

/// The symbolic sizes and strides of every tensor value of the export
/// 'graph', and its symbolic integer values, parsed from their sympy strings.
SymShapes symbolicShapes(const torch::_export::Graph& graph);

/// True for the elementwise ops whose output is the broadcast of their tensor
/// operands' shapes.
bool broadcastsOperands(std::string_view target);

/// True for the ops whose output has the shape of input 'self'.
bool keepsSelfShape(std::string_view target);

/// Adds shapes for the tensor values of 'waveGraph' that the export does not
/// describe -- the outputs of nodes the graph rewrites created -- where the
/// node's rule gives them from its inputs' shapes. Returns how many it added.
int32_t inferSymShapes(SymShapes& shapes, WaveGraph& waveGraph);

/// How many tensor values of 'waveGraph' have dims fixed at compile time, how
/// many have export expressions (split into input-dim and data-dependent
/// ones), and how many have neither: values the graph rewrites created, whose
/// expressions have to be inferred.
std::string symShapeCoverage(const SymShapes& shapes, WaveGraph& waveGraph);

} // namespace torch::wave
