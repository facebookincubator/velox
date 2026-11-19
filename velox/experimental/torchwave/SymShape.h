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
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <folly/container/F14Map.h>
#include <torch/csrc/utils/generated_serialization_types.h> // @manual=//caffe2:torch-cpp

namespace torch::nativert {
class Node;
class Value;
} // namespace torch::nativert

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
  /// For a view: the value whose storage it views, never itself a view, and
  /// where it starts in that storage, in elements. Empty for a value with
  /// storage of its own.
  std::string base;
  SymExpr offset{SymExpr::constant(0)};

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
  /// The export's shapes, for rules that line operands up with the dims the
  /// export traced, when these shapes are derived apart from the export's.
  const SymShapes* exported{nullptr};
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

/// The name under which SymShapes keeps the shape of element 'index' of the
/// list value 'list'.
std::string listElementName(std::string_view list, size_t index);

/// The shapes of the elements of list value 'list': the operands of its
/// ListPack, else the shapes recorded under listElementName. Empty when the
/// list's length is not known; null for an element without a shape.
std::vector<const SymShape*> listElementShapes(
    const SymShapes& shapes,
    const torch::nativert::Value* list);

/// The element count of 'shape': the product of its dims.
SymExpr numelOf(const SymShape& shape);

/// A one-dim contiguous shape of 'length' elements.
SymShape vectorShape(const SymExpr& length);

/// Adds to 'shapes' what the node's outputs' shapes are, from its inputs'
/// shapes: an output's shape under the output's name, a list output's
/// elements under listElementName. Returns how many shapes it added.
using SymShapeRule = std::function<
    int32_t(SymShapes& shapes, const torch::nativert::Node& node)>;

/// Registers the rule inferSymShapes applies to nodes of 'target', for ops
/// defined outside the core.
void registerSymShapeRule(std::string_view target, SymShapeRule rule);

/// Registers the rules of the aten view, shape and reduction ops.
void registerAtenShapeRules();

/// Expression builders that fold constants.
SymExpr symAdd(const SymExpr& lhs, const SymExpr& rhs);
SymExpr symMul(const SymExpr& lhs, const SymExpr& rhs);
SymExpr symMax(const SymExpr& lhs, const SymExpr& rhs);
SymExpr symFloorDiv(const SymExpr& lhs, const SymExpr& rhs);

/// The value 'expr' had when the export was traced, where its symbols are
/// graph-input dims named d<c> for a dim traced at c, or hostDimSymbols;
/// nullopt when it refers to another symbol.
std::optional<int64_t> tracedValue(const SymExpr& expr);

/// A symbol for dim 'dim' of the value named 'value', whose extent the host
/// learns when the op that makes it returns, traced at 'traced' in the
/// export.
SymExpr hostDimSymbol(std::string_view value, size_t dim, int64_t traced);

/// True when 'symbol' is a hostDimSymbol.
bool isHostDimSymbol(std::string_view symbol);

/// True when 'lhs' and 'rhs' are the same expression, or have the same traced
/// value. Exports specialize dims to their traced values, so a dim that is a
/// constant in one place is an input dim symbol in another.
bool sameDim(const SymExpr& lhs, const SymExpr& rhs);

/// A symbol for element 'index' of the integer tensor named 'tensor': an
/// extent the contents of a device tensor decide, such as one share of
/// cumulative lengths a kernel writes.
SymExpr elementSymbol(std::string_view tensor, int64_t index);

/// The tensor name and index an elementSymbol names, or nullopt for another
/// symbol.
std::optional<std::pair<std::string, int64_t>> parseElementSymbol(
    std::string_view symbol);

/// A shape with 'dims', laid out densely.
SymShape denseShape(std::vector<SymExpr> dims);

struct ReserveArgs;
struct Shape;

/// The operands of a SymReserveFn: what a ReserveFn reads, as expressions.
struct SymReserveArgs {
  /// Tensor group i, one shape per tensor; null for a tensor without one.
  std::vector<std::vector<const SymShape*>> tensors;
  /// Integer operands; nullopt for an absent optional one.
  std::vector<std::optional<SymExpr>> scalars;
  std::vector<std::vector<SymExpr>> intLists;
  /// Threads per block of the launch the reserve sizes for.
  int32_t blockSize{0};

  /// Tensor 'element' of group 'group', or null.
  const SymShape* tensor(size_t group, size_t element = 0) const {
    return group < tensors.size() && element < tensors[group].size()
        ? tensors[group][element]
        : nullptr;
  }
};

/// The symbolic form of a ReserveFn: the shapes of the outputs, one per
/// output tensor, 'numOut' of them. Returns false when an operand it needs
/// has no expression.
using SymReserveFn = bool (*)(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t numOut);

/// The type of a ReserveFn, declared here without Reserve.h.
using ReserveFnKey = bool (*)(const ReserveArgs&, Shape*, int32_t);

/// Gives the symbolic form of a ReserveFn, which sizing on the device
/// evaluates instead of calling the function on the host.
void registerSymReserve(ReserveFnKey fn, SymReserveFn sym);

/// The symbolic form of 'fn', or null.
SymReserveFn symReserveOf(ReserveFnKey fn);

/// SymReserveFns many ReserveFns share. Each output is one-dim:
/// tensor 0's element count, that plus one, scalar 0, tensor 0's first dim,
/// scalar 0 for every output, and tensor (0, i)'s element count for output i.
bool symNumel(const SymReserveArgs& args, std::vector<SymShape>& out, int32_t);
bool symNumelPlusOne(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t);
bool symScalarLength(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t);
bool symFirstDim(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t);
bool symRepeatedScalar(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t numOut);
bool symListNumel(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t numOut);

/// Adds what one node's rule gives for its outputs. Returns how many shapes
/// it added.
int32_t inferNodeShapes(SymShapes& shapes, const torch::nativert::Node& node);

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
