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

#include "velox/experimental/torchwave/SymShape.h"

#include "velox/experimental/torchwave/StaticDims.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <fmt/format.h>
#include <folly/CppAttributes.h>
#include <folly/container/F14Set.h>
#include <algorithm>
#include <cctype>

namespace torch::wave {

namespace {

// A recursive-descent reader of the srepr grammar the exporter writes:
//   expr   := Head '(' args ')'
//   Head   := Symbol | Integer | Add | Mul | Max | Min | Pow | FloorDiv | Mod
//   Symbol takes a quoted name and keyword flags; Integer an integer literal.
class Reader {
 public:
  explicit Reader(std::string_view text) : text_(text) {}

  std::optional<SymExpr> expr() {
    skipSpace();
    const auto head = identifier();
    if (head.empty() || !consume('(')) {
      return std::nullopt;
    }
    if (head == "Integer") {
      auto value = integer();
      if (!value || !consume(')')) {
        return std::nullopt;
      }
      return SymExpr::constant(*value);
    }
    if (head == "Symbol") {
      auto name = quoted();
      if (!name) {
        return std::nullopt;
      }
      // Keyword flags such as integer=True, positive=True.
      while (consume(',')) {
        skipSpace();
        identifier();
        if (!consume('=')) {
          return std::nullopt;
        }
        skipSpace();
        identifier();
      }
      if (!consume(')')) {
        return std::nullopt;
      }
      return SymExpr{.kind = SymExpr::Kind::kSymbol, .symbol = *name};
    }
    SymExpr::Kind kind;
    if (head == "Add") {
      kind = SymExpr::Kind::kAdd;
    } else if (head == "Mul") {
      kind = SymExpr::Kind::kMul;
    } else if (head == "Max") {
      kind = SymExpr::Kind::kMax;
    } else if (head == "Min") {
      kind = SymExpr::Kind::kMin;
    } else if (head == "Pow") {
      kind = SymExpr::Kind::kPow;
    } else if (head == "FloorDiv") {
      kind = SymExpr::Kind::kFloorDiv;
    } else if (head == "Mod") {
      kind = SymExpr::Kind::kMod;
    } else {
      return std::nullopt;
    }
    SymExpr result{.kind = kind};
    do {
      auto arg = expr();
      if (!arg) {
        return std::nullopt;
      }
      result.args.push_back(std::move(*arg));
    } while (consume(','));
    if (!consume(')') || result.args.empty()) {
      return std::nullopt;
    }
    return result;
  }

  bool atEnd() {
    skipSpace();
    return pos_ == text_.size();
  }

 private:
  void skipSpace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char c) {
    skipSpace();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  std::string_view identifier() {
    const auto start = pos_;
    while (pos_ < text_.size() &&
           (std::isalnum(static_cast<unsigned char>(text_[pos_])) ||
            text_[pos_] == '_')) {
      ++pos_;
    }
    return text_.substr(start, pos_ - start);
  }

  std::optional<int64_t> integer() {
    skipSpace();
    const auto start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ == start) {
      return std::nullopt;
    }
    return std::stoll(std::string(text_.substr(start, pos_ - start)));
  }

  std::optional<std::string> quoted() {
    skipSpace();
    if (pos_ >= text_.size() || (text_[pos_] != '\'' && text_[pos_] != '"')) {
      return std::nullopt;
    }
    const char quote = text_[pos_++];
    const auto start = pos_;
    while (pos_ < text_.size() && text_[pos_] != quote) {
      ++pos_;
    }
    if (pos_ >= text_.size()) {
      return std::nullopt;
    }
    return std::string(text_.substr(start, pos_++ - start));
  }

  std::string_view text_;
  size_t pos_{0};
};

int64_t floorDiv(int64_t dividend, int64_t divisor) {
  const auto quotient = dividend / divisor;
  return (dividend % divisor != 0 && ((dividend < 0) != (divisor < 0)))
      ? quotient - 1
      : quotient;
}

} // namespace

std::optional<SymExpr> parseSympy(std::string_view text) {
  Reader reader(text);
  auto result = reader.expr();
  if (!result || !reader.atEnd()) {
    return std::nullopt;
  }
  return result;
}

std::optional<int64_t> SymExpr::evaluate(
    const folly::F14FastMap<std::string, int64_t>& symbols) const {
  switch (kind) {
    case Kind::kConst:
      return value;
    case Kind::kSymbol: {
      auto it = symbols.find(symbol);
      if (it == symbols.end()) {
        return std::nullopt;
      }
      return it->second;
    }
    case Kind::kAdd:
    case Kind::kMul:
    case Kind::kMax:
    case Kind::kMin:
    case Kind::kPow:
    case Kind::kFloorDiv:
    case Kind::kMod:
      break;
  }
  std::vector<int64_t> values;
  values.reserve(args.size());
  for (const auto& arg : args) {
    auto v = arg.evaluate(symbols);
    if (!v) {
      return std::nullopt;
    }
    values.push_back(*v);
  }
  switch (kind) {
    case Kind::kAdd: {
      int64_t sum = 0;
      for (auto v : values) {
        sum += v;
      }
      return sum;
    }
    case Kind::kMul: {
      int64_t product = 1;
      for (auto v : values) {
        product *= v;
      }
      return product;
    }
    case Kind::kMax:
      return *std::max_element(values.begin(), values.end());
    case Kind::kMin:
      return *std::min_element(values.begin(), values.end());
    case Kind::kPow: {
      if (values.size() != 2 || values[1] < 0) {
        return std::nullopt;
      }
      int64_t result = 1;
      for (int64_t i = 0; i < values[1]; ++i) {
        result *= values[0];
      }
      return result;
    }
    case Kind::kFloorDiv:
      if (values.size() != 2 || values[1] == 0) {
        return std::nullopt;
      }
      return floorDiv(values[0], values[1]);
    case Kind::kMod:
      if (values.size() != 2 || values[1] == 0) {
        return std::nullopt;
      }
      return values[0] - floorDiv(values[0], values[1]) * values[1];
    case Kind::kConst:
    case Kind::kSymbol:
      break;
  }
  return std::nullopt;
}

void SymExpr::collectSymbols(std::vector<std::string>& out) const {
  if (kind == Kind::kSymbol) {
    if (std::find(out.begin(), out.end(), symbol) == out.end()) {
      out.push_back(symbol);
    }
    return;
  }
  for (const auto& arg : args) {
    arg.collectSymbols(out);
  }
}

std::string SymExpr::toString() const {
  switch (kind) {
    case Kind::kConst:
      return std::to_string(value);
    case Kind::kSymbol:
      return symbol;
    case Kind::kAdd:
    case Kind::kMul:
    case Kind::kMax:
    case Kind::kMin:
    case Kind::kPow:
    case Kind::kFloorDiv:
    case Kind::kMod:
      break;
  }
  auto join = [&](const char* separator) {
    std::string out;
    for (size_t i = 0; i < args.size(); ++i) {
      if (i > 0) {
        out += separator;
      }
      const bool nested =
          args[i].kind != Kind::kConst && args[i].kind != Kind::kSymbol;
      out += nested ? "(" + args[i].toString() + ")" : args[i].toString();
    }
    return out;
  };
  switch (kind) {
    case Kind::kAdd:
      return join(" + ");
    case Kind::kMul:
      return join("*");
    case Kind::kPow:
      return join("**");
    case Kind::kMax:
      return "Max(" + join(", ") + ")";
    case Kind::kMin:
      return "Min(" + join(", ") + ")";
    case Kind::kFloorDiv:
      return "FloorDiv(" + join(", ") + ")";
    case Kind::kMod:
      return "Mod(" + join(", ") + ")";
    case Kind::kConst:
    case Kind::kSymbol:
      break;
  }
  return "?";
}

bool SymShape::dependsOn(char prefix) const {
  std::vector<std::string> symbols;
  for (const auto* list : {&dims, &strides}) {
    for (const auto& expr : *list) {
      expr.collectSymbols(symbols);
    }
  }
  return std::any_of(symbols.begin(), symbols.end(), [&](const auto& name) {
    return !name.empty() && name[0] == prefix;
  });
}

void addSymShape(
    SymShapes& shapes,
    const std::string& name,
    const std::vector<SymDimPart>& dims,
    const std::vector<SymDimPart>& strides) {
  SymShape shape;
  auto convert = [&](const std::vector<SymDimPart>& parts,
                     std::vector<SymExpr>& out) {
    for (const auto& part : parts) {
      if (const auto* value = std::get_if<int64_t>(&part)) {
        out.push_back(SymExpr::constant(*value));
        continue;
      }
      auto expr = parseSympy(std::get<std::string>(part));
      if (!expr) {
        return false;
      }
      out.push_back(std::move(*expr));
    }
    return true;
  };
  if (!convert(dims, shape.dims) || !convert(strides, shape.strides)) {
    shapes.unparsed.push_back(name);
    return;
  }
  shapes.byName.emplace(name, std::move(shape));
}

void addSymScalar(
    SymShapes& shapes,
    const std::string& name,
    const SymDimPart& value) {
  if (const auto* constant = std::get_if<int64_t>(&value)) {
    shapes.scalarByName.emplace(name, SymExpr::constant(*constant));
    return;
  }
  auto expr = parseSympy(std::get<std::string>(value));
  if (!expr) {
    shapes.unparsed.push_back(name);
    return;
  }
  shapes.scalarByName.emplace(name, std::move(*expr));
}

SymShapes symbolicShapes(const torch::_export::Graph& graph) {
  auto parts = [](const std::vector<torch::_export::SymInt>& values) {
    std::vector<SymDimPart> result;
    for (const auto& value : values) {
      if (value.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
        result.emplace_back(value.get_as_expr().get_expr_str());
      } else {
        result.emplace_back(value.get_as_int());
      }
    }
    return result;
  };
  SymShapes result;
  for (const auto& [name, meta] : graph.get_tensor_values()) {
    addSymShape(
        result, name, parts(meta.get_sizes()), parts(meta.get_strides()));
  }
  for (const auto& [name, value] : graph.get_sym_int_values()) {
    addSymScalar(result, name, parts({value})[0]);
  }
  return result;
}

namespace {

SymExpr product(const SymExpr& lhs, const SymExpr& rhs) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return SymExpr::constant(lhs.value * rhs.value);
  }
  if (lhs.isConstant() && lhs.value == 1) {
    return rhs;
  }
  if (rhs.isConstant() && rhs.value == 1) {
    return lhs;
  }
  return SymExpr{.kind = SymExpr::Kind::kMul, .args = {lhs, rhs}};
}

std::vector<SymExpr> contiguousStrides(const std::vector<SymExpr>& dims) {
  std::vector<SymExpr> strides(dims.size(), SymExpr::constant(1));
  for (auto i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = product(strides[i + 1], dims[i + 1]);
  }
  return strides;
}

bool sameExpr(const SymExpr& lhs, const SymExpr& rhs) {
  return lhs.toString() == rhs.toString();
}

// Right-aligned broadcast of symbolic dims; nullopt where two operands have
// different non-1 extents that the expressions do not show to be equal.
std::optional<std::vector<SymExpr>> broadcastDims(
    const std::vector<const SymShape*>& operands) {
  size_t rank = 0;
  for (const auto* shape : operands) {
    rank = std::max(rank, shape->dims.size());
  }
  std::vector<SymExpr> result(rank, SymExpr::constant(1));
  for (size_t i = 0; i < rank; ++i) {
    auto& out = result[rank - 1 - i];
    for (const auto* shape : operands) {
      if (i >= shape->dims.size()) {
        continue;
      }
      const auto& dim = shape->dims[shape->dims.size() - 1 - i];
      if (dim.isConstant() && dim.value == 1) {
        continue;
      }
      if (out.isConstant() && out.value == 1) {
        out = dim;
      } else if (!sameExpr(out, dim)) {
        return std::nullopt;
      }
    }
  }
  return result;
}

} // namespace

bool keepsSelfShape(std::string_view target) {
  static const folly::F14FastSet<std::string_view> kTargets{
      "torch.ops.aten.add.Scalar",
      "torch.ops.aten.sub.Scalar",
      "torch.ops.aten.mul.Scalar",
      "torch.ops.aten.div.Scalar",
      "torch.ops.aten.rsub.Scalar",
      "torch.ops.aten.remainder.Scalar",
      "torch.ops.aten.neg.default",
      "torch.ops.aten.abs.default",
      "torch.ops.aten.bitwise_not.default",
      "torch.ops.aten.to.dtype",
      "torch.ops.aten._to_copy.default",
      "torch.ops.aten.clone.default",
      "torch.ops.aten.ones_like.default",
      "torch.ops.aten.zeros_like.default",
      "torch.ops.aten.full_like.default",
      "torch.ops.aten.clamp.default",
      "torch.ops.aten.cumsum.default",
      "tw.cumsum_cg",
  };
  return kTargets.count(target) != 0;
}

bool broadcastsOperands(std::string_view target) {
  static const folly::F14FastSet<std::string_view> kTargets{
      "torch.ops.aten.add.Tensor",
      "torch.ops.aten.sub.Tensor",
      "torch.ops.aten.mul.Tensor",
      "torch.ops.aten.eq.Tensor",
      "torch.ops.aten.ne.Tensor",
      "torch.ops.aten.lt.Tensor",
      "torch.ops.aten.le.Tensor",
      "torch.ops.aten.gt.Tensor",
      "torch.ops.aten.ge.Tensor",
      "torch.ops.aten.bitwise_and.Tensor",
      "torch.ops.aten.bitwise_or.Tensor",
      "torch.ops.aten.where.self",
  };
  return kTargets.count(target) != 0;
}

namespace {

const SymShape* FOLLY_NULLABLE shapeOf(const SymShapes& shapes, ValueCP value) {
  if (value == nullptr) {
    return nullptr;
  }
  auto it = shapes.byName.find(std::string(value->name()));
  return it != shapes.byName.end() ? &it->second : nullptr;
}

std::optional<SymExpr> scalarOf(const SymShapes& shapes, NodeCP node) {
  const auto* input = node->tryGetInput("end");
  if (input != nullptr && input->value != nullptr) {
    auto it = shapes.scalarByName.find(std::string(input->value->name()));
    if (it != shapes.scalarByName.end()) {
      return it->second;
    }
    return std::nullopt;
  }
  const auto* attr = node->tryGetAttribute("end");
  if (attr != nullptr && std::holds_alternative<int64_t>(attr->value)) {
    return SymExpr::constant(std::get<int64_t>(attr->value));
  }
  return std::nullopt;
}

std::optional<SymShape> inferNode(const SymShapes& shapes, NodeCP node) {
  const auto target = node->target();
  if (target == "tw.arange_mem") {
    auto end = scalarOf(shapes, node);
    if (!end) {
      return std::nullopt;
    }
    return SymShape{.dims = {*end}, .strides = {SymExpr::constant(1)}};
  }
  // Row reductions keep the leading dims: any_rows the first 'keep', sum_rows
  // all but the last.
  if (target == "tw.any_rows" || target == "tw.sum_rows") {
    const auto* self = node->tryGetInput("self");
    const auto* shape =
        self != nullptr ? shapeOf(shapes, self->value) : nullptr;
    if (shape == nullptr) {
      return std::nullopt;
    }
    auto keep = static_cast<int64_t>(shape->dims.size()) - 1;
    if (target == "tw.any_rows") {
      const auto* attr = node->tryGetAttribute("keep");
      if (attr == nullptr || !std::holds_alternative<int64_t>(attr->value)) {
        return std::nullopt;
      }
      keep = std::get<int64_t>(attr->value);
    }
    if (keep < 1 || keep > static_cast<int64_t>(shape->dims.size())) {
      return std::nullopt;
    }
    std::vector<SymExpr> dims(shape->dims.begin(), shape->dims.begin() + keep);
    return SymShape{.dims = dims, .strides = contiguousStrides(dims)};
  }
  // Gathers that are elementwise over their index list: one element of
  // 'self' per broadcast index position.
  if (target == "tw.index_elt_one" || target == "tw.index_elt_one_default") {
    const auto* indices = node->tryGetInput("indices");
    const auto* pack = indices != nullptr && indices->value != nullptr
        ? indices->value->producer()
        : nullptr;
    if (pack == nullptr || pack->target() != "prim.ListPack") {
      return std::nullopt;
    }
    std::vector<const SymShape*> operands;
    for (const auto& input : pack->inputs()) {
      const auto* shape = shapeOf(shapes, input.value);
      if (shape == nullptr) {
        return std::nullopt;
      }
      operands.push_back(shape);
    }
    auto dims = broadcastDims(operands);
    if (!dims) {
      return std::nullopt;
    }
    return SymShape{.dims = *dims, .strides = contiguousStrides(*dims)};
  }
  if (keepsSelfShape(target)) {
    const auto* self = node->tryGetInput("self");
    if (self == nullptr) {
      self = node->tryGetInput("input");
    }
    const auto* shape =
        self != nullptr ? shapeOf(shapes, self->value) : nullptr;
    if (shape == nullptr) {
      return std::nullopt;
    }
    return SymShape{
        .dims = shape->dims, .strides = contiguousStrides(shape->dims)};
  }
  if (broadcastsOperands(target)) {
    std::vector<const SymShape*> operands;
    for (const auto& input : node->inputs()) {
      if (input.value == nullptr ||
          input.value->type().kind() != nativert::Type::Kind::Tensor) {
        continue;
      }
      const auto* shape = shapeOf(shapes, input.value);
      if (shape == nullptr) {
        return std::nullopt;
      }
      operands.push_back(shape);
    }
    auto dims = broadcastDims(operands);
    if (!dims) {
      return std::nullopt;
    }
    return SymShape{.dims = *dims, .strides = contiguousStrides(*dims)};
  }
  return std::nullopt;
}

} // namespace

int32_t inferSymShapes(SymShapes& shapes, WaveGraph& waveGraph) {
  int32_t numAdded = 0;
  // A rewrite's replacement equals the value it replaced. Chains of
  // replacements resolve over repeated passes.
  for (bool changed = true; changed;) {
    changed = false;
    for (const auto& [newName, oldName] : waveGraph.replacements()) {
      if (shapes.byName.count(newName) == 0) {
        auto it = shapes.byName.find(oldName);
        if (it != shapes.byName.end()) {
          shapes.byName.emplace(newName, it->second);
          ++numAdded;
          changed = true;
        }
      }
      if (shapes.scalarByName.count(newName) == 0) {
        auto it = shapes.scalarByName.find(oldName);
        if (it != shapes.scalarByName.end()) {
          shapes.scalarByName.emplace(newName, it->second);
          changed = true;
        }
      }
    }
  }
  auto infer = [&](const nativert::Node& node) {
    const auto& outputs = node.outputs();
    // A cooperative-grid scan adds its per-block partials as a second output.
    const size_t numOutputs = node.target() == "tw.cumsum_cg" ? 2 : 1;
    if (outputs.size() != numOutputs ||
        outputs[0]->type().kind() != nativert::Type::Kind::Tensor ||
        shapes.byName.count(std::string(outputs[0]->name())) != 0) {
      return false;
    }
    auto shape = inferNode(shapes, &node);
    if (!shape) {
      return false;
    }
    shapes.byName.emplace(std::string(outputs[0]->name()), std::move(*shape));
    ++numAdded;
    return true;
  };
  for (const auto& node : waveGraph.graph()->nodes()) {
    infer(node);
  }
  // Nodes of variants' graphs (a cg scan) are reached through their values.
  for (bool changed = true; changed;) {
    changed = false;
    for (const auto& [id, value] : waveGraph.idToValue()) {
      if (value->producer() != nullptr && infer(*value->producer())) {
        changed = true;
      }
    }
  }
  // The total a repeat_interleave head returns is the length of the final's
  // output along the interleave axis. The head and final can live in a
  // variant's graph, so they are found through their values.
  for (const auto& [id, value] : waveGraph.idToValue()) {
    const auto* producer = value->producer();
    if (producer == nullptr ||
        (producer->target() != "tw.repeat_interleave_final" &&
         producer->target() != "tw.repeat_interleave_index_final") ||
        producer->outputs().size() != 1) {
      continue;
    }
    const auto& node = *producer;
    const auto* total = node.tryGetInput("total");
    const auto* shape = shapeOf(shapes, node.outputs()[0]);
    if (total == nullptr || total->value == nullptr || shape == nullptr ||
        shapes.scalarByName.count(std::string(total->value->name())) != 0) {
      continue;
    }
    int64_t axis = 0;
    if (const auto* dim = node.tryGetAttribute("dim"); dim != nullptr &&
        std::holds_alternative<int64_t>(dim->value) &&
        std::get<int64_t>(dim->value) >= 0) {
      axis = std::get<int64_t>(dim->value);
    }
    if (axis < static_cast<int64_t>(shape->dims.size())) {
      shapes.scalarByName.emplace(
          std::string(total->value->name()), shape->dims[axis]);
    }
  }
  return numAdded;
}

std::string symShapeCoverage(const SymShapes& shapes, WaveGraph& waveGraph) {
  int32_t numTensors = 0;
  int32_t numStatic = 0;
  int32_t numInputDims = 0;
  int32_t numDataDependent = 0;
  int32_t numUnknown = 0;
  std::vector<std::string> unknownTargets;
  for (const auto& [id, value] : waveGraph.idToValue()) {
    if (value->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    ++numTensors;
    auto it = shapes.byName.find(std::string(value->name()));
    if (it != shapes.byName.end()) {
      if (it->second.dependsOn('u')) {
        ++numDataDependent;
      } else if (it->second.dependsOn('s')) {
        ++numInputDims;
      } else {
        ++numStatic;
      }
      continue;
    }
    if (isFullyStatic(waveGraph.types(), value)) {
      ++numStatic;
      continue;
    }
    ++numUnknown;
    if (value->producer() != nullptr) {
      unknownTargets.emplace_back(value->producer()->target());
    }
  }
  std::sort(unknownTargets.begin(), unknownTargets.end());
  std::string byTarget;
  for (size_t i = 0; i < unknownTargets.size();) {
    size_t j = i;
    while (j < unknownTargets.size() &&
           unknownTargets[j] == unknownTargets[i]) {
      ++j;
    }
    byTarget += fmt::format(" {} x{}", unknownTargets[i], j - i);
    i = j;
  }
  return fmt::format(
      "Symbolic shapes: {} tensor values; {} static, {} by input dims, {} "
      "data-dependent, {} without an expression; {} export expressions did not "
      "parse\n  without an expression, by producer:{}\n",
      numTensors,
      numStatic,
      numInputDims,
      numDataDependent,
      numUnknown,
      shapes.unparsed.size(),
      byTarget);
}

} // namespace torch::wave
