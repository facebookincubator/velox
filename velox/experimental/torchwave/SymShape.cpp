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
// different non-1 extents that sameDim does not take to be equal. Of two
// extents that are the same dim, the one that is not a constant is kept.
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
      } else if (!sameDim(out, dim)) {
        return std::nullopt;
      } else if (out.isConstant()) {
        out = dim;
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
  // An in-place copy returns 'self', strides and storage included.
  if (target == "torch.ops.aten.copy_.default") {
    const auto* self = node->tryGetInput("self");
    const auto* shape =
        self != nullptr ? shapeOf(shapes, self->value) : nullptr;
    if (shape == nullptr) {
      return std::nullopt;
    }
    return *shape;
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

namespace {

folly::F14FastMap<std::string, SymShapeRule>& symShapeRules() {
  static folly::F14FastMap<std::string, SymShapeRule> rules;
  return rules;
}

// A ListUnpack's outputs are its list's elements. The elements are copied
// first: adding to 'shapes' can move the ones it holds.
int32_t unpackRule(SymShapes& shapes, const nativert::Node& node) {
  if (node.inputs().empty()) {
    return 0;
  }
  std::vector<std::optional<SymShape>> elements;
  for (const auto* element :
       listElementShapes(shapes, node.inputs()[0].value)) {
    elements.push_back(
        element != nullptr ? std::optional<SymShape>(*element) : std::nullopt);
  }
  int32_t numAdded = 0;
  for (size_t i = 0; i < node.outputs().size() && i < elements.size(); ++i) {
    const auto& element = elements.at(i);
    if (element.has_value() &&
        shapes.byName.emplace(std::string(node.outputs()[i]->name()), *element)
            .second) {
      ++numAdded;
    }
  }
  return numAdded;
}

} // namespace

std::string listElementName(std::string_view list, size_t index) {
  return fmt::format("{}[{}]", list, index);
}

std::vector<const SymShape*> listElementShapes(
    const SymShapes& shapes,
    const nativert::Value* list) {
  std::vector<const SymShape*> result;
  if (list == nullptr) {
    return result;
  }
  const auto* producer = list->producer();
  if (producer != nullptr && producer->target() == "prim.ListPack") {
    for (const auto& input : producer->inputs()) {
      result.push_back(shapeOf(shapes, input.value));
    }
    return result;
  }
  for (size_t i = 0;; ++i) {
    auto it = shapes.byName.find(listElementName(list->name(), i));
    if (it == shapes.byName.end()) {
      break;
    }
    result.push_back(&it->second);
  }
  return result;
}

SymExpr numelOf(const SymShape& shape) {
  SymExpr result = SymExpr::constant(1);
  for (const auto& dim : shape.dims) {
    result = product(result, dim);
  }
  return result;
}

SymShape vectorShape(const SymExpr& length) {
  return SymShape{.dims = {length}, .strides = {SymExpr::constant(1)}};
}

namespace {

// Adds 'coefficient' times 'expr' to the terms of a sum, flattening nested
// sums and splitting a constant factor off a product.
void addTerms(
    const SymExpr& expr,
    int64_t coefficient,
    int64_t& constant,
    std::vector<std::pair<SymExpr, int64_t>>& terms) {
  if (expr.isConstant()) {
    constant += coefficient * expr.value;
    return;
  }
  if (expr.kind == SymExpr::Kind::kAdd) {
    for (const auto& arg : expr.args) {
      addTerms(arg, coefficient, constant, terms);
    }
    return;
  }
  if (expr.kind == SymExpr::Kind::kMul && expr.args.size() == 2) {
    for (size_t i = 0; i < 2; ++i) {
      if (expr.args[i].isConstant()) {
        addTerms(
            expr.args[1 - i],
            coefficient * expr.args[i].value,
            constant,
            terms);
        return;
      }
    }
  }
  const auto text = expr.toString();
  for (auto& [term, existing] : terms) {
    if (term.toString() == text) {
      existing += coefficient;
      return;
    }
  }
  terms.emplace_back(expr, coefficient);
}

} // namespace

SymExpr symAdd(const SymExpr& lhs, const SymExpr& rhs) {
  int64_t constant = 0;
  std::vector<std::pair<SymExpr, int64_t>> terms;
  addTerms(lhs, 1, constant, terms);
  addTerms(rhs, 1, constant, terms);
  std::optional<SymExpr> sum;
  auto append = [&](SymExpr term) {
    sum = sum.has_value()
        ? SymExpr{.kind = SymExpr::Kind::kAdd, .args = {*sum, std::move(term)}}
        : std::move(term);
  };
  for (auto& [term, coefficient] : terms) {
    if (coefficient == 1) {
      append(term);
    } else if (coefficient != 0) {
      append(product(SymExpr::constant(coefficient), term));
    }
  }
  if (constant != 0 || !sum.has_value()) {
    append(SymExpr::constant(constant));
  }
  return *sum;
}

SymExpr symMul(const SymExpr& lhs, const SymExpr& rhs) {
  return product(lhs, rhs);
}

SymExpr symMax(const SymExpr& lhs, const SymExpr& rhs) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return SymExpr::constant(std::max(lhs.value, rhs.value));
  }
  return SymExpr{.kind = SymExpr::Kind::kMax, .args = {lhs, rhs}};
}

namespace {

// The factors of a product, constants folded into 'constant'.
void collectFactors(
    const SymExpr& expr,
    int64_t& constant,
    std::vector<SymExpr>& factors) {
  if (expr.isConstant()) {
    constant *= expr.value;
  } else if (expr.kind == SymExpr::Kind::kMul) {
    for (const auto& arg : expr.args) {
      collectFactors(arg, constant, factors);
    }
  } else {
    factors.push_back(expr);
  }
}

} // namespace

// Factors common to both sides cancel, which is exact for the positive
// extents these expressions describe.
SymExpr symFloorDiv(const SymExpr& lhs, const SymExpr& rhs) {
  int64_t numeratorConstant = 1;
  int64_t divisorConstant = 1;
  std::vector<SymExpr> numerator;
  std::vector<SymExpr> divisor;
  collectFactors(lhs, numeratorConstant, numerator);
  collectFactors(rhs, divisorConstant, divisor);
  if (divisorConstant <= 0 || numeratorConstant < 0) {
    return SymExpr{.kind = SymExpr::Kind::kFloorDiv, .args = {lhs, rhs}};
  }
  for (auto it = divisor.begin(); it != divisor.end();) {
    auto match = std::find_if(numerator.begin(), numerator.end(), [&](auto& f) {
      return sameExpr(f, *it);
    });
    if (match != numerator.end()) {
      numerator.erase(match);
      it = divisor.erase(it);
    } else {
      ++it;
    }
  }
  if (numeratorConstant % divisorConstant == 0) {
    numeratorConstant /= divisorConstant;
    divisorConstant = 1;
  }
  SymExpr top = SymExpr::constant(numeratorConstant);
  for (const auto& factor : numerator) {
    top = product(top, factor);
  }
  SymExpr bottom = SymExpr::constant(divisorConstant);
  for (const auto& factor : divisor) {
    bottom = product(bottom, factor);
  }
  if (bottom.isConstant() && bottom.value == 1) {
    return top;
  }
  if (top.isConstant() && bottom.isConstant()) {
    return SymExpr::constant(top.value / bottom.value);
  }
  return SymExpr{.kind = SymExpr::Kind::kFloorDiv, .args = {top, bottom}};
}

namespace {

constexpr std::string_view kHostDimPrefix{"h:"};

// The traced extent at the end of a hostDimSymbol, or nullopt.
std::optional<int64_t> hostDimTraced(std::string_view symbol) {
  const auto last = symbol.rfind(':');
  if (!isHostDimSymbol(symbol) || last == std::string_view::npos ||
      last + 1 >= symbol.size()) {
    return std::nullopt;
  }
  const auto digits = symbol.substr(last + 1);
  if (!std::all_of(digits.begin(), digits.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
      })) {
    return std::nullopt;
  }
  return std::stoll(std::string(digits));
}

} // namespace

SymExpr hostDimSymbol(std::string_view value, size_t dim, int64_t traced) {
  return SymExpr{
      .kind = SymExpr::Kind::kSymbol,
      .symbol = fmt::format("{}{}:{}:{}", kHostDimPrefix, value, dim, traced)};
}

bool isHostDimSymbol(std::string_view symbol) {
  return symbol.starts_with(kHostDimPrefix);
}

std::optional<int64_t> tracedValue(const SymExpr& expr) {
  if (expr.isConstant()) {
    return expr.value;
  }
  std::vector<std::string> names;
  expr.collectSymbols(names);
  folly::F14FastMap<std::string, int64_t> traced;
  for (const auto& name : names) {
    if (auto host = hostDimTraced(name)) {
      traced[name] = *host;
      continue;
    }
    if (name.size() < 2 || name[0] != 'd' ||
        !std::all_of(name.begin() + 1, name.end(), [](char c) {
          return std::isdigit(static_cast<unsigned char>(c));
        })) {
      return std::nullopt;
    }
    traced[name] = std::stoll(name.substr(1));
  }
  return expr.evaluate(traced);
}

SymExpr elementSymbol(std::string_view tensor, int64_t index) {
  return SymExpr{
      .kind = SymExpr::Kind::kSymbol,
      .symbol = fmt::format("e:{}:{}", tensor, index)};
}

std::optional<std::pair<std::string, int64_t>> parseElementSymbol(
    std::string_view symbol) {
  const auto last = symbol.rfind(':');
  if (symbol.size() < 5 || symbol.substr(0, 2) != "e:" ||
      last == std::string_view::npos || last < 2) {
    return std::nullopt;
  }
  const auto index = symbol.substr(last + 1);
  if (index.empty() || !std::all_of(index.begin(), index.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
      })) {
    return std::nullopt;
  }
  return std::make_pair(
      std::string(symbol.substr(2, last - 2)), std::stoll(std::string(index)));
}

bool sameDim(const SymExpr& lhs, const SymExpr& rhs) {
  if (sameExpr(lhs, rhs)) {
    return true;
  }
  const auto lhsTraced = tracedValue(lhs);
  return lhsTraced.has_value() && lhsTraced == tracedValue(rhs);
}

SymShape denseShape(std::vector<SymExpr> dims) {
  auto strides = contiguousStrides(dims);
  return SymShape{.dims = std::move(dims), .strides = std::move(strides)};
}

namespace {

folly::F14FastMap<ReserveFnKey, SymReserveFn>& symReserves() {
  static folly::F14FastMap<ReserveFnKey, SymReserveFn> reserves;
  return reserves;
}

} // namespace

void registerSymReserve(ReserveFnKey fn, SymReserveFn sym) {
  symReserves()[fn] = sym;
}

SymReserveFn symReserveOf(ReserveFnKey fn) {
  auto it = symReserves().find(fn);
  return it != symReserves().end() ? it->second : nullptr;
}

bool symNumel(const SymReserveArgs& args, std::vector<SymShape>& out, int32_t) {
  const auto* input = args.tensor(0);
  if (input == nullptr) {
    return false;
  }
  out.at(0) = vectorShape(numelOf(*input));
  return true;
}

bool symNumelPlusOne(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t) {
  const auto* input = args.tensor(0);
  if (input == nullptr) {
    return false;
  }
  out.at(0) = vectorShape(symAdd(numelOf(*input), SymExpr::constant(1)));
  return true;
}

bool symScalarLength(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t) {
  if (args.scalars.empty() || !args.scalars[0].has_value()) {
    return false;
  }
  out.at(0) = vectorShape(*args.scalars[0]);
  return true;
}

bool symFirstDim(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t) {
  const auto* input = args.tensor(0);
  if (input == nullptr || input->dims.empty()) {
    return false;
  }
  out.at(0) = vectorShape(input->dims[0]);
  return true;
}

bool symRepeatedScalar(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t numOut) {
  if (args.scalars.empty() || !args.scalars[0].has_value()) {
    return false;
  }
  for (int32_t i = 0; i < numOut; ++i) {
    out.at(i) = vectorShape(*args.scalars[0]);
  }
  return true;
}

bool symListNumel(
    const SymReserveArgs& args,
    std::vector<SymShape>& out,
    int32_t numOut) {
  for (int32_t i = 0; i < numOut; ++i) {
    const auto* input = args.tensor(0, i);
    if (input == nullptr) {
      return false;
    }
    out.at(i) = vectorShape(numelOf(*input));
  }
  return true;
}

void registerSymShapeRule(std::string_view target, SymShapeRule rule) {
  symShapeRules()[std::string(target)] = std::move(rule);
}

int32_t inferNodeShapes(SymShapes& shapes, const nativert::Node& node) {
  if (node.target() == "prim.ListUnpack") {
    return unpackRule(shapes, node);
  }
  auto rule = symShapeRules().find(std::string(node.target()));
  if (rule != symShapeRules().end()) {
    return rule->second(shapes, node);
  }
  const auto& outputs = node.outputs();
  // A cooperative-grid scan adds its per-block partials as a second output.
  const size_t numOutputs = node.target() == "tw.cumsum_cg" ? 2 : 1;
  if (outputs.size() != numOutputs ||
      outputs[0]->type().kind() != nativert::Type::Kind::Tensor ||
      shapes.byName.count(std::string(outputs[0]->name())) != 0) {
    return 0;
  }
  auto shape = inferNode(shapes, &node);
  if (!shape) {
    return 0;
  }
  shapes.byName.emplace(std::string(outputs[0]->name()), std::move(*shape));
  return 1;
}

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
    const auto added = inferNodeShapes(shapes, node);
    numAdded += added;
    return added > 0;
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
