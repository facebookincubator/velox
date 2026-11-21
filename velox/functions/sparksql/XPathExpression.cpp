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
#include "velox/functions/sparksql/XPathExpression.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <variant>

#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <libxml/xpathInternals.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/sparksql/XPathContext.h"
#include "velox/functions/sparksql/XPathValue.h"

namespace facebook::velox::functions::sparksql::xpath {
namespace {

// Preserve the JDK's two name tests: ordinary walkers use local names, while
// eligible descendant iterators use the DOM's full qualified name.
class NamespaceUnawarePath {
 public:
  explicit NamespaceUnawarePath(std::string_view path) : path_(path) {
    const std::string terminatedPath(path);
    const auto* base = reinterpret_cast<const xmlChar*>(terminatedPath.c_str());
    std::unique_ptr<xmlXPathParserContext, decltype(&xmlXPathFreeParserContext)>
        parser{
            xmlXPathNewParserContext(base, nullptr), xmlXPathFreeParserContext};
    VELOX_CHECK_NOT_NULL(parser.get(), "Failed to tokenize XPath");
    const auto readName = [&](size_t start) {
      parser->cur = base + start;
      XmlStringPtr name{xmlXPathParseNCName(parser.get()), xmlFree};
      VELOX_CHECK_NOT_NULL(name.get(), "Failed to tokenize XPath name");
      return static_cast<size_t>(parser->cur - base);
    };
    for (size_t offset = 0; offset < path.size();) {
      const auto byte = path[offset];
      if (byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n') {
        ++offset;
        continue;
      }
      const auto start = offset++;
      auto kind = TokenKind::kSymbol;
      if (byte == '\'' || byte == '"') {
        kind = TokenKind::kLiteral;
        while (offset < path.size() && path[offset] != byte) {
          ++offset;
        }
        ++offset;
      } else if (
          (byte >= '0' && byte <= '9') ||
          (byte == '.' && offset < path.size() && path[offset] >= '0' &&
           path[offset] <= '9')) {
        kind = TokenKind::kNumber;
        while (offset < path.size() &&
               ((path[offset] >= '0' && path[offset] <= '9') ||
                path[offset] == '.')) {
          ++offset;
        }
      } else if (
          (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
          byte == '_' || static_cast<unsigned char>(byte) >= 0x80) {
        kind = TokenKind::kName;
        offset = readName(start);
        if (offset + 1 < path.size() && path[offset] == ':' &&
            path[offset + 1] != ':') {
          prefixes_.emplace_back(path.substr(start, offset - start));
          ++offset;
          // A QName wildcard ends at '*', even without whitespace before an
          // operator such as "or" or "div".
          offset = path[offset] == '*' ? offset + 1 : readName(offset);
        }
      } else if (
          offset < path.size() &&
          ((byte == '/' && path[offset] == '/') ||
           (byte == ':' && path[offset] == ':') ||
           (byte == '.' && path[offset] == '.') ||
           ((byte == '<' || byte == '>' || byte == '!') &&
            path[offset] == '='))) {
        ++offset;
      }
      const auto text = path.substr(start, offset - start);
      tokens_.push_back({text, start, kind, numPoolTokens_});
      // The JDK lexer splits QNames and two-character comparisons, but not
      // axis separators or "..". These indices affect proximity analysis.
      numPoolTokens_ +=
          kind == TokenKind::kName && text.find(':') != std::string_view::npos
          ? 3
          : (text == "//" || text == "!=" || text == "<=" || text == ">=") ? 2
                                                                           : 1;
    }
    tokens_.push_back({{}, path.size(), TokenKind::kEnd});
    std::sort(prefixes_.begin(), prefixes_.end());
    prefixes_.erase(
        std::unique(prefixes_.begin(), prefixes_.end()), prefixes_.end());
  }

  std::optional<std::string> rewrite() {
    try {
      expression(1);
      if (current().kind != TokenKind::kEnd) {
        throw InvalidPlan{};
      }
      operations_[1] = operations_.size();
      operations_.resize(operations_.size() + 4, 0);
      for (const auto& rewrite : rewrites_) {
        if (const auto* index = std::get_if<size_t>(&rewrite)) {
          const auto& path = paths_[*index];
          rewriteSteps(path.start, path.steps, path.iteratorOrder);
        } else {
          const auto& numeric = std::get<NumericExpression>(rewrite);
          auto left = numeric.operation == kNegate
              ? "0"
              : render(numeric.start, numeric.leftEnd);
          auto right = render(numeric.rightStart, numeric.end);
          addEdit(
              numeric.start,
              numeric.end,
              fmt::format(
                  "_velox_numeric({},{},{})", left, right, numeric.operation));
        }
      }
      return render(0, path_.size());
    } catch (const InvalidPlan&) {
      return std::nullopt;
    }
  }

  bool usesInternalFunctions() const {
    for (size_t i = 0; i + 1 < tokens_.size(); ++i) {
      if (tokens_[i].kind == TokenKind::kName && tokens_[i + 1].text == "(" &&
          (tokens_[i].text == "_velox_namespaces" ||
           tokens_[i].text == "_velox_namespace_step" ||
           tokens_[i].text == "_velox_first_step" ||
           tokens_[i].text == "_velox_literal" ||
           tokens_[i].text == "_velox_numeric")) {
        return true;
      }
    }
    return false;
  }

  const std::vector<std::string>& namespaceSelectors() const {
    return namespaceSelectors_;
  }

  const std::vector<std::string>& prefixes() const {
    return prefixes_;
  }

  const std::vector<double>& literals() const {
    return literals_;
  }

 private:
  enum class TokenKind { kName, kNumber, kLiteral, kSymbol, kEnd };

  struct Token {
    std::string_view text;
    size_t offset;
    TokenKind kind;
    int64_t poolIndex{0};
  };

  struct NumericExpression {
    size_t start;
    size_t leftEnd;
    size_t rightStart;
    size_t end;
    int operation;
  };

  class InvalidPlan : public std::exception {};

  struct Step {
    std::string_view axis;
    size_t nameToken;
    bool genericNode{true};
    bool hasPredicates{false};
    size_t testToken{0};
    size_t start{0};
    size_t testEnd{0};
    size_t end{0};
    size_t operation{0};
  };

  struct Path {
    size_t start;
    std::vector<Step> steps;
    bool iteratorOrder{false};
  };

  struct Edit {
    size_t end;
    std::string text;
  };

  const Token& current() const {
    return tokens_[position_];
  }

  void addEdit(size_t start, size_t end, std::string text) {
    VELOX_CHECK(
        edits_.emplace(start, Edit{end, std::move(text)}).second,
        "Duplicate XPath normalization");
  }

  std::string render(size_t start, size_t end) {
    std::string result;
    size_t copied = start;
    for (auto edit = edits_.lower_bound(start);
         edit != edits_.end() && edit->first < end;) {
      VELOX_CHECK_GE(edit->first, copied, "Overlapping XPath normalization");
      VELOX_CHECK_LE(edit->second.end, end, "Overlapping XPath normalization");
      result.append(path_.substr(copied, edit->first - copied));
      result.append(edit->second.text);
      copied = edit->second.end;
      edit = edits_.erase(edit);
    }
    result.append(path_.substr(copied, end - copied));
    return result;
  }

  bool take(std::string_view text) {
    if (current().text != text) {
      return false;
    }
    ++position_;
    return true;
  }

  void require(std::string_view text) {
    if (!take(text)) {
      throw InvalidPlan{};
    }
  }

  static int precedence(std::string_view text) {
    if (text == "or") {
      return 1;
    }
    if (text == "and") {
      return 2;
    }
    if (text == "=" || text == "!=") {
      return 3;
    }
    if (text == "<" || text == ">" || text == "<=" || text == ">=") {
      return 4;
    }
    if (text == "+" || text == "-") {
      return 5;
    }
    if (text == "*" || text == "div" || text == "mod") {
      return 6;
    }
    return text == "|" ? 8 : 0;
  }

  size_t appendOperation(int code) {
    const auto start = operations_.size();
    operations_.insert(operations_.end(), {code, 0});
    return start;
  }

  void finishOperation(size_t start, bool terminate = false) {
    if (terminate) {
      operations_.push_back(kEnd);
    }
    operations_[start + 1] = operations_.size() - start;
  }

  void insertOperation(size_t start, int code) {
    operations_.insert(operations_.begin() + start, {code, 0});
    for (auto& path : paths_) {
      for (auto& step : path.steps) {
        if (step.operation >= start) {
          step.operation += 2;
        }
      }
    }
  }

  static int binaryOperation(std::string_view operation) {
    static constexpr std::pair<std::string_view, int> codes[] = {
        {"or", kOr},
        {"and", kAnd},
        {"!=", kNotEqual},
        {"=", kEqual},
        {"<=", kLessEqual},
        {"<", kLess},
        {">=", kGreaterEqual},
        {">", kGreater},
        {"+", kAdd},
        {"-", kSubtract},
        {"*", kMultiply},
        {"div", kDivide},
        {"mod", kModulo},
        {"|", kUnion},
    };
    for (const auto& [name, code] : codes) {
      if (operation == name) {
        return code;
      }
    }
    VELOX_UNREACHABLE("Unknown XPath operation");
  }

  int64_t expression(int minimum) {
    const auto start = operations_.size();
    const auto textStart = current().offset;
    int64_t left;
    if (take("-")) {
      const auto rightStart = current().offset;
      appendOperation(kNegate);
      expression(7);
      finishOperation(start);
      rewrites_.emplace_back(
          NumericExpression{
              textStart, textStart, rightStart, current().offset, kNegate});
      left = kNegate;
    } else {
      left = pathExpression();
    }
    for (int level = precedence(current().text); level >= minimum;
         level = precedence(current().text)) {
      const auto code = binaryOperation(current().text);
      const auto leftEnd = current().offset;
      ++position_;
      const auto rightStart = current().offset;
      if (code != kUnion || left != kUnion) {
        insertOperation(start, code);
      }
      expression(level <= 2 ? level : level + 1);
      finishOperation(start);
      if (code >= kNotEqual && code <= kModulo) {
        rewrites_.emplace_back(
            NumericExpression{
                textStart, leftEnd, rightStart, current().offset, code});
      }
      left = code;
    }
    return left;
  }

  bool startsStep() const {
    const auto text = current().text;
    if (text == "." || text == ".." || text == "@" || text == "*") {
      return true;
    }
    if (current().kind != TokenKind::kName) {
      return false;
    }
    return tokens_[position_ + 1].text != "(" || text == "node" ||
        text == "text" || text == "comment" || text == "processing-instruction";
  }

  static int axisOperation(std::string_view axis) {
    static constexpr std::string_view axes[] = {
        "ancestor",
        "ancestor-or-self",
        "attribute",
        "child",
        "descendant",
        "descendant-or-self",
        "following",
        "following-sibling",
        "parent",
        "preceding",
        "preceding-sibling",
        "self",
        "namespace",
        "root",
    };
    for (size_t i = 0; i < std::size(axes); ++i) {
      if (axis == axes[i]) {
        return kAncestor + static_cast<int>(i);
      }
    }
    VELOX_UNREACHABLE("Unknown XPath axis");
  }

  Step genericStep(std::string_view axis) {
    Step result{axis, tokens_.size()};
    result.operation = appendOperation(axisOperation(axis));
    operations_.insert(
        operations_.end(), {4, axis == "root" ? kRootTest : kNodeTest});
    finishOperation(result.operation);
    return result;
  }

  void predicates() {
    while (take("[")) {
      const auto start = appendOperation(kPredicate);
      expression(1);
      require("]");
      finishOperation(start, true);
    }
  }

  int64_t pathExpression() {
    const auto start = current().offset;
    const auto operation = operations_.size();
    std::vector<Step> steps;
    if (current().text == "/" || current().text == "//") {
      appendOperation(kLocation);
      steps.push_back(genericStep("root"));
    } else if (!startsStep()) {
      const auto result = primary();
      if (current().text != "[" && current().text != "/" &&
          current().text != "//") {
        return result;
      }
      insertOperation(operation, kLocation);
      predicates();
      steps.push_back({"filter", tokens_.size()});
      steps.back().operation = operation + 2;
    } else {
      appendOperation(kLocation);
      steps.push_back(step());
    }
    while (current().text == "/" || current().text == "//") {
      if (take("//")) {
        steps.push_back(genericStep("descendant-or-self"));
      } else {
        require("/");
      }
      if (startsStep()) {
        steps.push_back(step());
      }
    }
    finishOperation(operation, true);
    paths_.push_back({start, std::move(steps)});
    rewrites_.emplace_back(paths_.size() - 1);
    return kLocation;
  }

  void rewriteSteps(
      size_t start,
      const std::vector<Step>& steps,
      bool iteratorOrder) {
    if (usesQualifiedNames(steps)) {
      const auto first = steps.front().axis;
      bool includeSelf = first == "self" || first == "descendant-or-self" ||
          (first == "root" && steps[1].axis == "descendant-or-self");
      for (const auto& step : steps) {
        if (step.axis == "child") {
          includeSelf = false;
        }
      }
      std::string axis = first == "root" ? "/" : "";
      axis += includeSelf ? "descendant-or-self::" : "descendant::";
      addEdit(start, tokens_[steps.back().testToken].offset, std::move(axis));
    } else {
      for (const auto& step : steps) {
        if (step.nameToken < tokens_.size() && step.axis != "namespace") {
          const auto& token = tokens_[step.nameToken];
          addEdit(
              token.offset,
              token.offset + token.text.size(),
              fmt::format("*[local-name()='{}']", token.text));
        }
      }
      for (size_t i = 0; i < steps.size(); ++i) {
        const auto& step = steps[i];
        if (step.axis != "namespace") {
          continue;
        }
        auto prefix = render(start, step.start);
        while (!prefix.empty() &&
               std::string_view(" \t\r\n").find(prefix.back()) !=
                   std::string_view::npos) {
          prefix.pop_back();
        }
        if (prefix.ends_with("//")) {
          prefix.resize(prefix.size() - 2);
          prefix += "/descendant-or-self::node()";
        } else if (prefix.ends_with("/") && prefix != "/") {
          prefix.pop_back();
        }
        if (prefix.empty()) {
          prefix = ".";
        }
        if (iteratorOrder) {
          rewriteFirstMatch(start, steps, i, std::move(prefix));
          return;
        }
        auto selector = namespaceSelector(step);
        // Keep selectors separate so nesting never requotes rewritten paths.
        const auto selectorIndex = namespaceSelectors_.size();
        namespaceSelectors_.push_back(std::move(selector));
        addEdit(
            start,
            step.end,
            fmt::format(
                "_velox_namespace_step({},{},0)", prefix, selectorIndex));
      }
    }
  }

  std::string namespaceSelector(const Step& step) {
    const auto test = tokens_[step.testToken].text;
    std::string selector = "_velox_namespaces()";
    if (!step.genericNode && test != "*") {
      if (tokens_[step.testToken + 1].text == "(") {
        selector += "[false()]";
      } else {
        const auto colon = test.find(':');
        const auto name =
            colon == std::string_view::npos ? test : test.substr(colon + 1);
        if (name != "*") {
          selector += fmt::format("[local-name()='{}']", name);
        }
      }
    }
    selector += render(step.testEnd, step.end);
    return selector;
  }

  // Keep each namespace candidate's continuation separate so later steps do
  // not reorder results from different candidates before asNode() sees them.
  void rewriteFirstMatch(
      size_t start,
      const std::vector<Step>& steps,
      size_t firstNamespace,
      std::string prefix) {
    std::vector<std::string> selectors;
    for (size_t i = firstNamespace; i < steps.size(); ++i) {
      const auto& step = steps[i];
      if (step.axis == "namespace") {
        selectors.push_back(namespaceSelector(step));
      } else if (step.end == 0) {
        VELOX_CHECK_EQ(step.axis, "descendant-or-self");
        selectors.emplace_back("descendant-or-self::node()");
      } else {
        selectors.push_back(render(step.start, step.end));
      }
    }
    auto continuation = std::move(selectors.back());
    for (size_t i = selectors.size() - 1; i > 0; --i) {
      const auto selectorIndex = namespaceSelectors_.size();
      namespaceSelectors_.push_back(std::move(continuation));
      continuation = fmt::format(
          "_velox_first_step({},{})", selectors[i - 1], selectorIndex);
    }
    const auto selectorIndex = namespaceSelectors_.size();
    namespaceSelectors_.push_back(std::move(continuation));
    addEdit(
        start,
        steps.back().end,
        fmt::format("_velox_first_step({},{})", prefix, selectorIndex));
  }

  static int functionId(std::string_view name) {
    static constexpr std::string_view names[] = {
        "current",
        "last",
        "position",
        "count",
        "id",
        "key",
        "",
        "local-name",
        "namespace-uri",
        "name",
        "generate-id",
        "not",
        "true",
        "false",
        "boolean",
        "number",
        "floor",
        "ceiling",
        "round",
        "sum",
        "string",
        "starts-with",
        "contains",
        "substring-before",
        "substring-after",
        "normalize-space",
        "translate",
        "concat",
        "",
        "substring",
        "string-length",
        "system-property",
        "lang",
        "function-available",
        "element-available",
        "document-location",
        "unparsed-entity-uri",
        "here",
    };
    for (size_t i = 0; i < std::size(names); ++i) {
      if (name == names[i]) {
        return static_cast<int>(i);
      }
    }
    // Unknown functions must reach libxml2's evaluation-time diagnostics.
    return -1;
  }

  int64_t primary() {
    const auto token = current();
    ++position_;
    const auto start = operations_.size();
    if (token.kind == TokenKind::kLiteral) {
      appendOperation(kLiteral);
      operations_.push_back(token.poolIndex);
    } else if (token.kind == TokenKind::kNumber) {
      appendOperation(kNumber);
      operations_.push_back(token.poolIndex);
      // Parse once while retaining the literal's JDK operation-map identity.
      const auto index = literals_.size();
      literals_.push_back(xpathStringNumber(token.text));
      addEdit(
          token.offset,
          token.offset + token.text.size(),
          fmt::format("_velox_literal({})", index));
    } else if (token.text == "$") {
      appendOperation(kVariable);
      const auto& name = current();
      const auto qualified = name.text.find(':') != std::string_view::npos;
      operations_.insert(
          operations_.end(),
          {qualified ? name.poolIndex : static_cast<int64_t>(kEmpty),
           name.poolIndex + (qualified ? 2 : 0)});
      ++position_;
    } else if (token.text == "(") {
      appendOperation(kGroup);
      expression(1);
      require(")");
    } else {
      VELOX_CHECK(token.kind == TokenKind::kName, "Failed to normalize XPath");
      require("(");
      if (token.text.find(':') != std::string_view::npos) {
        appendOperation(kExtensionFunction);
        operations_.insert(
            operations_.end(), {token.poolIndex, token.poolIndex + 2});
      } else {
        appendOperation(kFunction);
        operations_.push_back(functionId(token.text));
      }
      if (!take(")")) {
        do {
          const auto argument = appendOperation(kArgument);
          expression(1);
          auto expressionStart = argument + 2;
          while (operations_[expressionStart] == kGroup) {
            expressionStart += 2;
          }
          // These JDK functions use asNode(), not a document-ordered node-set.
          if ((token.text == "name" || token.text == "local-name" ||
               token.text == "namespace-uri") &&
              operations_[expressionStart] == kLocation) {
            paths_.back().iteratorOrder = true;
          }
          finishOperation(argument);
        } while (take(","));
        require(")");
      }
      finishOperation(start, true);
      return operations_[start];
    }
    finishOperation(start);
    return operations_[start];
  }

  Step step() {
    Step result{"child", tokens_.size()};
    result.start = current().offset;
    int testType = kNodeTest;
    std::vector<int64_t> arguments;
    if (take(".")) {
      result.axis = "self";
    } else if (take("..")) {
      result.axis = "parent";
    } else {
      if (take("@")) {
        result.axis = "attribute";
      } else if (tokens_[position_ + 1].text == "::") {
        result.axis = current().text;
        position_ += 2;
      }
      const auto test = position_++;
      result.testToken = test;
      result.genericNode = false;
      if (take("(")) {
        const auto text = tokens_[test].text;
        result.genericNode = text == "node";
        testType = text == "node" ? kNodeTest
            : text == "text"      ? kTextTest
            : text == "comment"   ? kCommentTest
                                  : kProcessingInstructionTest;
        if (current().kind == TokenKind::kLiteral) {
          arguments.push_back(current().poolIndex);
          ++position_;
        }
        require(")");
      } else {
        testType = kNameTest;
        const auto& token = tokens_[test];
        const auto qualified = token.text.find(':') != std::string_view::npos;
        arguments = {
            qualified ? token.poolIndex : static_cast<int64_t>(kEmpty),
            token.text.ends_with("*") ? static_cast<int64_t>(kWildcard)
                                      : token.poolIndex + (qualified ? 2 : 0)};
        if (token.kind == TokenKind::kName && !qualified) {
          result.nameToken = test;
        }
      }
    }
    result.testEnd = current().offset;
    result.operation = appendOperation(axisOperation(result.axis));
    operations_.insert(
        operations_.end(),
        {4 + static_cast<int64_t>(arguments.size()), testType});
    operations_.insert(operations_.end(), arguments.begin(), arguments.end());
    result.hasPredicates = current().text == "[";
    predicates();
    finishOperation(result.operation);
    result.end = current().offset;
    return result;
  }

  int64_t operation(int64_t position) const {
    if (position < 0 || static_cast<size_t>(position) >= operations_.size()) {
      throw InvalidPlan{};
    }
    return operations_[position];
  }

  int64_t nextOperation(int64_t position) const {
    return position + operation(position + 1);
  }

  bool functionProximity(int64_t position, std::vector<int64_t>& active) const {
    const auto id = operation(position + 2);
    if (id == 1 || id == 2) {
      return true;
    }
    const auto end = position + operation(position + 1) - 1;
    for (auto argument = position + 3; argument < end;) {
      if (innerProximity(argument + 2, active)) {
        return true;
      }
      const auto next = nextOperation(argument);
      if (next <= argument) {
        throw InvalidPlan{};
      }
      argument = next;
    }
    return false;
  }

  bool innerProximity(int64_t position, std::vector<int64_t>& active) const {
    if (std::find(active.begin(), active.end(), position) != active.end()) {
      throw InvalidPlan{};
    }
    active.push_back(position);
    SCOPE_EXIT {
      active.pop_back();
    };
    const auto code = operation(position);
    if (code == kVariable || code == kNumber || code == kLiteral ||
        code == kLocation) {
      return false;
    }
    if (code == kArgument) {
      return innerProximity(position + 2, active);
    }
    if (code == kFunction) {
      return functionProximity(position, active);
    }
    if (code >= kEqual && code <= kGreater) {
      // Preserve the JDK's opcode-based child lookup for descendant-iterator
      // selection.
      const auto left = code + 2;
      const auto right = nextOperation(left);
      return innerProximity(left, active) || innerProximity(right, active);
    }
    return true;
  }

  bool predicateProximity(int64_t position) const {
    const auto code = operation(position);
    if (code == kLocation) {
      return false;
    }
    std::vector<int64_t> active;
    if (code == kFunction) {
      return functionProximity(position, active);
    }
    if (code >= kEqual && code <= kGreater) {
      const auto left = position + 2;
      const auto right = nextOperation(left);
      return innerProximity(left, active) || innerProximity(right, active);
    }
    return true;
  }

  bool usesQualifiedNames(const std::vector<Step>& steps) const {
    if (steps.size() < 2) {
      return false;
    }
    bool descendant = false;
    bool descendantOrSelf = false;
    bool self = false;
    for (size_t i = 0; i < steps.size(); ++i) {
      if (i == 3) {
        return false;
      }
      const auto& step = steps[i];
      if (step.axis != "filter") {
        auto predicate = static_cast<int64_t>(step.operation) +
            operation(step.operation + 2);
        while (operation(predicate) == kPredicate) {
          if (predicateProximity(predicate + 2)) {
            return false;
          }
          predicate = nextOperation(predicate);
        }
      }
      if (i + 1 < steps.size() && (!step.genericNode || step.hasPredicates)) {
        return false;
      }
      if (step.axis == "root" || step.axis == "self") {
        if (i != 0) {
          return false;
        }
        self = step.axis == "self";
      } else if (
          step.axis == "descendant" || step.axis == "descendant-or-self") {
        if (i == 2) {
          return false;
        }
        descendant = true;
        descendantOrSelf =
            descendantOrSelf || step.axis == "descendant-or-self";
      } else if (step.axis == "child") {
        if (!descendantOrSelf && !(descendant && self)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  std::string_view path_;
  std::vector<Token> tokens_;
  std::vector<int64_t> operations_{kXPath, 2};
  std::vector<Path> paths_;
  // Postorder keeps nested operators inside their path predicates and
  // selectors.
  std::vector<std::variant<size_t, NumericExpression>> rewrites_;
  std::map<size_t, Edit> edits_;
  std::vector<std::string> namespaceSelectors_;
  std::vector<std::string> prefixes_;
  std::vector<double> literals_;
  int64_t numPoolTokens_{0};
  size_t position_{0};
};

} // namespace

std::optional<XPathExpression> normalizeXPath(std::string_view path) {
  NamespaceUnawarePath normalized(path);
  if (normalized.usesInternalFunctions()) {
    return std::nullopt;
  }
  auto rewritten = normalized.rewrite();
  if (!rewritten.has_value()) {
    return std::nullopt;
  }
  return XPathExpression{
      std::move(*rewritten),
      normalized.namespaceSelectors(),
      normalized.prefixes(),
      normalized.literals()};
}

} // namespace facebook::velox::functions::sparksql::xpath
