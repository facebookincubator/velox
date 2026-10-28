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
#include "velox/functions/sparksql/XPathUtil.h"

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <libxml/SAX2.h>
#include <libxml/parser.h>
#include <libxml/xmlerror.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"

// Parser and XPath state, including the last error, must be per-thread.
#ifndef LIBXML_THREAD_ENABLED
#error "Spark XPath functions require libxml2 built with thread support"
#endif

#ifndef LIBXML_SAX1_ENABLED
#error "Spark XPath functions require libxml2 built with SAX1 support"
#endif

#if LIBXML_VERSION >= 21400
#error "Spark XPath functions require libxml2 earlier than 2.14"
#endif

namespace facebook::velox::functions::sparksql {
namespace xpath {
namespace {

using CompiledXPathPtr =
    std::unique_ptr<xmlXPathCompExpr, decltype(&xmlXPathFreeCompExpr)>;
using XPathObjectPtr =
    std::unique_ptr<xmlXPathObject, decltype(&xmlXPathFreeObject)>;
using XmlStringPtr = std::unique_ptr<xmlChar, decltype(xmlFree)>;

struct CompiledXPath {
  CompiledXPathPtr expression{nullptr, xmlXPathFreeCompExpr};
  std::vector<CompiledXPathPtr> namespaceSelectors;
};

struct XPathEvaluationState {
  std::exception_ptr error;
  std::optional<int> selectorErrorCode;
  const std::vector<CompiledXPathPtr>* namespaceSelectors{nullptr};
  std::string_view path;
};

struct XmlParserState {
  bool undeclaredEntity{false};
  std::exception_ptr error;
};

void ensureLibxml2Initialized() {
  struct Libxml2Init {
    Libxml2Init() {
      xmlInitParser();
    }
  };
  static Libxml2Init instance;
}

// Match the structured-error callback type from the libxml2 headers.
#if LIBXML_VERSION >= 21200
void ignoreStructuredXmlError(void*, const xmlError*) {}
void recordStructuredXmlError(void* context, const xmlError* error) {
#else
void ignoreStructuredXmlError(void*, xmlError*) {}
void recordStructuredXmlError(void* context, xmlError* error) {
#endif
  if (error != nullptr &&
      (error->code == XML_ERR_UNDECLARED_ENTITY ||
       error->code == XML_WAR_UNDECLARED_ENTITY)) {
    auto* parser = static_cast<xmlParserCtxtPtr>(context);
    static_cast<XmlParserState*>(parser->sax->_private)->undeclaredEntity =
        true;
  }
}

void recordGenericXmlError(void* context, const char*, ...) {
  recordStructuredXmlError(
      context, xmlCtxtGetLastError(static_cast<xmlParserCtxtPtr>(context)));
}

// Use SAX1 names without namespace validation, but reuse libxml2's DOM builder.
void startNamespaceUnawareElement(
    void* context,
    const xmlChar* name,
    const xmlChar** attributes) noexcept {
  auto* parser = static_cast<xmlParserCtxtPtr>(context);
  try {
    std::vector<const xmlChar*> rewritten;
    if (attributes != nullptr) {
      for (size_t i = 0; attributes[i] != nullptr; i += 2) {
        const auto* value = attributes[i + 1];
        rewritten.insert(
            rewritten.end(),
            {attributes[i], nullptr, nullptr, value, value + xmlStrlen(value)});
      }
    }
    xmlSAX2StartElementNs(
        context,
        name,
        nullptr,
        nullptr,
        0,
        nullptr,
        static_cast<int>(rewritten.size() / 5),
        0,
        rewritten.data());
  } catch (...) {
    static_cast<XmlParserState*>(parser->sax->_private)->error =
        std::current_exception();
    xmlStopParser(parser);
  }
}

void endNamespaceUnawareElement(void* context, const xmlChar* name) {
  xmlSAX2EndElementNs(context, name, nullptr, nullptr);
}

static_assert(
    std::is_convertible_v<
        decltype(&recordStructuredXmlError),
        xmlStructuredErrorFunc>,
    "libxml2 structured error callback signature mismatch");

// Keep diagnostics context-local and disable external resolvers.
void installLocalXmlErrorHandler(xmlParserCtxtPtr ctxt, XmlParserState& state) {
  VELOX_CHECK_NOT_NULL(ctxt);
  VELOX_CHECK_NOT_NULL(ctxt->sax);
  VELOX_CHECK_EQ(
      ctxt->sax->initialized, XML_SAX2_MAGIC, "XML parser context is not SAX2");
  ctxt->sax->_private = &state;
  ctxt->sax->serror = recordStructuredXmlError;
  ctxt->sax->error = recordGenericXmlError;
  ctxt->sax->warning = recordGenericXmlError;
  ctxt->sax->fatalError = recordGenericXmlError;
  ctxt->sax->startElement = startNamespaceUnawareElement;
  ctxt->sax->endElement = endNamespaceUnawareElement;
  ctxt->sax->startElementNs = nullptr;
  ctxt->sax->endElementNs = nullptr;
  ctxt->sax->resolveEntity = nullptr;
  ctxt->sax->externalSubset = nullptr;
#if LIBXML_VERSION >= 21300
  xmlCtxtSetErrorHandler(ctxt, recordStructuredXmlError, ctxt);
#endif
}

// Parser and XPath failures update the thread-local xmlLastError even when
// the report itself is swallowed. Restore the caller's error around evaluation.
class LibxmlLastErrorGuard {
 public:
  LibxmlLastErrorGuard() {
    ensureLibxml2Initialized();
    // 2.12+ returns const xmlError* for a mutable thread-local object.
#if LIBXML_VERSION >= 21200
    error_ = const_cast<xmlError*>(xmlGetLastError());
#else
    error_ = xmlGetLastError();
#endif
    if (error_ != nullptr) {
      // Transfer the owned strings: copying can truncate the message or fail
      // to allocate, including while unwinding an allocation failure.
      saved_ = std::exchange(*error_, xmlError{});
    }
  }

  ~LibxmlLastErrorGuard() noexcept {
    xmlResetLastError();
    if (error_ != nullptr) {
      *error_ = saved_;
    }
  }

  LibxmlLastErrorGuard(const LibxmlLastErrorGuard&) = delete;
  LibxmlLastErrorGuard& operator=(const LibxmlLastErrorGuard&) = delete;
  LibxmlLastErrorGuard(LibxmlLastErrorGuard&&) = delete;
  LibxmlLastErrorGuard& operator=(LibxmlLastErrorGuard&&) = delete;

 private:
  xmlError saved_{};
  xmlError* error_{nullptr};
};

bool isUnsafeErrorCodePoint(int32_t codePoint) {
  return codePoint < 0x20 || (codePoint >= 0x7f && codePoint <= 0x9f) ||
      (codePoint >= 0x200b && codePoint <= 0x200f) ||
      (codePoint >= 0x2028 && codePoint <= 0x202e) ||
      (codePoint >= 0x2060 && codePoint <= 0x206f) || codePoint == 0xfeff;
}

// Maximum number of path bytes echoed in an error message.
constexpr size_t kMaxXPathErrorBytes = 128;

// Keep exception text bounded, valid UTF-8, and free of control characters.
// The path is caller-controlled and may be logged with the user error.
std::string boundedXPathForError(std::string_view path) {
  const size_t limit = std::min(path.size(), kMaxXPathErrorBytes);
  std::string bounded;
  bounded.reserve(limit + 3);
  size_t i = 0;
  while (i < limit) {
    int32_t codePoint;
    const auto sequence =
        tryGetUtf8CharLength(path.data() + i, path.size() - i, codePoint);
    if (sequence <= 0) {
      bounded.push_back('?');
      ++i;
      continue;
    }
    // A code point that would cross the cap is omitted rather than split.
    if (i + sequence > limit) {
      break;
    }
    if (isUnsafeErrorCodePoint(codePoint)) {
      bounded.push_back('?');
    } else {
      bounded.append(path.data() + i, sequence);
    }
    i += sequence;
  }
  if (i < path.size()) {
    bounded.append("...");
  }
  return bounded;
}

// Allocation failures and a missing diagnostic are engine failures rather than
// bad input, so they are not raised as user errors that TRY would hide.
Status invalidXmlStatus(int code) {
  // Do not echo parser text. It can repeat document bytes or a system
  // identifier from the input. No code means the parser returned no document
  // and no diagnostic (for example, the input buffer could not be created).
  if (code == XML_ERR_NO_MEMORY || code == XML_ERR_OK) {
    VELOX_FAIL("Failed to parse XML document");
  }
  return Status::UserError("Invalid XML document (libxml2 code {})", code);
}

Status invalidXPathStatus(std::string_view path, int code) {
  if (code == XML_ERR_NO_MEMORY || code == XML_XPATH_MEMORY_ERROR ||
      code == XML_ERR_OK) {
    VELOX_FAIL("Failed to evaluate XPath");
  }
  const std::string shown = boundedXPathForError(path);
  return Status::UserError("Invalid XPath: {} (libxml2 code {})", shown, code);
}

constexpr bool isXPathNameByte(char byte) {
  const auto value = static_cast<unsigned char>(byte);
  return value >= 0x80 || (value >= 'a' && value <= 'z') ||
      (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') ||
      value == '_' || value == '-' || value == '.';
}

// Bind path prefixes to the empty URI without importing document namespaces.
// Spark's namespace-unaware nodes never match these prefixed name tests.
void registerNamespaceUnawarePrefixes(
    xmlXPathContextPtr context,
    std::string_view path) {
  char quote = '\0';
  for (size_t i = 0; i < path.size(); ++i) {
    const char current = path[i];
    if (quote != '\0') {
      if (current == quote) {
        quote = '\0';
      }
      continue;
    }
    if (current == '\'' || current == '"') {
      quote = current;
      continue;
    }
    if (current == ':' && i > 0 && i + 1 < path.size() && path[i - 1] != ':' &&
        path[i + 1] != ':' && isXPathNameByte(path[i - 1]) &&
        (isXPathNameByte(path[i + 1]) || path[i + 1] == '*')) {
      size_t start = i - 1;
      while (start > 0 && isXPathNameByte(path[start - 1])) {
        --start;
      }
      const std::string prefix(path.substr(start, i - start));
      VELOX_CHECK_EQ(
          xmlXPathRegisterNs(
              context,
              reinterpret_cast<const xmlChar*>(prefix.c_str()),
              reinterpret_cast<const xmlChar*>("")),
          0,
          "Failed to register XPath prefix");
    }
  }
}

// Preserve the JDK's two name tests: ordinary walkers use local names, while
// eligible descendant iterators use the DOM's full qualified name.
class NamespaceUnawarePath {
 public:
  explicit NamespaceUnawarePath(std::string_view path) : path_(path) {
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
      } else if (isXPathNameByte(byte) && byte != '.' && byte != '-') {
        kind = TokenKind::kName;
        while (offset < path.size()) {
          if (isXPathNameByte(path[offset])) {
            ++offset;
          } else if (
              path[offset] == ':' &&
              (offset + 1 == path.size() || path[offset + 1] != ':')) {
            ++offset;
            if (offset < path.size() && path[offset] == '*') {
              ++offset;
            }
          } else {
            break;
          }
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
  }

  folly::Expected<std::string, Status> rewrite() {
    try {
      expression(1);
      if (current().kind != TokenKind::kEnd) {
        throw InvalidPlan{};
      }
      operations_[1] = operations_.size();
      operations_.resize(operations_.size() + 4, 0);
      for (const auto& path : paths_) {
        rewriteSteps(path.start, path.steps, path.iteratorOrder);
      }
      return render(0, path_.size());
    } catch (const InvalidPlan&) {
      return folly::makeUnexpected(
          Status::UserError("Invalid XPath: {}", boundedXPathForError(path_)));
    }
  }

  bool usesInternalFunctions() const {
    for (size_t i = 0; i + 1 < tokens_.size(); ++i) {
      if (tokens_[i].kind == TokenKind::kName && tokens_[i + 1].text == "(" &&
          (tokens_[i].text == "_velox_namespaces" ||
           tokens_[i].text == "_velox_namespace_step" ||
           tokens_[i].text == "_velox_first_step")) {
        return true;
      }
    }
    return false;
  }

  const std::vector<std::string>& namespaceSelectors() const {
    return namespaceSelectors_;
  }

 private:
  enum class TokenKind { kName, kNumber, kLiteral, kSymbol, kEnd };

  struct Token {
    std::string_view text;
    size_t offset;
    TokenKind kind;
    int64_t poolIndex{0};
  };

  // Preserve operation layout, not just expression dependencies. JDK 17/21
  // WalkerFactory indexes the map by opcode for nested comparisons, so the
  // surrounding expression can change descendant-iterator selection.
  enum Operation {
    kEnd = -1,
    kEmpty = -2,
    kWildcard = -3,
    kXPath = 1,
    kOr = 2,
    kAnd = 3,
    kNotEqual = 4,
    kEqual = 5,
    kLessEqual = 6,
    kLess = 7,
    kGreaterEqual = 8,
    kGreater = 9,
    kAdd = 10,
    kSubtract = 11,
    kMultiply = 12,
    kDivide = 13,
    kModulo = 14,
    kNegate = 16,
    kUnion = 20,
    kLiteral = 21,
    kVariable = 22,
    kGroup = 23,
    kExtensionFunction = 24,
    kFunction = 25,
    kArgument = 26,
    kNumber = 27,
    kLocation = 28,
    kPredicate = 29,
    kNameTest = 34,
    kRootTest = 35,
    kAncestor = 37,
    kCommentTest = 1030,
    kTextTest = 1031,
    kProcessingInstructionTest = 1032,
    kNodeTest = 1033,
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
    int64_t left;
    if (take("-")) {
      appendOperation(kNegate);
      expression(7);
      finishOperation(start);
      left = kNegate;
    } else {
      left = pathExpression();
    }
    for (int level = precedence(current().text); level >= minimum;
         level = precedence(current().text)) {
      const auto code = binaryOperation(current().text);
      ++position_;
      if (code != kUnion || left != kUnion) {
        insertOperation(start, code);
      }
      expression(level <= 2 ? level : level + 1);
      finishOperation(start);
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
  std::map<size_t, Edit> edits_;
  std::vector<std::string> namespaceSelectors_;
  int64_t numPoolTokens_{0};
  size_t position_{0};
};

// libxml2 formats numbers with 15 significant digits and sometimes an
// exponent. XPath strings require decimal notation without losing the
// digits needed to distinguish the underlying double.
std::string numberToXPathString(double number) {
  if (std::isnan(number)) {
    return "NaN";
  }
  if (std::isinf(number)) {
    return number < 0 ? "-Infinity" : "Infinity";
  }
  if (number == 0) {
    return "0";
  }

  std::string value = fmt::format("{}", number);
  const auto exponentOffset = value.find('e');
  if (exponentOffset == std::string::npos) {
    return value;
  }

  auto* exponentStart = value.data() + exponentOffset + 1;
  const auto* end = value.data() + value.size();
  if (*exponentStart == '+') {
    ++exponentStart;
  }
  int exponent;
  const auto parsed = std::from_chars(exponentStart, end, exponent);
  VELOX_CHECK(
      parsed.ec == std::errc{} && parsed.ptr == end,
      "Failed to format XPath number");
  value.resize(exponentOffset);
  const size_t sign = value.front() == '-' ? 1 : 0;
  const auto point = value.find('.');
  const int decimalPosition =
      static_cast<int>(
          (point == std::string::npos ? value.size() : point) - sign) +
      exponent;
  if (point != std::string::npos) {
    value.erase(point, 1);
  }
  const int digits = static_cast<int>(value.size() - sign);
  if (decimalPosition <= 0) {
    value.insert(sign, "0." + std::string(-decimalPosition, '0'));
  } else if (decimalPosition < digits) {
    value.insert(sign + decimalPosition, 1, '.');
  } else {
    value.append(decimalPosition - digits, '0');
  }
  return value;
}

// Normalize numeric string arguments before delegating to libxml2 so explicit
// string(), implicit conversions, and the UDF's final conversion agree.
// substring's position and length arguments must remain numbers.
template <xmlXPathFunction function, bool firstArgumentOnly = false>
void callWithXPathNumberStrings(
    xmlXPathParserContextPtr ctxt,
    int nargs) noexcept {
  try {
    if (nargs > 0 && nargs <= ctxt->valueNr) {
      const int first = ctxt->valueNr - nargs;
      const int end = firstArgumentOnly ? first + 1 : ctxt->valueNr;
      for (int i = first; i < end; ++i) {
        auto* argument = ctxt->valueTab[i];
        if (argument->type != XPATH_NUMBER) {
          continue;
        }
        const auto value = numberToXPathString(argument->floatval);
        auto* copy = xmlStrdup(reinterpret_cast<const xmlChar*>(value.c_str()));
        if (copy == nullptr) {
          xmlXPathSetError(ctxt, XPATH_MEMORY_ERROR);
          return;
        }
        argument->stringval = copy;
        argument->type = XPATH_STRING;
      }
    }
    function(ctxt, nargs);
  } catch (...) {
    // Never unwind C++ exceptions through libxml2. The evaluator rethrows the
    // original exception after libxml2 has released its evaluation stack.
    static_cast<XPathEvaluationState*>(ctxt->context->userData)->error =
        std::current_exception();
    xmlXPathSetError(ctxt, XPATH_EXPR_ERROR);
  }
}

// Strip qualified names as the JDK does even when the DOM is namespace-unaware.
void namespaceUnawareLocalName(xmlXPathParserContextPtr context, int nargs) {
  auto* node = nargs == 0 ? context->context->node : nullptr;
  if (nargs == 1 && context->value != nullptr &&
      (context->value->type == XPATH_NODESET ||
       context->value->type == XPATH_XSLT_TREE)) {
    const auto* nodes = context->value->nodesetval;
    if (nodes != nullptr && nodes->nodeNr > 0) {
      node = nodes->nodeTab[0];
    }
  }
  const bool namespaceNode =
      node != nullptr && node->type == XML_NAMESPACE_DECL;
  xmlXPathLocalNameFunction(context, nargs);
  if (namespaceNode || context->error != XPATH_EXPRESSION_OK ||
      context->value == nullptr || context->value->type != XPATH_STRING) {
    return;
  }
  auto* value = context->value->stringval;
  if (value != nullptr) {
    if (const auto* colon = xmlStrchr(value, ':')) {
      std::memmove(value, colon + 1, xmlStrlen(colon + 1) + 1);
    }
  }
}

xmlXPathFunction lookupXPathStringFunction(
    void*,
    const xmlChar* name,
    const xmlChar* namespaceUri);

void sortNamespaceNodes(xmlNodeSetPtr nodes) {
  if (nodes == nullptr || nodes->nodeNr < 2) {
    return;
  }
  // libxml2 does not order namespace copies by their declaring owners.
  std::sort(
      nodes->nodeTab,
      nodes->nodeTab + nodes->nodeNr,
      [](xmlNodePtr left, xmlNodePtr right) {
        const auto* lhs = reinterpret_cast<xmlNsPtr>(left);
        const auto* rhs = reinterpret_cast<xmlNsPtr>(right);
        if (lhs->next == rhs->next) {
          // The declaration slot precedes the implicit xml slot on its owner.
          const auto* xml = reinterpret_cast<const xmlChar*>("xml");
          return !xmlStrEqual(lhs->prefix, xml) &&
              xmlStrEqual(rhs->prefix, xml);
        }
        return xmlXPathCmpNodes(
                   reinterpret_cast<xmlNodePtr>(lhs->next),
                   reinterpret_cast<xmlNodePtr>(rhs->next)) > 0;
      });
}

// DOM2DTM groups namespace-unaware declarations under one expanded-name ID.
// The last declaration replaces that slot, while the implicit xml binding has
// its own slot. Inherited namespace nodes retain their declaring element.
void namespaceUnawareNamespaces(xmlXPathParserContextPtr parser, int nargs) {
  if (nargs != 0) {
    xmlXPathSetArityError(parser);
    return;
  }
  XPathObjectPtr result{xmlXPathNewNodeSet(nullptr), xmlXPathFreeObject};
  if (!result) {
    xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
    return;
  }
  auto* context = parser->context;
  auto* node = context->node;
  if (node != nullptr && node->type == XML_ELEMENT_NODE) {
    auto* root = xmlDocGetRootElement(context->doc);
    xmlNsPtr selected = nullptr;
    auto* owner = node;
    for (; owner != nullptr && owner->type == XML_ELEMENT_NODE;
         owner = owner->parent) {
      for (auto* declaration = owner->nsDef; declaration != nullptr;
           declaration = declaration->next) {
        selected = declaration;
      }
      if (selected != nullptr) {
        break;
      }
    }
    // Without a root declaration, DOM2DTM creates the implicit xml slot first.
    const bool xmlFirst = root->nsDef == nullptr;
    if (!xmlFirst && selected != nullptr &&
        xmlXPathNodeSetAddNs(result->nodesetval, owner, selected) < 0) {
      xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
      return;
    }
    bool explicitXml = false;
    for (auto* declaration = root->nsDef; declaration != nullptr;
         declaration = declaration->next) {
      explicitXml = explicitXml ||
          xmlStrEqual(declaration->prefix,
                      reinterpret_cast<const xmlChar*>("xml"));
    }
    if (!explicitXml) {
      xmlNs xmlNamespace{};
      xmlNamespace.type = XML_LOCAL_NAMESPACE;
      xmlNamespace.href = XML_XML_NAMESPACE;
      xmlNamespace.prefix = reinterpret_cast<const xmlChar*>("xml");
      if (xmlXPathNodeSetAddNs(result->nodesetval, root, &xmlNamespace) < 0) {
        xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
        return;
      }
    }
    if (xmlFirst && selected != nullptr &&
        xmlXPathNodeSetAddNs(result->nodesetval, owner, selected) < 0) {
      xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
      return;
    }
  }
  if (valuePush(parser, result.get()) < 0) {
    xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
    return;
  }
  result.release();
}

// Apply namespace-axis predicates separately for each input node before merging
// results. First-match continuations instead stop at the first nonempty result.
template <bool firstMatch>
void namespaceUnawareStep(xmlXPathParserContextPtr parser, int nargs) noexcept {
  try {
    if (nargs != (firstMatch ? 2 : 3)) {
      xmlXPathSetArityError(parser);
      return;
    }
    const bool iteratorOrder = firstMatch || xmlXPathPopBoolean(parser);
    const auto selectorIndex = xmlXPathPopNumber(parser);
    XPathObjectPtr source{valuePop(parser), xmlXPathFreeObject};
    if (parser->error != XPATH_EXPRESSION_OK) {
      return;
    }
    if (!source || source->type != XPATH_NODESET) {
      xmlXPathSetTypeError(parser);
      return;
    }
    auto* state = static_cast<XPathEvaluationState*>(parser->context->userData);
    VELOX_CHECK_NOT_NULL(state->namespaceSelectors);
    const auto& selectors = *state->namespaceSelectors;
    VELOX_CHECK_GE(selectorIndex, 0);
    VELOX_CHECK_LT(selectorIndex, selectors.size());
    auto* compiled = selectors[static_cast<size_t>(selectorIndex)].get();
    XPathObjectPtr result{xmlXPathNewNodeSet(nullptr), xmlXPathFreeObject};
    VELOX_CHECK_NOT_NULL(result.get(), "Failed to create XPath result");
    if (source->nodesetval != nullptr && source->nodesetval->nodeNr > 0) {
      std::unique_ptr<xmlXPathContext, decltype(&xmlXPathFreeContext)> context{
          xmlXPathNewContext(parser->context->doc), xmlXPathFreeContext};
      VELOX_CHECK_NOT_NULL(context.get(), "Failed to create XPath context");
      context->error = ignoreStructuredXmlError;
      context->userData = parser->context->userData;
      context->contextSize = 1;
      context->proximityPosition = 1;
      xmlXPathRegisterFuncLookup(
          context.get(), lookupXPathStringFunction, nullptr);
      registerNamespaceUnawarePrefixes(context.get(), state->path);
      for (int i = 0; i < source->nodesetval->nodeNr; ++i) {
        context->node = source->nodesetval->nodeTab[i];
        XPathObjectPtr selected{
            xmlXPathCompiledEval(compiled, context.get()), xmlXPathFreeObject};
        if (!selected) {
          if (!state->selectorErrorCode.has_value()) {
            state->selectorErrorCode = context->lastError.code;
          }
          xmlXPathSetError(
              parser,
              (context->lastError.code == XML_XPATH_MEMORY_ERROR ||
               context->lastError.code == XML_ERR_NO_MEMORY)
                  ? XPATH_MEMORY_ERROR
                  : XPATH_EXPR_ERROR);
          return;
        }
        VELOX_CHECK_EQ(
            static_cast<int>(selected->type), static_cast<int>(XPATH_NODESET));
        if constexpr (firstMatch) {
          if (selected->nodesetval != nullptr &&
              selected->nodesetval->nodeNr > 0) {
            result = std::move(selected);
            break;
          }
        } else if (selected->nodesetval != nullptr) {
          for (int j = 0; j < selected->nodesetval->nodeNr; ++j) {
            auto* node = selected->nodesetval->nodeTab[j];
            VELOX_CHECK_EQ(
                static_cast<int>(node->type),
                static_cast<int>(XML_NAMESPACE_DECL));
            auto* ns = reinterpret_cast<xmlNsPtr>(node);
            // Unlike merge, insertion retains the nodeset on allocation
            // failure.
            if (xmlXPathNodeSetAddNs(
                    result->nodesetval,
                    reinterpret_cast<xmlNodePtr>(ns->next),
                    ns) < 0) {
              xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
              return;
            }
          }
        }
      }
    }
    if (!iteratorOrder) {
      sortNamespaceNodes(result->nodesetval);
    }
    if (valuePush(parser, result.get()) < 0) {
      xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
      return;
    }
    result.release();
  } catch (...) {
    static_cast<XPathEvaluationState*>(parser->context->userData)->error =
        std::current_exception();
    xmlXPathSetError(parser, XPATH_EXPR_ERROR);
  }
}

xmlXPathFunction lookupXPathStringFunction(
    void*,
    const xmlChar* name,
    const xmlChar* namespaceUri) {
  if (namespaceUri != nullptr) {
    return nullptr;
  }
  static const std::pair<const char*, xmlXPathFunction> functions[] = {
      {"local-name", namespaceUnawareLocalName},
      {"_velox_namespaces", namespaceUnawareNamespaces},
      {"_velox_namespace_step", namespaceUnawareStep<false>},
      {"_velox_first_step", namespaceUnawareStep<true>},
      {"string", callWithXPathNumberStrings<xmlXPathStringFunction>},
      {"concat", callWithXPathNumberStrings<xmlXPathConcatFunction>},
      {"contains", callWithXPathNumberStrings<xmlXPathContainsFunction>},
      {"starts-with", callWithXPathNumberStrings<xmlXPathStartsWithFunction>},
      {"string-length",
       callWithXPathNumberStrings<xmlXPathStringLengthFunction>},
      {"substring",
       callWithXPathNumberStrings<xmlXPathSubstringFunction, true>},
      {"substring-before",
       callWithXPathNumberStrings<xmlXPathSubstringBeforeFunction>},
      {"substring-after",
       callWithXPathNumberStrings<xmlXPathSubstringAfterFunction>},
      {"normalize-space",
       callWithXPathNumberStrings<xmlXPathNormalizeFunction>},
      {"translate", callWithXPathNumberStrings<xmlXPathTranslateFunction>},
      {"id", callWithXPathNumberStrings<xmlXPathIdFunction>},
      {"lang", callWithXPathNumberStrings<xmlXPathLangFunction>},
  };
  for (const auto& [functionName, function] : functions) {
    if (xmlStrEqual(name, reinterpret_cast<const xmlChar*>(functionName))) {
      return function;
    }
  }
  return nullptr;
}

// Reject retained entity references before XPath can expand them via node
// string values. Do not walk an entity reference's children: they point into
// the DTD rather than the document tree. Predefined and numeric references
// are already text nodes and remain supported.
bool hasEntityReferences(xmlNodePtr root) {
  for (auto* node = root; node != nullptr;) {
    if (node->type == XML_ENTITY_REF_NODE) {
      return true;
    }
    if (node->type == XML_ELEMENT_NODE) {
      for (auto* attribute = node->properties; attribute != nullptr;
           attribute = attribute->next) {
        for (auto* value = attribute->children; value != nullptr;
             value = value->next) {
          if (value->type == XML_ENTITY_REF_NODE) {
            return true;
          }
        }
      }
    }
    if (node->children != nullptr) {
      node = node->children;
      continue;
    }
    while (node != root && node->next == nullptr) {
      node = node->parent;
    }
    if (node == root) {
      break;
    }
    node = node->next;
  }
  return false;
}

bool jdkNameLess(const xmlChar* leftName, const xmlChar* rightName) {
  std::string_view left(reinterpret_cast<const char*>(leftName));
  std::string_view right(reinterpret_cast<const char*>(rightName));
  while (!left.empty() && !right.empty()) {
    int32_t leftPoint;
    int32_t rightPoint;
    const auto leftBytes =
        tryGetUtf8CharLength(left.data(), left.size(), leftPoint);
    const auto rightBytes =
        tryGetUtf8CharLength(right.data(), right.size(), rightPoint);
    VELOX_CHECK_GT(leftBytes, 0);
    VELOX_CHECK_GT(rightBytes, 0);
    if (leftPoint != rightPoint) {
      // UTF-16 places supplementary characters before BMP characters above
      // the surrogate range, unlike UTF-8 byte order.
      const auto leftRank = leftPoint >= 0x10000 ? 0xd800 : leftPoint;
      const auto rightRank = rightPoint >= 0x10000 ? 0xd800 : rightPoint;
      return leftRank == rightRank ? leftPoint < rightPoint
                                   : leftRank < rightRank;
    }
    left.remove_prefix(leftBytes);
    right.remove_prefix(rightBytes);
  }
  return left.empty() && !right.empty();
}

// Order attributes as the JDK DOM does and keep declarations off the attribute
// axis. The SAX builder has already preserved namespace-unaware qualified
// names.
void makeNamespaceUnaware(xmlNodePtr root) {
  std::vector<xmlAttrPtr> attributes;
  for (auto* node = root; node != nullptr;) {
    if (node->type == XML_ELEMENT_NODE) {
      attributes.clear();
      for (auto* attribute = node->properties; attribute != nullptr;
           attribute = attribute->next) {
        attributes.push_back(attribute);
      }
      // The JDK DOM enumerates attributes in qualified-name order.
      if (attributes.size() > 1) {
        std::sort(
            attributes.begin(), attributes.end(), [](auto* left, auto* right) {
              return jdkNameLess(left->name, right->name);
            });
        node->properties = attributes.front();
        for (size_t i = 0; i < attributes.size(); ++i) {
          attributes[i]->prev = i == 0 ? nullptr : attributes[i - 1];
          attributes[i]->next =
              i + 1 == attributes.size() ? nullptr : attributes[i + 1];
        }
      }
      auto** nextNamespace = &node->nsDef;
      while (*nextNamespace != nullptr) {
        nextNamespace = &(*nextNamespace)->next;
      }
      for (auto* attribute : attributes) {
        const std::string_view name(
            reinterpret_cast<const char*>(attribute->name));
        if (name == "xmlns" || name.starts_with("xmlns:")) {
          XmlStringPtr value{
              xmlNodeGetContent(reinterpret_cast<xmlNodePtr>(attribute)),
              xmlFree};
          VELOX_CHECK_NOT_NULL(value.get(), "Failed to read XML namespace");
          // Older libxml2 skips the canonical xml binding in xmlNewNs.
          // Preserve every declaration, including that explicit binding.
          std::unique_ptr<xmlNs, decltype(&xmlFreeNs)> declaration{
              xmlNewNs(nullptr, nullptr, nullptr), xmlFreeNs};
          VELOX_CHECK_NOT_NULL(
              declaration.get(), "Failed to create XML namespace");
          declaration->href = value.release();
          if (name != "xmlns") {
            declaration->prefix = xmlStrdup(attribute->name + 6);
            VELOX_CHECK_NOT_NULL(
                declaration->prefix, "Failed to create XML namespace prefix");
          }
          *nextNamespace = declaration.release();
          nextNamespace = &(*nextNamespace)->next;
          VELOX_CHECK_EQ(xmlRemoveProp(attribute), 0);
        }
      }
    }
    if (node->children != nullptr) {
      node = node->children;
      continue;
    }
    while (node != root && node->next == nullptr) {
      node = node->parent;
    }
    if (node == root) {
      break;
    }
    node = node->next;
  }
}

// Retains compiled paths across rows without sharing them between threads.
// Compiled expressions are not tracked by memory pools, so bound both the entry
// count and the path length, evicting an arbitrary entry when full so that new
// constant paths remain cacheable on long-lived threads.
class CompiledXPathCache {
 public:
  const CompiledXPath* find(std::string_view path) const {
    auto it = cache_.find(path);
    return it == cache_.end() ? nullptr : it->second.get();
  }

  // Moves ownership into the cache only when the path fits. Otherwise the
  // caller's unique_ptr frees the expression after this evaluation. Called
  // only on a miss, so eviction never frees 'compiled'.
  void store(std::string path, std::unique_ptr<CompiledXPath>& compiled) {
    if (path.size() > kMaxCachedXPathBytes) {
      return;
    }
    if (cache_.size() >= kMaxCompiledXPathCacheEntries) {
      cache_.erase(cache_.begin());
    }
    cache_.try_emplace(std::move(path), std::move(compiled));
  }

 private:
  struct TransparentStringHash {
    using is_transparent = void;

    size_t operator()(std::string_view value) const {
      return std::hash<std::string_view>{}(value);
    }

    size_t operator()(const std::string& value) const {
      return (*this)(std::string_view(value));
    }
  };

  static constexpr size_t kMaxCompiledXPathCacheEntries = 32;
  static constexpr size_t kMaxCachedXPathBytes = 4'096;

  std::unordered_map<
      std::string,
      std::unique_ptr<CompiledXPath>,
      TransparentStringHash,
      std::equal_to<>>
      cache_;
};

CompiledXPathCache& compiledXPathCache() {
  static thread_local CompiledXPathCache cache;
  return cache;
}

// Owns the document and its evaluation context. The document node is the XPath
// context node, as with Spark's DocumentBuilder/XPath combination, so root,
// parent, wildcard, and relative-path semantics match Spark.
class XmlXPathEvaluator {
 public:
  explicit XmlXPathEvaluator(std::string_view xml) {
    if (xml.size() > static_cast<size_t>(INT_MAX)) {
      status_ = Status::UserError("XML document is too large");
      return;
    }
    if (xml.find('\0') != std::string_view::npos) {
      status_ = Status::UserError("Invalid XML document");
      return;
    }
    // Spark's character Reader does not consume a byte-order mark in the
    // prolog, unlike libxml2's byte-oriented input.
    if (xml.starts_with("\xef\xbb\xbf")) {
      status_ = Status::UserError("Invalid XML document");
      return;
    }

    // Parse Spark UTF-8 strings without reinterpreting encoding declarations.
    // Merge CDATA with adjacent text and disable network and external-entity
    // access while retaining context-local diagnostics.
    int options = XML_PARSE_NONET | XML_PARSE_IGNORE_ENC | XML_PARSE_NOCDATA;
#if LIBXML_VERSION >= 21300
    options |= XML_PARSE_NO_XXE;
#endif
    using ParserCtxtPtr =
        std::unique_ptr<xmlParserCtxt, decltype(&xmlFreeParserCtxt)>;
    XmlParserState parserState;
    ParserCtxtPtr parser{xmlNewParserCtxt(), xmlFreeParserCtxt};
    VELOX_CHECK_NOT_NULL(parser, "Failed to create XML parser context");
    installLocalXmlErrorHandler(parser.get(), parserState);
    doc_.reset(xmlCtxtReadMemory(
        parser.get(),
        xml.data(),
        static_cast<int>(xml.size()),
        nullptr,
        "UTF-8",
        options));
    if (parserState.error) {
      std::rethrow_exception(parserState.error);
    }
    const xmlError* parseError = xmlCtxtGetLastError(parser.get());
    const int parseErrorCode =
        parseError == nullptr ? XML_ERR_OK : parseError->code;
    // Before libxml2 2.13, an allocation failure, including a text node over
    // the size limit, stops the parser without clearing wellFormed, so a
    // truncated document can be returned.
    const bool outOfMemory = parser->errNo == XML_ERR_NO_MEMORY;
    if (doc_ == nullptr || outOfMemory) {
      status_ =
          invalidXmlStatus(outOfMemory ? XML_ERR_NO_MEMORY : parseErrorCode);
      return;
    }

    // Any retained general entity reference returns NULL, including one left
    // when an external subset is not loaded. That document has no internal
    // subset. Some libxml2 versions replace an undeclared attribute reference
    // with an empty value while reporting it in the parser diagnostic instead
    // of retaining an entity-reference node.
    blockedEntities_ = parserState.undeclaredEntity ||
        hasEntityReferences(xmlDocGetRootElement(doc_.get()));
    if (!blockedEntities_) {
      makeNamespaceUnaware(xmlDocGetRootElement(doc_.get()));
    }

    ctx_.reset(xmlXPathNewContext(doc_.get()));
    VELOX_CHECK_NOT_NULL(ctx_, "Failed to create XPath context");
    ctx_->node = reinterpret_cast<xmlNodePtr>(doc_.get());
    ctx_->contextSize = 1;
    ctx_->proximityPosition = 1;
    ctx_->error = ignoreStructuredXmlError;
    ctx_->userData = &evaluationState_;
    xmlXPathRegisterFuncLookup(ctx_.get(), lookupXPathStringFunction, nullptr);
  }

  folly::Expected<XPathObjectPtr, Status> eval(std::string_view path) {
    if (!status_.ok()) {
      return folly::makeUnexpected(status_);
    }
    // Reject embedded NUL because libxml2 consumes NUL-terminated expressions.
    if (path.find('\0') != std::string::npos) {
      return folly::makeUnexpected(
          Status::UserError("Invalid XPath: {}", boundedXPathForError(path)));
    }
    registerNamespaceUnawarePrefixes(ctx_.get(), path);

    auto& cache = compiledXPathCache();
    const CompiledXPath* compiled = cache.find(path);
    std::unique_ptr<CompiledXPath> owned;
    std::string ownedPath;
    if (compiled == nullptr) {
      ownedPath.assign(path);
      owned = std::make_unique<CompiledXPath>();
      owned->expression.reset(xmlXPathCtxtCompile(
          ctx_.get(), reinterpret_cast<const xmlChar*>(ownedPath.c_str())));
      if (!owned->expression) {
        return folly::makeUnexpected(
            invalidXPathStatus(path, ctx_->lastError.code));
      }
      NamespaceUnawarePath normalized(path);
      if (normalized.usesInternalFunctions()) {
        return folly::makeUnexpected(
            Status::UserError("Invalid XPath: {}", boundedXPathForError(path)));
      }
      const auto rewritten = normalized.rewrite();
      if (rewritten.hasError()) {
        return folly::makeUnexpected(rewritten.error());
      }
      if (rewritten.value() != path) {
        owned->expression.reset(xmlXPathCtxtCompile(
            ctx_.get(),
            reinterpret_cast<const xmlChar*>(rewritten.value().c_str())));
        VELOX_CHECK_NOT_NULL(
            owned->expression.get(), "Failed to normalize XPath");
      }
      for (const auto& selector : normalized.namespaceSelectors()) {
        CompiledXPathPtr selectorExpression{
            xmlXPathCtxtCompile(
                ctx_.get(), reinterpret_cast<const xmlChar*>(selector.c_str())),
            xmlXPathFreeCompExpr};
        VELOX_CHECK_NOT_NULL(
            selectorExpression.get(), "Failed to compile XPath selector");
        owned->namespaceSelectors.push_back(std::move(selectorExpression));
      }
      compiled = owned.get();
      cache.store(std::move(ownedPath), owned);
    }

    XPathObjectPtr result{nullptr, xmlXPathFreeObject};
    if (blockedEntities_) {
      return result;
    }
    evaluationState_.namespaceSelectors = &compiled->namespaceSelectors;
    evaluationState_.path = path;
    result.reset(xmlXPathCompiledEval(compiled->expression.get(), ctx_.get()));
    if (evaluationState_.error) {
      std::rethrow_exception(evaluationState_.error);
    }
    if (!result) {
      const auto code = ctx_->lastError.code;
      return folly::makeUnexpected(invalidXPathStatus(
          path,
          (code == XML_ERR_OK || code == XML_ERR_NO_MEMORY ||
           code == XML_XPATH_MEMORY_ERROR)
              ? code
              : evaluationState_.selectorErrorCode.value_or(code)));
    }
    return result;
  }

 private:
  // Declared first so it is destroyed last and sees every libxml2 call above.
  LibxmlLastErrorGuard lastErrorGuard_;
  XPathEvaluationState evaluationState_;
  std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> doc_{nullptr, xmlFreeDoc};
  std::unique_ptr<xmlXPathContext, decltype(&xmlXPathFreeContext)> ctx_{
      nullptr,
      xmlXPathFreeContext};
  Status status_;
  bool blockedEntities_{false};
};

} // namespace

XPathStringValue::XPathStringValue(std::string value)
    : value_(std::move(value)) {}

XPathStringValue::XPathStringValue(void* value, size_t size)
    : xmlValue_(value), xmlSize_(size) {}

XPathStringValue::~XPathStringValue() {
  if (xmlValue_ != nullptr) {
    xmlFree(xmlValue_);
  }
}

XPathStringValue::XPathStringValue(XPathStringValue&& other) noexcept
    : value_(std::move(other.value_)),
      xmlValue_(std::exchange(other.xmlValue_, nullptr)),
      xmlSize_(std::exchange(other.xmlSize_, 0)) {}

XPathStringValue& XPathStringValue::operator=(
    XPathStringValue&& other) noexcept {
  if (this != &other) {
    if (xmlValue_ != nullptr) {
      xmlFree(xmlValue_);
    }
    value_ = std::move(other.value_);
    xmlValue_ = std::exchange(other.xmlValue_, nullptr);
    xmlSize_ = std::exchange(other.xmlSize_, 0);
  }
  return *this;
}

std::string_view XPathStringValue::view() const {
  if (xmlValue_ != nullptr) {
    return {
        reinterpret_cast<const char*>(xmlValue_),
        xmlSize_,
    };
  }
  return value_;
}

folly::Expected<std::optional<bool>, Status> evalBoolean(
    std::string_view xml,
    std::string_view path) {
  if (xml.empty() || path.empty()) {
    return std::optional<bool>{};
  }
  XmlXPathEvaluator evaluator(xml);
  auto evaluated = evaluator.eval(path);
  if (evaluated.hasError()) {
    return folly::makeUnexpected(std::move(evaluated.error()));
  }
  auto& result = evaluated.value();
  if (!result) {
    return std::optional<bool>{};
  }
  return std::optional<bool>{xmlXPathCastToBoolean(result.get()) != 0};
}

folly::Expected<std::optional<XPathStringValue>, Status> evalString(
    std::string_view xml,
    std::string_view path) {
  if (xml.empty() || path.empty()) {
    return std::optional<XPathStringValue>{};
  }
  XmlXPathEvaluator evaluator(xml);
  auto evaluated = evaluator.eval(path);
  if (evaluated.hasError()) {
    return folly::makeUnexpected(std::move(evaluated.error()));
  }
  auto& result = evaluated.value();
  if (!result) {
    return std::optional<XPathStringValue>{};
  }
  if (result->type == XPATH_NUMBER) {
    XPathStringValue value(numberToXPathString(result->floatval));
    return std::optional<XPathStringValue>{std::move(value)};
  }

  auto* value = xmlXPathCastToString(result.get());
  VELOX_CHECK_NOT_NULL(value, "Failed to convert XPath result to string");
  // An empty node-set casts to "", matching XPathConstants.STRING in Spark.
  XPathStringValue resultValue(
      value, std::strlen(reinterpret_cast<const char*>(value)));
  return std::optional<XPathStringValue>{std::move(resultValue)};
}

} // namespace xpath
} // namespace facebook::velox::functions::sparksql
