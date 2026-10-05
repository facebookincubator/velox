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
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
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
#else
void ignoreStructuredXmlError(void*, xmlError*) {}
#endif

static_assert(
    std::is_convertible_v<
        decltype(&ignoreStructuredXmlError),
        xmlStructuredErrorFunc>,
    "libxml2 structured error callback signature mismatch");

// Keep diagnostics context-local and disable external resolvers.
void installLocalXmlErrorHandler(xmlParserCtxtPtr ctxt) {
  VELOX_CHECK_NOT_NULL(ctxt);
  VELOX_CHECK_NOT_NULL(ctxt->sax);
  VELOX_CHECK_EQ(
      ctxt->sax->initialized, XML_SAX2_MAGIC, "XML parser context is not SAX2");
  ctxt->sax->serror = ignoreStructuredXmlError;
  ctxt->sax->error = nullptr;
  ctxt->sax->warning = nullptr;
  ctxt->sax->fatalError = nullptr;
  ctxt->sax->resolveEntity = nullptr;
  ctxt->sax->externalSubset = nullptr;
#if LIBXML_VERSION >= 21300
  xmlCtxtSetErrorHandler(ctxt, ignoreStructuredXmlError, nullptr);
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

constexpr bool hasUnregisteredPrefix(std::string_view path) {
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
      return true;
    }
  }
  return false;
}

static_assert(hasUnregisteredPrefix("x:a/x:b"));
static_assert(hasUnregisteredPrefix("f:g()"));
static_assert(hasUnregisteredPrefix("$p:v"));
static_assert(hasUnregisteredPrefix("ns:*"));
static_assert(!hasUnregisteredPrefix("child::node()"));
static_assert(hasUnregisteredPrefix("descendant-or-self::x:b"));
static_assert(!hasUnregisteredPrefix("\"a:b\""));
static_assert(!hasUnregisteredPrefix("'a:b'"));
static_assert(!hasUnregisteredPrefix("'a:b"));

bool isUnregisteredPrefixError(std::string_view path, int code) {
  if (code == XML_XPATH_UNDEF_PREFIX_ERROR) {
    return true;
  }
#if LIBXML_VERSION < 21300
  return code == XML_ERR_OK && hasUnregisteredPrefix(path);
#else
  return false;
#endif
}

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
    *static_cast<std::exception_ptr*>(ctxt->context->userData) =
        std::current_exception();
    xmlXPathSetError(ctxt, XPATH_EXPR_ERROR);
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

// Retains compiled paths across rows without sharing them between threads.
// Compiled expressions are not tracked by memory pools, so bound both the entry
// count and the path length, evicting an arbitrary entry when full so that new
// constant paths remain cacheable on long-lived threads.
class CompiledXPathCache {
 public:
  xmlXPathCompExprPtr find(std::string_view path) const {
    auto it = cache_.find(path);
    return it == cache_.end() ? nullptr : it->second.get();
  }

  // Moves ownership into the cache only when the path fits. Otherwise the
  // caller's unique_ptr frees the expression after this evaluation. Called
  // only on a miss, so eviction never frees 'compiled'.
  void store(std::string path, CompiledXPathPtr& compiled) {
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
      CompiledXPathPtr,
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
    ParserCtxtPtr parser{xmlNewParserCtxt(), xmlFreeParserCtxt};
    VELOX_CHECK_NOT_NULL(parser, "Failed to create XML parser context");
    installLocalXmlErrorHandler(parser.get());
    doc_.reset(xmlCtxtReadMemory(
        parser.get(),
        xml.data(),
        static_cast<int>(xml.size()),
        nullptr,
        "UTF-8",
        options));
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
    blockedEntities_ = parseErrorCode == XML_ERR_UNDECLARED_ENTITY ||
        parseErrorCode == XML_WAR_UNDECLARED_ENTITY ||
        hasEntityReferences(xmlDocGetRootElement(doc_.get()));

    ctx_.reset(xmlXPathNewContext(doc_.get()));
    VELOX_CHECK_NOT_NULL(ctx_, "Failed to create XPath context");
    ctx_->node = reinterpret_cast<xmlNodePtr>(doc_.get());
    ctx_->contextSize = 1;
    ctx_->proximityPosition = 1;
    ctx_->error = ignoreStructuredXmlError;
    ctx_->userData = &conversionError_;
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

    auto& cache = compiledXPathCache();
    xmlXPathCompExprPtr compiled = cache.find(path);
    CompiledXPathPtr owned{nullptr, xmlXPathFreeCompExpr};
    std::string ownedPath;
    if (compiled == nullptr) {
      ownedPath.assign(path);
      owned.reset(xmlXPathCtxtCompile(
          ctx_.get(), reinterpret_cast<const xmlChar*>(ownedPath.c_str())));
      if (!owned) {
        if (isUnregisteredPrefixError(path, ctx_->lastError.code)) {
          return XPathObjectPtr{nullptr, xmlXPathFreeObject};
        }
        return folly::makeUnexpected(
            invalidXPathStatus(path, ctx_->lastError.code));
      }
      compiled = owned.get();
      cache.store(std::move(ownedPath), owned);
    }

    XPathObjectPtr result{nullptr, xmlXPathFreeObject};
    if (blockedEntities_) {
      return result;
    }
    result.reset(xmlXPathCompiledEval(compiled, ctx_.get()));
    if (conversionError_) {
      std::rethrow_exception(conversionError_);
    }
    // Namespace-aware libxml2 cannot match Spark's namespace-unaware parser.
    // Preserve NULL for unregistered prefixes, but do not swallow other
    // evaluation failures (unknown functions, variables, argument/type errors).
    if (!result) {
      if (isUnregisteredPrefixError(path, ctx_->lastError.code)) {
        return result;
      }
      return folly::makeUnexpected(
          invalidXPathStatus(path, ctx_->lastError.code));
    }
    return result;
  }

 private:
  // Declared first so it is destroyed last and sees every libxml2 call above.
  LibxmlLastErrorGuard lastErrorGuard_;
  std::exception_ptr conversionError_;
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
