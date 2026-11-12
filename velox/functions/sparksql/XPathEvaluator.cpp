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
#include "velox/functions/sparksql/XPathEvaluator.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <libxml/SAX2.h>
#include <libxml/parser.h>
#include <libxml/xmlerror.h>
#include <libxml/xpathInternals.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/sparksql/XPathContext.h"
#include "velox/functions/sparksql/XPathExpression.h"
#include "velox/functions/sparksql/XPathValue.h"

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

namespace facebook::velox::functions::sparksql::xpath {
namespace {

struct CompiledXPath {
  CompiledXPathPtr expression{nullptr, xmlXPathFreeCompExpr};
  std::vector<CompiledXPathPtr> namespaceSelectors;
  std::vector<std::string> prefixes;
  std::vector<double> literals;
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

// Bind only prefixes identified by the expression tokenizer. Spark's
// namespace-unaware nodes never match these prefixed name tests.
void registerNamespaceUnawarePrefixes(
    xmlXPathContextPtr context,
    const std::vector<std::string>& prefixes) {
  for (const auto& prefix : prefixes) {
    VELOX_CHECK_EQ(
        xmlXPathRegisterNs(
            context,
            reinterpret_cast<const xmlChar*>(prefix.c_str()),
            reinterpret_cast<const xmlChar*>("")),
        0,
        "Failed to register XPath prefix");
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
      VELOX_CHECK_NOT_NULL(state->prefixes);
      registerNamespaceUnawarePrefixes(context.get(), *state->prefixes);
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
  };
  for (const auto& [functionName, function] : functions) {
    if (xmlStrEqual(name, reinterpret_cast<const xmlChar*>(functionName))) {
      return function;
    }
  }
  return lookupXPathValueFunction(nullptr, name, namespaceUri);
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
      auto normalized = normalizeXPath(path);
      if (!normalized.has_value()) {
        return folly::makeUnexpected(
            Status::UserError("Invalid XPath: {}", boundedXPathForError(path)));
      }
      owned->prefixes = std::move(normalized->prefixes);
      owned->literals = std::move(normalized->literals);
      if (normalized->text != path) {
        owned->expression.reset(xmlXPathCtxtCompile(
            ctx_.get(),
            reinterpret_cast<const xmlChar*>(normalized->text.c_str())));
        VELOX_CHECK_NOT_NULL(
            owned->expression.get(), "Failed to normalize XPath");
      }
      for (const auto& selector : normalized->namespaceSelectors) {
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
    registerNamespaceUnawarePrefixes(ctx_.get(), compiled->prefixes);
    evaluationState_.namespaceSelectors = &compiled->namespaceSelectors;
    evaluationState_.prefixes = &compiled->prefixes;
    evaluationState_.literals = &compiled->literals;
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

folly::Expected<std::optional<std::string>, Status> evalString(
    std::string_view xml,
    std::string_view path) {
  if (xml.empty() || path.empty()) {
    return std::optional<std::string>{};
  }
  XmlXPathEvaluator evaluator(xml);
  auto evaluated = evaluator.eval(path);
  if (evaluated.hasError()) {
    return folly::makeUnexpected(std::move(evaluated.error()));
  }
  auto& result = evaluated.value();
  if (!result) {
    return std::optional<std::string>{};
  }
  return std::optional<std::string>{xpathResultString(result.get())};
}

} // namespace facebook::velox::functions::sparksql::xpath
