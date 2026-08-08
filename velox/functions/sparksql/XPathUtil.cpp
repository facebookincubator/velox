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

#include <climits>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "velox/common/base/Exceptions.h"

#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

namespace facebook::velox::functions::sparksql {
namespace xpath {

namespace {

/// No-op error handler to suppress libxml2 stderr output.
/// In a threaded libxml2 build (the default on Linux/macOS) the generic error
/// handler is per-thread global state, so it is installed once per thread (see
/// ensureLibxml2Initialized). Parse errors are additionally silenced per call
/// via XML_PARSE_NOERROR | XML_PARSE_NOWARNING on every thread. Suppression is
/// intentional: invalid XPath syntax is surfaced as a user error and other
/// invalid input as SQL NULL, not as stderr noise.
void suppressError(void* /*ctx*/, const char* /*msg*/, ...) {}

/// Thread-safe one-time libxml2 initialization (Meyers singleton).
/// Suppresses libxml2 error output to stderr.
///
/// Security notes:
/// - XML_PARSE_NONET prevents network access (XXE mitigation).
/// - XML_PARSE_NOENT is NOT set, so entity substitution is off.
/// - XML_PARSE_DTDLOAD is NOT set, so external DTDs are not loaded.
/// - libxml2 has a built-in entity expansion depth limit (default 40) which
///   mitigates billion-laughs style attacks. No additional configuration
///   needed.
void ensureLibxml2Initialized() {
  // Process-wide one-time parser initialization (thread-safe Meyers singleton).
  struct Libxml2Init {
    Libxml2Init() {
      xmlInitParser();
    }
  };
  static Libxml2Init instance;

  // libxml2's generic error handler lives in per-thread global state in
  // threaded builds (the default configuration on Linux/macOS), so the stderr
  // suppressor must be installed on every thread that evaluates XPath, not just
  // the first one to initialize. A thread_local guard installs it once per
  // thread; without this, XPath evaluation diagnostics on other worker threads
  // could still leak to stderr. (The structured-error API,
  // xmlSetStructuredErrorFunc, is the modern alternative but has the same
  // per-thread install requirement.)
  static thread_local bool errorHandlerInstalled = false;
  if (!errorHandlerInstalled) {
    xmlSetGenericErrorFunc(nullptr, suppressError);
    errorHandlerInstalled = true;
  }
}

/// Rewrite absolute paths in an XPath expression to go through the
/// synthetic wrapper root "_r".  Handles "/" at start or after XPath
/// operators/delimiters (with optional whitespace), but skips "//"
/// (descendant-or-self axis).
///
/// Known limitations (heuristic approach):
/// - String literals are skipped (both ' and " delimited), so "/" inside a
///   literal (e.g. contains(x, "/a/b")) is preserved verbatim. The backward
///   operator scan below inspects only characters outside string literals:
///   a '/' is processed only when inQuote == 0 (quotes balanced up to that
///   point), and the scan stops at the first non-whitespace char to its left,
///   which therefore cannot lie inside a literal. So quoted operator
///   characters can never trigger a spurious absolute-path rewrite.
/// - The multiplication operator '*' is intentionally NOT treated as an
///   absolute-path boundary, because '*' is overwhelmingly the wildcard
///   name-test ("a/*", "*/b"). As a result a contrived "3 * /a/b" leaves the
///   "/a/b" unrewritten and diverges from Spark; adding '*' would corrupt the
///   common wildcard case ("*/b" -> "*/_r/b") and is not worth that trade-off.
/// - Complex axis steps (e.g., "child::a/b") are not explicitly handled
///   but work naturally since they don't start with "/".
/// - Nested predicates with absolute paths (e.g., "a[/b=1]") are handled
///   by the operator-boundary detection.
std::string rewriteAbsolutePaths(const std::string& path) {
  if (path.empty()) {
    return path;
  }

  // Helper: check if position i is preceded by a word operator (or, and, div,
  // mod). These are XPath 1.0 keyword operators that can precede absolute
  // paths.
  auto endsWithWordOperator = [&](size_t pos) -> bool {
    // pos points to the last non-whitespace char before '/'.
    // Check if it ends with "or", "and", "div", or "mod" that is acting as a
    // binary operator (not an element name). Per the XPath 1.0 ExprToken
    // disambiguation rule, an NCName is an OperatorName only when the preceding
    // token is NOT one of '@', '::', '(', '[', ',', another Operator (which
    // includes '/' and '//'), or the start of the expression. Equivalently, the
    // word is an operator only when its left operand has just ended, i.e. the
    // immediately preceding character closes an operand: whitespace, ')', ']',
    // '*', or a string-literal quote. This avoids misclassifying a location
    // step whose element is literally named or/and/div/mod (e.g. the very
    // common "div") as an operator, which would otherwise splice the wrapper
    // root into the middle of a relative path ("a/div/b" -> "a/div/_r/b").
    static const std::vector<std::string_view> kOps = {
        "or", "and", "div", "mod"};
    auto isOperandEnd = [](char c) -> bool {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ')' ||
          c == ']' || c == '*' || c == '\'' || c == '"';
    };
    for (const auto& op : kOps) {
      if (pos + 1 >= op.size()) {
        size_t start = pos + 1 - op.size();
        if (path.compare(start, op.size(), op) == 0) {
          // A leading word (start == 0) can never be a binary operator.
          if (start > 0 && isOperandEnd(path[start - 1])) {
            return true;
          }
        }
      }
    }
    return false;
  };

  std::string result;
  result.reserve(path.size() + 16);
  // Tracks the active string-literal delimiter (0 when not inside a literal).
  // XPath string literals may contain '/' that must never be rewritten, e.g.
  // a[b='(/c)'] - without this the embedded "/c" would be treated as an
  // absolute path and corrupted to "/_r/c".
  char inQuote = 0;
  for (size_t i = 0; i < path.size(); ++i) {
    const char c = path[i];
    if (inQuote != 0) {
      result += c;
      if (c == inQuote) {
        inQuote = 0;
      }
      continue;
    }
    if (c == '\'' || c == '"') {
      inQuote = c;
      result += c;
      continue;
    }
    if (c == '/') {
      // Skip "//" - descendant-or-self axis should not be rewritten.
      if (i + 1 < path.size() && path[i + 1] == '/') {
        result += c;
        continue;
      }
      // Check if this is an absolute path: at start, or after operator/paren
      // (with optional whitespace).
      bool isAbsolute = (i == 0);
      if (!isAbsolute) {
        // Scan backwards past whitespace to find the preceding non-space char.
        size_t j = i - 1;
        while (j > 0 &&
               (path[j] == ' ' || path[j] == '\t' || path[j] == '\n' ||
                path[j] == '\r')) {
          --j;
        }
        char prev = path[j];
        // Operators/delimiters that can precede an absolute path:
        // ( - function call or grouping
        // | - union operator
        // , - function argument separator
        // = != < > - comparison operators
        // + - - arithmetic operators
        // [ - predicate start
        if (prev == '(' || prev == '|' || prev == ',' || prev == '=' ||
            prev == '<' || prev == '>' || prev == '+' || prev == '-' ||
            prev == '[' || prev == '!' ||
            (j == 0 &&
             (prev == ' ' || prev == '\t' || prev == '\n' || prev == '\r'))) {
          isAbsolute = true;
        }
        // Check for XPath word operators (or, and, div, mod).
        if (!isAbsolute && endsWithWordOperator(j)) {
          isAbsolute = true;
        }
      }
      if (isAbsolute) {
        result += "/_r/";
      } else {
        result += '/';
      }
    } else {
      result += c;
    }
  }
  return result;
}

/// Thread-local cache of compiled XPath expressions, keyed by the ORIGINAL
/// (pre-rewrite) path string. Spark passes a constant path literal, so the same
/// expression is evaluated for every row; compiling it once per thread and
/// reusing the immutable xmlXPathCompExpr avoids a per-row rewrite + compile.
///
/// The cache is thread_local so each worker thread owns its compiled
/// expressions - libxml2 compiled expressions are reused read-only within a
/// thread (the intended xmlXPathCompile/xmlXPathCompiledEval pattern) and are
/// never shared across threads - and frees them at thread exit.
///
/// The number of retained expressions is capped: the functions also accept a
/// non-constant (per-row) path argument, so a high-cardinality path column
/// would otherwise grow the map without bound for the thread's lifetime. Once
/// the cap is reached the cache stops taking ownership; the caller then owns
/// and frees any expression it compiled (see eval()), keeping memory bounded
/// while the common constant-path case stays cached.
class CompiledXPathCache {
 public:
  /// Returns the cached compiled expression for `path`, or nullptr if it is not
  /// cached. The returned pointer is owned by the cache; the caller must NOT
  /// free it. It stays valid for the thread's lifetime.
  xmlXPathCompExprPtr find(const std::string& path) const {
    auto it = cache_.find(path);
    return it == cache_.end() ? nullptr : it->second;
  }

  /// Attempts to store `compiled` under `path`. Returns true if the cache took
  /// ownership (the caller must NOT free it); returns false if the cache is
  /// full, in which case the caller retains ownership and must free `compiled`.
  bool store(const std::string& path, xmlXPathCompExprPtr compiled) {
    if (cache_.size() >= kMaxEntries) {
      return false;
    }
    cache_.emplace(path, compiled);
    return true;
  }

  ~CompiledXPathCache() {
    for (auto& entry : cache_) {
      xmlXPathFreeCompExpr(entry.second);
    }
  }

 private:
  // Generous bound: a query uses very few distinct constant paths, so this is
  // only reached by a pathological non-constant path column, for which we fall
  // back to compile-per-row rather than grow without limit.
  static constexpr size_t kMaxEntries = 1024;
  std::unordered_map<std::string, xmlXPathCompExprPtr> cache_;
};

CompiledXPathCache& compiledXPathCache() {
  static thread_local CompiledXPathCache cache;
  return cache;
}

/// RAII wrapper for libxml2 XML document and XPath context.
///
/// SPARK-FAITHFUL ERROR SEMANTICS (XP-BUG-1):
/// Spark's UDFXPathUtil throws RuntimeException for:
///   - Invalid XML input (malformed tags, encoding errors)
///   - Invalid XPath expressions (syntax errors)
/// To avoid a NULL-vs-throw divergence on the offloaded path, this
/// implementation also throws (VELOX_USER_FAIL) for these two cases instead of
/// returning NULL; see evalBoolean/evalString. Null and
/// empty inputs are short-circuited to NULL earlier (XPathFunctions.h, matching
/// UDFXPathUtil.eval's null/empty guards), and valid input that yields an empty
/// node-set still returns NULL/"" exactly as Spark does, so only genuinely
/// invalid input throws. The throw is data-dependent (depends on row contents),
/// matching Spark's per-row failure behavior.
///
/// Remaining accepted divergences (intentionally NOT converged):
///   - XP-BUG-2 (XXE/DOCTYPE): external-entity references that Spark would
///     resolve (and throw on) are stripped for security; converging would
///     reintroduce XXE/SSRF. See the DOCTYPE-strip comment in the constructor.
///   - XP-BUG-3 (namespaces): libxml2 is always namespace-aware whereas Spark
///     parses namespace-unaware; see the namespace comment in the constructor.
///
/// Java compatibility: Spark uses Java's XPath API where relative paths like
/// "a/b" are evaluated against the document and automatically resolve through
/// the document element.  libxml2 2.14+ does not support this - relative paths
/// from the document node return empty nodesets.  To match Java semantics, we
/// wrap the input XML in a synthetic root element ("<_r>...</_r>") and set the
/// XPath context to that wrapper so that "a/b" correctly selects child "a" of
/// the wrapper.  Absolute paths (starting with "/") are rewritten to go through
/// the wrapper (e.g. "/a/b" -> "/_r/a/b").
class XmlXPathEvaluator {
 public:
  XmlXPathEvaluator(const char* xml, size_t xmlLen) {
    ensureLibxml2Initialized();
    // Guard against overflow: wrapped size is xmlLen + 9 (<_r> + </_r>).
    // For inputs this large we leave doc_/ctx_ null so isValid() is false and
    // the evaluator reports neither valid nor raw-malformed; evalBoolean/
    // evalString therefore return NULL rather than throwing. This is a
    // documented divergence from Spark (which would attempt to parse), but it
    // is safe and unreachable in practice - Gluten string inputs are far below
    // 2 GiB.
    static constexpr size_t kWrapOverhead =
        9; // strlen("<_r>") + strlen("</_r>")
    if (xmlLen > static_cast<size_t>(INT_MAX) - kWrapOverhead) {
      return;
    }

    // Validate that the RAW (unwrapped) input is well-formed XML before we wrap
    // it in the synthetic <_r> root. Without this check the wrapper would make
    // non-well-formed input parse successfully - e.g. bare text ("not xml") or
    // trailing content after the root element ("<a/><b/>") - and the function
    // would return empty-result defaults instead of failing. Spark parses the
    // unwrapped document (Java DocumentBuilder) and throws RuntimeException on
    // such input; we mirror that by flagging the input malformed so the eval*
    // paths throw a user error (see XP-BUG-1 above). This is the authoritative
    // signal for Spark's throw, because Spark's success/throw is decided solely
    // by this raw parse - failures of the later wrapped parse (e.g. stripped
    // DOCTYPE/entity, XP-BUG-2) are NOT raw-malformed and keep returning NULL.
    {
      xmlDocPtr rawDoc = xmlReadMemory(
          xml,
          static_cast<int>(xmlLen),
          nullptr,
          nullptr,
          XML_PARSE_NOERROR | XML_PARSE_NOWARNING | XML_PARSE_NONET);
      if (rawDoc == nullptr) {
        xmlMalformed_ = true;
        return;
      }
      xmlFreeDoc(rawDoc);
    }

    // Wrap in synthetic root to match Java XPath evaluation semantics.
    // Strip XML declaration (<?xml ...?>) and DOCTYPE if present - these
    // are invalid inside a wrapper element. Only strip at the start of
    // the document (true prolog position).
    //
    // SPARK-DIVERGENCE: XP-BUG-2 (DOCTYPE strip / XXE handling)
    // Stripping the DOCTYPE changes external-entity (XXE) semantics relative
    // to Spark's JVM XPath: SYSTEM/external entity references that Spark would
    // attempt to resolve (and throw on) are instead removed, so evaluation
    // proceeds and returns NULL or a value rather than throwing. This is an
    // intentional, security-motivated divergence:
    //   - DTD loading and entity substitution are disabled at parse time
    //     (XML_PARSE_NOENT and XML_PARSE_DTDLOAD are NOT set) and network
    //     access is blocked (XML_PARSE_NONET), preventing XXE/SSRF.
    //   - Removing the DOCTYPE here lets otherwise-valid documents parse inside
    //     the synthetic wrapper without pulling in an external/internal subset.
    // A further consequence: if the input both DEFINES an entity in the
    // internal subset AND REFERENCES it in the body (e.g.
    // "<!DOCTYPE foo [<!ENTITY x \"hi\">]><foo>&x;</foo>"), the raw
    // well-formedness check passes (the internal subset is in scope there) but
    // the wrapped re-parse fails because the entity definition has been
    // stripped, so the function returns NULL. Spark expands the internal entity
    // and returns its value. This is the same accepted NULL-vs-throw/expand
    // divergence (see XP-BUG-1); internal-entity references in xpath inputs are
    // rare in practice.
    std::string_view xmlView(xml, xmlLen);
    std::string stripped;

    // Strip leading whitespace for prolog detection.
    size_t start = 0;
    while (start < xmlView.size() &&
           (xmlView[start] == ' ' || xmlView[start] == '\t' ||
            xmlView[start] == '\n' || xmlView[start] == '\r')) {
      ++start;
    }

    // Strip <?xml ...?> declaration if at prolog position. Require a
    // whitespace char after "<?xml" so we match only the XML declaration
    // (spec: "<?xml" S "version"...) and not a processing instruction whose
    // target merely begins with "xml", e.g. "<?xml-stylesheet ...?>".
    //
    // SPARK-DIVERGENCE: XP-BUG-2 (declared encoding is dropped)
    // Removing the declaration discards any encoding="..." attribute, so the
    // wrapped re-parse below always assumes UTF-8. For a document that declares
    // a non-UTF-8 encoding this can diverge from the raw parse (which honors
    // the declaration): bytes that are invalid UTF-8 make the re-parse fail ->
    // SQL NULL, and bytes that happen to be valid UTF-8 are decoded as
    // different characters. This is unreachable inside Gluten because Spark
    // StringType inputs are already UTF-8; it is documented here for the
    // standalone case.
    if (xmlView.substr(start, 5) == "<?xml" && start + 5 < xmlView.size() &&
        (xmlView[start + 5] == ' ' || xmlView[start + 5] == '\t' ||
         xmlView[start + 5] == '\n' || xmlView[start + 5] == '\r')) {
      auto end = xmlView.find("?>", start);
      if (end != std::string_view::npos) {
        stripped.append(xmlView.data() + end + 2, xmlView.size() - end - 2);
        xmlView = stripped;
      }
    }

    // Strip <!DOCTYPE ...> if present (must come before root element).
    // Handles internal subsets: <!DOCTYPE foo [<!ENTITY x "val">]>
    // by tracking bracket nesting depth before looking for closing '>'.
    // Skips brackets inside quoted strings (single or double quotes) to
    // handle cases like <!DOCTYPE foo [<!ENTITY x SYSTEM "file://[t]">]>, and
    // treats comments (<!-- ... -->) and PIs (<? ... ?>) inside the internal
    // subset as opaque so a '[', ']' or '>' within them does not corrupt the
    // bracket depth or terminate the scan early.
    {
      // Advance past prolog Misc content (whitespace, comments, and
      // processing instructions). XML's prolog grammar permits Misc*
      // (S | Comment | PI) before the doctypedecl, so a <!DOCTYPE preceded by
      // a comment or PI (e.g. "<!-- c --><!DOCTYPE foo>...") must still be
      // detected; otherwise the DOCTYPE would survive into the synthetic
      // wrapper (invalid there) and the re-parse would spuriously fail. The
      // skipped comments/PIs are left in place (they are valid inside <_r>).
      // An unterminated comment/PI stops the scan; such input is malformed and
      // was already rejected by the raw well-formedness parse above.
      size_t dtdStart = 0;
      while (dtdStart < xmlView.size()) {
        const char c = xmlView[dtdStart];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
          ++dtdStart;
        } else if (xmlView.substr(dtdStart, 4) == "<!--") {
          const size_t commentEnd = xmlView.find("-->", dtdStart + 4);
          if (commentEnd == std::string_view::npos) {
            break;
          }
          dtdStart = commentEnd + 3;
        } else if (xmlView.substr(dtdStart, 2) == "<?") {
          const size_t piEnd = xmlView.find("?>", dtdStart + 2);
          if (piEnd == std::string_view::npos) {
            break;
          }
          dtdStart = piEnd + 2;
        } else {
          break;
        }
      }
      if (xmlView.substr(dtdStart, 9) == "<!DOCTYPE") {
        size_t i = dtdStart + 9;
        int bracketDepth = 0;
        char inQuote = 0; // 0 = not in quotes, '"' or '\'' = in that quote
        size_t end = std::string_view::npos;
        while (i < xmlView.size()) {
          char c = xmlView[i];
          if (inQuote) {
            if (c == inQuote) {
              inQuote = 0;
            }
          } else if (xmlView.substr(i, 4) == "<!--") {
            // Comments inside the internal subset are opaque: a '[', ']' or
            // '>' within a comment must not affect bracket depth or be taken
            // as the DOCTYPE terminator. Fast-forward past "-->".
            const size_t commentEnd = xmlView.find("-->", i + 4);
            if (commentEnd == std::string_view::npos) {
              break; // unterminated (malformed); leave end unset -> no strip.
            }
            i = commentEnd + 3;
            continue;
          } else if (xmlView.substr(i, 2) == "<?") {
            // Processing instructions inside the internal subset are likewise
            // opaque. Fast-forward past "?>".
            const size_t piEnd = xmlView.find("?>", i + 2);
            if (piEnd == std::string_view::npos) {
              break; // unterminated (malformed); leave end unset -> no strip.
            }
            i = piEnd + 2;
            continue;
          } else if (c == '"' || c == '\'') {
            inQuote = c;
          } else if (c == '[') {
            ++bracketDepth;
          } else if (c == ']') {
            --bracketDepth;
          } else if (c == '>' && bracketDepth == 0) {
            end = i;
            break;
          }
          ++i;
        }
        if (end != std::string_view::npos) {
          std::string tmp;
          tmp.append(xmlView.data(), dtdStart);
          tmp.append(xmlView.data() + end + 1, xmlView.size() - end - 1);
          stripped = std::move(tmp);
          xmlView = stripped;
        }
      }
    }

    static constexpr std::string_view kWrapOpen = "<_r>";
    static constexpr std::string_view kWrapClose = "</_r>";
    wrapped_.reserve(kWrapOpen.size() + xmlView.size() + kWrapClose.size());
    wrapped_.append(kWrapOpen);
    wrapped_.append(xmlView);
    wrapped_.append(kWrapClose);

    // Security: XML_PARSE_NONET prevents network access (XXE mitigation).
    // Entity substitution is off by default (XML_PARSE_NOENT not set).
    // DTD loading is off by default (XML_PARSE_DTDLOAD not set).
    doc_ = xmlReadMemory(
        wrapped_.data(),
        static_cast<int>(wrapped_.size()),
        nullptr,
        nullptr,
        XML_PARSE_NOERROR | XML_PARSE_NOWARNING | XML_PARSE_NONET);
    if (doc_) {
      ctx_ = xmlXPathNewContext(doc_);
      if (ctx_) {
        ctx_->node = xmlDocGetRootElement(doc_);
        // SPARK-DIVERGENCE: XP-BUG-3 (XML namespace handling)
        // libxml2 always parses namespace-aware, and we register no namespace
        // prefixes on the XPath context. Spark/Hive (UDFXPathUtil) parse with a
        // namespace-UNAWARE DocumentBuilder, so element names are matched
        // literally. As a result, for namespaced input this implementation
        // diverges from Spark:
        //   - A prefixed step (e.g. "x:a/x:b") references an unregistered
        //     prefix -> XPath evaluation fails -> the function returns NULL.
        //   - Default-namespace input (<a xmlns="..">) is not matched by an
        //     unprefixed step ("a/b") because the element is in a non-null
        //     namespace -> empty node-set (e.g. xpath_string -> "").
        // This is an accepted, documented divergence. Registering namespaces
        // is intentionally not attempted because Spark's namespace-unaware
        // matching cannot be reproduced via libxml2's always-namespace-aware
        // XPath engine.
      }
    }
  }

  ~XmlXPathEvaluator() {
    if (ctx_) {
      xmlXPathFreeContext(ctx_);
    }
    if (doc_) {
      xmlFreeDoc(doc_);
    }
  }

  XmlXPathEvaluator(const XmlXPathEvaluator&) = delete;
  XmlXPathEvaluator& operator=(const XmlXPathEvaluator&) = delete;

  bool isValid() const {
    return doc_ != nullptr && ctx_ != nullptr;
  }

  /// True when the RAW (unwrapped) input failed to parse as well-formed XML.
  /// This is the Spark-faithful signal for a thrown error: Spark's
  /// DocumentBuilder.parse fails on exactly this input. Distinct from a generic
  /// !isValid(), which also covers wrapped-only parse failures (XP-BUG-2) that
  /// must keep returning NULL rather than throwing.
  bool xmlMalformed() const {
    return xmlMalformed_;
  }

  /// Evaluate XPath expression. Caller must free result with
  /// xmlXPathFreeObject.
  ///
  /// Distinguishes an XPath *syntax* error from an *evaluation-time* failure so
  /// callers can honor Spark's error semantics precisely:
  ///   - A syntax error (e.g. "///[invalid") fails xmlXPathCompile; Spark's
  ///     xpath.compile throws RuntimeException, so we set pathInvalid=true and
  ///     callers throw.
  ///   - A syntactically valid path that fails at evaluation (e.g. an
  ///     unregistered namespace prefix, XP-BUG-3) returns nullptr with
  ///     pathInvalid=false; callers keep returning NULL (documented divergence,
  ///     not a throw).
  xmlXPathObjectPtr eval(const std::string& path, bool& pathInvalid) {
    pathInvalid = false;
    if (!isValid()) {
      return nullptr;
    }
    // Reuse a thread-local compiled expression for this (constant) path; the
    // cache rewrites absolute paths through the wrapper root and compiles once
    // per thread. The compiled expression is context-free, so it is evaluated
    // here against this row's per-document context. If the cache is full (only
    // reached by a high-cardinality non-constant path), we own the freshly
    // compiled expression and free it after evaluating this row.
    auto& cache = compiledXPathCache();
    xmlXPathCompExprPtr compiled = cache.find(path);
    bool owned = false;
    if (compiled == nullptr) {
      const std::string rewritten = rewriteAbsolutePaths(path);
      compiled =
          xmlXPathCompile(reinterpret_cast<const xmlChar*>(rewritten.c_str()));
      if (compiled == nullptr) {
        pathInvalid = true;
        return nullptr;
      }
      // Hand ownership to the cache when there is room; otherwise keep it and
      // free it below after evaluation.
      owned = !cache.store(path, compiled);
    }
    xmlXPathObjectPtr result = xmlXPathCompiledEval(compiled, ctx_);
    if (owned) {
      xmlXPathFreeCompExpr(compiled);
    }
    return result;
  }

 private:
  std::string wrapped_;
  xmlDocPtr doc_ = nullptr;
  xmlXPathContextPtr ctx_ = nullptr;
  bool xmlMalformed_ = false;
};

// malformed XML and on invalid XPath expressions. These helpers raise the
// equivalent user-facing error so the offloaded Velox path matches Spark
// instead of silently returning NULL. They are only reached for genuinely
// invalid input (null/empty are handled in XPathFunctions.h; valid-but-empty
// results return NULL upstream).
[[noreturn]] void throwInvalidXml() {
  VELOX_USER_FAIL("Invalid XML document");
}

[[noreturn]] void throwInvalidXPath(const std::string& path) {
  VELOX_USER_FAIL("Invalid XPath '{}'", path);
}

} // namespace

std::optional<bool>
evalBoolean(const char* xml, size_t xmlLen, const char* path, size_t pathLen) {
  XmlXPathEvaluator evaluator(xml, xmlLen);
  if (!evaluator.isValid()) {
    if (evaluator.xmlMalformed()) {
      throwInvalidXml();
    }
    return std::nullopt;
  }

  std::string pathStr(path, pathLen);
  bool pathInvalid = false;
  xmlXPathObjectPtr result = evaluator.eval(pathStr, pathInvalid);
  if (pathInvalid) {
    throwInvalidXPath(pathStr);
  }
  if (!result) {
    return std::nullopt;
  }

  bool val = xmlXPathCastToBoolean(result) != 0;
  xmlXPathFreeObject(result);
  return val;
}

std::optional<std::string>
evalString(const char* xml, size_t xmlLen, const char* path, size_t pathLen) {
  XmlXPathEvaluator evaluator(xml, xmlLen);
  if (!evaluator.isValid()) {
    if (evaluator.xmlMalformed()) {
      throwInvalidXml();
    }
    return std::nullopt;
  }

  std::string pathStr(path, pathLen);
  bool pathInvalid = false;
  xmlXPathObjectPtr result = evaluator.eval(pathStr, pathInvalid);
  if (pathInvalid) {
    throwInvalidXPath(pathStr);
  }
  if (!result) {
    return std::nullopt;
  }

  xmlChar* str = xmlXPathCastToString(result);
  xmlXPathFreeObject(result);

  if (!str) {
    return std::nullopt;
  }
  // An empty node-set casts to "" (not null), so a no-match returns the empty
  // string rather than NULL. This matches Spark's UDFXPathUtil.evalString,
  // which evaluates with XPathConstants.STRING and likewise yields "" for a
  // non-matching path (NULL is reserved for null/empty inputs, handled upstream
  // in XPathFunctions.h).
  std::string ret(reinterpret_cast<const char*>(str));
  xmlFree(str);
  return ret;
}

} // namespace xpath
} // namespace facebook::velox::functions::sparksql
