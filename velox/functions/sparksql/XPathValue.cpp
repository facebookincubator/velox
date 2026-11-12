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
#include "velox/functions/sparksql/XPathValue.h"

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <fast_float/fast_float.h>
#include <fmt/format.h>
#include <libxml/xpathInternals.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/sparksql/XPathContext.h"
#include "velox/functions/sparksql/XPathExpression.h"

namespace facebook::velox::functions::sparksql::xpath {
namespace {

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

// Keep id() and lang() numeric arguments consistent with string conversions.
template <xmlXPathFunction function>
void callWithXPathNumberStrings(
    xmlXPathParserContextPtr ctxt,
    int nargs) noexcept {
  try {
    if (nargs > 0 && nargs <= ctxt->valueNr) {
      const int first = ctxt->valueNr - nargs;
      for (int i = first; i < ctxt->valueNr; ++i) {
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

struct InvalidXPathString {};

// Keep unpaired UTF-16 surrogates as WTF-8 inside XPath. Replacing them here
// would break concatenation and searches across substring boundaries.
std::u16string xpathUtf16(std::string_view value) {
  std::u16string result;
  while (!value.empty()) {
    int32_t codePoint;
    auto bytes = tryGetUtf8CharLength(value.data(), value.size(), codePoint);
    if (bytes <= 0 && value.size() >= 3 &&
        static_cast<unsigned char>(value[0]) == 0xed &&
        (static_cast<unsigned char>(value[1]) & 0xe0) == 0xa0 &&
        (static_cast<unsigned char>(value[2]) & 0xc0) == 0x80) {
      codePoint = ((value[0] & 0x0f) << 12) | ((value[1] & 0x3f) << 6) |
          (value[2] & 0x3f);
      bytes = 3;
    }
    if (bytes <= 0) {
      throw InvalidXPathString{};
    }
    if (codePoint >= 0x10000) {
      codePoint -= 0x10000;
      result.push_back(static_cast<char16_t>(0xd800 + (codePoint >> 10)));
      result.push_back(static_cast<char16_t>(0xdc00 + (codePoint & 0x3ff)));
    } else {
      result.push_back(static_cast<char16_t>(codePoint));
    }
    value.remove_prefix(bytes);
  }
  return result;
}

std::string xpathUtf8(std::u16string_view value) {
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    uint32_t codePoint = value[i];
    if (codePoint >= 0xd800 && codePoint <= 0xdbff && i + 1 < value.size() &&
        value[i + 1] >= 0xdc00 && value[i + 1] <= 0xdfff) {
      codePoint =
          0x10000 + ((codePoint - 0xd800) << 10) + (value[++i] - 0xdc00);
    }
    if (codePoint < 0x80) {
      result.push_back(static_cast<char>(codePoint));
    } else if (codePoint < 0x800) {
      result.push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else if (codePoint < 0x10000) {
      result.push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else {
      result.push_back(static_cast<char>(0xf0 | (codePoint >> 18)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    }
  }
  return result;
}

std::u16string xpathString(xmlXPathObjectPtr value) {
  if (value->type == XPATH_NUMBER) {
    return xpathUtf16(numberToXPathString(value->floatval));
  }
  XmlStringPtr text{xmlXPathCastToString(value), xmlFree};
  VELOX_CHECK_NOT_NULL(text.get(), "Failed to convert XPath string");
  return xpathUtf16(reinterpret_cast<const char*>(text.get()));
}

double xpathStringNumber(std::string_view value) {
  while (!value.empty() && static_cast<unsigned char>(value.front()) <= 0x20) {
    value.remove_prefix(1);
  }
  while (!value.empty() && static_cast<unsigned char>(value.back()) <= 0x20) {
    value.remove_suffix(1);
  }
  size_t position = value.starts_with("-") ? 1 : 0;
  size_t digits = 0;
  bool point = false;
  for (; position < value.size(); ++position) {
    const auto byte = value[position];
    if (byte >= '0' && byte <= '9') {
      ++digits;
    } else if (byte == '.' && !point) {
      point = true;
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (digits == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double result;
  const auto* end = value.data() + value.size();
  const auto parsed = fast_float::from_chars(value.data(), end, result);
  VELOX_CHECK(
      parsed.ec == std::errc{} || parsed.ec == std::errc::result_out_of_range,
      "Failed to parse XPath decimal");
  VELOX_CHECK(parsed.ptr == end, "Failed to parse complete XPath decimal");
  // fast_float sets signed infinity or zero on overflow or underflow.
  return result;
}

double xpathNodeNumber(xmlNodePtr node) {
  XmlStringPtr text{xmlXPathCastNodeToString(node), xmlFree};
  VELOX_CHECK_NOT_NULL(text.get(), "Failed to convert XPath node");
  return xpathStringNumber(reinterpret_cast<const char*>(text.get()));
}

double xpathNumber(xmlXPathObjectPtr value) {
  if (value->type == XPATH_NUMBER) {
    return value->floatval;
  }
  if (value->type == XPATH_BOOLEAN) {
    return value->boolval ? 1 : 0;
  }
  XmlStringPtr text{xmlXPathCastToString(value), xmlFree};
  VELOX_CHECK_NOT_NULL(text.get(), "Failed to convert XPath number");
  return xpathStringNumber(reinterpret_cast<const char*>(text.get()));
}

bool pushXPathObject(xmlXPathParserContextPtr parser, XPathObjectPtr value) {
  VELOX_CHECK_NOT_NULL(value.get(), "Failed to create XPath value");
  if (valuePush(parser, value.get()) < 0) {
    xmlXPathSetError(parser, XPATH_MEMORY_ERROR);
    return false;
  }
  value.release();
  return true;
}

bool pushXPathNumber(xmlXPathParserContextPtr parser, double value) {
  return pushXPathObject(
      parser, XPathObjectPtr{xmlXPathNewFloat(value), xmlXPathFreeObject});
}

void pushXPathString(
    xmlXPathParserContextPtr parser,
    std::u16string_view value) {
  const auto text = xpathUtf8(value);
  VELOX_CHECK_LE(
      text.size(), static_cast<size_t>(INT_MAX), "XPath string is too large");
  pushXPathObject(
      parser,
      XPathObjectPtr{
          xmlXPathNewString(reinterpret_cast<const xmlChar*>(text.c_str())),
          xmlXPathFreeObject});
}

template <xmlXPathFunction function>
void protectXPathFunction(xmlXPathParserContextPtr parser, int nargs) noexcept {
  try {
    function(parser, nargs);
  } catch (const InvalidXPathString&) {
    xmlXPathSetError(parser, XPATH_INVALID_CHAR_ERROR);
  } catch (...) {
    static_cast<XPathEvaluationState*>(parser->context->userData)->error =
        std::current_exception();
    xmlXPathSetError(parser, XPATH_EXPR_ERROR);
  }
}

void strictXPathNumber(xmlXPathParserContextPtr parser, int nargs) {
  if (nargs == 0) {
    pushXPathNumber(parser, xpathNodeNumber(parser->context->node));
  } else if (nargs == 1) {
    XPathObjectPtr value{valuePop(parser), xmlXPathFreeObject};
    VELOX_CHECK_NOT_NULL(value.get());
    pushXPathNumber(parser, xpathNumber(value.get()));
  } else {
    xmlXPathSetArityError(parser);
  }
}

template <xmlXPathFunction function>
void strictXPathNumericArgument(xmlXPathParserContextPtr parser, int nargs) {
  if (nargs != 1) {
    xmlXPathSetArityError(parser);
    return;
  }
  strictXPathNumber(parser, nargs);
  if (parser->error == XPATH_EXPRESSION_OK) {
    function(parser, nargs);
  }
}

bool isXPathNodeSet(xmlXPathObjectPtr value) {
  return value->type == XPATH_NODESET || value->type == XPATH_XSLT_TREE;
}

void strictXPathSum(xmlXPathParserContextPtr parser, int nargs) {
  if (nargs != 1) {
    xmlXPathSetArityError(parser);
    return;
  }
  XPathObjectPtr value{valuePop(parser), xmlXPathFreeObject};
  VELOX_CHECK_NOT_NULL(value.get());
  if (!isXPathNodeSet(value.get())) {
    xmlXPathSetTypeError(parser);
    return;
  }
  double sum = 0;
  if (value->nodesetval != nullptr) {
    for (int i = 0; i < value->nodesetval->nodeNr; ++i) {
      sum += xpathNodeNumber(value->nodesetval->nodeTab[i]);
    }
  }
  pushXPathNumber(parser, sum);
}

template <typename Predicate>
bool anyXPathNumber(xmlXPathObjectPtr value, Predicate&& predicate) {
  if (!isXPathNodeSet(value)) {
    return predicate(xpathNumber(value));
  }
  if (value->nodesetval != nullptr) {
    for (int i = 0; i < value->nodesetval->nodeNr; ++i) {
      if (predicate(xpathNodeNumber(value->nodesetval->nodeTab[i]))) {
        return true;
      }
    }
  }
  return false;
}

template <typename Predicate>
bool anyXPathString(xmlXPathObjectPtr value, Predicate&& predicate) {
  if (!isXPathNodeSet(value)) {
    return predicate(xpathString(value));
  }
  if (value->nodesetval != nullptr) {
    for (int i = 0; i < value->nodesetval->nodeNr; ++i) {
      XmlStringPtr text{
          xmlXPathCastNodeToString(value->nodesetval->nodeTab[i]), xmlFree};
      VELOX_CHECK_NOT_NULL(text.get(), "Failed to convert XPath node");
      if (predicate(xpathUtf16(reinterpret_cast<const char*>(text.get())))) {
        return true;
      }
    }
  }
  return false;
}

bool compareXPathNumbers(int operation, double left, double right) {
  switch (operation) {
    case kEqual:
      return left == right;
    case kNotEqual:
      return left != right;
    case kLess:
      return left < right;
    case kLessEqual:
      return left <= right;
    case kGreater:
      return left > right;
    case kGreaterEqual:
      return left >= right;
    default:
      VELOX_UNREACHABLE("Unknown XPath comparison");
  }
}

void strictXPathOperation(xmlXPathParserContextPtr parser, int nargs) {
  if (nargs != 3) {
    xmlXPathSetArityError(parser);
    return;
  }
  const auto code = xmlXPathPopNumber(parser);
  VELOX_CHECK_GE(code, static_cast<int>(kNotEqual));
  VELOX_CHECK_LE(code, static_cast<int>(kNegate));
  const auto operation = static_cast<int>(code);
  XPathObjectPtr right{valuePop(parser), xmlXPathFreeObject};
  XPathObjectPtr left{valuePop(parser), xmlXPathFreeObject};
  VELOX_CHECK_NOT_NULL(left.get());
  VELOX_CHECK_NOT_NULL(right.get());
  if (operation <= kGreater) {
    const bool equality = operation == kEqual || operation == kNotEqual;
    const bool boolean =
        left->type == XPATH_BOOLEAN || right->type == XPATH_BOOLEAN;
    bool result;
    if (boolean &&
        (equality || isXPathNodeSet(left.get()) ||
         isXPathNodeSet(right.get()))) {
      result = compareXPathNumbers(
          operation,
          xmlXPathCastToBoolean(left.get()) != 0,
          xmlXPathCastToBoolean(right.get()) != 0);
    } else if (
        equality && left->type != XPATH_NUMBER && right->type != XPATH_NUMBER) {
      result = anyXPathString(left.get(), [&](std::u16string_view lhs) {
        return anyXPathString(right.get(), [&](std::u16string_view rhs) {
          return operation == kEqual ? lhs == rhs : lhs != rhs;
        });
      });
    } else {
      result = anyXPathNumber(left.get(), [&](double lhs) {
        return anyXPathNumber(right.get(), [&](double rhs) {
          return compareXPathNumbers(operation, lhs, rhs);
        });
      });
    }
    pushXPathObject(
        parser, XPathObjectPtr{xmlXPathNewBoolean(result), xmlXPathFreeObject});
    return;
  }
  if (!pushXPathNumber(parser, xpathNumber(left.get())) ||
      !pushXPathNumber(parser, xpathNumber(right.get()))) {
    return;
  }
  switch (operation) {
    case kAdd:
      xmlXPathAddValues(parser);
      break;
    case kSubtract:
      xmlXPathSubValues(parser);
      break;
    case kMultiply:
      xmlXPathMultValues(parser);
      break;
    case kDivide:
      xmlXPathDivValues(parser);
      break;
    case kModulo:
      xmlXPathModValues(parser);
      break;
    case kNegate: {
      XPathObjectPtr argument{valuePop(parser), xmlXPathFreeObject};
      XPathObjectPtr unused{valuePop(parser), xmlXPathFreeObject};
      pushXPathNumber(parser, -argument->floatval);
      break;
    }
    default:
      VELOX_UNREACHABLE("Unknown XPath arithmetic operation");
  }
}

// Java's rounded long is converted back to double before substring indexing.
double xpathRoundedLong(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  constexpr double kLongLimit = 0x1p63;
  if (value >= kLongLimit || value <= -kLongLimit) {
    return std::copysign(kLongLimit, value);
  }
  const auto integral = std::floor(value);
  return integral + (value - integral >= 0.5 ? 1 : 0);
}

int32_t xpathInt(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  if (value >= INT_MAX) {
    return INT_MAX;
  }
  if (value <= INT_MIN) {
    return INT_MIN;
  }
  return static_cast<int32_t>(value);
}

enum class XPathStringFunction {
  kString,
  kLength,
  kConcat,
  kContains,
  kStartsWith,
  kBefore,
  kAfter,
  kSubstring,
  kNormalize,
  kTranslate,
};

template <XPathStringFunction function>
void utf16XPathFunction(xmlXPathParserContextPtr parser, int nargs) {
  const bool optionalArgument = function == XPathStringFunction::kString ||
      function == XPathStringFunction::kLength ||
      function == XPathStringFunction::kNormalize;
  const bool validArity = optionalArgument          ? nargs >= 0 && nargs <= 1
      : function == XPathStringFunction::kConcat    ? nargs >= 2
      : function == XPathStringFunction::kSubstring ? nargs == 2 || nargs == 3
      : function == XPathStringFunction::kTranslate ? nargs == 3
                                                    : nargs == 2;
  if (!validArity) {
    xmlXPathSetArityError(parser);
    return;
  }
  std::vector<XPathObjectPtr> arguments;
  arguments.reserve(nargs);
  for (int i = 0; i < nargs; ++i) {
    arguments.emplace_back(valuePop(parser), xmlXPathFreeObject);
    VELOX_CHECK_NOT_NULL(arguments.back().get());
  }
  std::reverse(arguments.begin(), arguments.end());
  std::u16string value;
  if (nargs == 0) {
    XmlStringPtr text{xmlXPathCastNodeToString(parser->context->node), xmlFree};
    VELOX_CHECK_NOT_NULL(text.get(), "Failed to convert XPath node");
    value = xpathUtf16(reinterpret_cast<const char*>(text.get()));
  } else {
    value = xpathString(arguments[0].get());
  }
  if constexpr (function == XPathStringFunction::kLength) {
    pushXPathNumber(parser, static_cast<double>(value.size()));
    return;
  } else if constexpr (function == XPathStringFunction::kConcat) {
    for (int i = 1; i < nargs; ++i) {
      value += xpathString(arguments[i].get());
    }
  } else if constexpr (function == XPathStringFunction::kSubstring) {
    auto start = xpathNumber(arguments[1].get());
    start = std::isnan(start) ? -1'000'000 : xpathRoundedLong(start);
    int64_t begin = start > 0 ? xpathInt(start) - 1 : 0;
    auto end = static_cast<int64_t>(value.size());
    if (nargs == 3) {
      const auto limit =
          xpathInt(xpathRoundedLong(xpathNumber(arguments[2].get())) + start);
      // Java int subtraction wraps after a saturating floating-point cast.
      end = limit == INT_MIN ? INT_MAX : limit - 1;
    }
    begin = std::min<int64_t>(begin, value.size());
    end = std::clamp<int64_t>(end, 0, value.size());
    if (begin > end) {
      xmlXPathSetError(parser, XPATH_INVALID_OPERAND);
      return;
    }
    value = value.substr(begin, end - begin);
  } else if constexpr (function == XPathStringFunction::kNormalize) {
    std::u16string normalized;
    bool space = false;
    for (const auto unit : value) {
      if (unit == u' ' || unit == u'\t' || unit == u'\r' || unit == u'\n') {
        space = !normalized.empty();
      } else {
        if (space) {
          normalized.push_back(u' ');
        }
        normalized.push_back(unit);
        space = false;
      }
    }
    value = std::move(normalized);
  } else if constexpr (function == XPathStringFunction::kTranslate) {
    const auto from = xpathString(arguments[1].get());
    const auto to = xpathString(arguments[2].get());
    std::u16string translated;
    for (const auto unit : value) {
      const auto position = from.find(unit);
      if (position == std::u16string::npos) {
        translated.push_back(unit);
      } else if (position < to.size()) {
        translated.push_back(to[position]);
      }
    }
    value = std::move(translated);
  } else if constexpr (function != XPathStringFunction::kString) {
    const auto needle = xpathString(arguments[1].get());
    const auto position = value.find(needle);
    if constexpr (
        function == XPathStringFunction::kContains ||
        function == XPathStringFunction::kStartsWith) {
      const bool result = function == XPathStringFunction::kContains
          ? position != std::u16string::npos
          : position == 0;
      pushXPathObject(
          parser,
          XPathObjectPtr{xmlXPathNewBoolean(result), xmlXPathFreeObject});
      return;
    } else if (position == std::u16string::npos) {
      value.clear();
    } else if constexpr (function == XPathStringFunction::kBefore) {
      value.resize(position);
    } else {
      value = value.substr(position + needle.size());
    }
  }
  pushXPathString(parser, value);
}

} // namespace

xmlXPathFunction lookupXPathValueFunction(
    void*,
    const xmlChar* name,
    const xmlChar* namespaceUri) {
  if (namespaceUri != nullptr) {
    return nullptr;
  }
  static const std::pair<const char*, xmlXPathFunction> functions[] = {
      {"_velox_numeric", protectXPathFunction<strictXPathOperation>},
      {"number", protectXPathFunction<strictXPathNumber>},
      {"sum", protectXPathFunction<strictXPathSum>},
      {"floor",
       protectXPathFunction<strictXPathNumericArgument<xmlXPathFloorFunction>>},
      {"ceiling",
       protectXPathFunction<
           strictXPathNumericArgument<xmlXPathCeilingFunction>>},
      {"round",
       protectXPathFunction<strictXPathNumericArgument<xmlXPathRoundFunction>>},
      {"string",
       protectXPathFunction<utf16XPathFunction<XPathStringFunction::kString>>},
      {"concat",
       protectXPathFunction<utf16XPathFunction<XPathStringFunction::kConcat>>},
      {"contains",
       protectXPathFunction<
           utf16XPathFunction<XPathStringFunction::kContains>>},
      {"starts-with",
       protectXPathFunction<
           utf16XPathFunction<XPathStringFunction::kStartsWith>>},
      {"string-length",
       protectXPathFunction<utf16XPathFunction<XPathStringFunction::kLength>>},
      {"substring",
       protectXPathFunction<
           utf16XPathFunction<XPathStringFunction::kSubstring>>},
      {"substring-before",
       protectXPathFunction<utf16XPathFunction<XPathStringFunction::kBefore>>},
      {"substring-after",
       protectXPathFunction<utf16XPathFunction<XPathStringFunction::kAfter>>},
      {"normalize-space",
       protectXPathFunction<
           utf16XPathFunction<XPathStringFunction::kNormalize>>},
      {"translate",
       protectXPathFunction<
           utf16XPathFunction<XPathStringFunction::kTranslate>>},
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

std::string xpathResultString(xmlXPathObjectPtr result) {
  if (result->type == XPATH_NUMBER) {
    return numberToXPathString(result->floatval);
  }
  XmlStringPtr value{xmlXPathCastToString(result), xmlFree};
  VELOX_CHECK_NOT_NULL(value.get(), "Failed to convert XPath result to string");
  const std::string_view text(reinterpret_cast<const char*>(value.get()));
  std::string replaced;
  size_t copied = 0;
  for (size_t i = 0; i + 2 < text.size(); ++i) {
    if (static_cast<unsigned char>(text[i]) == 0xed &&
        (static_cast<unsigned char>(text[i + 1]) & 0xe0) == 0xa0 &&
        (static_cast<unsigned char>(text[i + 2]) & 0xc0) == 0x80) {
      replaced.append(text.substr(copied, i - copied));
      replaced.push_back('?');
      i += 2;
      copied = i + 1;
    }
  }
  if (copied != 0) {
    replaced.append(text.substr(copied));
    return replaced;
  }
  return std::string(text);
}

} // namespace facebook::velox::functions::sparksql::xpath
