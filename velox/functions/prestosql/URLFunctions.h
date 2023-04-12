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

#include <boost/regex.hpp>
#include <cctype>
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

namespace {
FOLLY_ALWAYS_INLINE StringView submatch(const boost::cmatch& match, int idx) {
  const auto& sub = match[idx];
  return StringView(sub.first, sub.length());
}

template <typename TInString>
bool parse(const TInString& rawUrl, boost::cmatch& match) {
  static const boost::regex kUriRegex(
      "([a-zA-Z][a-zA-Z0-9+.-]*):" // scheme:
      "([^?#]*)" // authority and path
      "(?:\\?([^#]*))?" // ?query
      "(?:#(.*))?"); // #fragment

  return boost::regex_match(
      rawUrl.data(), rawUrl.data() + rawUrl.size(), match, kUriRegex);
}

FOLLY_ALWAYS_INLINE unsigned char toHex(unsigned char c) {
  return c < 10 ? (c + '0') : (c + 'A' - 10);
}

FOLLY_ALWAYS_INLINE void charEscape(unsigned char c, char* output) {
  output[0] = '%';
  output[1] = toHex(c / 16);
  output[2] = toHex(c % 16);
}

/// Escapes ``input`` by encoding it so that it can be safely included in
/// URL query parameter names and values:
///
///  * Alphanumeric characters are not encoded.
///  * The characters ``.``, ``-``, ``*`` and ``_`` are not encoded.
///  * The ASCII space character is encoded as ``+``.
///  * All other characters are converted to UTF-8 and the bytes are encoded
///    as the string ``%XX`` where ``XX`` is the uppercase hexadecimal
///    value of the UTF-8 byte.
template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE void urlEscape(TOutString& output, const TInString& input) {
  auto inputSize = input.size();
  output.reserve(inputSize * 3);

  auto inputBuffer = input.data();
  auto outputBuffer = output.data();

  size_t outIndex = 0;
  for (auto i = 0; i < inputSize; ++i) {
    unsigned char p = inputBuffer[i];

    if ((p >= 'a' && p <= 'z') || (p >= 'A' && p <= 'Z') ||
        (p >= '0' && p <= '9') || p == '-' || p == '_' || p == '.' ||
        p == '*') {
      outputBuffer[outIndex++] = p;
    } else if (p == ' ') {
      outputBuffer[outIndex++] = '+';
    } else {
      charEscape(p, outputBuffer + outIndex);
      outIndex += 3;
    }
  }
  output.resize(outIndex);
}

template <typename TOutString, typename TInString>
FOLLY_ALWAYS_INLINE void urlUnescape(
    TOutString& output,
    const TInString& input) {
  auto inputSize = input.size();
  output.reserve(inputSize);

  auto outputBuffer = output.data();
  const char* p = input.data();
  const char* end = p + inputSize;
  char buf[3];
  buf[2] = '\0';
  char* endptr;
  for (; p < end; ++p) {
    if (*p == '+') {
      *outputBuffer++ = ' ';
    } else if (*p == '%') {
      if (p + 2 < end) {
        buf[0] = p[1];
        buf[1] = p[2];
        int val = strtol(buf, &endptr, 16);
        if (endptr == buf + 2) {
          *outputBuffer++ = (char)val;
          p += 2;
        } else {
          VELOX_USER_FAIL(
              "Illegal hex characters in escape (%) pattern: {}", buf);
        }
      } else {
        VELOX_USER_FAIL("Incomplete trailing escape (%) pattern");
      }
    } else {
      *outputBuffer++ = *p;
    }
  }
  output.resize(outputBuffer - output.data());
}

} // namespace

bool matchAuthorityAndPath(
    const boost::cmatch& urlMatch,
    boost::cmatch& authAndPathMatch,
    boost::cmatch& authorityMatch,
    bool& hasAuthority);

template <typename T>
struct UrlExtractProtocolFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
    } else {
      result.setNoCopy(submatch(match, 1));
    }
    return true;
  }
};

template <typename T>
struct UrlExtractFragmentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
    } else {
      result.setNoCopy(submatch(match, 4));
    }
    return true;
  }
};

template <typename T>
struct UrlExtractHostFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
      return true;
    }
    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority;

    if (matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority) &&
        hasAuthority) {
      result.setNoCopy(submatch(authorityMatch, 3));
    } else {
      result.setEmpty();
    }
    return true;
  }
};

template <typename T>
struct UrlExtractPortFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      return false;
    }

    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority;
    if (matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority) &&
        hasAuthority) {
      auto port = submatch(authorityMatch, 4);
      if (!port.empty()) {
        try {
          result = to<int64_t>(port);
          return true;
        } catch (folly::ConversionError const&) {
        }
      }
    }
    return false;
  }
};

template <typename T>
struct UrlExtractPathFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
      return true;
    }

    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority;

    if (matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority)) {
      if (hasAuthority) {
        result.setNoCopy(submatch(authAndPathMatch, 2));
      } else {
        result.setNoCopy(submatch(match, 2));
      }
    }

    return true;
  }
};

template <typename T>
struct UrlExtractQueryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
      return true;
    }

    auto query = submatch(match, 3);
    result.setNoCopy(query);
    return true;
  }
};

template <typename T>
struct UrlExtractParameterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url,
      const arg_type<Varchar>& param) {
    boost::cmatch match;
    if (!parse(url, match)) {
      result.setEmpty();
      return false;
    }

    auto query = submatch(match, 3);
    if (!query.empty()) {
      // Parse query string.
      static const boost::regex kQueryParamRegex(
          "(^|&)" // start of query or start of parameter "&"
          "([^=&]*)=?" // parameter name and "=" if value is expected
          "([^=&]*)" // parameter value
          "(?=(&|$))" // forward reference, next should be end of query or
                      // start of next parameter
      );

      const boost::cregex_iterator begin(
          query.data(), query.data() + query.size(), kQueryParamRegex);
      boost::cregex_iterator end;

      for (auto it = begin; it != end; ++it) {
        if (it->length(2) != 0) { // key shouldnt be empty.
          auto key = submatch((*it), 2);
          if (param.compare(key) == 0) {
            auto value = submatch((*it), 3);
            result.setNoCopy(value);
            return true;
          }
        }
      }
    }

    return false;
  }
};

template <typename T>
struct UrlEncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varbinary>& input) {
    urlEscape(result, input);
  }
};

template <typename T>
struct UrlDecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varbinary>& input) {
    urlUnescape(result, input);
  }
};

} // namespace facebook::velox::functions
