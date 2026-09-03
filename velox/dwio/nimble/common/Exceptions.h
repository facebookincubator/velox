/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "folly/FixedString.h"
#include "folly/Preprocessor.h"
#include "velox/dwio/nimble/common/ExceptionHelper.h"
#include "velox/dwio/nimble/common/NimbleException.h"

// Standard errors used throughout the codebase.

namespace facebook::nimble {

namespace detail {

// Struct containing the arguments needed to throw a Nimble exception
struct NimbleCheckFailArgs {
  const char* file;
  size_t line;
  const char* function;
  const char* expression;
  std::string_view errorCode;
  bool isRetryable;
};

// Out-of-line nimbleCheckFail implementations to reduce binary bloat
template <typename Exception, typename Msg>
[[noreturn]] void nimbleCheckFail(const NimbleCheckFailArgs& args, Msg msg) {
  throw Exception(
      args.file,
      args.line,
      args.function,
      args.expression,
      msg,
      args.errorCode,
      args.isRetryable);
}

// NimbleCheckFailStringType helps us pass by reference to
// NimbleCheckFail exactly when the string type is std::string.
template <typename T>
struct NimbleCheckFailStringType;

template <>
struct NimbleCheckFailStringType<CompileTimeEmptyString> {
  using type = CompileTimeEmptyString;
};

template <>
struct NimbleCheckFailStringType<const char*> {
  using type = const char*;
};

template <>
struct NimbleCheckFailStringType<std::string> {
  using type = const std::string&;
};

// Declare explicit instantiations of nimbleCheckFail for the given
// exceptionType. Just the signatures go in this macro, not the definitions.
#define NIMBLE_DECLARE_CHECK_FAIL_TEMPLATES(exceptionType)                     \
  namespace detail {                                                           \
  extern template void nimbleCheckFail<exceptionType, const char*>(            \
      const NimbleCheckFailArgs&,                                              \
      const char*);                                                            \
  extern template void nimbleCheckFail<exceptionType, const std::string&>(     \
      const NimbleCheckFailArgs&,                                              \
      const std::string&);                                                     \
  extern template void nimbleCheckFail<exceptionType, CompileTimeEmptyString>( \
      const NimbleCheckFailArgs&,                                              \
      CompileTimeEmptyString);                                                 \
  }

// Define explicit instantiations of nimbleCheckFail for the given
// exceptionType. The actual template instantiations go in this macro.
#define NIMBLE_DEFINE_CHECK_FAIL_TEMPLATES(exceptionType)               \
  namespace detail {                                                    \
  template void nimbleCheckFail<exceptionType, const char*>(            \
      const NimbleCheckFailArgs&,                                       \
      const char*);                                                     \
  template void nimbleCheckFail<exceptionType, const std::string&>(     \
      const NimbleCheckFailArgs&,                                       \
      const std::string&);                                              \
  template void nimbleCheckFail<exceptionType, CompileTimeEmptyString>( \
      const NimbleCheckFailArgs&,                                       \
      CompileTimeEmptyString);                                          \
  }
} // namespace detail

// Declare template instantiations for common exception types
NIMBLE_DECLARE_CHECK_FAIL_TEMPLATES(::facebook::nimble::NimbleUserError);
NIMBLE_DECLARE_CHECK_FAIL_TEMPLATES(::facebook::nimble::NimbleInternalError);

// Base throw implementation - Velox-style with NimbleCheckFailArgs
#define _NIMBLE_THROW_IMPL(exception, exprStr, errorCode, retryable, ...)     \
  do {                                                                        \
    /* GCC 9.2.1 doesn't accept this code with constexpr. */                  \
    static const ::facebook::nimble::detail::NimbleCheckFailArgs              \
        nimbleCheckFailArgs = {                                               \
            __FILE__, __LINE__, __FUNCTION__, exprStr, errorCode, retryable}; \
    auto message = ::facebook::nimble::errorMessage(__VA_ARGS__);             \
    ::facebook::nimble::detail::nimbleCheckFail<                              \
        exception,                                                            \
        typename ::facebook::nimble::detail::NimbleCheckFailStringType<       \
            decltype(message)>::type>(nimbleCheckFailArgs, message);          \
  } while (0)

#define NIMBLE_RAISE_USER_ERROR(expression, code, retryable, ...) \
  _NIMBLE_THROW_IMPL(                                             \
      ::facebook::nimble::NimbleUserError,                        \
      expression,                                                 \
      code,                                                       \
      retryable,                                                  \
      ##__VA_ARGS__)

#define NIMBLE_RAISE_INTERNAL_ERROR(expression, code, retryable, ...) \
  _NIMBLE_THROW_IMPL(                                                 \
      ::facebook::nimble::NimbleInternalError,                        \
      expression,                                                     \
      code,                                                           \
      retryable,                                                      \
      ##__VA_ARGS__)

// Check internal preconditions and invariants - Velox-style implementation
#define _NIMBLE_CHECK_AND_THROW_IMPL(                             \
    exprStr, expr, exception, errorCode, retryable, ...)          \
  if (UNLIKELY(!(expr))) {                                        \
    _NIMBLE_THROW_IMPL(                                           \
        exception, exprStr, errorCode, retryable, ##__VA_ARGS__); \
  }

#define _NIMBLE_CHECK_IMPL(expr, exprStr, ...)         \
  _NIMBLE_CHECK_AND_THROW_IMPL(                        \
      exprStr,                                         \
      expr,                                            \
      ::facebook::nimble::NimbleInternalError,         \
      ::facebook::nimble::error_code::InvalidArgument, \
      false,                                           \
      ##__VA_ARGS__)

#define NIMBLE_CHECK(expr, ...) _NIMBLE_CHECK_IMPL(expr, #expr, ##__VA_ARGS__)

#define NIMBLE_USER_CHECK(expr, ...) \
  _NIMBLE_USER_CHECK_IMPL(expr, #expr, ##__VA_ARGS__)

// Verify an expected file format conditions.
// the file is corrupted (e.g. passed magic number and version verification, but
// got unexpected format). This will trigger a user error.
#define _NIMBLE_CHECK_FILE_IMPL(condition, conditionString, ...) \
  if (UNLIKELY(!(condition))) {                                  \
    NIMBLE_RAISE_USER_ERROR(                                     \
        conditionString,                                         \
        ::facebook::nimble::error_code::CorruptedFile,           \
        /* retryable */ false,                                   \
        __VA_ARGS__);                                            \
  }

#define NIMBLE_CHECK_FILE(condition, ...) \
  _NIMBLE_CHECK_FILE_IMPL(condition, #condition, ##__VA_ARGS__)

// Should be raised when we don't expect to hit a code path, but we did. This
// means a bug in Nimble.
#define NIMBLE_UNREACHABLE(...)                        \
  NIMBLE_RAISE_INTERNAL_ERROR(                         \
      "",                                              \
      ::facebook::nimble::error_code::UnreachableCode, \
      /* retryable */ false,                           \
      __VA_ARGS__);

// Should be raised in places where we still didn't implement the required
// functionality, but intend to do so in the future. This raises an internal
// error to indicate users needs this functionality, but we don't provide it.
#define NIMBLE_NOT_IMPLEMENTED(...)                   \
  NIMBLE_RAISE_INTERNAL_ERROR(                        \
      "",                                             \
      ::facebook::nimble::error_code::NotImplemented, \
      /* retryable */ false,                          \
      __VA_ARGS__);

// Should be raised in places where we don't support a functionality, and have
// no intention to support it in the future. This raises a user error, as the
// user should not expect this functionality to exist in the first place.
#define NIMBLE_UNSUPPORTED(...)                     \
  NIMBLE_RAISE_USER_ERROR(                          \
      "",                                           \
      ::facebook::nimble::error_code::NotSupported, \
      /* retryable */ false,                        \
      __VA_ARGS__);

// Incompatible Encoding errors are used in Nimble's encoding optimization, to
// indicate that an attempted encoding is incompatible with the data and should
// be avoided.
#define NIMBLE_INCOMPATIBLE_ENCODING(...)                   \
  NIMBLE_RAISE_USER_ERROR(                                  \
      "",                                                   \
      ::facebook::nimble::error_code::IncompatibleEncoding, \
      /* retryable */ false,                                \
      __VA_ARGS__);

// Should be used in "catch all" exception handlers, where we can't classify the
// error correctly. These errors mean that we are missing error classification.
#define NIMBLE_UNKNOWN(...)                    \
  NIMBLE_RAISE_INTERNAL_ERROR(                 \
      "",                                      \
      ::facebook::nimble::error_code::Unknown, \
      /* retryable */ true,                    \
      __VA_ARGS__);

// Comparison macros - Velox-style
#define _NIMBLE_CHECK_OP_WITH_USER_FMT_HELPER(  \
    implmacro, expr1, expr2, op, user_fmt, ...) \
  implmacro(                                    \
      (expr1)op(expr2),                         \
      #expr1 " " #op " " #expr2,                \
      "({} vs. {}) " user_fmt,                  \
      expr1,                                    \
      expr2,                                    \
      ##__VA_ARGS__)

#define _NIMBLE_CHECK_OP_HELPER(implmacro, expr1, expr2, op, ...) \
  do {                                                            \
    if constexpr (FOLLY_PP_DETAIL_NARGS(__VA_ARGS__) > 0) {       \
      _NIMBLE_CHECK_OP_WITH_USER_FMT_HELPER(                      \
          implmacro, expr1, expr2, op, __VA_ARGS__);              \
    } else {                                                      \
      implmacro(                                                  \
          (expr1)op(expr2),                                       \
          #expr1 " " #op " " #expr2,                              \
          "({} vs. {})",                                          \
          expr1,                                                  \
          expr2);                                                 \
    }                                                             \
  } while (0)

#define _NIMBLE_CHECK_OP(expr1, expr2, op, ...) \
  _NIMBLE_CHECK_OP_HELPER(_NIMBLE_CHECK_IMPL, expr1, expr2, op, ##__VA_ARGS__)

#define _NIMBLE_CHECK_FILE_OP(expr1, expr2, op, ...) \
  _NIMBLE_CHECK_OP_HELPER(                           \
      _NIMBLE_CHECK_FILE_IMPL, expr1, expr2, op, ##__VA_ARGS__)

#define _NIMBLE_USER_CHECK_IMPL(expr, exprStr, ...)    \
  _NIMBLE_CHECK_AND_THROW_IMPL(                        \
      exprStr,                                         \
      expr,                                            \
      ::facebook::nimble::NimbleUserError,             \
      ::facebook::nimble::error_code::InvalidArgument, \
      /* retryable */ false,                           \
      ##__VA_ARGS__)

#define _NIMBLE_USER_CHECK_OP(expr1, expr2, op, ...) \
  _NIMBLE_CHECK_OP_HELPER(                           \
      _NIMBLE_USER_CHECK_IMPL, expr1, expr2, op, ##__VA_ARGS__)

// Comparison check macros - internal errors
#define NIMBLE_CHECK_GT(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, >, ##__VA_ARGS__)
#define NIMBLE_CHECK_GE(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, >=, ##__VA_ARGS__)
#define NIMBLE_CHECK_LT(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, <, ##__VA_ARGS__)
#define NIMBLE_CHECK_LE(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, <=, ##__VA_ARGS__)
#define NIMBLE_CHECK_EQ(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, ==, ##__VA_ARGS__)
#define NIMBLE_CHECK_NE(e1, e2, ...) _NIMBLE_CHECK_OP(e1, e2, !=, ##__VA_ARGS__)

// Comparison check macros - corrupted file errors
#define NIMBLE_CHECK_FILE_GT(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, >, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_GE(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, >=, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_LT(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, <, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_LE(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, <=, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_EQ(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, ==, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_NE(e1, e2, ...) \
  _NIMBLE_CHECK_FILE_OP(e1, e2, !=, ##__VA_ARGS__)

// Comparison check macros - user errors
#define NIMBLE_USER_CHECK_GT(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, >, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_GE(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, >=, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_LT(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, <, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_LE(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, <=, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_EQ(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, ==, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_NE(e1, e2, ...) \
  _NIMBLE_USER_CHECK_OP(e1, e2, !=, ##__VA_ARGS__)

// Null pointer checks
#define NIMBLE_CHECK_NULL(e, ...) NIMBLE_CHECK((e) == nullptr, ##__VA_ARGS__)
#define NIMBLE_CHECK_NOT_NULL(e, ...) \
  NIMBLE_CHECK((e) != nullptr, ##__VA_ARGS__)
#define NIMBLE_CHECK_FILE_NOT_NULL(e, ...) \
  NIMBLE_CHECK_FILE((e) != nullptr, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_NULL(e, ...) \
  NIMBLE_USER_CHECK((e) == nullptr, ##__VA_ARGS__)
#define NIMBLE_USER_CHECK_NOT_NULL(e, ...) \
  NIMBLE_USER_CHECK((e) != nullptr, ##__VA_ARGS__)

// Failure macros without conditions
#define NIMBLE_FAIL(...)                            \
  NIMBLE_RAISE_INTERNAL_ERROR(                      \
      "",                                           \
      ::facebook::nimble::error_code::InvalidState, \
      /* retryable */ false,                        \
      __VA_ARGS__)

#define NIMBLE_USER_FAIL(...)                          \
  NIMBLE_RAISE_USER_ERROR(                             \
      "",                                              \
      ::facebook::nimble::error_code::InvalidArgument, \
      /* retryable */ false,                           \
      __VA_ARGS__)

// Debug variants
#ifndef NDEBUG
#define NIMBLE_DCHECK(condition, ...) NIMBLE_CHECK(condition, ##__VA_ARGS__)
#define NIMBLE_DCHECK_GT(e1, e2, ...) NIMBLE_CHECK_GT(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_GE(e1, e2, ...) NIMBLE_CHECK_GE(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_LT(e1, e2, ...) NIMBLE_CHECK_LT(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_LE(e1, e2, ...) NIMBLE_CHECK_LE(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_EQ(e1, e2, ...) NIMBLE_CHECK_EQ(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_NE(e1, e2, ...) NIMBLE_CHECK_NE(e1, e2, ##__VA_ARGS__)
#define NIMBLE_DCHECK_NULL(e, ...) NIMBLE_CHECK_NULL(e, ##__VA_ARGS__)
#define NIMBLE_DCHECK_NOT_NULL(e, ...) NIMBLE_CHECK_NOT_NULL(e, ##__VA_ARGS__)

#define NIMBLE_DEBUG_ONLY
#else
#define NIMBLE_DCHECK(condition, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_GT(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_GE(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_LT(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_LE(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_EQ(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_NE(e1, e2, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_NULL(e, ...) NIMBLE_CHECK(true, "")
#define NIMBLE_DCHECK_NOT_NULL(e, ...) NIMBLE_CHECK(true, "")

#define NIMBLE_DEBUG_ONLY [[maybe_unused]]
#endif

} // namespace facebook::nimble
