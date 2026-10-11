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

#include <memory>
#include <sstream>

#include <fmt/ostream.h>
#include <folly/Preprocessor.h>

#include "velox/common/base/ExceptionHelper.h"
#include "velox/common/base/FmtStdFormatters.h"
#include "velox/common/base/VeloxException.h"

namespace facebook::velox {
namespace detail {

struct VeloxCheckFailArgs {
  const char* file;
  size_t line;
  const char* function;
  const char* expression;
  const char* errorSource;
  const char* errorCode;
  bool isRetriable;
};

// veloxCheckFail is defined as a separate helper function rather than
// a macro or inline `throw` expression to allow the compiler *not* to
// inline it when it is large. Having an out-of-line error path helps
// otherwise-small functions that call error-checking macros stay
// small and thus stay eligible for inlining.

// Overload without messageTemplate: message IS the template (0 or 1 args).
template <typename Exception, typename StringType>
[[noreturn]] void veloxCheckFail(const VeloxCheckFailArgs& args, StringType s) {
  static_assert(
      !std::is_same_v<StringType, std::string>,
      "BUG: we should not pass std::string by value to veloxCheckFail");
  if constexpr (!std::is_same_v<Exception, VeloxUserError>) {
    LOG(ERROR) << "Line: " << args.file << ":" << args.line
               << ", Function:" << args.function
               << ", Expression: " << args.expression << " " << s
               << ", Source: " << args.errorSource
               << ", ErrorCode: " << args.errorCode;
  }

  ++threadNumVeloxThrow();
  throw Exception(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable);
}

// Overload with messageTemplate: template is the format string before
// interpolation (>=2 args, format string + arguments). The template should
// be a string literal; for runtime format strings, use the single-arg
// VELOX_FAIL("{}", runtimeStr) pattern instead.
template <typename Exception, typename StringType>
[[noreturn]] void veloxCheckFail(
    const VeloxCheckFailArgs& args,
    StringType s,
    CompileTimeStringLiteral messageTemplate) {
  static_assert(
      !std::is_same_v<StringType, std::string>,
      "BUG: we should not pass std::string by value to veloxCheckFail");
  if constexpr (!std::is_same_v<Exception, VeloxUserError>) {
    LOG(ERROR) << "Line: " << args.file << ":" << args.line
               << ", Function:" << args.function
               << ", Expression: " << args.expression << " " << s
               << ", Source: " << args.errorSource
               << ", ErrorCode: " << args.errorCode;
  }

  ++threadNumVeloxThrow();
  throw Exception(
      args.file,
      args.line,
      args.function,
      args.expression,
      s,
      args.errorSource,
      args.errorCode,
      args.isRetriable,
      messageTemplate);
}

// VeloxCheckFailStringType helps us pass by reference to
// veloxCheckFail exactly when the string type is std::string.
template <typename T>
struct VeloxCheckFailStringType;

template <>
struct VeloxCheckFailStringType<CompileTimeEmptyString> {
  using type = CompileTimeEmptyString;
};

template <>
struct VeloxCheckFailStringType<const char*> {
  using type = const char*;
};

template <>
struct VeloxCheckFailStringType<std::string> {
  using type = const std::string&;
};

// Declare explicit instantiations of veloxCheckFail for the given
// exceptionType. Just like normal function declarations (prototypes),
// this allows the compiler to assume that they are defined elsewhere
// and simply insert a function call for the linker to fix up, rather
// than emitting a definition of these templates into every
// translation unit they are used in.
#define DECLARE_CHECK_FAIL_TEMPLATES(exception_type)                           \
  namespace detail {                                                           \
  extern template void veloxCheckFail<exception_type, CompileTimeEmptyString>( \
      const VeloxCheckFailArgs& args,                                          \
      CompileTimeEmptyString);                                                 \
  extern template void veloxCheckFail<exception_type, const char*>(            \
      const VeloxCheckFailArgs& args,                                          \
      const char*);                                                            \
  extern template void veloxCheckFail<exception_type, const std::string&>(     \
      const VeloxCheckFailArgs& args,                                          \
      const std::string&);                                                     \
  extern template void veloxCheckFail<exception_type, CompileTimeEmptyString>( \
      const VeloxCheckFailArgs& args,                                          \
      CompileTimeEmptyString,                                                  \
      CompileTimeStringLiteral);                                               \
  extern template void veloxCheckFail<exception_type, const char*>(            \
      const VeloxCheckFailArgs& args,                                          \
      const char*,                                                             \
      CompileTimeStringLiteral);                                               \
  extern template void veloxCheckFail<exception_type, const std::string&>(     \
      const VeloxCheckFailArgs& args,                                          \
      const std::string&,                                                      \
      CompileTimeStringLiteral);                                               \
  } // namespace detail

// Definitions corresponding to DECLARE_CHECK_FAIL_TEMPLATES. Should
// only be used in Exceptions.cpp.
#define DEFINE_CHECK_FAIL_TEMPLATES(exception_type)                           \
  template void veloxCheckFail<exception_type, CompileTimeEmptyString>(       \
      const VeloxCheckFailArgs& args, CompileTimeEmptyString);                \
  template void veloxCheckFail<exception_type, const char*>(                  \
      const VeloxCheckFailArgs& args, const char*);                           \
  template void veloxCheckFail<exception_type, const std::string&>(           \
      const VeloxCheckFailArgs& args, const std::string&);                    \
  template void veloxCheckFail<exception_type, CompileTimeEmptyString>(       \
      const VeloxCheckFailArgs& args,                                         \
      CompileTimeEmptyString,                                                 \
      CompileTimeStringLiteral);                                              \
  template void veloxCheckFail<exception_type, const char*>(                  \
      const VeloxCheckFailArgs& args, const char*, CompileTimeStringLiteral); \
  template void veloxCheckFail<exception_type, const std::string&>(           \
      const VeloxCheckFailArgs& args,                                         \
      const std::string&,                                                     \
      CompileTimeStringLiteral);

} // namespace detail

// Macro arg-count detection for dispatching between the no-template
// and with-template overloads of veloxCheckFail.
// 0 or 1 args: message IS the template (no-template overload).
// >=2 args: first arg is the format string template (with-template overload).
#define _VELOX_NARGS_IMPL( \
    _0,                    \
    _1,                    \
    _2,                    \
    _3,                    \
    _4,                    \
    _5,                    \
    _6,                    \
    _7,                    \
    _8,                    \
    _9,                    \
    _10,                   \
    _11,                   \
    _12,                   \
    _13,                   \
    _14,                   \
    _15,                   \
    _16,                   \
    N,                     \
    ...)                   \
  N
#define _VELOX_NARGS(...) \
  _VELOX_NARGS_IMPL(      \
      dummy,              \
      ##__VA_ARGS__,      \
      16,                 \
      15,                 \
      14,                 \
      13,                 \
      12,                 \
      11,                 \
      10,                 \
      9,                  \
      8,                  \
      7,                  \
      6,                  \
      5,                  \
      4,                  \
      3,                  \
      2,                  \
      1,                  \
      0)

// Extract the first argument from __VA_ARGS__ (the format string) as a
// CompileTimeStringLiteral. Only used for >=2 args path.
#define _VELOX_MSG_TEMPLATE_PICK(_1, _2, ...) _2
#define _VELOX_MSG_TEMPLATE(...)               \
  ::facebook::velox::CompileTimeStringLiteral( \
      _VELOX_MSG_TEMPLATE_PICK("", ##__VA_ARGS__, ""))

// _VELOX_THROW_IMPL dispatches to the no-template or with-template path
// based on the number of user-supplied message arguments.
#define _VELOX_THROW_IMPL_BODY_NO_TEMPLATE(                           \
    exception, exprStr, errorSource, errorCode, isRetriable, ...)     \
  do {                                                                \
    /* GCC 9.2.1 doesn't accept this code with constexpr. */          \
    static const ::facebook::velox::detail::VeloxCheckFailArgs        \
        veloxCheckFailArgs = {                                        \
            __FILE__,                                                 \
            __LINE__,                                                 \
            __FUNCTION__,                                             \
            exprStr,                                                  \
            errorSource,                                              \
            errorCode,                                                \
            isRetriable};                                             \
    auto message = ::facebook::velox::errorMessage(__VA_ARGS__);      \
    ::facebook::velox::detail::veloxCheckFail<                        \
        exception,                                                    \
        typename ::facebook::velox::detail::VeloxCheckFailStringType< \
            decltype(message)>::type>(veloxCheckFailArgs, message);   \
  } while (0)

#define _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE(                           \
    exception, exprStr, errorSource, errorCode, isRetriable, ...)       \
  do {                                                                  \
    /* GCC 9.2.1 doesn't accept this code with constexpr. */            \
    static const ::facebook::velox::detail::VeloxCheckFailArgs          \
        veloxCheckFailArgs = {                                          \
            __FILE__,                                                   \
            __LINE__,                                                   \
            __FUNCTION__,                                               \
            exprStr,                                                    \
            errorSource,                                                \
            errorCode,                                                  \
            isRetriable};                                               \
    auto message = ::facebook::velox::errorMessage(__VA_ARGS__);        \
    ::facebook::velox::detail::veloxCheckFail<                          \
        exception,                                                      \
        typename ::facebook::velox::detail::VeloxCheckFailStringType<   \
            decltype(message)>::type>(                                  \
        veloxCheckFailArgs, message, _VELOX_MSG_TEMPLATE(__VA_ARGS__)); \
  } while (0)

#define _VELOX_THROW_DISPATCH_0 _VELOX_THROW_IMPL_BODY_NO_TEMPLATE
#define _VELOX_THROW_DISPATCH_1 _VELOX_THROW_IMPL_BODY_NO_TEMPLATE
#define _VELOX_THROW_DISPATCH_2 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_3 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_4 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_5 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_6 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_7 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_8 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_9 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_10 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_11 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_12 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_13 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_14 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_15 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE
#define _VELOX_THROW_DISPATCH_16 _VELOX_THROW_IMPL_BODY_WITH_TEMPLATE

#define _VELOX_THROW_CONCAT2(a, b) a##b
#define _VELOX_THROW_CONCAT(a, b) _VELOX_THROW_CONCAT2(a, b)
#define _VELOX_THROW_SELECT(n) _VELOX_THROW_CONCAT(_VELOX_THROW_DISPATCH_, n)

#define _VELOX_THROW_IMPL(                                        \
    exception, exprStr, errorSource, errorCode, isRetriable, ...) \
  _VELOX_THROW_SELECT(_VELOX_NARGS(__VA_ARGS__))(                 \
      exception, exprStr, errorSource, errorCode, isRetriable, ##__VA_ARGS__)

#define _VELOX_CHECK_AND_THROW_IMPL(                                    \
    expr, exprStr, exception, errorSource, errorCode, isRetriable, ...) \
  do {                                                                  \
    if (UNLIKELY(!(expr))) {                                            \
      _VELOX_THROW_IMPL(                                                \
          exception,                                                    \
          exprStr,                                                      \
          errorSource,                                                  \
          errorCode,                                                    \
          isRetriable,                                                  \
          __VA_ARGS__);                                                 \
    }                                                                   \
  } while (0)

#define _VELOX_THROW(exception, ...) \
  _VELOX_THROW_IMPL(exception, "", ##__VA_ARGS__)

DECLARE_CHECK_FAIL_TEMPLATES(::facebook::velox::VeloxRuntimeError)

#define _VELOX_CHECK_IMPL(expr, exprStr, ...)                       \
  _VELOX_CHECK_AND_THROW_IMPL(                                      \
      expr,                                                         \
      exprStr,                                                      \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kInvalidState.c_str(),         \
      /* isRetriable */ false,                                      \
      ##__VA_ARGS__)

/// Throws VeloxRuntimeError when functions receive input values out of the
/// supported range. This should only be used when we want to force TRY() to not
/// suppress the error.
#define VELOX_CHECK_UNSUPPORTED_INPUT_UNCATCHABLE(expr, ...)                   \
  do {                                                                         \
    if (UNLIKELY(!(expr))) {                                                   \
      _VELOX_THROW_IMPL(                                                       \
          ::facebook::velox::VeloxRuntimeError,                                \
          #expr,                                                               \
          ::facebook::velox::error_source::kErrorSourceRuntime.c_str(),        \
          ::facebook::velox::error_code::kUnsupportedInputUncatchable.c_str(), \
          /* isRetriable */ false,                                             \
          __VA_ARGS__);                                                        \
    }                                                                          \
  } while (0)

// If the caller passes a custom message (4 *or more* arguments), we
// have to construct a format string from ours ("({} vs. {})") plus
// theirs by adding a space and shuffling arguments. If they don't (exactly 3
// arguments), we can just pass our own format string and arguments straight
// through.

#define _VELOX_CHECK_OP_WITH_USER_FMT_HELPER(   \
    implmacro, expr1, expr2, op, user_fmt, ...) \
  implmacro(                                    \
      (expr1)op(expr2),                         \
      #expr1 " " #op " " #expr2,                \
      "({} vs. {}) " user_fmt,                  \
      expr1,                                    \
      expr2,                                    \
      ##__VA_ARGS__)

#define _VELOX_CHECK_OP_HELPER(implmacro, expr1, expr2, op, ...) \
  do {                                                           \
    if constexpr (FOLLY_PP_DETAIL_NARGS(__VA_ARGS__) > 0) {      \
      _VELOX_CHECK_OP_WITH_USER_FMT_HELPER(                      \
          implmacro, expr1, expr2, op, __VA_ARGS__);             \
    } else {                                                     \
      implmacro(                                                 \
          (expr1)op(expr2),                                      \
          #expr1 " " #op " " #expr2,                             \
          "({} vs. {})",                                         \
          expr1,                                                 \
          expr2);                                                \
    }                                                            \
  } while (0)

#define _VELOX_CHECK_OP(expr1, expr2, op, ...) \
  _VELOX_CHECK_OP_HELPER(_VELOX_CHECK_IMPL, expr1, expr2, op, ##__VA_ARGS__)

#define _VELOX_USER_CHECK_IMPL(expr, exprStr, ...)               \
  _VELOX_CHECK_AND_THROW_IMPL(                                   \
      expr,                                                      \
      exprStr,                                                   \
      ::facebook::velox::VeloxUserError,                         \
      ::facebook::velox::error_source::kErrorSourceUser.c_str(), \
      ::facebook::velox::error_code::kInvalidArgument.c_str(),   \
      /* isRetriable */ false,                                   \
      ##__VA_ARGS__)

#define _VELOX_USER_CHECK_OP(expr1, expr2, op, ...) \
  _VELOX_CHECK_OP_HELPER(                           \
      _VELOX_USER_CHECK_IMPL, expr1, expr2, op, ##__VA_ARGS__)

// For all below macros, an additional message can be passed using a
// format string and arguments, as with `fmt::format`.
#define VELOX_CHECK(expr, ...) _VELOX_CHECK_IMPL(expr, #expr, ##__VA_ARGS__)
#define VELOX_CHECK_GT(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, >, ##__VA_ARGS__)
#define VELOX_CHECK_GE(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, >=, ##__VA_ARGS__)
#define VELOX_CHECK_LT(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, <, ##__VA_ARGS__)
#define VELOX_CHECK_LE(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, <=, ##__VA_ARGS__)
#define VELOX_CHECK_EQ(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, ==, ##__VA_ARGS__)
#define VELOX_CHECK_NE(e1, e2, ...) _VELOX_CHECK_OP(e1, e2, !=, ##__VA_ARGS__)
#define VELOX_CHECK_NULL(e, ...) VELOX_CHECK(e == nullptr, ##__VA_ARGS__)
#define VELOX_CHECK_NOT_NULL(e, ...) VELOX_CHECK(e != nullptr, ##__VA_ARGS__)

#define VELOX_CHECK_OK(expr)                          \
  do {                                                \
    ::facebook::velox::Status _s = (expr);            \
    _VELOX_CHECK_IMPL(_s.ok(), #expr, _s.toString()); \
  } while (false)

#define VELOX_UNSUPPORTED(...)                                   \
  _VELOX_THROW(                                                  \
      ::facebook::velox::VeloxUserError,                         \
      ::facebook::velox::error_source::kErrorSourceUser.c_str(), \
      ::facebook::velox::error_code::kUnsupported.c_str(),       \
      /* isRetriable */ false,                                   \
      ##__VA_ARGS__)

#define VELOX_ARITHMETIC_ERROR(...)                              \
  _VELOX_THROW(                                                  \
      ::facebook::velox::VeloxUserError,                         \
      ::facebook::velox::error_source::kErrorSourceUser.c_str(), \
      ::facebook::velox::error_code::kArithmeticError.c_str(),   \
      /* isRetriable */ false,                                   \
      ##__VA_ARGS__)

#define VELOX_SCHEMA_MISMATCH_ERROR(...)                         \
  _VELOX_THROW(                                                  \
      ::facebook::velox::VeloxUserError,                         \
      ::facebook::velox::error_source::kErrorSourceUser.c_str(), \
      ::facebook::velox::error_code::kSchemaMismatch.c_str(),    \
      /* isRetriable */ false,                                   \
      ##__VA_ARGS__)

#define VELOX_FILE_NOT_FOUND_ERROR(...)                             \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kFileNotFound.c_str(),         \
      /* isRetriable */ false,                                      \
      ##__VA_ARGS__)

#define VELOX_UNREACHABLE(...)                                      \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kUnreachableCode.c_str(),      \
      /* isRetriable */ false,                                      \
      ##__VA_ARGS__)

#ifndef NDEBUG
#define VELOX_DCHECK(expr, ...) VELOX_CHECK(expr, ##__VA_ARGS__)
#define VELOX_DCHECK_GT(e1, e2, ...) VELOX_CHECK_GT(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_GE(e1, e2, ...) VELOX_CHECK_GE(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_LT(e1, e2, ...) VELOX_CHECK_LT(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_LE(e1, e2, ...) VELOX_CHECK_LE(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_EQ(e1, e2, ...) VELOX_CHECK_EQ(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_NE(e1, e2, ...) VELOX_CHECK_NE(e1, e2, ##__VA_ARGS__)
#define VELOX_DCHECK_NULL(e, ...) VELOX_CHECK_NULL(e, ##__VA_ARGS__)
#define VELOX_DCHECK_NOT_NULL(e, ...) VELOX_CHECK_NOT_NULL(e, ##__VA_ARGS__)
#define VELOX_DEBUG_ONLY
#else
#define VELOX_DCHECK(expr, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_GT(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_GE(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_LT(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_LE(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_EQ(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_NE(e1, e2, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_NULL(e, ...) VELOX_CHECK(true)
#define VELOX_DCHECK_NOT_NULL(e, ...) VELOX_CHECK(true)
#define VELOX_DEBUG_ONLY [[maybe_unused]]
#endif

#define VELOX_FAIL(...)                                             \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kInvalidState.c_str(),         \
      /* isRetriable */ false,                                      \
      ##__VA_ARGS__)

/// Throws VeloxRuntimeError when functions receive input values out of the
/// supported range. This should only be used when we want to force TRY() to not
/// suppress the error.
#define VELOX_FAIL_UNSUPPORTED_INPUT_UNCATCHABLE(...)                      \
  _VELOX_THROW(                                                            \
      ::facebook::velox::VeloxRuntimeError,                                \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(),        \
      ::facebook::velox::error_code::kUnsupportedInputUncatchable.c_str(), \
      /* isRetriable */ false,                                             \
      ##__VA_ARGS__)

#define VELOX_TRACE_LIMIT_EXCEEDED(...)                             \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kTraceLimitExceeded.c_str(),   \
      /* isRetriable */ true,                                       \
      ##__VA_ARGS__)

DECLARE_CHECK_FAIL_TEMPLATES(::facebook::velox::VeloxUserError)

// For all below macros, an additional message can be passed using a
// format string and arguments, as with `fmt::format`.
#define VELOX_USER_CHECK(expr, ...) \
  _VELOX_USER_CHECK_IMPL(expr, #expr, ##__VA_ARGS__)
#define VELOX_USER_CHECK_GT(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, >, ##__VA_ARGS__)
#define VELOX_USER_CHECK_GE(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, >=, ##__VA_ARGS__)
#define VELOX_USER_CHECK_LT(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, <, ##__VA_ARGS__)
#define VELOX_USER_CHECK_LE(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, <=, ##__VA_ARGS__)
#define VELOX_USER_CHECK_EQ(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, ==, ##__VA_ARGS__)
#define VELOX_USER_CHECK_NE(e1, e2, ...) \
  _VELOX_USER_CHECK_OP(e1, e2, !=, ##__VA_ARGS__)
#define VELOX_USER_CHECK_NULL(e, ...) \
  VELOX_USER_CHECK(e == nullptr, ##__VA_ARGS__)
#define VELOX_USER_CHECK_NOT_NULL(e, ...) \
  VELOX_USER_CHECK(e != nullptr, ##__VA_ARGS__)

#ifndef NDEBUG
#define VELOX_USER_DCHECK(expr, ...) VELOX_USER_CHECK(expr, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_GT(e1, e2, ...) \
  VELOX_USER_CHECK_GT(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_GE(e1, e2, ...) \
  VELOX_USER_CHECK_GE(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_LT(e1, e2, ...) \
  VELOX_USER_CHECK_LT(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_LE(e1, e2, ...) \
  VELOX_USER_CHECK_LE(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_EQ(e1, e2, ...) \
  VELOX_USER_CHECK_EQ(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_NE(e1, e2, ...) \
  VELOX_USER_CHECK_NE(e1, e2, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_NOT_NULL(e, ...) \
  VELOX_USER_CHECK_NOT_NULL(e, ##__VA_ARGS__)
#define VELOX_USER_DCHECK_NULL(e, ...) VELOX_USER_CHECK_NULL(e, ##__VA_ARGS__)
#else
#define VELOX_USER_DCHECK(expr, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_GT(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_GE(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_LT(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_LE(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_EQ(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_NE(e1, e2, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_NULL(e, ...) VELOX_USER_CHECK(true)
#define VELOX_USER_DCHECK_NOT_NULL(e, ...) VELOX_USER_CHECK(true)
#endif

#define VELOX_USER_FAIL(...)                                     \
  _VELOX_THROW(                                                  \
      ::facebook::velox::VeloxUserError,                         \
      ::facebook::velox::error_source::kErrorSourceUser.c_str(), \
      ::facebook::velox::error_code::kInvalidArgument.c_str(),   \
      /* isRetriable */ false,                                   \
      ##__VA_ARGS__)

#define VELOX_NYI(...)                                              \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kNotImplemented.c_str(),       \
      /* isRetriable */ false,                                      \
      ##__VA_ARGS__)

} // namespace facebook::velox
