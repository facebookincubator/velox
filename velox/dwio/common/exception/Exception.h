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

#include "velox/common/base/VeloxException.h"

namespace facebook::velox::dwio {
namespace common::exception {

class ExceptionLogger {
 public:
  virtual ~ExceptionLogger() = default;

  virtual void logException(
      const char* file,
      size_t line,
      const char* function,
      const char* expression,
      const char* message) = 0;

  virtual void logWarning(
      const char* file,
      size_t line,
      const char* function,
      const char* expression,
      const char* message) = 0;
};

bool registerExceptionLogger(std::unique_ptr<ExceptionLogger> logger);

ExceptionLogger* getExceptionLogger();

class LoggedException : public velox::VeloxException {
 public:
  explicit LoggedException(
      const std::string& errorMessage,
      const std::string& errorSource =
          ::facebook::velox::error_source::kErrorSourceExternal,
      const std::string& errorCode = ::facebook::velox::error_code::kUnknown,
      const bool isRetriable = false)
      : VeloxException(
            nullptr,
            0,
            nullptr,
            "",
            errorMessage,
            errorSource,
            errorCode,
            isRetriable) {
    logException();
  }

  LoggedException(
      const char* file,
      size_t line,
      const char* function,
      const char* expression,
      const std::string& errorMessage,
      const std::string& errorSource,
      const std::string& errorCode)
      : VeloxException(
            file,
            line,
            function,
            expression,
            errorMessage,
            errorSource,
            errorCode,
            false) {
    logException();
  }

  // Carries an explicit message template, kept distinct from the interpolated
  // 'errorMessage' so VeloxException::messageTemplate() stays low-cardinality
  // for error aggregation.
  LoggedException(
      const char* file,
      size_t line,
      const char* function,
      const char* expression,
      const std::string& errorMessage,
      const std::string& errorSource,
      const std::string& errorCode,
      velox::CompileTimeStringLiteral messageTemplate)
      : VeloxException(
            file,
            line,
            function,
            expression,
            errorMessage,
            errorSource,
            errorCode,
            /*isRetriable=*/false,
            messageTemplate) {
    logException();
  }

 private:
  void logException() {
    auto logger = getExceptionLogger();
    if (logger) {
      logger->logException(
          file(),
          line(),
          function(),
          failingExpression().data(),
          VeloxException::what());
    }
  }
};

} // namespace common::exception

#define DWIO_WARN_IF(e, ...)                                                \
  ({                                                                        \
    auto const& _tmp = (e);                                                 \
    if (_tmp) {                                                             \
      auto logger =                                                         \
          ::facebook::velox::dwio::common::exception::getExceptionLogger(); \
      if (logger) {                                                         \
        logger->logWarning(                                                 \
            __FILE__,                                                       \
            __LINE__,                                                       \
            __FUNCTION__,                                                   \
            #e,                                                             \
            ::folly::to<std::string>(__VA_ARGS__).c_str());                 \
      }                                                                     \
    }                                                                       \
  })

#define DWIO_WARN(...) DWIO_WARN_IF(true, ##__VA_ARGS__)

#define DWIO_WARN_EVERY_N(n, ...)          \
  static size_t LOG_OCCURRENCES_MOD_N = 0; \
  if (++LOG_OCCURRENCES_MOD_N > n)         \
    LOG_OCCURRENCES_MOD_N -= n;            \
  if (LOG_OCCURRENCES_MOD_N == 1)          \
    DWIO_WARN(__VA_ARGS__);

/*
 * Use ENFORCE(expr) or ENFORCE(expr, message) whenever you want to
 * make sure that expr is nonzero. If it is zero, an exception is
 * thrown. The type yielded by ENFORCE is the same as expr, so it acts
 * like an identity function, which makes it convenient to insert
 * anywhere. For example:
 *
 * auto file = ENFORCE(fopen("file.txt", "r"), "Can't find file.txt");
 * auto p = ENFORCE(dynamic_cast<Type*>(pointer));
 *
 * In case the expression is nonzero, the message, if any, is not
 * evaluated so you can plant in there a costly string concatenation:
 *
 * string fn = "file.txt"
 * auto file = ENFORCE(fopen(fn.c_str(), "r"), "Can't find " + fn);
 *
 * The ENFORCE macro stores the file, name, and function into the
 * FBException thrown.
 */
#define DWIO_ENFORCE_CUSTOM(                            \
    exception, expression, errorSource, errorCode, ...) \
  ({                                                    \
    auto const& _tmp = (expression);                    \
    _tmp ? _tmp                                         \
         : throw exception(                             \
               __FILE__,                                \
               __LINE__,                                \
               __FUNCTION__,                            \
               #expression,                             \
               ::folly::to<std::string>(__VA_ARGS__),   \
               errorSource,                             \
               errorCode);                              \
  })

/*
Unconditionally throws an exception derived from VeloxException
containing information about the file, line, and function where it happened.
 */
#define DWIO_EXCEPTION_CUSTOM(exception, errorSource, errorCode, ...) \
  (throw exception(                                                   \
      __FILE__,                                                       \
      __LINE__,                                                       \
      __FUNCTION__,                                                   \
      "",                                                             \
      ::folly::to<std::string>(__VA_ARGS__),                          \
      errorSource,                                                    \
      errorCode))

#define DWIO_RAISE(...)                                          \
  DWIO_EXCEPTION_CUSTOM(                                         \
      facebook::velox::dwio::common::exception::LoggedException, \
      ::facebook::velox::error_source::kErrorSourceExternal,     \
      ::facebook::velox::error_code::kUnknown,                   \
      ##__VA_ARGS__)

#define DWIO_ENSURE(expr, ...)                                   \
  DWIO_ENFORCE_CUSTOM(                                           \
      facebook::velox::dwio::common::exception::LoggedException, \
      expr,                                                      \
      ::facebook::velox::error_source::kErrorSourceExternal,     \
      ::facebook::velox::error_code::kUnknown,                   \
      ##__VA_ARGS__)

// Like DWIO_RAISE, but the first argument is an fmt format string and the rest
// are interpolated into it. The format string, not the interpolated result,
// becomes VeloxException::messageTemplate(), so identical failures aggregate
// together regardless of their interpolated values.
#define DWIO_RAISE_FMT(fmtString, ...)                               \
  throw ::facebook::velox::dwio::common::exception::LoggedException( \
      __FILE__,                                                      \
      __LINE__,                                                      \
      __FUNCTION__,                                                  \
      "",                                                            \
      ::facebook::velox::errorMessage(fmtString, ##__VA_ARGS__),     \
      ::facebook::velox::error_source::kErrorSourceExternal,         \
      ::facebook::velox::error_code::kUnknown,                       \
      ::facebook::velox::CompileTimeStringLiteral(fmtString))

// DWIO_ENSURE with a stable message template. See DWIO_RAISE_FMT.
#define DWIO_ENSURE_FMT(expr, fmtString, ...)   \
  do {                                          \
    if (!(expr)) {                              \
      DWIO_RAISE_FMT(fmtString, ##__VA_ARGS__); \
    }                                           \
  } while (0)

#define DWIO_ENSURE_NOT_NULL(p, ...) \
  DWIO_ENSURE(p != nullptr, "[Null pointer]: ", ##__VA_ARGS__);

#define DWIO_ENSURE_NE(l, r, ...)         \
  DWIO_ENSURE(                            \
      l != r,                             \
      "[Equality Constraint Violation: ", \
      l,                                  \
      "!=",                               \
      r,                                  \
      "]: ",                              \
      ##__VA_ARGS__);

#define DWIO_ENSURE_EQ(l, r, ...)         \
  DWIO_ENSURE(                            \
      l == r,                             \
      "[Equality Constraint Violation: ", \
      l,                                  \
      "==",                               \
      r,                                  \
      "]: ",                              \
      ##__VA_ARGS__);

#define DWIO_ENSURE_LT(l, r, ...)      \
  DWIO_ENSURE(                         \
      l < r,                           \
      "[Range Constraint Violation: ", \
      l,                               \
      "<",                             \
      r,                               \
      "]: ",                           \
      ##__VA_ARGS__);

#define DWIO_ENSURE_LE(l, r, ...)      \
  DWIO_ENSURE(                         \
      l <= r,                          \
      "[Range Constraint Violation: ", \
      l,                               \
      "<=",                            \
      r,                               \
      "]: ",                           \
      ##__VA_ARGS__);

#define DWIO_ENSURE_GT(l, r, ...)      \
  DWIO_ENSURE(                         \
      l > r,                           \
      "[Range Constraint Violation: ", \
      l,                               \
      ">",                             \
      r,                               \
      "]: ",                           \
      ##__VA_ARGS__);

#define DWIO_ENSURE_GE(l, r, ...)      \
  DWIO_ENSURE(                         \
      l >= r,                          \
      "[Range Constraint Violation: ", \
      l,                               \
      ">=",                            \
      r,                               \
      "]: ",                           \
      ##__VA_ARGS__);

// Template-stable counterparts of the constraint macros above. The compared
// values are passed as fmt arguments, so 'messageTemplate()' stays constant
// across failures instead of embedding the runtime values. 'fmtString' is a
// format literal for extra context and may be "" when none is needed; any
// following arguments interpolate into it.
#define DWIO_ENSURE_NOT_NULL_FMT(p, fmtString, ...) \
  DWIO_ENSURE_FMT((p) != nullptr, "[Null pointer]: " fmtString, ##__VA_ARGS__)

#define DWIO_ENSURE_EQ_FMT(l, r, fmtString, ...)               \
  DWIO_ENSURE_FMT(                                             \
      (l) == (r),                                              \
      "[Equality Constraint Violation: {} == {}]: " fmtString, \
      (l),                                                     \
      (r),                                                     \
      ##__VA_ARGS__)

#define DWIO_ENSURE_NE_FMT(l, r, fmtString, ...)               \
  DWIO_ENSURE_FMT(                                             \
      (l) != (r),                                              \
      "[Equality Constraint Violation: {} != {}]: " fmtString, \
      (l),                                                     \
      (r),                                                     \
      ##__VA_ARGS__)

#define DWIO_ENSURE_LT_FMT(l, r, fmtString, ...)           \
  DWIO_ENSURE_FMT(                                         \
      (l) < (r),                                           \
      "[Range Constraint Violation: {} < {}]: " fmtString, \
      (l),                                                 \
      (r),                                                 \
      ##__VA_ARGS__)

#define DWIO_ENSURE_LE_FMT(l, r, fmtString, ...)            \
  DWIO_ENSURE_FMT(                                          \
      (l) <= (r),                                           \
      "[Range Constraint Violation: {} <= {}]: " fmtString, \
      (l),                                                  \
      (r),                                                  \
      ##__VA_ARGS__)

#define DWIO_ENSURE_GT_FMT(l, r, fmtString, ...)           \
  DWIO_ENSURE_FMT(                                         \
      (l) > (r),                                           \
      "[Range Constraint Violation: {} > {}]: " fmtString, \
      (l),                                                 \
      (r),                                                 \
      ##__VA_ARGS__)

#define DWIO_ENSURE_GE_FMT(l, r, fmtString, ...)            \
  DWIO_ENSURE_FMT(                                          \
      (l) >= (r),                                           \
      "[Range Constraint Violation: {} >= {}]: " fmtString, \
      (l),                                                  \
      (r),                                                  \
      ##__VA_ARGS__)

} // namespace facebook::velox::dwio
