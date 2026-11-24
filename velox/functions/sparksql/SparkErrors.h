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

#include <folly/FixedString.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"

/// Helpers for Spark functions to report errors that map to a Spark error
/// condition (see Spark's error-conditions.json). The condition is attached to
/// the error as error code "SPARK:<condition>", e.g. "SPARK:DIVIDE_BY_ZERO".
/// The error message is unchanged. Engines such as Gluten use the error code to
/// raise the matching Spark exception.

namespace facebook::velox::functions::sparksql::spark_error_code {
using namespace folly::string_literals;

/// Spark error condition DIVIDE_BY_ZERO.
inline constexpr auto kDivideByZero = "SPARK:DIVIDE_BY_ZERO"_fs;

} // namespace facebook::velox::functions::sparksql::spark_error_code

/// Same as _VELOX_RETURN_IMPL with Status::UserError, but sets 'errorCode', one
/// of the spark_error_code constants. When error details are skipped (e.g.
/// under TRY), no code is set to keep TRY cheap.
#define _SPARK_USER_RETURN_IMPL(errorCode, expr, exprStr, ...)                 \
  do {                                                                         \
    if (expr) [[unlikely]] {                                                   \
      if (::facebook::velox::threadSkipErrorDetails()) {                       \
        return ::facebook::velox::Status::UserError();                         \
      }                                                                        \
      auto message = ::facebook::velox::errorMessage(__VA_ARGS__);             \
      return ::facebook::velox::Status::UserError(                             \
                 ::facebook::velox::internal::generateError(message, exprStr)) \
          .withErrorCode((errorCode).toStdString());                           \
    }                                                                          \
  } while (0)

#define _SPARK_USER_RETURN_OP(errorCode, expr1, expr2, op, user_fmt, ...) \
  _SPARK_USER_RETURN_IMPL(                                                \
      errorCode,                                                          \
      (expr1)op(expr2),                                                   \
      #expr1 " " #op " " #expr2,                                          \
      "({} vs. {}) " user_fmt,                                            \
      expr1,                                                              \
      expr2,                                                              \
      ##__VA_ARGS__)

/// Returns a user error Status with 'errorCode' (a spark_error_code constant)
/// if 'expr' is true. An error message can be passed as with
/// VELOX_USER_RETURN.
#define SPARK_USER_RETURN(errorCode, expr, ...) \
  _SPARK_USER_RETURN_IMPL(errorCode, expr, #expr, ##__VA_ARGS__)

/// Same as VELOX_USER_RETURN_EQ(e1, e2, user_fmt, ...) but sets 'errorCode' (a
/// spark_error_code constant). A message format string is required.
#define SPARK_USER_RETURN_EQ(errorCode, e1, e2, user_fmt, ...) \
  _SPARK_USER_RETURN_OP(errorCode, e1, e2, ==, user_fmt, ##__VA_ARGS__)
