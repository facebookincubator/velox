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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"

/// Helpers for Spark functions to report errors that map to a Spark error
/// condition (see Spark's error-conditions.json). The condition is attached to
/// the error as error code "SPARK:<condition>", e.g. "SPARK:DIVIDE_BY_ZERO".
/// The error message is unchanged. Engines such as Gluten use the error code to
/// raise the matching Spark exception.

#define _SPARK_ERROR_CODE(condition) "SPARK:" condition

/// Same as _VELOX_RETURN_IMPL with Status::UserError, but sets the error code
/// to "SPARK:<condition>". 'condition' must be a string literal. When error
/// details are skipped (e.g. under TRY), no code is set to keep TRY cheap.
#define _SPARK_USER_RETURN_IMPL(condition, expr, exprStr, ...)                 \
  do {                                                                         \
    if (expr) [[unlikely]] {                                                   \
      if (::facebook::velox::threadSkipErrorDetails()) {                       \
        return ::facebook::velox::Status::UserError();                         \
      }                                                                        \
      auto message = ::facebook::velox::errorMessage(__VA_ARGS__);             \
      return ::facebook::velox::Status::UserError(                             \
                 ::facebook::velox::internal::generateError(message, exprStr)) \
          .withErrorCode(_SPARK_ERROR_CODE(condition));                        \
    }                                                                          \
  } while (0)

#define _SPARK_USER_RETURN_OP(condition, expr1, expr2, op, user_fmt, ...) \
  _SPARK_USER_RETURN_IMPL(                                                \
      condition,                                                          \
      (expr1)op(expr2),                                                   \
      #expr1 " " #op " " #expr2,                                          \
      "({} vs. {}) " user_fmt,                                            \
      expr1,                                                              \
      expr2,                                                              \
      ##__VA_ARGS__)

/// Returns a user error Status with error code "SPARK:<condition>" if 'expr'
/// is true. An error message can be passed as with VELOX_USER_RETURN.
#define SPARK_USER_RETURN(condition, expr, ...) \
  _SPARK_USER_RETURN_IMPL(condition, expr, #expr, ##__VA_ARGS__)

/// Same as VELOX_USER_RETURN_EQ(e1, e2, user_fmt, ...) but sets the error code
/// to "SPARK:<condition>". A message format string is required.
#define SPARK_USER_RETURN_EQ(condition, e1, e2, user_fmt, ...) \
  _SPARK_USER_RETURN_OP(condition, e1, e2, ==, user_fmt, ##__VA_ARGS__)
