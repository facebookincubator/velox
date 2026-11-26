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

#include <string>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"
#include "velox/functions/sparksql/SparkErrorCodes.h"

/// Helpers for Spark functions to report errors that map to a Spark error
/// condition. The condition is attached to the error as an error code from
/// SparkErrorCodes.h, e.g. "SPARK:DIVIDE_BY_ZERO". The error message is
/// unchanged. Engines such as Gluten use the error code to raise the matching
/// Spark exception.

namespace facebook::velox::functions::sparksql::detail {

/// Builds the user error Status returned by the SPARK_USER_RETURN* macros.
/// Without a message (error details are skipped, e.g. under TRY), no code is
/// attached to keep TRY cheap.
struct SparkUserError {
  const char* errorCode;

  Status operator()() const {
    return Status::UserError();
  }

  Status operator()(std::string message) const {
    return Status::UserError(std::move(message)).withErrorCode(errorCode);
  }
};

} // namespace facebook::velox::functions::sparksql::detail

/// Same as _VELOX_USER_RETURN_IMPL, but sets 'errorCode', one of the
/// spark_error_code constants.
#define _SPARK_USER_RETURN_IMPL(errorCode, expr, exprStr, ...)        \
  _VELOX_RETURN_IMPL(                                                 \
      expr,                                                           \
      exprStr,                                                        \
      ::facebook::velox::functions::sparksql::detail::SparkUserError{ \
          (errorCode).c_str()},                                       \
      ##__VA_ARGS__)

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
