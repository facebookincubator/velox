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

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/URLFunctions.h"
#include "velox/functions/sparksql/AnsiMode.h"

namespace facebook::velox::functions::sparksql {

/// url_decode(value) -> varchar
/// url_decode(value, failOnError) -> varchar
///
/// Decodes the URL encoded ``value``. On a malformed escape, the one-argument
/// form and the two-argument form with ``failOnError`` true throw a user error,
/// and the two-argument form with ``failOnError`` false returns NULL. The
/// two-argument form matches Spark's UrlCodec.decode(src, failOnError).
template <typename T>
struct UrlDecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    functions::detail::urlUnescape<out_type<Varchar>, arg_type<Varchar>, true>(
        result, input);
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input,
      bool failOnError) {
    const auto status = functions::detail::
        tryUrlUnescape<out_type<Varchar>, arg_type<Varchar>, true>(
            result, input);
    if (FOLLY_UNLIKELY(!status.ok())) {
      ansiUserFail(failOnError, "{}", status.message());
      return false;
    }
    return true;
  }
};

} // namespace facebook::velox::functions::sparksql
