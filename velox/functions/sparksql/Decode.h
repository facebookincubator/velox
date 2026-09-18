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

#include "velox/common/base/Status.h"
#include "velox/expression/StringWriter.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {
namespace detail {

Status decode(
    const StringView& input,
    const StringView& charset,
    exec::StringWriter& out);

} // namespace detail

/// Spark decode(binary, charset) returns UTF-8 text using replacement-mode
/// decoding. Supports the canonical names US-ASCII, ISO-8859-1, UTF-8,
/// UTF-16BE, UTF-16LE, UTF-16 and UTF-32 (ASCII case-insensitive).
/// Unsupported names return a user error. Nulls use default null propagation.
template <typename TExec>
struct DecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      const arg_type<Varbinary>& input,
      const arg_type<Varchar>& charset) {
    return detail::decode(input, charset, result);
  }
};

} // namespace facebook::velox::functions::sparksql
