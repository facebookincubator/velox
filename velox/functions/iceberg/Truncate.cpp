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

#include "velox/functions/iceberg/Truncate.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions::iceberg {
namespace {

template <typename TExec>
struct TruncateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const int32_t* /*length*/,
      const arg_type<Varchar>* /*input*/) {
    if (inputTypes[1]->kind() == TypeKind::VARBINARY) {
      inputIsVarbinary = true;
    }
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE Status call(T& out, int32_t length, T input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    out = input - ((input % length) + length) % length;
    return Status::OK();
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      int32_t length,
      const arg_type<Varchar>& input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    const auto truncatedLength = inputIsVarbinary
        ? stringImpl::cappedByteLength<true>(input, length)
        : stringImpl::cappedByteLength<false>(input, length);
    result += StringView(input.data(), truncatedLength);
    return Status::OK();
  }

  FOLLY_ALWAYS_INLINE Status callAscii(
      out_type<Varchar>& result,
      int32_t length,
      const arg_type<Varchar>& input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    auto truncatedLength = stringImpl::cappedLength<true>(input, length);
    result += StringView(input.data(), truncatedLength);
    return Status::OK();
  }

 private:
  bool inputIsVarbinary = false;
};
} // namespace

void registerTruncateFunctions(const std::string& prefix) {
  registerFunction<TruncateFunction, int8_t, int32_t, int8_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int16_t, int32_t, int16_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, int32_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int64_t, int32_t, int64_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, Varchar, int32_t, Varchar>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, Varbinary, int32_t, Varbinary>(
      {prefix + "truncate"});
  registerFunction<
      TruncateFunction,
      LongDecimal<P1, S1>,
      int32_t,
      LongDecimal<P1, S1>>({prefix + "truncate"});

  registerFunction<
      TruncateFunction,
      ShortDecimal<P1, S1>,
      int32_t,
      ShortDecimal<P1, S1>>({prefix + "truncate"});
}

} // namespace facebook::velox::functions::iceberg
