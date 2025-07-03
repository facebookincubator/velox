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

#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/iceberg/TruncateFunction.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::iceberg {
namespace {

template <typename T>
struct TruncateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const int32_t* /*length*/,
      const arg_type<Varchar>* /*input*/) {
    if (inputTypes[1]->kind() == TypeKind::VABINARY) {
      inputIsVarbinary = true;
    }
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE Status
  call(int32_t& out, const int32_t& length, const TInput& input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    out = input - ((input % length) + length) % length;
    return Status::OK();
  }

  FOLLY_ALWAYS_INLINE Status
  call(int32_t& out, const int32_t& length, const arg_type<Varchar>& input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    auto truncatedLength = inputisVarbinary
        ? stringImpl::cappedLength<true>(input, length)
        : stringImpl::cappedLength<false>(input, length);
    out = result.setNoCopy(StringView(input.data(), truncatedLength));
    return Status::OK();
  }

  FOLLY_ALWAYS_INLINE Status callAscii(
      int32_t& out,
      const int32_t& length,
      const arg_type<Varchar>& input) {
    VELOX_RETURN_IF(
        length <= 0,
        Status::UserError("Invalid truncate width: {} (must be > 0)", length));
    auto truncatedLength = stringImpl::cappedLength<true>(input, length);
    out = result.setNoCopy(StringView(input.data(), truncatedLength));
    return Status::OK();
  }

 private:
  bool inputIsVarbinary = false;
};
} // namespace

void registerTruncateFunctions(const std::string& prefix) {
  registerFunction<TruncateFunction, int32_t, int32_t, int8_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, int16_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, int32_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, int64_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, int128_t>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, Varchar>(
      {prefix + "truncate"});
  registerFunction<TruncateFunction, int32_t, int32_t, Varbinary>(
      {prefix + "truncate"});
}

} // namespace facebook::velox::functions::iceberg
