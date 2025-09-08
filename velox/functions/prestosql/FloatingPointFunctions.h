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

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/HashCodeUtils.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions {

/// xxhash64(real) → bigint
/// Return a xxhash64 of input Real
template <typename T>
struct XxHash64RealFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<float>& input) {
    // Hash the float value directly
    result = XXH64(&input, sizeof(input), 0);
  }
};

/// xxhash64(double) → bigint
/// Return a xxhash64 of input Double
template <typename T>
struct XxHash64DoubleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<double>& input) {
    // Hash the double value directly
    result = XXH64(&input, sizeof(input), 0);
  }
};

/// hash_code(real) → bigint
template <typename T>
struct HashCodeRealFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<float>& input) {
    result = HashCodeUtils::hashReal(input);
  }
};

/// hash_code(double) → bigint
template <typename T>
struct HashCodeDoubleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE
  void call(out_type<int64_t>& result, const arg_type<double>& input) {
    result = HashCodeUtils::hashDouble(input);
  }
};

} // namespace facebook::velox::functions
