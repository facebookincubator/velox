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

#include "velox/functions/iceberg/BucketFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/iceberg/util/Murmur3_32HashFunction.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::iceberg {
namespace {

FOLLY_ALWAYS_INLINE int apply(int numBuckets, int hashedValue) {
  return (hashedValue & INT_MAX) % numBuckets;
}

template <typename T>
struct BucketDecimalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(int32_t& out, const int32_t& numBuckets, const TInput& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto length = DecimalUtil::getByteArrayLength(input);
    char bytes[length];
    DecimalUtil::toByteArray(input, bytes);
    const auto hash = util::Murmur3_32HashFunction::hashString(bytes, length);
    out = apply(numBuckets, hash);
  }
};

template <typename T>
struct BucketFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(int32_t& out, const int32_t& numBuckets, const TInput& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto hash = util::Murmur3_32HashFunction::hashBigint(input);
    out = apply(numBuckets, hash);
  }

  FOLLY_ALWAYS_INLINE void call(
      int32_t& out,
      const int32_t& numBuckets,
      const arg_type<Varchar>& input) {
    VELOX_USER_CHECK_NE(numBuckets, 0, "Remainder cannot be zero");
    const auto hash =
        util::Murmur3_32HashFunction::hashString(input.data(), input.size());
    out = apply(numBuckets, hash);
  }

  FOLLY_ALWAYS_INLINE void call(
      int32_t& out,
      const int32_t& numBuckets,
      const arg_type<Timestamp>& input) {
    const auto hash =
        util::Murmur3_32HashFunction::hashBigint(input.toMicros());
    out = apply(numBuckets, hash);
  }
};
} // namespace

void registerBucketFunctions(const std::string& prefix) {
  registerFunction<BucketFunction, int32_t, int32_t, int32_t>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, int64_t>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Varchar>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Date>({prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Timestamp>(
      {prefix + "bucket"});
  registerFunction<BucketFunction, int32_t, int32_t, Varbinary>(
      {prefix + "bucket"});

  registerFunction<
      BucketDecimalFunction,
      int32_t,
      int32_t,
      LongDecimal<P1, S1>>({prefix + "bucket"});

  registerFunction<
      BucketDecimalFunction,
      int32_t,
      int32_t,
      ShortDecimal<P1, S1>>({prefix + "bucket"});
}

} // namespace facebook::velox::functions::iceberg
