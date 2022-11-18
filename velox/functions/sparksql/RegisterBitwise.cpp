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
#include "velox/functions/sparksql/Bitwise.h"
#include "velox/functions/sparksql/RegisterBitwise.h"

#include "velox/functions/lib/RegistrationHelpers.h"

namespace facebook::velox::functions::sparksql {

namespace {
template <template <class> class T>
void registerBitwiseBinaryIntegral(const std::vector<std::string>& aliases) {
  // Use left input type as result type.
  registerFunction<T, int8_t, int8_t, int8_t>(aliases);
  registerFunction<T, int16_t, int16_t, int16_t>(aliases);
  registerFunction<T, int32_t, int32_t, int32_t>(aliases);
  registerFunction<T, int64_t, int64_t, int64_t>(aliases);
}
} // namespace

void registerBitwiseFunctions(const std::string& prefix) {
  registerBitwiseBinaryIntegral<BitwiseAndFunction>({prefix + "bitwise_and"});
  registerBitwiseBinaryIntegral<BitwiseOrFunction>({prefix + "bitwise_or"});
}

} // namespace facebook::velox::functions::sparksql
