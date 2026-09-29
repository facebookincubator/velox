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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Bitwise.h"

namespace facebook::velox::functions {
namespace {
template <template <class> class T>
void registerBitwiseBinaryIntegral(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner) {
  registerFunction<T, int64_t, int8_t, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int16_t, int16_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int32_t, int32_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int64_t, int64_t>(
      aliases, {}, true, defaultOwner);
}

template <template <class> class T>
void registerBitwiseUnaryIntegral(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner) {
  registerFunction<T, int64_t, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int16_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int32_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int64_t>(aliases, {}, true, defaultOwner);
}

template <template <class> class T>
void registerShift(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner) {
  registerFunction<T, int8_t, int8_t, int32_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int16_t, int16_t, int32_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int32_t, int32_t, int32_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int64_t, int32_t>(
      aliases, {}, true, defaultOwner);
}
} // namespace

void registerBitwiseFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerBitwiseBinaryIntegral<BitwiseAndFunction>(
      {prefix + "bitwise_and"}, defaultOwner);
  registerBitwiseUnaryIntegral<BitwiseNotFunction>(
      {prefix + "bitwise_not"}, defaultOwner);
  registerBitwiseBinaryIntegral<BitwiseOrFunction>(
      {prefix + "bitwise_or"}, defaultOwner);
  registerBitwiseBinaryIntegral<BitwiseXorFunction>(
      {prefix + "bitwise_xor"}, defaultOwner);
  registerBitwiseBinaryIntegral<BitCountFunction>(
      {prefix + "bit_count"}, defaultOwner);
  registerFunction<
      BitwiseArithmeticShiftRightFunction,
      int64_t,
      int64_t,
      int64_t>(
      {prefix + "bitwise_arithmetic_shift_right"}, {}, true, defaultOwner);
  registerShift<BitwiseLeftShiftFunction>(
      {prefix + "bitwise_left_shift"}, defaultOwner);
  registerShift<BitwiseRightShiftFunction>(
      {prefix + "bitwise_right_shift"}, defaultOwner);
  registerShift<BitwiseRightShiftArithmeticFunction>(
      {prefix + "bitwise_right_shift_arithmetic"}, defaultOwner);
  registerFunction<
      BitwiseLogicalShiftRightFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>(
      {prefix + "bitwise_logical_shift_right"}, {}, true, defaultOwner);
  registerFunction<
      BitwiseShiftLeftFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_shift_left"}, {}, true, defaultOwner);
}

} // namespace facebook::velox::functions
