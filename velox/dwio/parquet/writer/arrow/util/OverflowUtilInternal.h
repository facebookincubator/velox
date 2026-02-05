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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

// "Safe-math.h" includes <intsafe.h> from the Windows headers.
#include "arrow/util/windows_compatibility.h"
// #Include "arrow/vendored/portable-snippets/safe-math.h".
#include "velox/dwio/parquet/writer/arrow/util/safe-math.h"
// clang-format off (avoid include reordering)
#include "arrow/util/windows_fixup.h"
// clang-format on.

namespace arrow {
namespace internal {

// Define functions AddWithOverflow, SubtractWithOverflow, MultiplyWithOverflow.
// With the signature `bool(T u, T v, T* out)` where T is an integer type.
// On overflow, these functions return true.  Otherwise, false is returned.
// And `out` is updated with the result of the operation.

#define OP_WITH_OVERFLOW(funcName, psnipOp, Type, PsnipType)             \
  [[nodiscard]] static inline bool funcName(Type u, Type v, Type* out) { \
    return !psnipSafe_##PsnipType##_##psnipOp(out, u, v);                \
  }

#define OPS_WITH_OVERFLOW(funcName, psnipOp)            \
  OP_WITH_OVERFLOW(funcName, psnipOp, int8_t, int8)     \
  OP_WITH_OVERFLOW(funcName, psnipOp, int16_t, int16)   \
  OP_WITH_OVERFLOW(funcName, psnipOp, int32_t, int32)   \
  OP_WITH_OVERFLOW(funcName, psnipOp, int64_t, int64)   \
  OP_WITH_OVERFLOW(funcName, psnipOp, uint8_t, uint8)   \
  OP_WITH_OVERFLOW(funcName, psnipOp, uint16_t, uint16) \
  OP_WITH_OVERFLOW(funcName, psnipOp, uint32_t, uint32) \
  OP_WITH_OVERFLOW(funcName, psnipOp, uint64_t, uint64)

OPS_WITH_OVERFLOW(addWithOverflow, add)
OPS_WITH_OVERFLOW(SubtractWithOverflow, sub)
OPS_WITH_OVERFLOW(multiplyWithOverflow, mul)
OPS_WITH_OVERFLOW(DivideWithOverflow, div)

#undef OP_WITH_OVERFLOW
#undef OPS_WITH_OVERFLOW

// Define function NegateWithOverflow with the signature `bool(T u, T* out)`.
// Where T is a signed integer type.  On overflow, these functions return true.
// Otherwise, false is returned and `out` is updated with the result of the.
// Operation.

#define UNARY_OP_WITH_OVERFLOW(funcName, psnipOp, Type, PsnipType) \
  [[nodiscard]] static inline bool funcName(Type u, Type* out) {   \
    return !psnipSafe_##PsnipType##_##psnipOp(out, u);             \
  }

#define SIGNED_UNARY_OPS_WITH_OVERFLOW(funcName, psnipOp)   \
  UNARY_OP_WITH_OVERFLOW(funcName, psnipOp, int8_t, int8)   \
  UNARY_OP_WITH_OVERFLOW(funcName, psnipOp, int16_t, int16) \
  UNARY_OP_WITH_OVERFLOW(funcName, psnipOp, int32_t, int32) \
  UNARY_OP_WITH_OVERFLOW(funcName, psnipOp, int64_t, int64)

SIGNED_UNARY_OPS_WITH_OVERFLOW(NegateWithOverflow, neg)

#undef UNARY_OP_WITH_OVERFLOW
#undef SIGNED_UNARY_OPS_WITH_OVERFLOW

/// Signed addition with well-defined behaviour on overflow (as unsigned)
template <typename SignedInt>
SignedInt safeSignedAdd(SignedInt u, SignedInt v) {
  using UnsignedInt = typename std::make_unsigned<SignedInt>::type;
  return static_cast<SignedInt>(
      static_cast<UnsignedInt>(u) + static_cast<UnsignedInt>(v));
}

/// Signed subtraction with well-defined behaviour on overflow (as unsigned)
template <typename SignedInt>
SignedInt safeSignedSubtract(SignedInt u, SignedInt v) {
  using UnsignedInt = typename std::make_unsigned<SignedInt>::type;
  return static_cast<SignedInt>(
      static_cast<UnsignedInt>(u) - static_cast<UnsignedInt>(v));
}

/// Signed negation with well-defined behaviour on overflow (as unsigned)
template <typename SignedInt>
SignedInt safeSignedNegate(SignedInt u) {
  using UnsignedInt = typename std::make_unsigned<SignedInt>::type;
  return static_cast<SignedInt>(~static_cast<UnsignedInt>(u) + 1);
}

/// Signed left shift with well-defined behaviour on negative numbers or.
/// Overflow.
template <typename SignedInt, typename Shift>
SignedInt safeLeftShift(SignedInt u, Shift shift) {
  using UnsignedInt = typename std::make_unsigned<SignedInt>::type;
  return static_cast<SignedInt>(static_cast<UnsignedInt>(u) << shift);
}

} // namespace internal
} // namespace arrow
