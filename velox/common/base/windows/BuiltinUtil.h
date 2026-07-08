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

#ifdef _MSC_VER
// Windows/MSVC compatibility for GCC builtin functions

#include <intrin.h>
#include <limits>
#include <type_traits>

// Forward declarations for Int128 types
namespace facebook::velox {
class Int128;
class UInt128;
} // namespace facebook::velox

namespace facebook::velox::windows {

// Type trait to check if T is Int128 or UInt128
template <typename T>
struct is_velox_int128 : std::false_type {};

template <>
struct is_velox_int128<::facebook::velox::Int128> : std::true_type {};

template <>
struct is_velox_int128<::facebook::velox::UInt128> : std::true_type {};

// MSVC implementations of GCC builtin functions
template<typename T>
inline bool builtin_add_overflow(T a, T b, T* result) {
    static_assert(std::is_integral_v<T> || is_velox_int128<T>::value, "T must be an integral type");
    
    if constexpr (is_velox_int128<T>::value) {
        // For Int128/UInt128, perform addition with overflow detection
        *result = a + b;
        // Signed overflow: both operands same sign, result different sign
        return ((a ^ b) >= 0) && ((*result ^ a) < 0);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        // For signed int64_t, detect signed overflow (not unsigned carry)
        *result = static_cast<int64_t>(
            static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
        // Signed overflow: both operands same sign, result different sign
        return ((a ^ b) >= 0) && ((*result ^ a) < 0);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
#if defined(_M_ARM64)
        *result = a + b;
        return *result < a;
#else
        return _addcarry_u64(0, a, b, result);
#endif
    } else if constexpr (std::is_same_v<T, int32_t>) {
        // For signed int32_t, use 64-bit promotion to detect overflow
        int64_t wide = static_cast<int64_t>(a) + static_cast<int64_t>(b);
        *result = static_cast<int32_t>(wide);
        return wide != static_cast<int64_t>(*result);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
#if defined(_M_ARM64)
        *result = a + b;
        return *result < a;
#else
        return _addcarry_u32(0, a, b, result);
#endif
    } else {
        // Fallback for other types
        if constexpr (std::is_signed_v<T>) {
            if ((b > 0 && a > std::numeric_limits<T>::max() - b) ||
                (b < 0 && a < std::numeric_limits<T>::min() - b)) {
                return true; // overflow
            }
        } else {
            if (a > std::numeric_limits<T>::max() - b) {
                return true; // overflow
            }
        }
        *result = a + b;
        return false;
    }
}

template<typename T>
inline bool builtin_sub_overflow(T a, T b, T* result) {
    static_assert(std::is_integral_v<T> || is_velox_int128<T>::value, "T must be an integral type");
    
    if constexpr (is_velox_int128<T>::value) {
        // For Int128/UInt128, perform subtraction with overflow detection
        *result = a - b;
        // Signed overflow: operands different sign, result sign differs from a
        return ((a ^ b) < 0) && ((*result ^ a) < 0);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        // For signed int64_t, detect signed overflow (not unsigned borrow)
        *result = static_cast<int64_t>(
            static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
        // Signed overflow: operands different sign, result sign differs from a
        return ((a ^ b) < 0) && ((*result ^ a) < 0);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
#if defined(_M_ARM64)
        *result = a - b;
        return a < b;
#else
        return _subborrow_u64(0, a, b, result);
#endif
    } else if constexpr (std::is_same_v<T, int32_t>) {
        // For signed int32_t, use 64-bit promotion to detect overflow
        int64_t wide = static_cast<int64_t>(a) - static_cast<int64_t>(b);
        *result = static_cast<int32_t>(wide);
        return wide != static_cast<int64_t>(*result);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
#if defined(_M_ARM64)
        *result = a - b;
        return a < b;
#else
        return _subborrow_u32(0, a, b, result);
#endif
    } else {
        // Fallback for other types
        if constexpr (std::is_signed_v<T>) {
            if ((b > 0 && a < std::numeric_limits<T>::min() + b) ||
                (b < 0 && a > std::numeric_limits<T>::max() + b)) {
                return true; // overflow
            }
        } else {
            if (a < b) {
                return true; // underflow for unsigned
            }
        }
        *result = a - b;
        return false;
    }
}

template<typename T>
inline bool builtin_mul_overflow(T a, T b, T* result) {
    static_assert(std::is_integral_v<T> || is_velox_int128<T>::value, "T must be an integral type");
    
    if constexpr (is_velox_int128<T>::value) {
        // For Int128/UInt128, detect multiplication overflow via back-check
        *result = a * b;
        // If neither is zero, check: result / b == a
        if (b != 0 && *result / b != a) {
            return true;
        }
        // Special case: a * b where both are nonzero and result is zero
        // means overflow if neither operand is zero (can't happen for Int128
        // since we already check result / b == a, so this is covered)
        return false;
    } else if constexpr (std::is_same_v<T, int64_t>) {
#if defined(_M_ARM64)
        *result = a * b;
        __int64 high = __mulh(a, b);
        return high != ((*result < 0) ? -1LL : 0LL);
#else
        __int64 high;
        *result = _mul128(a, b, &high);
        return high != ((*result < 0) ? -1LL : 0LL);
#endif
    } else if constexpr (std::is_same_v<T, uint64_t>) {
#if defined(_M_ARM64)
        *result = a * b;
        return __umulh(a, b) != 0;
#else
        unsigned __int64 high;
        *result = _umul128(a, b, &high);
        return high != 0;
#endif
    } else {
        // Fallback for other types using promotion to larger type
        if constexpr (sizeof(T) < sizeof(int64_t)) {
            using Promoted = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
            Promoted promoted_result = static_cast<Promoted>(a) * static_cast<Promoted>(b);
            
            if (promoted_result > std::numeric_limits<T>::max() || 
                promoted_result < std::numeric_limits<T>::min()) {
                return true; // overflow
            }
            
            *result = static_cast<T>(promoted_result);
            return false;
        } else {
            // For types as large as int64_t, use simple check
            if (a != 0 && b != 0) {
                T max_val = std::numeric_limits<T>::max();
                if constexpr (std::is_signed_v<T>) {
                    T min_val = std::numeric_limits<T>::min();
                    if ((a > 0 && b > 0 && a > max_val / b) ||
                        (a < 0 && b < 0 && a < max_val / b) ||
                        (a > 0 && b < 0 && b < min_val / a) ||
                        (a < 0 && b > 0 && a < min_val / b)) {
                        return true; // overflow
                    }
                } else {
                    if (a > max_val / b) {
                        return true; // overflow
                    }
                }
            }
            *result = a * b;
            return false;
        }
    }
}

// Overload for mixed signed/unsigned types (e.g., int64_t * uint64_t)
template<typename T, typename U>
inline bool builtin_mul_overflow(T a, U b, T* result) {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, 
                  "T and U must be integral types");
    
    // Handle mixed signed/unsigned by converting to the signed type's width
    if constexpr (std::is_signed_v<T> && std::is_unsigned_v<U>) {
        // Check if b fits in the range that won't cause issues
        if (b > static_cast<U>(std::numeric_limits<T>::max())) {
            return true; // Multiplying by a value larger than max signed could overflow
        }
        return builtin_mul_overflow(a, static_cast<T>(b), result);
    } else if constexpr (std::is_unsigned_v<T> && std::is_signed_v<U>) {
        // Check if U is negative
        if (b < 0) {
            return true; // Cannot multiply unsigned by negative without overflow/underflow
        }
        return builtin_mul_overflow(a, static_cast<T>(b), result);
    } else {
        // Both same signedness but different types - convert to result type
        return builtin_mul_overflow(a, static_cast<T>(b), result);
    }
}

// Overload for mixed signed/unsigned types in addition (e.g., int64_t + uint64_t)
template<typename T, typename U>
inline bool builtin_add_overflow(T a, U b, T* result) {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>, 
                  "T and U must be integral types");
    
    // Convert to result type and check
    if constexpr (std::is_signed_v<T> && std::is_unsigned_v<U>) {
        if (b > static_cast<U>(std::numeric_limits<T>::max())) {
            return true; // b is too large to safely add
        }
        return builtin_add_overflow(a, static_cast<T>(b), result);
    } else if constexpr (std::is_unsigned_v<T> && std::is_signed_v<U>) {
        if (b < 0) {
            // Adding a negative to unsigned - check underflow
            if (static_cast<T>(-b) > a) {
                return true;
            }
            *result = a - static_cast<T>(-b);
            return false;
        }
        return builtin_add_overflow(a, static_cast<T>(b), result);
    } else {
        return builtin_add_overflow(a, static_cast<T>(b), result);
    }
}

// Overload for mixed signed/unsigned types in subtraction (e.g., int32_t - int8_t)
template<typename T, typename U>
inline bool builtin_sub_overflow(T a, U b, T* result) {
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>,
                  "T and U must be integral types");

    if constexpr (std::is_signed_v<T> && std::is_unsigned_v<U>) {
        if (b > static_cast<U>(std::numeric_limits<T>::max())) {
            return true; // b is too large to safely subtract
        }
        return builtin_sub_overflow(a, static_cast<T>(b), result);
    } else if constexpr (std::is_unsigned_v<T> && std::is_signed_v<U>) {
        if (b < 0) {
            // Subtracting a negative is addition - check overflow.
            if (static_cast<T>(-b) > std::numeric_limits<T>::max() - a) {
                return true;
            }
            *result = a + static_cast<T>(-b);
            return false;
        }
        return builtin_sub_overflow(a, static_cast<T>(b), result);
    } else {
        return builtin_sub_overflow(a, static_cast<T>(b), result);
    }
}

} // namespace facebook::velox::windows

// Define macros that replace GCC builtins with our implementations
#define __builtin_add_overflow facebook::velox::windows::builtin_add_overflow
#define __builtin_sub_overflow facebook::velox::windows::builtin_sub_overflow  
#define __builtin_mul_overflow facebook::velox::windows::builtin_mul_overflow

#endif // _MSC_VER