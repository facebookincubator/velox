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

#include <cstdint>

namespace torch::wave {

// Casts __half and __nv_bfloat16 to float for arithmetic so that mixed-type
// expressions (e.g. int32_t * __half) do not hit ambiguous operator overloads
// from cuda_fp16.h / cuda_bf16.h.
template <typename T>
__device__ inline T arithCast(T x) {
  return x;
}
__device__ inline float arithCast(__half x) {
  return __half2float(x);
}
__device__ inline float arithCast(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Binary arithmetic.
//
// The return type is T, the FIRST operand's type, because GraphOptimizer's
// promotion pass casts every tensor operand (and the output) to PyTorch's
// promoted dtype before codegen. So a1 already carries the op's output dtype,
// and returning T stores the result at that dtype. The other operands get
// their own template types (T2, TAlpha) because they need not be tensors of
// that dtype: the second operand of a *.Scalar op, and the alpha of add/sub,
// are scalar constants emitted as C++ literals that the promotion pass does not
// unify. They are folded in via arithCast + ordinary C++ promotion inside the
// expression, then the result converts back to T on return.
template <typename T, typename T2, typename TAlpha>
__device__ inline T __add(T a1, T2 a2, TAlpha alpha) {
  return arithCast(a1) + arithCast(a2) * arithCast(alpha);
}

template <typename T, typename T2, typename TAlpha>
__device__ inline T __sub(T a1, T2 a2, TAlpha alpha) {
  return arithCast(a1) - arithCast(a2) * arithCast(alpha);
}

template <typename T, typename T2 = T>
__device__ inline T __mul(T a1, T2 a2) {
  return arithCast(a1) * arithCast(a2);
}

template <>
__device__ inline bool __mul<bool, bool>(bool a1, bool a2) {
  return a1 & a2;
}

template <typename T, typename T2 = T>
__device__ inline T __div(T a1, T2 a2) {
  return arithCast(a1) / arithCast(a2);
}

// Integer true division matching PyTorch semantics: div(int, int) -> double.
__device__ inline double __div(int64_t a1, int64_t a2) {
  return static_cast<double>(a1) / static_cast<double>(a2);
}

__device__ inline double __div(int32_t a1, int32_t a2) {
  return static_cast<double>(a1) / static_cast<double>(a2);
}

// Python/_operator.floordiv on integers: floor(a / b), rounding toward
// negative infinity (C++ integer division truncates toward zero, so adjust
// when the remainder is nonzero and the operand signs differ).
template <typename T, typename T2 = T>
__device__ inline int64_t __floordiv(T a1, T2 a2) {
  int64_t a = static_cast<int64_t>(a1);
  int64_t b = static_cast<int64_t>(a2);
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) {
    --q;
  }
  return q;
}

// Two-arg add/sub for scalar Python _operator.add / _operator.sub, which have
// no alpha (unlike aten.add/aten.sub). Unlike the tensor ops above there is no
// promotion pass and no anchoring tensor, so the result takes the deduced
// promoted type of the operands -- exactly what plain `a1 + a2` yields -- which
// is correct for SymInt, SymFloat, and mixed int/float operands. Forcing a
// fixed type (e.g. int64) here would truncate a SymFloat operand.
template <typename T, typename T2 = T>
__device__ inline auto __opadd(T a1, T2 a2) {
  return arithCast(a1) + arithCast(a2);
}

template <typename T, typename T2 = T>
__device__ inline auto __opsub(T a1, T2 a2) {
  return arithCast(a1) - arithCast(a2);
}

// PyTorch remainder: result = a - floor(a/b) * b, same sign as divisor.
// C++ % truncates toward zero (sign of dividend), so we adjust.
template <typename T, typename T2 = T>
__device__ inline T __remainder(T a1, T2 a2) {
  T r = a1 % a2;
  if (r != 0 && ((r ^ a2) < 0)) {
    r += a2;
  }
  return r;
}

__device__ inline float __remainder(float a1, float a2) {
  float r = fmodf(a1, a2);
  if (r != 0.0f && ((r < 0) != (a2 < 0))) {
    r += a2;
  }
  return r;
}

__device__ inline double __remainder(double a1, double a2) {
  double r = fmod(a1, a2);
  if (r != 0.0 && ((r < 0) != (a2 < 0))) {
    r += a2;
  }
  return r;
}

template <typename T, typename T2 = T>
__device__ inline T __fmod(T a1, T2 a2) {
  return a1 % a2;
}

__device__ inline float __fmod(float a1, float a2) {
  return fmodf(a1, a2);
}

__device__ inline double __fmod(double a1, double a2) {
  return fmod(a1, a2);
}

template <typename T, typename T2 = T>
__device__ inline T __pow(T a1, T2 a2) {
  return powf(static_cast<float>(a1), static_cast<float>(a2));
}

__device__ inline float __pow(float a1, float a2) {
  return powf(a1, a2);
}

__device__ inline double __pow(double a1, double a2) {
  return pow(a1, a2);
}

// Comparison.

template <typename T, typename T2 = T>
__device__ inline bool __eq(T a1, T2 a2) {
  return a1 == a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __ne(T a1, T2 a2) {
  return a1 != a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __lt(T a1, T2 a2) {
  return a1 < a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __le(T a1, T2 a2) {
  return a1 <= a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __gt(T a1, T2 a2) {
  return a1 > a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __ge(T a1, T2 a2) {
  return a1 >= a2;
}

// Bitwise.

template <typename T>
__device__ inline T __bitwise_and(T a1, T a2) {
  return a1 & a2;
}

template <typename T>
__device__ inline T __bitwise_or(T a1, T a2) {
  return a1 | a2;
}

template <typename T>
__device__ inline T __bitwise_xor(T a1, T a2) {
  return a1 ^ a2;
}

template <typename T>
__device__ inline T __bitwise_not(T a1) {
  return ~a1;
}

template <>
__device__ inline bool __bitwise_not<bool>(bool a1) {
  return !a1;
}

// Logical.

template <typename T>
__device__ inline bool __logical_and(T a1, T a2) {
  return static_cast<bool>(a1) && static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_or(T a1, T a2) {
  return static_cast<bool>(a1) || static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_xor(T a1, T a2) {
  return static_cast<bool>(a1) != static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_not(T a1) {
  return !static_cast<bool>(a1);
}

// Unary math.

template <typename T>
__device__ inline T __abs(T a1) {
  return abs(a1);
}

__device__ inline float __abs(float a1) {
  return fabsf(a1);
}

__device__ inline double __abs(double a1) {
  return fabs(a1);
}

template <typename T>
__device__ inline T __neg(T a1) {
  return -a1;
}

__device__ inline float __ceil(float a1) {
  return ceilf(a1);
}

__device__ inline double __ceil(double a1) {
  return ceil(a1);
}

__device__ inline float __floor(float a1) {
  return floorf(a1);
}

__device__ inline double __floor(double a1) {
  return floor(a1);
}

__device__ inline float __round(float a1) {
  return roundf(a1);
}

__device__ inline double __round(double a1) {
  return round(a1);
}

__device__ inline float __trunc(float a1) {
  return truncf(a1);
}

__device__ inline double __trunc(double a1) {
  return trunc(a1);
}

template <typename T>
__device__ inline T __sign(T a1) {
  return (a1 > T(0)) - (a1 < T(0));
}

__device__ inline float __sqrt(float a1) {
  return sqrtf(a1);
}

__device__ inline double __sqrt(double a1) {
  return sqrt(a1);
}

__device__ inline float __rsqrt(float a1) {
  return rsqrtf(a1);
}

__device__ inline double __rsqrt(double a1) {
  return 1.0 / sqrt(a1);
}

__device__ inline float __reciprocal(float a1) {
  return 1.0f / a1;
}

__device__ inline double __reciprocal(double a1) {
  return 1.0 / a1;
}

__device__ inline float __exp(float a1) {
  return expf(a1);
}

__device__ inline double __exp(double a1) {
  return exp(a1);
}

__device__ inline float __log(float a1) {
  return logf(a1);
}

__device__ inline double __log(double a1) {
  return log(a1);
}

__device__ inline float __log2(float a1) {
  return log2f(a1);
}

__device__ inline double __log2(double a1) {
  return log2(a1);
}

__device__ inline float __log10(float a1) {
  return log10f(a1);
}

__device__ inline double __log10(double a1) {
  return log10(a1);
}

__device__ inline float __log1p(float a1) {
  return log1pf(a1);
}

__device__ inline double __log1p(double a1) {
  return log1p(a1);
}

// Trigonometric.

__device__ inline float __sin(float a1) {
  return sinf(a1);
}

__device__ inline double __sin(double a1) {
  return sin(a1);
}

__device__ inline float __cos(float a1) {
  return cosf(a1);
}

__device__ inline double __cos(double a1) {
  return cos(a1);
}

__device__ inline float __tan(float a1) {
  return tanf(a1);
}

__device__ inline double __tan(double a1) {
  return tan(a1);
}

__device__ inline float __asin(float a1) {
  return asinf(a1);
}

__device__ inline double __asin(double a1) {
  return asin(a1);
}

__device__ inline float __acos(float a1) {
  return acosf(a1);
}

__device__ inline double __acos(double a1) {
  return acos(a1);
}

__device__ inline float __atan(float a1) {
  return atanf(a1);
}

__device__ inline double __atan(double a1) {
  return atan(a1);
}

__device__ inline float __atan2(float a1, float a2) {
  return atan2f(a1, a2);
}

__device__ inline double __atan2(double a1, double a2) {
  return atan2(a1, a2);
}

__device__ inline float __sinh(float a1) {
  return sinhf(a1);
}

__device__ inline double __sinh(double a1) {
  return sinh(a1);
}

__device__ inline float __cosh(float a1) {
  return coshf(a1);
}

__device__ inline double __cosh(double a1) {
  return cosh(a1);
}

__device__ inline float __tanh(float a1) {
  return tanhf(a1);
}

__device__ inline double __tanh(double a1) {
  return tanh(a1);
}

// Activation functions.

template <typename T>
__device__ inline T __relu(T a1) {
  return a1 > T(0) ? a1 : T(0);
}

__device__ inline float __sigmoid(float a1) {
  return 1.0f / (1.0f + expf(-a1));
}

__device__ inline double __sigmoid(double a1) {
  return 1.0 / (1.0 + exp(-a1));
}

// logit (inverse sigmoid): log(z / (1 - z)). The optional aten eps clamps z to
// [eps, 1 - eps]; a negative eps (set by resolveLogitDefault when eps is None)
// means no clamp. Non-floating inputs are not meaningful for logit; the
// template fallback is only there so codegen compiles for any T.
template <typename T>
__device__ inline T __logit(T a1, double /*eps*/) {
  return a1;
}

__device__ inline float __logit(float a1, double eps) {
  float z = a1;
  if (eps >= 0.0) {
    float e = static_cast<float>(eps);
    z = fminf(fmaxf(a1, e), 1.0f - e);
  }
  return logf(z / (1.0f - z));
}

__device__ inline double __logit(double a1, double eps) {
  double z = a1;
  if (eps >= 0.0) {
    z = fmin(fmax(a1, eps), 1.0 - eps);
  }
  return log(z / (1.0 - z));
}

template <
    bool kHasMin = true,
    bool kHasMax = true,
    typename T,
    typename TLo = T,
    typename THi = T>
__device__ inline T __clamp(T a1, TLo lo, THi hi) {
  T result = a1;
  if constexpr (kHasMin) {
    result = result < static_cast<T>(lo) ? static_cast<T>(lo) : result;
  }
  if constexpr (kHasMax) {
    result = result > static_cast<T>(hi) ? static_cast<T>(hi) : result;
  }
  return result;
}

// In-place variants. The mutated argument is bound by reference so the write
// lands directly in the self tensor's storage (the codegen passes
// storage<T>(self)[idx], an lvalue). PyTorch preserves these ops in
// non-functionalized graphs (e.g. feature-transform normalization:
// x.add_(mean).mul_(inv_std), x[:, :k].clamp_(lo, hi)), so torchwave must
// execute them. The computation reuses the functional helper and assigns the
// result back to self (keeping self's dtype, matching in-place semantics); the
// returned value also flows to the result register when the output is used.

template <typename T, typename T2, typename TAlpha>
__device__ inline T __add_(T& a1, T2 a2, TAlpha alpha) {
  a1 = __add<T, T2, TAlpha>(a1, a2, alpha);
  return a1;
}

template <typename T, typename T2, typename TAlpha>
__device__ inline T __sub_(T& a1, T2 a2, TAlpha alpha) {
  a1 = __sub<T, T2, TAlpha>(a1, a2, alpha);
  return a1;
}

template <typename T, typename T2 = T>
__device__ inline T __mul_(T& a1, T2 a2) {
  a1 = __mul<T, T2>(a1, a2);
  return a1;
}

template <typename T, typename T2 = T>
__device__ inline T __div_(T& a1, T2 a2) {
  a1 = static_cast<T>(__div<T, T2>(a1, a2));
  return a1;
}

template <
    bool kHasMin = true,
    bool kHasMax = true,
    typename T,
    typename TLo = T,
    typename THi = T>
__device__ inline T __clamp_(T& a1, TLo lo, THi hi) {
  a1 = __clamp<kHasMin, kHasMax, T, TLo, THi>(a1, lo, hi);
  return a1;
}

// NaN/Inf replacement.

template <typename T>
__device__ inline T
__nan_to_num(T a1, double nan_val, double posinf_val, double neginf_val) {
  return a1;
}

__device__ inline float
__nan_to_num(float a1, double nan_val, double posinf_val, double neginf_val) {
  if (isnan(a1)) {
    return static_cast<float>(nan_val);
  }
  if (isinf(a1)) {
    return a1 > 0 ? static_cast<float>(posinf_val)
                  : static_cast<float>(neginf_val);
  }
  return a1;
}

__device__ inline double
__nan_to_num(double a1, double nan_val, double posinf_val, double neginf_val) {
  if (isnan(a1)) {
    return nan_val;
  }
  if (isinf(a1)) {
    return a1 > 0 ? posinf_val : neginf_val;
  }
  return a1;
}

// Min/max.

template <typename T>
__device__ inline T __minimum(T a1, T a2) {
  return a1 < a2 ? a1 : a2;
}

template <typename T, typename U>
__device__ inline auto __minimum(T a1, U a2) -> decltype(a1 + a2) {
  return a1 < a2 ? a1 : a2;
}

template <typename T>
__device__ inline T __maximum(T a1, T a2) {
  return a1 > a2 ? a1 : a2;
}

template <typename T, typename U>
__device__ inline auto __maximum(T a1, U a2) -> decltype(a1 + a2) {
  return a1 > a2 ? a1 : a2;
}

// Item.

template <typename T>
__device__ inline T __item(Tensor* self) {
  return *storage<T>(self);
}

// Where.

template <typename T>
__device__ inline T __where(bool c, T x, T y) {
  return c ? x : y;
}

// Zero / One.

template <typename T>
__device__ inline T __zero() {
  return T(0);
}

template <typename T>
__device__ inline T __one() {
  return T(1);
}

template <typename T, typename U>
__device__ inline T __one(U /*ignore*/) {
  return T(1);
}

// Arange.

template <typename T>
__device__ inline T __arange_start(int64_t idx, T start) {
  return idx + start;
}

template <typename T>
__device__ inline T __arange(int64_t idx) {
  return __arange_start<T>(idx, T(0));
}

// Shape query.

__device__ inline int64_t __sym_size(Tensor* self, int64_t dim) {
  return self->dims[dim < 0 ? dim + self->rank : dim];
}

__device__ inline int64_t __numel(Tensor* self) {
  return self->numEl;
}

// Index gather: returns source[indices[0][i]] for 1D indexing.
// Takes a TensorList with a single index tensor.
template <typename T, typename TIdx>
__device__ inline T __index1d(Tensor* source, TIdx index) {
  return storage<T>(source)[index * source->strides[0]];
}

// Elementwise index_put variants with scalar indices in registers.
template <typename T>
__device__ inline T __index_put_elt_one(
    Tensor* dest,
    int32_t idx0,
    T value,
    bool accumulate,
    BlockInfo& block) {
  if (idx0 >= 0 && idx0 < dest->dims[0]) {
    auto* dst = storage<T>(dest);
    auto offset = indexOffset(dest, idx0);
    if (accumulate) {
      dst[offset] += value;
    } else {
      dst[offset] = value;
    }
  } else if (block.debugInfo) {
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = 0;
    block.debugInfo->extra[1] = idx0;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

template <typename T>
__device__ inline T __index_put_elt_two(
    Tensor* dest,
    int32_t idx0,
    int32_t idx1,
    T value,
    bool accumulate,
    BlockInfo& block) {
  if (idx0 >= 0 && idx0 < dest->dims[0] && idx1 >= 0 && idx1 < dest->dims[1]) {
    auto* dst = storage<T>(dest);
    auto offset = indexOffset(dest, idx0, idx1);
    if (accumulate) {
      dst[offset] += value;
    } else {
      dst[offset] = value;
    }
  } else if (block.debugInfo) {
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = idx0 < 0 || idx0 >= dest->dims[0] ? 0 : 1;
    block.debugInfo->extra[1] = idx0 < 0 || idx0 >= dest->dims[0] ? idx0 : idx1;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

template <typename T>
__device__ inline T __index_put_elt_three(
    Tensor* dest,
    int32_t idx0,
    int32_t idx1,
    int32_t idx2,
    T value,
    bool accumulate,
    BlockInfo& block) {
  if (idx0 >= 0 && idx0 < dest->dims[0] && idx1 >= 0 && idx1 < dest->dims[1] &&
      idx2 >= 0 && idx2 < dest->dims[2]) {
    auto* dst = storage<T>(dest);
    auto offset = indexOffset(dest, idx0, idx1, idx2);
    if (accumulate) {
      dst[offset] += value;
    } else {
      dst[offset] = value;
    }
  } else if (block.debugInfo) {
    int32_t dim = idx0 < 0 || idx0 >= dest->dims[0] ? 0
        : idx1 < 0 || idx1 >= dest->dims[1]         ? 1
                                                    : 2;
    int32_t badIdx = dim == 0 ? idx0 : (dim == 1 ? idx1 : idx2);
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = dim;
    block.debugInfo->extra[1] = badIdx;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

// Fused slice_scatter along 'dim' (functional/out-of-place). The elementwise
// loop iterates every element of the output (which has self's shape), so 'idx'
// is the row-major offset of one output element and the device function returns
// that element's value. Positions inside the slice [start, start + len*step)
// along 'dim' (stride 'step') come from 'src'; all other positions pass through
// 'self' unchanged. 'len' (the slice length) is taken from src, which is
// authoritative. Both 'self' and 'src' are read at computed offsets through
// their own strides, so non-contiguous operands (e.g. a column-slice view) are
// handled; both are whole tensors and must be materialized before this op (the
// randomAccess argument flag enforces that, as for the index-gather source).
// The output is contiguous, so 'idx' decomposes directly into per-dim
// coordinates via self's dims. A slice that does not fit in 'self' along 'dim'
// (an out-of-range start/end, possibly injected) is reported like
// __index_put_elt_*.
template <typename T>
__device__ inline T __slice_scatter(
    uint32_t idx,
    Tensor* self,
    Tensor* src,
    int32_t dim,
    int32_t start,
    int32_t /*end*/,
    int32_t step,
    BlockInfo& block) {
  int64_t dimSize = self->dims[dim];
  int64_t len = src->dims[dim];
  int64_t st = static_cast<int64_t>(step) > 0 ? static_cast<int64_t>(step) : 1;
  // The whole slice must fit in 'self' along 'dim'; otherwise report and pass
  // self through unchanged. An empty slice is a no-op (all positions pass
  // through).
  int64_t lastPos = static_cast<int64_t>(start) + (len - 1) * st;
  if (len >= 1 && (start < 0 || lastPos >= dimSize) && block.debugInfo) {
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = dim;
    block.debugInfo->extra[1] = start;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  // Decompose the contiguous output offset 'idx' into per-dim coordinates using
  // the output dims (== self dims), innermost dim first.
  int64_t coord[kMaxDims];
  int64_t rem = static_cast<int64_t>(idx);
  for (int d = self->rank - 1; d >= 0; --d) {
    coord[d] = rem % self->dims[d];
    rem /= self->dims[d];
  }
  int64_t pos = coord[dim];
  int64_t rel = pos - static_cast<int64_t>(start);
  bool inSlice = len >= 1 && start >= 0 && lastPos < dimSize && rel >= 0 &&
      rel < len * st && rel % st == 0;
  if (inSlice) {
    int64_t sliceJ = rel / st;
    int64_t srcOff = 0;
    for (int d = 0; d < src->rank; ++d) {
      int64_t c = d == dim ? sliceJ : coord[d];
      srcOff += c * src->strides[d];
    }
    return storage<T>(src)[srcOff];
  }
  int64_t selfOff = 0;
  for (int d = 0; d < self->rank; ++d) {
    selfOff += coord[d] * self->strides[d];
  }
  return storage<T>(self)[selfOff];
}

// Elementwise index gather variants with scalar indices in registers.
template <typename T>
__device__ inline T
__index_elt_one(Tensor* source, int32_t idx0, BlockInfo& block) {
  if (idx0 >= 0 && idx0 < source->dims[0]) {
    return storage<T>(source)[indexOffset(source, idx0)];
  }
  if (block.debugInfo) {
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = 0;
    block.debugInfo->extra[1] = idx0;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

// Like __index_elt_one but returns 'deflt' for an out-of-range index instead of
// flagging an error. 'deflt' has its own template type U (a second type
// template parameter after the source element type T) so a scalar constant of a
// possibly different type is converted to the element type T on return.
template <typename T, typename U>
__device__ inline T
__index_elt_one_default(Tensor* source, int32_t idx0, U deflt) {
  if (idx0 >= 0 && idx0 < source->dims[0]) {
    return storage<T>(source)[indexOffset(source, idx0)];
  }
  return static_cast<T>(deflt);
}

template <typename T>
__device__ inline T
__index_elt_two(Tensor* source, int32_t idx0, int32_t idx1, BlockInfo& block) {
  if (idx0 >= 0 && idx0 < source->dims[0] && idx1 >= 0 &&
      idx1 < source->dims[1]) {
    return storage<T>(source)[indexOffset(source, idx0, idx1)];
  }
  if (block.debugInfo) {
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = idx0 < 0 || idx0 >= source->dims[0] ? 0 : 1;
    block.debugInfo->extra[1] =
        idx0 < 0 || idx0 >= source->dims[0] ? idx0 : idx1;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

template <typename T>
__device__ inline T __index_elt_three(
    Tensor* source,
    int32_t idx0,
    int32_t idx1,
    int32_t idx2,
    BlockInfo& block) {
  if (idx0 >= 0 && idx0 < source->dims[0] && idx1 >= 0 &&
      idx1 < source->dims[1] && idx2 >= 0 && idx2 < source->dims[2]) {
    return storage<T>(source)[indexOffset(source, idx0, idx1, idx2)];
  }
  if (block.debugInfo) {
    int32_t dim = idx0 < 0 || idx0 >= source->dims[0] ? 0
        : idx1 < 0 || idx1 >= source->dims[1]         ? 1
                                                      : 2;
    int32_t badIdx = dim == 0 ? idx0 : (dim == 1 ? idx1 : idx2);
    block.debugInfo->line = __LINE__;
    block.debugInfo->extra[0] = dim;
    block.debugInfo->extra[1] = badIdx;
    SET_MSG(block.debugInfo, "Bad idx\0");
  }
  return T();
}

// Fused index gather from TensorList. Supports 1D to kMaxDims indexing.
// Each index tensor selects a coordinate along its dimension. The linear
// offset is sum(idx[dim] * source->strides[dim]).
template <typename T>
__device__ void __indexgather(
    Tensor* source,
    TensorList* indices,
    Tensor* output,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    for (int32_t dim = 0; dim < indices->size; ++dim) {
      auto* t = indices->tensors[dim];
      assert(
          (t->elementType == kScalarTypeInt ||
           t->elementType == kScalarTypeLong) &&
          "index tensor must be int or long");
    }
  }
  __syncthreads();
  auto n = indices->tensors[0]->numEl;
  auto* src = storage<T>(source);
  auto* dst = storage<T>(output);
  for (uint32_t i = block.blockInOp * blockDim.x + threadIdx.x; i < n;
       i += block.numBlocksInOp * blockDim.x) {
    int32_t offset = 0;
    bool valid = true;
    for (int32_t dim = 0; dim < indices->size; ++dim) {
      auto* idxTensor = indices->tensors[dim];
      int32_t idx;
      if (idxTensor->elementType == kScalarTypeLong) {
        idx = static_cast<int32_t>(storage<int64_t>(idxTensor)[i]);
      } else {
        idx = storage<int32_t>(idxTensor)[i];
      }
      if (idx < 0 || idx >= source->dims[dim]) {
        valid = false;
        if (block.debugInfo) {
          block.debugInfo->line = __LINE__;
          block.debugInfo->extra[0] = dim;
          block.debugInfo->extra[1] = idx;
          SET_MSG(block.debugInfo, "Bad idx\0");
        }
        break;
      }
      offset += idx * source->strides[dim];
    }
    if (valid) {
      dst[i] = src[offset];
    }
  }
}

// Fused elementwise index_select along 'dim' (functional/out-of-place). The
// enclosing elementwise loop iterates every element of the whole expression's
// output 'out'; 'idx' is the logical row-major index of one output element. The
// index_select result has 'source's shape with dimension 'dim' resized to the
// index length; that result broadcasts to 'out'. Decompose idx into per-dim
// coordinates using out's dims, right-align the result within out
// ('dimOffset'), broadcast any size-1 result dimension to coordinate 0, replace
// the coordinate along 'dim' with the selected index value, and read 'source'
// at the resulting offset. This mirrors ATen's advanced-indexing kernel:
//   offset = sum(coord[d] * source->strides[d], d != dim)
//          + index[coord[dim]] * source->strides[dim]
// (see ATen native/cuda/IndexKernel.cu). 'source' and 'index' are whole tensors
// read at computed offsets (so they carry randomAccess and are materialized
// before this op). An out-of-range index is reported like __index_elt_*.
template <typename T, int32_t kDim>
__device__ inline T __index_select(
    uint32_t idx,
    Tensor* source,
    Tensor* index,
    Tensor* out,
    BlockInfo& block) {
  int32_t dim = kDim < 0 ? kDim + source->rank : kDim;
  // Decompose the logical output index into per-dim coordinates using the
  // enclosing expression's output shape (innermost dim first).
  int32_t coord[kMaxDims];
  uint32_t rem = idx;
  for (int d = out->rank - 1; d >= 0; --d) {
    auto dimSize = static_cast<uint32_t>(out->dims[d]);
    coord[d] = static_cast<int32_t>(rem % dimSize);
    rem /= dimSize;
  }
  int32_t dimOffset = out->rank - source->rank;
  // int64 offset (like ATen's IndexKernel) so coord*stride accumulation does
  // not overflow for large tensors.
  int64_t offset = 0;
  int32_t pos = 0;
  for (int d = 0; d < source->rank; ++d) {
    int32_t oDimSize = d == dim ? index->dims[0] : source->dims[d];
    int32_t c = oDimSize == 1 ? 0 : coord[d + dimOffset];
    if (d == dim) {
      pos = c;
    } else {
      offset += static_cast<int64_t>(c) * source->strides[d];
    }
  }
  int64_t selected = index->elementType == kScalarTypeLong
      ? storage<int64_t>(index)[pos * index->strides[0]]
      : static_cast<int64_t>(storage<int32_t>(index)[pos * index->strides[0]]);
  if (selected < 0) {
    selected += source->dims[dim];
  }
  if (selected < 0 || selected >= source->dims[dim]) {
    if (block.debugInfo) {
      block.debugInfo->line = __LINE__;
      block.debugInfo->extra[0] = dim;
      block.debugInfo->extra[1] = selected;
      SET_MSG(block.debugInfo, "Bad idx\0");
    }
    return T();
  }
  offset += selected * source->strides[dim];
  return storage<T>(source)[offset];
}

} // namespace torch::wave
