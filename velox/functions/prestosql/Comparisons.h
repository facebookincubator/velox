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

#include "velox/common/base/CompareFlags.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {
#define VELOX_GEN_BINARY_EXPR(Name, Expr, TResult)                \
  template <typename T>                                           \
  struct Name {                                                   \
    VELOX_DEFINE_FUNCTION_TYPES(T);                               \
    template <typename TInput>                                    \
    FOLLY_ALWAYS_INLINE bool                                      \
    call(TResult& result, const TInput& lhs, const TInput& rhs) { \
      result = (Expr);                                            \
      return true;                                                \
    }                                                             \
  };

VELOX_GEN_BINARY_EXPR(NeqFunction, lhs != rhs, bool);
VELOX_GEN_BINARY_EXPR(LtFunction, lhs < rhs, bool);
VELOX_GEN_BINARY_EXPR(GtFunction, lhs > rhs, bool);
VELOX_GEN_BINARY_EXPR(LteFunction, lhs <= rhs, bool);
VELOX_GEN_BINARY_EXPR(GteFunction, lhs >= rhs, bool);

template <typename T>
struct DistinctFromFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(bool& result, const TInput& lhs, const TInput& rhs) {
    result = (lhs != rhs); // Return true if distinct.
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  callNullable(bool& result, const TInput* lhs, const TInput* rhs) {
    if (!lhs and !rhs) { // Both nulls -> not distinct -> false.
      result = false;
    } else if (!lhs or !rhs) { // Only one is null -> distinct -> true.
      result = true;
    } else { // Both not nulls - use usual comparison.
      call(result, *lhs, *rhs);
    }
  }
};

#undef VELOX_GEN_BINARY_EXPR

template <typename T>
struct EqFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Used for primitive inputs.
  template <typename TInput>
  void call(bool& out, const TInput& lhs, const TInput& rhs) {
    out = (lhs == rhs);
  }

  // For arbitrary nested complex types. Can return null.
  bool call(
      bool& out,
      const arg_type<Generic<T1>>& lhs,
      const arg_type<Generic<T1>>& rhs) {
    static constexpr CompareFlags kFlags = {
        false, false, /*euqalsOnly*/ true, true /*stopAtNull*/};
    auto result = lhs.compare(rhs, kFlags);
    if (!result.has_value()) {
      return false;
    }
    out = (result.value() == 0);
    return true;
  }
};

namespace {

template <typename Arch, typename ComparisonOp>
struct SimdComparator {
  template <typename T, bool isConstant>
  inline auto loadSimdData(const T* rawData, vector_size_t offset) {
    using d_type = xsimd::batch<T, Arch>;
    if constexpr (isConstant) {
      return xsimd::broadcast<T, Arch>(rawData[0]);
    }
    return d_type::load_unaligned(rawData + offset);
  }

  template <typename T, bool isLeftConstant, bool isRightConstant>
  void applySimdComparison(
      vector_size_t begin,
      vector_size_t end,
      const T* rawLhs,
      const T* rawRhs,
      uint8_t* rawResult) {
    using d_type = xsimd::batch<T, Arch>;
    constexpr auto numScalarElements = d_type::size;
    const auto vectorEnd = (end - begin) - (end - begin) % numScalarElements;

    if constexpr (numScalarElements == 2 || numScalarElements == 4) {
      for (auto i = begin; i < vectorEnd; i += 8) {
        rawResult[i / 8] = 0;
        for (auto j = 0; j < 8 && j < vectorEnd; j += numScalarElements) {
          auto left = loadSimdData<T, isLeftConstant>(rawLhs, i + j);
          auto right = loadSimdData<T, isRightConstant>(rawRhs, i + j);

          uint8_t res = simd::toBitMask(ComparisonOp()(left, right));
          rawResult[i / 8] |= res << j;
        }
      }
    } else {
      for (auto i = begin; i < vectorEnd; i += numScalarElements) {
        auto left = loadSimdData<T, isLeftConstant>(rawLhs, i);
        auto right = loadSimdData<T, isRightConstant>(rawRhs, i);

        auto res = simd::toBitMask(ComparisonOp()(left, right));
        if constexpr (numScalarElements == 8) {
          rawResult[i / 8] = res;
        } else if constexpr (numScalarElements == 16) {
          uint16_t* addr = reinterpret_cast<uint16_t*>(rawResult + i / 8);
          *addr = res;
        } else if constexpr (numScalarElements == 32) {
          uint32_t* addr = reinterpret_cast<uint32_t*>(rawResult + i / 8);
          *addr = res;
        } else {
          VELOX_FAIL("Unsupported number of scalar elements");
        }
      }
    }

    // Evaluate remaining values.
    for (auto i = vectorEnd; i < end; i++) {
      if constexpr (isRightConstant) {
        bits::setBit(rawResult, i, ComparisonOp()(rawLhs[i], rawRhs[0]));
      } else if constexpr (isLeftConstant) {
        bits::setBit(rawResult, i, ComparisonOp()(rawLhs[0], rawRhs[i]));
      } else {
        bits::setBit(rawResult, i, ComparisonOp()(rawLhs[i], rawRhs[i]));
      }
    }
  }

  template <
      TypeKind kind,
      typename std::enable_if_t<
          xsimd::has_simd_register<
              typename TypeTraits<kind>::NativeType>::value,
          int> = 0>
  void applyComparison(
      const SelectivityVector& rows,
      DecodedVector& lhs,
      DecodedVector& rhs,
      exec::EvalCtx* context,
      VectorPtr* result) {
    using T = typename TypeTraits<kind>::NativeType;

    auto rawRhs = rhs.template data<T>();
    auto rawLhs = lhs.template data<T>();
    auto rawResult =
        (*result)->asUnchecked<FlatVector<bool>>()->mutableRawValues<uint8_t>();

    auto isSimdizable = lhs.isIdentityMapping() && rhs.isIdentityMapping() &&
        rows.isAllSelected();

    if (!isSimdizable) {
      auto lhsIndices = lhs.indices();
      auto rhsIndices = rhs.indices();

      if (rhs.isConstantMapping()) {
        context->template applyToSelectedNoThrow(rows, [&](auto row) {
          bits::setBit(
              rawResult,
              row,
              ComparisonOp()(rawLhs[lhsIndices[row]], rawRhs[rhsIndices[0]]));
        });
      } else if (lhs.isConstantMapping()) {
        context->template applyToSelectedNoThrow(rows, [&](auto row) {
          bits::setBit(
              rawResult,
              row,
              ComparisonOp()(rawLhs[lhsIndices[0]], rawRhs[rhsIndices[row]]));
        });
      } else {
        context->template applyToSelectedNoThrow(rows, [&](auto row) {
          bits::setBit(
              rawResult,
              row,
              ComparisonOp()(rawLhs[lhsIndices[row]], rawRhs[rhsIndices[row]]));
        });
      }
      return;
    }

    if (lhs.isConstantMapping()) {
      applySimdComparison<T, true, false>(
          rows.begin(), rows.end(), rawLhs, rawRhs, rawResult);
    } else if (rhs.isConstantMapping()) {
      applySimdComparison<T, false, true>(
          rows.begin(), rows.end(), rawLhs, rawRhs, rawResult);
    } else {
      applySimdComparison<T, false, false>(
          rows.begin(), rows.end(), rawLhs, rawRhs, rawResult);
    }
  }

  template <
      TypeKind kind,
      typename std::enable_if_t<
          !xsimd::has_simd_register<
              typename TypeTraits<kind>::NativeType>::value,
          int> = 0>
  void applyComparison(
      const SelectivityVector& rows,
      DecodedVector& lhs,
      DecodedVector& rhs,
      exec::EvalCtx* context,
      VectorPtr* result) {
    VELOX_FAIL("Unsupported type for SIMD comparison");
  }
};

template <typename Arch, typename ComparisonOp>
class ComparisonSimdFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "Comparison requires two arguments");
    VELOX_CHECK_EQ(args[0]->typeKind(), args[1]->typeKind());
    VELOX_CHECK_EQ(outputType, BOOLEAN());

    BaseVector::ensureWritable(rows, BOOLEAN(), context->pool(), result);

    exec::LocalDecodedVector lhs(context, *args[0], rows);
    exec::LocalDecodedVector rhs(context, *args[1], rows);
    auto comparator = SimdComparator<Arch, ComparisonOp>{};

    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        comparator.template applyComparison,
        args[0]->typeKind(),
        rows,
        *lhs.get(),
        *rhs.get(),
        context,
        result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;

    for (const auto& inputType :
         {"tinyint", "smallint", "integer", "bigint", "real", "double"}) {
      signatures.push_back(exec::FunctionSignatureBuilder()
                               .returnType("boolean")
                               .argumentType(inputType)
                               .argumentType(inputType)
                               .build());
    }

    return signatures;
  }
};

} // namespace

template <typename T>
struct BetweenFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(
      bool& result,
      const TInput& value,
      const TInput& low,
      const TInput& high) {
    result = value >= low && value <= high;
    return true;
  }
};

#if XSIMD_WITH_AVX2

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_simd_comparison_eq,
    (ComparisonSimdFunction<xsimd::avx2, std::equal_to<>>::signatures()),
    (std::make_unique<ComparisonSimdFunction<xsimd::avx2, std::equal_to<>>>()));

#elif XSIMD_WITH_SSE2

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_simd_comparison_eq,
    (ComparisonSimdFunction<xsimd::sse2, std::equal_to<>>::signatures()),
    (std::make_unique<ComparisonSimdFunction<xsimd::sse2, std::equal_to<>>>()));

#elif SIMD_WITH_NEON

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_simd_comparison_eq,
    (ComparisonSimdFunction<xsimd::neon, std::equal_to<>>::signatures()),
    (std::make_unique<ComparisonSimdFunction<xsimd::neon, std::equal_to<>>>()));

#endif

} // namespace facebook::velox::functions
