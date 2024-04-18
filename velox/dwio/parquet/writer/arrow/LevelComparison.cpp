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

#include "velox/dwio/parquet/writer/arrow/LevelComparison.h"

#define PARQUET_IMPL_NAMESPACE standard
#include "velox/dwio/parquet/writer/arrow/LevelComparisonInc.h"
#undef PARQUET_IMPL_NAMESPACE

#include <vector>

#include "arrow/util/dispatch.h"

namespace facebook::velox::parquet::arrow::internal {

#if defined(ARROW_HAVE_RUNTIME_AVX2)
MinMax FindMinMaxAvx2(const int16_t* levels, int64_t num_levels);
uint64_t
GreaterThanBitmapAvx2(const int16_t* levels, int64_t num_levels, int16_t rhs);
#endif

namespace {

using ::arrow::internal::DispatchLevel;
using ::arrow::internal::DynamicDispatch;

// defined in level_comparison_avx2.cc

struct GreaterThanDynamicFunction {
  using FunctionType = decltype(&GreaterThanBitmap);

  static std::vector<std::pair<DispatchLevel, FunctionType>> implementations() {
    return {
        {DispatchLevel::NONE, standard::GreaterThanBitmapImpl}
#if defined(ARROW_HAVE_RUNTIME_AVX2)
        ,
        {DispatchLevel::AVX2, GreaterThanBitmapAvx2}
#endif
    };
  }
};

struct MinMaxDynamicFunction {
  using FunctionType = decltype(&FindMinMax);

  static std::vector<std::pair<DispatchLevel, FunctionType>> implementations() {
    return {
        {DispatchLevel::NONE, standard::FindMinMaxImpl}
#if defined(ARROW_HAVE_RUNTIME_AVX2)
        ,
        {DispatchLevel::AVX2, FindMinMaxAvx2}
#endif
    };
  }
};

} // namespace

uint64_t
GreaterThanBitmap(const int16_t* levels, int64_t num_levels, int16_t rhs) {
  static DynamicDispatch<GreaterThanDynamicFunction> dispatch;
  return dispatch.func(levels, num_levels, rhs);
}

MinMax FindMinMax(const int16_t* levels, int64_t num_levels) {
  static DynamicDispatch<MinMaxDynamicFunction> dispatch;
  return dispatch.func(levels, num_levels);
}

} // namespace facebook::velox::parquet::arrow::internal
