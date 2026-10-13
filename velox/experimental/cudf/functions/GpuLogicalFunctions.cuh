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

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/functions/GpuVariadicView.h"

#include "velox/common/base/Macros.h"
#include "velox/functions/Macros.h"
#include "velox/type/SimpleFunctionApi.h"

/// AND, OR, NOT and IS NULL for GPU SFI.
///
/// On the CPU these are special forms; here they are ordinary functions over
/// every row, so there is nothing to short-circuit. They follow Kleene logic:
/// one false makes a conjunction false and one true makes a disjunction true,
/// even beside nulls, so the result can be non-null when an input is null.
/// That is why they use callNullable(). AND and OR take a variadic pack because
/// the expression tree flattens `a AND b AND c` into one call.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// Conjunction: false if any term is false, else null if any is null, else
/// true.
template <typename T>
struct GpuAndFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE bool callNullable(
      bool& result,
      const arg_type<Variadic<bool>> terms) {
    bool sawNull = false;
    for (int32_t i = 0; i < terms.size(); ++i) {
      const auto term = terms.at(i);
      if (!term.has_value()) {
        sawNull = true;
      } else if (!term.value()) {
        result = false;
        return true;
      }
    }
    if (sawNull) {
      return false;
    }
    result = true;
    return true;
  }
};

/// Disjunction: true if any term is true, else null if any is null, else false.
template <typename T>
struct GpuOrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE bool callNullable(
      bool& result,
      const arg_type<Variadic<bool>> terms) {
    bool sawNull = false;
    for (int32_t i = 0; i < terms.size(); ++i) {
      const auto term = terms.at(i);
      if (!term.has_value()) {
        sawNull = true;
      } else if (term.value()) {
        result = true;
        return true;
      }
    }
    if (sawNull) {
      return false;
    }
    result = false;
    return true;
  }
};

/// Negation. A null input yields a null result.
template <typename T>
struct GpuNotFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE bool callNullable(bool& result, const bool* value) {
    if (value == nullptr) {
      return false;
    }
    result = !*value;
    return true;
  }
};

/// Never returns null.
template <typename T>
struct GpuIsNullFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE bool callNullable(bool& result, const bool* value) {
    result = value == nullptr;
    return true;
  }
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
