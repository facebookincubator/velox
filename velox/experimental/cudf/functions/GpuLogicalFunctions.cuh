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

#include "velox/common/base/Macros.h"
#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/functions/GpuVariadicView.h"
#include "velox/functions/Macros.h"
#include "velox/type/SimpleFunctionApi.h"

/// AND, OR, NOT and IS NULL for GPU SFI.
///
/// On the CPU these are special forms rather than registered functions: AND and
/// OR compile to ConjunctExpr, which short-circuits and reorders its terms by
/// measured selectivity, and IS NULL compiles to a special form so it can look
/// at an input's null mask without evaluating it as a value. None of that
/// machinery is reused here, and none of it needs to be -- a kernel evaluates
/// every row of every term regardless, so short-circuiting has nothing to skip
/// and term reordering nothing to gain.
///
/// What does carry over exactly is the three-valued truth table, and that is
/// the part worth being careful about. NULL in Kleene logic is "unknown", not
/// "false": a single false makes a conjunction false no matter how many unknowns
/// sit beside it, and a single true makes a disjunction true. So the result is
/// non-null even when an input is null, which is precisely the case default null
/// behaviour gets wrong -- hence callNullable() rather than call() throughout.
///
/// AND and OR take a variadic pack because that is how they arrive: the
/// expression tree flattens a chain, so `a AND b AND c` is one three-argument
/// call rather than nested pairs.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// Conjunction: false if any term is false, else null if any is null, else true.
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

/// Negation. Unknown negates to unknown, so a null input yields a null result --
/// which is also what default null behaviour would do, but stated explicitly
/// here so the three functions read alike.
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

/// The one function here whose output is never null: asking whether a value is
/// absent always has an answer.
template <typename T>
struct GpuIsNullFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE bool callNullable(bool& result, const bool* value) {
    result = value == nullptr;
    return true;
  }
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
