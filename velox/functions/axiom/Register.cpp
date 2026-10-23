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

#include "velox/functions/axiom/Register.h"

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"

namespace facebook::velox::functions::axiom {
namespace {

template <typename TExec>
struct BigintAddFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void call(int64_t& out, int64_t a, int64_t b) {
    out = checkedPlus(a, b);
  }
};

template <typename TExec>
struct ArrayAppendFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  bool callNullable(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>* array,
      const arg_type<Generic<T1>>* element) {
    if (array == nullptr) {
      return false;
    }
    out.reserve(array->size() + 1);
    out.add_items(*array);
    element != nullptr ? out.push_back(*element) : out.add_null();
    return true;
  }
};

template <typename TExec>
struct ArrayPrependFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  bool callNullable(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Generic<T1>>* element,
      const arg_type<Array<Generic<T1>>>* array) {
    if (array == nullptr) {
      return false;
    }
    out.reserve(array->size() + 1);
    element != nullptr ? out.push_back(*element) : out.add_null();
    out.add_items(*array);
    return true;
  }
};

} // namespace

void registerFunctions(std::string_view prefix) {
  const std::string name(prefix);
  registerFunction<BigintAddFunction, int64_t, int64_t, int64_t>(
      {name + "bigint_add"});
  registerFunction<
      ArrayAppendFunction,
      Array<Generic<T1>>,
      Array<Generic<T1>>,
      Generic<T1>>({name + "array_append"});
  registerFunction<
      ArrayPrependFunction,
      Array<Generic<T1>>,
      Generic<T1>,
      Array<Generic<T1>>>({name + "array_prepend"});
}

} // namespace facebook::velox::functions::axiom
