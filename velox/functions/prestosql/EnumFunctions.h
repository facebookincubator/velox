/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"

namespace facebook::velox::functions {

template <typename TExec>
struct EnumKeyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const arg_type<BigintEnum<E1>>* input) {
    VELOX_CHECK(
        inputTypes.size() == 1, "Expected 1 input type for enum_key function.");
    const auto& inputEnum = static_cast<const BigintEnumType&>(*inputTypes[0]);
    enumMap_ = inputEnum.flippedLongEnumMap();
  }

  void call(out_type<Varchar>& result, const arg_type<BigintEnum<E1>>& input) {
    result = BigintEnumType::keyAt(input, enumMap_);
  }

 private:
  std::unordered_map<int64_t, std::string> enumMap_;
};

} // namespace facebook::velox::functions
