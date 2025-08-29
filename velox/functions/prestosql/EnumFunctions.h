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

#include "velox/common/base/Status.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"
#include "velox/functions/prestosql/types/VarcharEnumType.h"

namespace facebook::velox::functions {

template <typename TExec>
struct EnumKeyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const arg_type<BigintEnum<E1>>* /*input*/) {
    VELOX_USER_CHECK_EQ(
        inputTypes.size(), 1, "Expected 1 input type for enum_key function.");
    enumPtr_ = inputTypes[0];
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const arg_type<VarcharEnum<E1>>* /*input*/) {
    VELOX_USER_CHECK_EQ(
        inputTypes.size(), 1, "Expected 1 input type for enum_key function.");
    enumPtr_ = inputTypes[0];
  }

  Status call(out_type<Varchar>& result, const int64_t& input) {
    auto bigintEnumPtr = asBigintEnum(enumPtr_);
    VELOX_USER_CHECK_NOT_NULL(
        bigintEnumPtr,
        "Input type for enum_key function must be a BigintEnumType.");
    auto keyAt = bigintEnumPtr->keyAt(input);
    if (!keyAt.has_value()) {
      return Status::UserError("Value '{}' not in enum 'BigintEnum'", input);
    }
    result = keyAt.value();
    return Status::OK();
  }

  Status call(out_type<Varchar>& result, const std::string& input) {
    auto varcharEnumPtr = asVarcharEnum(enumPtr_);
    VELOX_USER_CHECK_NOT_NULL(
        varcharEnumPtr,
        "Input type for enum_key function must be a VarcharEnumType.");
    auto keyAt = varcharEnumPtr->keyAt(input);
    if (!keyAt.has_value()) {
      return Status::UserError("Value '{}' not in enum 'VarcharEnum'", input);
    }
    result = keyAt.value();
    return Status::OK();
  }

 private:
  TypePtr enumPtr_;
};

} // namespace facebook::velox::functions
