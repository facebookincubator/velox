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

#include "velox/expression/FunctionCallToSpecialForm.h"

namespace facebook::velox::functions::sparksql {

/// named_struct(name1, val1, name2, val2, ...) -> struct
///
/// Creates a struct with the given field names and values. This function takes
/// an even number of arguments where odd-positioned arguments are field names
/// (must be constant VARCHAR expressions) and even-positioned arguments are
/// field values (can be any type).
///
/// Examples:
///   named_struct('id', 123, 'name', 'John') -> {id: 123, name: "John"}
///   named_struct('x', 1.5, 'y', 2.5, 'z', 3.5) -> {x: 1.5, y: 2.5, z: 3.5}
///
/// Field names must be constant VARCHAR expressions. Duplicate field names
/// are allowed (matching Spark behavior) but may cause ambiguity in downstream
/// operations. NULL values are allowed as field values but not as field names.
class NamedStructCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kNamedStruct = "named_struct";
};

} // namespace facebook::velox::functions::sparksql
