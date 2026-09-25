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

class FromCsvCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  /// Throws a "not supported" user error. The result type of from_csv cannot
  /// be inferred from the argument types alone; callers must supply the
  /// expected ROW type explicitly at call-site resolution.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Constructs the special-form expression for from_csv. Wraps a custom
  /// VectorFunction that parses each input CSV string into a row of the
  /// requested ROW type, returning NULL for malformed or oversized input
  /// (PERMISSIVE mode).
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kFromCsv = "from_csv";
};

} // namespace facebook::velox::functions::sparksql
