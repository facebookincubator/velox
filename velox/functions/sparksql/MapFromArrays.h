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

#include <string>

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

/// Implements Spark's map_from_arrays(array(K), array(V)) -> map(K,V). Entries
/// are inserted in array order. A repeated key raises DUPLICATED_MAP_KEY, or
/// under LAST_WIN overwrites the value of its first occurrence in place, so the
/// key keeps that first position.
class MapFromArraysFunction : public exec::VectorFunction {
 public:
  explicit MapFromArraysFunction(bool throwExceptionOnDuplicateMapKeys)
      : throwExceptionOnDuplicateMapKeys_{throwExceptionOnDuplicateMapKeys} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override;

  /// Returns the signatures of map_from_arrays.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures();

 private:
  const bool throwExceptionOnDuplicateMapKeys_;
};

/// Registers Spark's map_from_arrays under the given name.
void registerMapFromArraysFunction(const std::string& name);

} // namespace facebook::velox::functions::sparksql
