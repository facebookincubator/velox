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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions {

class CurrentTimestampFunction : public exec::VectorFunction {
public:
  explicit CurrentTimestampFunction(int64_t packed) : packed_(packed) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& /*args*/,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override;

  static std::unique_ptr<exec::VectorFunction> create(
    const std::string& name,
    const std::vector<TypePtr>& inputTypes,
    const core::QueryConfig& config);

private:
  const int64_t packed_;
};

/// Register the function with Velox
void registerCurrentTimestamp(const std::string& name);

} // namespace facebook::velox::functions
