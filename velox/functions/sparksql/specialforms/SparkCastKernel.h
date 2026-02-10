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

#include "velox/expression/PrestoCastKernel.h"

namespace facebook::velox::functions::sparksql {

// This class provides cast hooks following Spark semantics.
class SparkCastKernel : public exec::PrestoCastKernel {
 public:
  explicit SparkCastKernel(
      const velox::core::QueryConfig& config,
      bool allowOverflow);

  VectorPtr castToDate(
      const SelectivityVector& rows,
      const BaseVector& input,
      exec::EvalCtx& context,
      bool setNullInResultAtError) const override;

 private:
  StringView removeWhiteSpaces(const StringView& view) const;

  const core::QueryConfig& config_;

  // If true, the cast will truncate the overflow value to fit the target type.
  const bool allowOverflow_;
};
} // namespace facebook::velox::functions::sparksql
