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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

namespace facebook::velox::cudf_velox::sparksql {

/// Evaluates numeric spark_legacy_cast calls with non-ANSI overflow semantics.
class LegacyCastFunction : public CudfFunction {
 public:
  explicit LegacyCastFunction(const core::TypedExprPtr& expr);

  /// Accepts only numeric conversions whose Spark semantics are implemented.
  /// Literal inputs are unsupported; constant-encoded columns are supported.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  /// Casts a column, preserving nulls and returning null on decimal overflow.
  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // Includes the negative scale for cuDF fixed-point output types.
  cudf::data_type targetType_;

  // Decimal precision is not represented by cudf::data_type.
  int32_t targetPrecision_;

  // Numeric conversions requiring Spark rounding, saturation or overflow
  // checks.
  bool needsSparkKernel_;
};

} // namespace facebook::velox::cudf_velox::sparksql
