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
#include "velox/functions/prestosql/TimeIntervalYearMonth.h"

#include "velox/expression/VectorFunction.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {
namespace {

// Optimized vector function for Time +/- IntervalYearMonth
// This case is special because result = time (identity function), allowing
// for significant optimizations.
// For plus: supports both (time, interval) and (interval, time)
// For minus: only supports (time, interval) - not (interval, time)
class TimeIntervalYearMonthVectorFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VectorPtr& timeVector = args[0]->type()->isTime() ? args[0] : args[1];
    VELOX_DCHECK(timeVector->type()->equivalent(*TIME()));

    // Constant vector case
    // If time input is constant, create constant result - no iteration!
    if (timeVector->isConstantEncoding()) {
      auto constantTime = timeVector->as<ConstantVector<int64_t>>();
      if (constantTime->isNullAt(0)) {
        result = BaseVector::createNullConstant(
            outputType, rows.size(), context.pool());
      } else {
        auto value = constantTime->valueAt(0);
        result = BaseVector::createConstant(
            outputType, value, rows.size(), context.pool());
      }
      return;
    }

    // Single reference case
    // If input vector is singly referenced, reuse it directly - zero copy!
    if (!result && BaseVector::isVectorWritable(timeVector)) {
      result = std::move(timeVector); // Move input to result - zero copy!
      return;
    }

    // FALLBACK: Standard processing for other cases
    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->asFlatVector<int64_t>();
    // Fast path for flat vectors - use FlatVector::copy for efficient copying
    flatResult->copy(timeVector.get(), rows, nullptr);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>>
  signaturesPlus() {
    return {
        // Signature 1: (time, interval year to month) -> time
        exec::FunctionSignatureBuilder()
            .returnType("time")
            .argumentType("time")
            .argumentType("interval year to month")
            .build(),
        // Signature 2: (interval year to month, time) -> time
        exec::FunctionSignatureBuilder()
            .returnType("time")
            .argumentType("interval year to month")
            .argumentType("time")
            .build()};
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>>
  signaturesMinus() {
    return {// Only support: (time, interval year to month) -> time
            exec::FunctionSignatureBuilder()
                .returnType("time")
                .argumentType("time")
                .argumentType("interval year to month")
                .build()};
  }
};

} // namespace

void registerTimePlusIntervalYearMonth(
    std::string_view name,
    std::string_view defaultOwner) {
  exec::registerVectorFunction(
      name,
      TimeIntervalYearMonthVectorFunction::signaturesPlus(),
      std::make_unique<TimeIntervalYearMonthVectorFunction>(),
      {},
      /*overwrite=*/true,
      defaultOwner);
}

void registerTimeMinusIntervalYearMonth(
    std::string_view name,
    std::string_view defaultOwner) {
  exec::registerVectorFunction(
      name,
      TimeIntervalYearMonthVectorFunction::signaturesMinus(),
      std::make_unique<TimeIntervalYearMonthVectorFunction>(),
      {},
      /*overwrite=*/true,
      defaultOwner);
}

} // namespace facebook::velox::functions
