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
#include "velox/expression/RegisterSpecialForm.h"
#include "velox/expression/RowConstructor.h"
#include "velox/expression/SpecialFormRegistry.h"
#include "velox/functions/sparksql/specialforms/AtLeastNNonNulls.h"
#include "velox/functions/sparksql/specialforms/DecimalRound.h"
#include "velox/functions/sparksql/specialforms/MakeDecimal.h"
#include "velox/functions/sparksql/specialforms/SparkCastExpr.h"

namespace facebook::velox::functions {
void registerSparkSpecialFormFunctions() {
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_concat_row, exec::RowConstructorCallToSpecialForm::kRowConstructor);
}

namespace sparksql {
void registerSpecialFormGeneralFunctions(const std::string& prefix) {
  exec::registerFunctionCallToSpecialForms();
  exec::registerFunctionCallToSpecialForm(
      MakeDecimalCallToSpecialForm::kMakeDecimal,
      std::make_unique<MakeDecimalCallToSpecialForm>());
  exec::registerFunctionCallToSpecialForm(
      DecimalRoundCallToSpecialForm::kRoundDecimal,
      std::make_unique<DecimalRoundCallToSpecialForm>());
  exec::registerFunctionCallToSpecialForm(
      AtLeastNNonNullsCallToSpecialForm::kAtLeastNNonNulls,
      std::make_unique<AtLeastNNonNullsCallToSpecialForm>());
  registerSparkSpecialFormFunctions();
  registerFunctionCallToSpecialForm(
      "cast", std::make_unique<SparkCastCallToSpecialForm>());
  registerFunctionCallToSpecialForm(
      "try_cast", std::make_unique<SparkTryCastCallToSpecialForm>());
}
} // namespace sparksql
} // namespace facebook::velox::functions
