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

#include <functions/prestosql/types/JsonType.h>
#include "velox/expression/VectorWriters.h"

namespace facebook::velox::functions::sparksql {
namespace {

class ToJsonFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    JsonCastOperator::get()->castTo(
        *args.at(0), context, rows, outputType, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .returnType("varchar")
            .argumentType("T")
            .build(),
    };
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_to_json,
    ToJsonFunction::signatures(),
    std::make_unique<ToJsonFunction>());

} // namespace facebook::velox::functions::sparksql
