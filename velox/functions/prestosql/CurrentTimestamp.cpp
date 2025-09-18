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
#include "CurrentTimestamp.h"
#include "velox/expression/EvalCtx.h"
#include "velox/vector/FlatVector.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::functions {

using namespace facebook::velox;

void CurrentTimestampFunction::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& /* args */,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  // Allocate result vector of the right type and size.
  auto pool = context.pool();
  VectorPtr resultVec = BaseVector::create(outputType, rows.end(), pool);

  // Get a raw FlatVector pointer for writing values.
  auto* flat = resultVec->asFlatVector<int64_t>();

  // Fill only selected rows with the precomputed timestamp.
  rows.applyToSelected([&](vector_size_t row) {
    flat->set(row, packed_);
  });

  // Return the result to Velox by moving the VectorPtr.
  result = std::move(resultVec);
}

std::unique_ptr<exec::VectorFunction> CurrentTimestampFunction::create(
    const std::string& /*name*/,
    const std::vector<TypePtr>& /*argTypes*/,
    const core::QueryConfig& config) {
  auto tzID = tz::getTimeZoneID(config.sessionTimezone());
  Timestamp ts = Timestamp::now();
  int64_t packed = pack(ts, tzID);

  return std::make_unique<CurrentTimestampFunction>(packed);
}

void registerCurrentTimestamp(const std::string& name) {
  registerStatefulVectorFunction(
      name,
      {facebook::velox::functions::exec::FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .build()},
      [](const std::string& /*name*/,
         const std::vector<exec::VectorFunctionArg>& /*args*/,
         const core::QueryConfig& config)
         -> std::shared_ptr<facebook::velox::functions::exec::VectorFunction> {
          std::vector<facebook::velox::TypePtr> argTypes; // no arguments
        return std::shared_ptr<facebook::velox::functions::exec::VectorFunction>(
  CurrentTimestampFunction::create("", argTypes, config).release()
);},
      facebook::velox::functions::exec::VectorFunctionMetadataBuilder()
          .deterministic(false)
          .build(),
      true  // overwrite
  );
}

} // namespace facebook::velox::functions
