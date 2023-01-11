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
#include "velox/functions/sparksql/MightContain.h"

#include "velox/common/base/BloomFilter.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/vector/FlatVector.h"

#include <glog/logging.h>

namespace facebook::velox::functions::sparksql {
namespace {
class BloomFilterMightContainFunction final : public exec::VectorFunction {
  bool isDefaultNullBehavior() const final {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 2);
    context.ensureWritable(rows, BOOLEAN(), resultRef);
    auto& result = *resultRef->as<FlatVector<bool>>();
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto serialized = decodedArgs.at(0);
    auto value = decodedArgs.at(1);
    if (serialized->isConstantMapping() && serialized->isNullAt(0)) {
      rows.applyToSelected([&](int row) { result.setNull(row, true); });
      return;
    }

    if (serialized->isConstantMapping()) {
      BloomFilter<int64_t, false> output;
      auto serializedBloom = serialized->valueAt<StringView>(0);
      BloomFilter<int64_t, false>::deserialize(serializedBloom.data(), output);
      rows.applyToSelected([&](int row) {
        result.set(row, output.mayContain(value->valueAt<int64_t>(row)));
      });
      return;
    }

    rows.applyToSelected([&](int row) {
      BloomFilter<int64_t, false> output;
      auto serializedBloom = serialized->valueAt<StringView>(row);
      BloomFilter<int64_t, false>::deserialize(serializedBloom.data(), output);
      result.set(row, output.mayContain(value->valueAt<int64_t>(row)));
    });
  }
};
} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> mightContainSignatures() {
  return {exec::FunctionSignatureBuilder()
              .returnType("boolean")
              .argumentType("varbinary")
              .argumentType("bigint")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeMightContain(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  static const auto kHashFunction =
      std::make_shared<BloomFilterMightContainFunction>();
  return kHashFunction;
}

} // namespace facebook::velox::functions::sparksql