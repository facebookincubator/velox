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

namespace facebook::velox::functions::sparksql {
namespace {
template <typename Allocator = std::allocator<uint64_t>>
class BloomFilterMightContainFunction final : public exec::VectorFunction {
 public:
  explicit BloomFilterMightContainFunction(BloomFilter<Allocator> bloomFilter)
      : bloomFilter_(std::move(bloomFilter)) {}

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
    auto value = decodedArgs.at(1);
    uint64_t hashCodes[64];
    bool contains[64];
    if (!bloomFilter_.isSet()) {
      rows.applyToSelected([&](int row) { result.set(row, false); });
    } else {
      rows.applyToSelected(
          [&](int row) {
            auto contain = bloomFilter_.mayContain(
                folly::hasher<int64_t>()(value->valueAt<int64_t>(row)));
            result.set(row, contain);
          },
          [&](int begin, int end) {
            VELOX_CHECK_EQ(end - begin, 64);
            for (int i = begin; i < end; i++) {
              hashCodes[i - begin] =
                  folly::hasher<int64_t>()(value->valueAt<int64_t>(i));
            }
            bloomFilter_.mayContainForBatch(hashCodes, contains, end - begin);
            for (int i = begin; i < end; i++) {
              result.set(i, contains[i - begin]);
            }
          });
    }
  }

 private:
  BloomFilter<Allocator> bloomFilter_;
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
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  BaseVector* serialized = inputArgs[0].constantValue.get();
  if (!serialized) {
    VELOX_USER_FAIL(
        "{} requires first argument to be a constant of type VARBINARY", name);
  }
  BloomFilter bloomFilter;
  if (!serialized->isNullAt(0)) {
    bloomFilter.merge(
        serialized->as<ConstantVector<StringView>>()->valueAt(0).str().c_str());
  }
  return std::make_shared<BloomFilterMightContainFunction<>>(
      std::move(bloomFilter));
}

} // namespace facebook::velox::functions::sparksql
