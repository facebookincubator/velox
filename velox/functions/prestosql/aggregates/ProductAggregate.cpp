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
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

template <typename T, typename U>
class ProductAggregate {
 public:
  using InputType = Row<T>;

  using IntermediateType = U;

  using OutputType = U;

  struct AccumulatorType {
    U product_{1};

    AccumulatorType() = delete;

    explicit AccumulatorType(HashStringAllocator* /*allocator*/) {}

    void addInput(HashStringAllocator* /*allocator*/, exec::arg_type<T> value) {
      product_ *= value;
    }

    void combine(HashStringAllocator* /*allocator*/, exec::arg_type<U> other) {
      product_ *= other;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      out = product_;
      return true;
    }

    bool writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      out = product_;
      return true;
    }
  };

  static bool toIntermediate(exec::out_type<U>& out, exec::arg_type<T> in) {
    out = in;
    return true;
  }
};

} // namespace

void registerProductAggregate(const std::string& prefix) {
  const std::string name = prefix + kProduct;

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  for (const auto& inputType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType("bigint")
                             .intermediateType("bigint")
                             .argumentType(inputType)
                             .build());
  }

  for (const auto& inputType : {"real", "double"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType("double")
                             .intermediateType("double")
                             .argumentType(inputType)
                             .build());
  }

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        auto inputType = argTypes[0];
        if (exec::isRawInput(step)) {
          switch (inputType->kind()) {
            case TypeKind::TINYINT:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<int8_t, int64_t>>>(resultType);
            case TypeKind::SMALLINT:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<int16_t, int64_t>>>(resultType);
            case TypeKind::INTEGER:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<int32_t, int64_t>>>(resultType);
            case TypeKind::BIGINT:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<int64_t, int64_t>>>(resultType);
            case TypeKind::REAL:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<float, double>>>(resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<double, double>>>(resultType);
            default:
              VELOX_FAIL(
                  "Unknown input type for {} aggregation {}",
                  name,
                  inputType->kindName());
          }
        } else {
          switch (resultType->kind()) {
            case TypeKind::BIGINT:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<int64_t, int64_t>>>(resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  ProductAggregate<double, double>>>(resultType);
            default:
              VELOX_FAIL(
                  "Unsupported result type for final aggregation: {}",
                  resultType->kindName());
          }
        }
      });
}

} // namespace facebook::velox::aggregate::prestosql
