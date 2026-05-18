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

#include "velox/functions/sparksql/aggregates/BitmapOrAggAggregate.h"

#include <cstring>

#include "velox/common/base/BitUtil.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/aggregates/BitmapConstructAggAggregate.h"

namespace facebook::velox::functions::aggregate::sparksql {

namespace {

class BitmapOrAggAggregate {
 public:
  using InputType = Row<Varbinary>;

  using IntermediateType = Varbinary;

  using OutputType = Varbinary;

  // Non-default null behavior: the aggregate always produces a non-null result
  // (an all-zeros bitmap) even when all inputs are null or the group is empty.
  static constexpr bool default_null_behavior_ = false;

  struct AccumulatorType {
    uint8_t* data_{nullptr};
    HashStringAllocator::Header* header_{nullptr};

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        BitmapOrAggAggregate* /*fn*/) {}

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Varbinary> input) {
      if (!input.has_value()) {
        return false;
      }
      auto bitmap = input.value();
      VELOX_USER_CHECK_EQ(
          bitmap.size(),
          kBitmapNumBytes,
          "bitmap_or_agg expects exactly {} byte bitmaps, got {}",
          kBitmapNumBytes,
          bitmap.size());
      init(allocator);
      bits::orBits(
          reinterpret_cast<uint64_t*>(data_),
          reinterpret_cast<const uint64_t*>(bitmap.data()),
          0,
          kBitmapNumBits);
      return true;
    }

    bool combine(
        HashStringAllocator* allocator,
        exec::optional_arg_type<IntermediateType> other) {
      if (!other.has_value()) {
        return false;
      }
      auto serialized = other.value();
      VELOX_CHECK_EQ(
          serialized.size(),
          kBitmapNumBytes,
          "Unexpected intermediate bitmap size");
      init(allocator);
      bits::orBits(
          reinterpret_cast<uint64_t*>(data_),
          reinterpret_cast<const uint64_t*>(serialized.data()),
          0,
          kBitmapNumBits);
      return true;
    }

    bool writeIntermediateResult(
        bool /*nonNullGroup*/,
        exec::out_type<IntermediateType>& out) {
      return writeResult(out);
    }

    bool writeFinalResult(
        bool /*nonNullGroup*/,
        exec::out_type<OutputType>& out) {
      return writeResult(out);
    }

    void destroy(HashStringAllocator* allocator) {
      if (header_) {
        allocator->free(header_);
        header_ = nullptr;
        data_ = nullptr;
      }
    }

   private:
    bool writeResult(exec::out_type<Varbinary>& out) {
      out.resize(kBitmapNumBytes);
      if (data_) {
        std::memcpy(out.data(), data_, kBitmapNumBytes);
      } else {
        std::memset(out.data(), 0, kBitmapNumBytes);
      }
      return true;
    }

    void init(HashStringAllocator* allocator) {
      if (!data_) {
        header_ = allocator->allocate(kBitmapNumBytes);
        data_ = reinterpret_cast<uint8_t*>(header_->begin());
        std::memset(data_, 0, kBitmapNumBytes);
      }
    }
  };
};

} // namespace

exec::AggregateRegistrationResult registerBitmapOrAggAggregate(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .argumentType("varbinary")
          .intermediateType("varbinary")
          .returnType("varbinary")
          .build()};

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 1, "{} takes exactly one argument", name);
        return std::make_unique<
            exec::SimpleAggregateAdapter<BitmapOrAggAggregate>>(
            step, argTypes, resultType, &config);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
