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

#include "velox/functions/prestosql/types/P4HyperLogLogRegistration.h"

#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/P4HyperLogLogType.h"
#include "velox/functions/prestosql/types/fuzzer_utils/P4HyperLogLogInputGenerator.h"

namespace facebook::velox {
namespace {

class P4HyperLogLogCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    return other->equivalent(*HYPERLOGLOG());
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return other->equivalent(*HYPERLOGLOG());
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& /*resultType*/,
      VectorPtr& result) const override {
    context.ensureWritable(rows, P4HYPERLOGLOG(), result);

    if (input.type()->equivalent(*HYPERLOGLOG())) {
      // Cast from HYPERLOGLOG to P4HYPERLOGLOG - sparse to dense conversion
      castFromHyperLogLog(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to P4HyperLogLog not supported",
          input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->equivalent(*HYPERLOGLOG())) {
      // Cast to HYPERLOGLOG - direct copy (already dense)
      directCopy(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from P4HyperLogLog to {} not supported",
          resultType->toString());
    }
  }

 private:
  // Direct copy for P4HYPERLOGLOG to HYPERLOGLOG conversion.
  // No conversion needed since P4HyperLogLog is already in dense format.
  void directCopy(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* inputVector = input.as<SimpleVector<StringView>>();

    flatResult->acquireSharedStringBuffers(&input);
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto data = inputVector->valueAt(row);
      flatResult->setNoCopy(row, data);
    });
  }

  // The meaningful conversion: HYPERLOGLOG to P4HYPERLOGLOG (sparse to dense)
  static void castFromHyperLogLog(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* hllInput = input.as<SimpleVector<StringView>>();

    memory::MemoryPool* memoryPool = context.pool();
    using SparseHll = common::hll::SparseHll<memory::MemoryPool>;
    using DenseHll = common::hll::DenseHll<memory::MemoryPool>;
    using common::hll::DenseHlls;
    using common::hll::SparseHlls;

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto hllData = hllInput->valueAt(row);
      exec::StringWriter writer(flatResult, row);

      if (hllData.size() == 0 || hllData.data() == nullptr) {
        writer.resize(0);
        writer.finalize();
        return;
      }

      if (SparseHlls::canDeserialize(hllData.data())) {
        // Input is sparse - convert to dense
        int8_t indexBitLength =
            SparseHlls::deserializeIndexBitLength(hllData.data());

        SparseHll sparseHll(hllData.data(), memoryPool);
        DenseHll denseHll(indexBitLength, memoryPool);
        sparseHll.toDense(denseHll);

        int32_t serializedSize = denseHll.serializedSize();
        writer.resize(serializedSize);
        denseHll.serialize(writer.data());
      } else if (DenseHlls::canDeserialize(hllData.data())) {
        // Input is already dense - direct copy.
        writer.copy_from(hllData);
      } else {
        VELOX_USER_FAIL("Invalid HyperLogLog format");
      }

      writer.finalize();
    });
  }
};

class P4HyperLogLogTypeFactory : public CustomTypeFactory {
 public:
  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return P4HYPERLOGLOG();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<P4HyperLogLogCastOperator>();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::static_pointer_cast<AbstractInputGenerator>(
        std::make_shared<fuzzer::P4HyperLogLogInputGenerator>(
            config.seed_, config.nullRatio_, config.pool_));
  }
};
} // namespace
void registerP4HyperLogLogType() {
  registerCustomType(
      "p4hyperloglog", std::make_unique<const P4HyperLogLogTypeFactory>());
}
} // namespace facebook::velox
