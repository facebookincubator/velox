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
#include "velox/functions/sparksql/XPathFunctions.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/functions/sparksql/XPathUtil.h"

namespace facebook::velox::functions::sparksql {
namespace {

std::string_view toStringView(const StringView& value) {
  return {value.data(), value.size()};
}

/// Uses the vector API to support both row Status errors and successful NULLs.
class XPathBooleanFunction final : public exec::VectorFunction {
 public:
  /// Evaluates selected rows and records user errors in EvalCtx.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto* xml = decodedArgs.at(0);
    const auto* path = decodedArgs.at(1);
    auto localResult =
        BaseVector::create(outputType, rows.end(), context.pool());
    auto* flatResult = localResult->asFlatVector<bool>();

    rows.applyToSelected([&](vector_size_t row) {
      if (xml->isNullAt(row) || path->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto xmlValue = xml->valueAt<StringView>(row);
      const auto pathValue = path->valueAt<StringView>(row);
      auto evaluated =
          xpath::evalBoolean(toStringView(xmlValue), toStringView(pathValue));
      if (evaluated.hasError()) {
        context.setStatus(row, std::move(evaluated.error()));
        flatResult->setNull(row, true);
        return;
      }
      if (!evaluated.value().has_value()) {
        flatResult->setNull(row, true);
        return;
      }
      flatResult->set(row, *evaluated.value());
    });

    context.moveOrCopyResult(localResult, rows, result);
  }
};

/// Uses the vector API to support both row Status errors and successful NULLs.
class XPathStringFunction final : public exec::VectorFunction {
 public:
  /// Evaluates selected rows and records user errors in EvalCtx.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto* xml = decodedArgs.at(0);
    const auto* path = decodedArgs.at(1);
    auto localResult =
        BaseVector::create(outputType, rows.end(), context.pool());
    auto* flatResult = localResult->asFlatVector<StringView>();

    rows.applyToSelected([&](vector_size_t row) {
      if (xml->isNullAt(row) || path->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      const auto xmlValue = xml->valueAt<StringView>(row);
      const auto pathValue = path->valueAt<StringView>(row);
      auto evaluated =
          xpath::evalString(toStringView(xmlValue), toStringView(pathValue));
      if (evaluated.hasError()) {
        context.setStatus(row, std::move(evaluated.error()));
        flatResult->setNull(row, true);
        return;
      }
      if (!evaluated.value().has_value()) {
        flatResult->setNull(row, true);
        return;
      }
      const auto value = evaluated.value()->view();
      flatResult->set(row, StringView(value.data(), value.size()));
    });

    context.moveOrCopyResult(localResult, rows, result);
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> xpathSignatures(
    std::string_view returnType) {
  return {exec::FunctionSignatureBuilder()
              .returnType(std::string(returnType))
              .argumentType("varchar")
              .argumentType("varchar")
              .build()};
}

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> xpathBooleanSignatures() {
  return xpathSignatures("boolean");
}

std::unique_ptr<exec::VectorFunction> makeXPathBoolean() {
  return std::make_unique<XPathBooleanFunction>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>> xpathStringSignatures() {
  return xpathSignatures("varchar");
}

std::unique_ptr<exec::VectorFunction> makeXPathString() {
  return std::make_unique<XPathStringFunction>();
}

} // namespace facebook::velox::functions::sparksql
