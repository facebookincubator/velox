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

#include "velox/functions/sparksql/InputFileName.h"
#include <iostream>
#include <utility>
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/prestosql/URLFunctions.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions::sparksql {
namespace {
class InputFileName final : public exec::VectorFunction {
 public:
  InputFileName() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    BufferPtr values = AlignedBuffer::allocate<StringView>(1, context.pool());
    std::vector<BufferPtr> stringBuffers;
    stringBuffers.push_back(
        AlignedBuffer::allocate<char>(1024, context.pool()));
    auto localResult = std::make_shared<FlatVector<StringView>>(
        context.pool(),
        VARCHAR(),
        nullptr,
        1,
        values,
        std::move(stringBuffers));
    auto driverCtx = context.driverCtx();
    auto writer = exec::StringWriter<>(localResult.get(), 0);
    auto inputFileName = driverCtx != nullptr ? driverCtx->inputFileName : "";
    std::vector<uint64_t> doNotEncodeSymbolsBits(4);
    for (auto p : "!$&'()*+,;=/:@") {
      bits::setBit(doNotEncodeSymbolsBits.data(), static_cast<size_t>(p), true);
    }
    facebook::velox::functions::detail::urlEscape(
        writer, driverCtx->inputFileName, false, doNotEncodeSymbolsBits.data());
    writer.finalize();
    auto outFilename = localResult->valueAt(0);
    context.moveOrCopyResult(
        std::make_shared<ConstantVector<StringView>>(
            context.pool(),
            rows.end(),
            false /*isNull*/,
            VARCHAR(),
            std::move(outFilename)),
        rows,
        result);
  }
};
} // namespace
std::unique_ptr<exec::VectorFunction> makeInputFileName() {
  return std::make_unique<InputFileName>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
inputFileNameSignatures() {
  return {exec::FunctionSignatureBuilder().returnType("varchar").build()};
}
} // namespace facebook::velox::functions::sparksql
