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

#include <utility>

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {
namespace {

// concat_ws(separator, string1, ..., stringN) → varchar
// Returns the concatenation of string1, string2, ..., stringN connected by
// separator.
class ConcatWsFunction : public exec::VectorFunction {
 public:
  ConcatWsFunction(
      const std::string& name,
      const std::vector<exec::VectorFunctionArg>& inputArgs) {
    auto numArgs = inputArgs.size();
    VELOX_USER_CHECK(
        numArgs >= 2,
        "{} requires 2 arguments at least, but got {}",
        name,
        numArgs);

    BaseVector* constantPattern = inputArgs[0].constantValue.get();
    VELOX_USER_CHECK(
        constantPattern != nullptr,
        "{} requires first argument non-null",
        name);

    if (constantPattern->isNullAt(0)) {
      nullDelim_ = true;
      return;
    }

    delim_ =
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0).str();

    // Save constant values to constantStrings_.
    // Identify and combine consecutive constant inputs.
    argMapping_.reserve(numArgs - 1);
    constantStrings_.reserve(numArgs - 1);

    for (auto i = 1; i < numArgs; ++i) {
      argMapping_.push_back(i);

      const auto& arg = inputArgs[i];
      if (arg.constantValue) {
        std::string value = arg.constantValue->as<ConstantVector<StringView>>()
                                ->valueAt(0)
                                .str();

        column_index_t j = i + 1;
        for (; j < inputArgs.size(); ++j) {
          if (!inputArgs[j].constantValue) {
            break;
          }

          value += delim_ +
              inputArgs[j]
                  .constantValue->as<ConstantVector<StringView>>()
                  ->valueAt(0)
                  .str();
        }

        constantStrings_.push_back(std::string(value.data(), value.size()));

        i = j - 1;
      } else {
        constantStrings_.push_back(std::string());
      }
    }

    // Create StringViews for constant strings.
    constantStringViews_.reserve(numArgs - 1);
    for (const auto& constantString : constantStrings_) {
      constantStringViews_.push_back(
          StringView(constantString.data(), constantString.size()));
    }
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    auto flatResult = result->asFlatVector<StringView>();
    if (nullDelim_) {
      flatResult->nullAt(0) = true;
      return;
    }

    auto numArgs = argMapping_.size();

    std::vector<exec::LocalDecodedVector> decodedArgs;
    decodedArgs.reserve(numArgs);

    for (auto i = 0; i < numArgs; ++i) {
      auto index = argMapping_[i];
      if (constantStringViews_[i].empty()) {
        decodedArgs.emplace_back(context, *args[index], rows);
      } else {
        // Do not decode constant inputs.
        decodedArgs.emplace_back(context);
      }
    }

    // Calculate the combined size of the result strings.
    size_t totalResultBytes = 0;
    rows.applyToSelected([&](auto row) {
      auto isFirst = true;
      for (int i = 0; i < numArgs; i++) {
        auto value = constantStringViews_[i].empty()
            ? decodedArgs[i]->valueAt<StringView>(row)
            : constantStringViews_[i];
        if (!value.empty()) {
          if (isFirst) {
            isFirst = false;
          } else {
            totalResultBytes += delim_.size();
          }
          totalResultBytes += value.size();
        }
      }
    });

    // Allocate a string buffer.
    auto rawBuffer = flatResult->getRawStringBufferWithSpace(totalResultBytes);
    size_t offset = 0;
    rows.applyToSelected([&](int row) {
      const char* start = rawBuffer + offset;
      size_t combinedSize = 0;
      auto isFirst = true;
      for (int i = 0; i < numArgs; i++) {
        StringView value;
        if (constantStringViews_[i].empty()) {
          value = decodedArgs[i]->valueAt<StringView>(row);
        } else {
          value = constantStringViews_[i];
        }
        auto size = value.size();
        if (size > 0) {
          if (isFirst) {
            isFirst = false;
          } else {
            memcpy(rawBuffer + offset, delim_.data(), delim_.size());
            offset += delim_.size();
            combinedSize += delim_.size();
          }
          memcpy(rawBuffer + offset, value.data(), size);
          combinedSize += size;
          offset += size;
        }
      }
      flatResult->setNoCopy(row, StringView(start, combinedSize));
    });
  }

 private:
  const std::string delim_;
  bool nullDelim_ = false;
  std::vector<column_index_t> argMapping_;
  std::vector<std::string> constantStrings_;
  std::vector<StringView> constantStringViews_;
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // varchar, varchar,.. -> varchar.
  return {exec::FunctionSignatureBuilder()
              .returnType("varchar")
              .argumentType("varchar")
              .variableArity()
              .build()};
}

} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_concat_ws,
    ConcatWsFunction::signatures(),
    [](const auto& name,
       const auto& inputs,
       const core::QueryConfig& /*config*/) {
      return std::make_unique<ConcatWsFunction>(name, inputs);
    });
} // namespace facebook::velox::functions::sparksql