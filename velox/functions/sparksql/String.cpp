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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/StringEncodingUtils.h"
#include "velox/functions/lib/string/StringCore.h"

namespace facebook::velox::functions::sparksql {

using namespace stringCore;
namespace {

template <bool isAscii>
int32_t instr(
    const folly::StringPiece haystack,
    const folly::StringPiece needle) {
  int32_t offset = haystack.find(needle);
  if constexpr (isAscii) {
    return offset + 1;
  } else {
    // If the string is unicode, convert the byte offset to a codepoints.
    return offset == -1 ? 0 : lengthUnicode(haystack.data(), offset) + 1;
  }
}

class Instr : public exec::VectorFunction {
  bool ensureStringEncodingSetAtAllInputs() const override {
    return true;
  }

  void apply(
      const SelectivityVector& selected,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);
    VELOX_CHECK_EQ(args[0]->typeKind(), TypeKind::VARCHAR);
    VELOX_CHECK_EQ(args[1]->typeKind(), TypeKind::VARCHAR);
    exec::LocalDecodedVector haystack(context, *args[0], selected);
    exec::LocalDecodedVector needle(context, *args[1], selected);
    context.ensureWritable(selected, INTEGER(), result);
    auto* output = result->as<FlatVector<int32_t>>();

    if (isAscii(args[0].get(), selected)) {
      selected.applyToSelected([&](vector_size_t row) {
        auto h = haystack->valueAt<StringView>(row);
        auto n = needle->valueAt<StringView>(row);
        output->set(row, instr<true>(h, n));
      });
    } else {
      selected.applyToSelected([&](vector_size_t row) {
        auto h = haystack->valueAt<StringView>(row);
        auto n = needle->valueAt<StringView>(row);
        output->set(row, instr<false>(h, n));
      });
    }
  }
};

class Length : public exec::VectorFunction {
  bool ensureStringEncodingSetAtAllInputs() const override {
    return true;
  }

  void apply(
      const SelectivityVector& selected,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);
    VELOX_CHECK(
        args[0]->typeKind() == TypeKind::VARCHAR ||
        args[0]->typeKind() == TypeKind::VARBINARY);
    exec::LocalDecodedVector input(context, *args[0], selected);
    context.ensureWritable(selected, INTEGER(), result);
    auto* output = result->as<FlatVector<int32_t>>();

    if (args[0]->typeKind() == TypeKind::VARCHAR &&
        !isAscii(args[0].get(), selected)) {
      selected.applyToSelected([&](vector_size_t row) {
        const StringView str = input->valueAt<StringView>(row);
        output->set(row, lengthUnicode(str.data(), str.size()));
      });
    } else {
      selected.applyToSelected([&](vector_size_t row) {
        output->set(row, input->valueAt<StringView>(row).size());
      });
    }
  }
};

class ConcatWsVariableParameters : public exec::VectorFunction {
 public:
  explicit ConcatWsVariableParameters(const std::string& connector)
      : connector_{connector} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    std::vector<column_index_t> argMapping;
    std::vector<std::string> constantStrings;
    std::vector<StringView> constantStringViews;
    auto numArgs = args.size();

    // Save constant values to constantStrings_.
    // Identify and combine consecutive constant inputs.
    argMapping.reserve(numArgs - 1);
    constantStrings.reserve(numArgs - 1);
    for (auto i = 1; i < numArgs; ++i) {
      argMapping.push_back(i);
      if (args[i] && args[i]->as<ConstantVector<StringView>>() &&
          !args[i]->as<ConstantVector<StringView>>()->isNullAt(0)) {
        std::string value =
            args[i]->as<ConstantVector<StringView>>()->valueAt(0).str();
        column_index_t j = i + 1;
        for (; j < args.size(); ++j) {
          if (!args[j] || !args[j]->as<ConstantVector<StringView>>() ||
              args[j]->as<ConstantVector<StringView>>()->isNullAt(0)) {
            break;
          }

          value += connector_ +
              args[j]->as<ConstantVector<StringView>>()->valueAt(0).str();
        }
        constantStrings.push_back(std::string(value.data(), value.size()));
        i = j - 1;
      } else {
        constantStrings.push_back(std::string());
      }
    }

    // Create StringViews for constant strings.
    constantStringViews.reserve(numArgs - 1);
    for (const auto& constantString : constantStrings) {
      constantStringViews.push_back(
          StringView(constantString.data(), constantString.size()));
    }

    auto flatResult = result->asFlatVector<StringView>();
    auto numCols = argMapping.size();
    std::vector<exec::LocalDecodedVector> decodedArgs;
    decodedArgs.reserve(numCols);

    for (auto i = 0; i < numCols; ++i) {
      auto index = argMapping[i];
      if (constantStringViews[i].empty()) {
        decodedArgs.emplace_back(context, *args[index], rows);
      } else {
        // Do not decode constant inputs.
        decodedArgs.emplace_back(context);
      }
    }

    size_t totalResultBytes = 0;
    rows.applyToSelected([&](auto row) {
      auto isFirst = true;
      for (int i = 0; i < numCols; i++) {
        auto value = constantStringViews[i].empty()
            ? decodedArgs[i]->valueAt<StringView>(row)
            : constantStringViews[i];
        if (!value.empty()) {
          if (isFirst) {
            isFirst = false;
          } else {
            totalResultBytes += connector_.size();
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
      for (int i = 0; i < numCols; i++) {
        StringView value;
        if (constantStringViews[i].empty()) {
          value = decodedArgs[i]->valueAt<StringView>(row);
        } else {
          value = constantStringViews[i];
        }
        auto size = value.size();
        if (size > 0) {
          if (isFirst) {
            isFirst = false;
          } else {
            memcpy(rawBuffer + offset, connector_.data(), connector_.size());
            offset += connector_.size();
            combinedSize += connector_.size();
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
  const std::string connector_;
};

class ConcatWs : public exec::VectorFunction {
  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& selected,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto numArgs = args.size();
    VELOX_USER_CHECK(
        numArgs >= 1,
        "concat_ws requires one arguments at least, but got {}",
        numArgs);

    context.ensureWritable(selected, VARCHAR(), result);
    if (args[0]->isNullAt(0)) {
      selected.applyToSelected([&](int row) { result->setNull(row, true); });
      return;
    }

    if (numArgs == 1) {
      auto flatResult = result->asFlatVector<StringView>();
      selected.applyToSelected(
          [&](int row) { flatResult->setNoCopy(row, StringView("")); });
      return;
    }

    auto connectorVector = args[0]->as<ConstantVector<StringView>>();
    VELOX_USER_CHECK(
        connectorVector, "concat_ws function supports only constant connector");
    auto connector = connectorVector->valueAt(0).str();

    ConcatWsVariableParameters concatWsVariableParameters(connector);
    concatWsVariableParameters.apply(selected, args, nullptr, context, result);
  }
}
};

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> instrSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .returnType("INTEGER")
          .argumentType("VARCHAR")
          .argumentType("VARCHAR")
          .build(),
  };
}

std::shared_ptr<exec::VectorFunction> makeInstr(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  static const auto kInstrFunction = std::make_shared<Instr>();
  return kInstrFunction;
}

std::vector<std::shared_ptr<exec::FunctionSignature>> lengthSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .returnType("INTEGER")
          .argumentType("VARCHAR")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("INTEGER")
          .argumentType("VARBINARY")
          .build(),
  };
}

std::shared_ptr<exec::VectorFunction> makeLength(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  static const auto kLengthFunction = std::make_shared<Length>();
  return kLengthFunction;
}

std::vector<std::shared_ptr<exec::FunctionSignature>> concatWsSignatures() {
  return {
      // varchar, varchar,... -> varchar.
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .variableArity()
          .build(),
      // varchar, array(varchar) -> varchar.
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("array(varchar)")
          .build(),
  };
}

std::shared_ptr<exec::VectorFunction> makeConcatWs(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  static const auto kConcatWsFunction = std::make_shared<ConcatWs>();
  return kConcatWsFunction;
}

void encodeDigestToBase16(uint8_t* output, int digestSize) {
  static unsigned char const kHexCodes[] = "0123456789abcdef";
  for (int i = digestSize - 1; i >= 0; --i) {
    int digestChar = output[i];
    output[i * 2] = kHexCodes[(digestChar >> 4) & 0xf];
    output[i * 2 + 1] = kHexCodes[digestChar & 0xf];
  }
}

} // namespace facebook::velox::functions::sparksql
