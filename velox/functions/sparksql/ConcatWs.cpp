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
#include "velox/functions/sparksql/ConcatWs.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

namespace {
class ConcatWs : public exec::VectorFunction {
 public:
  explicit ConcatWs(const std::optional<std::string>& separator)
      : separator_(separator) {}

  bool isConstantSeparator() const {
    return separator_.has_value();
  }

  // Calculate the total number of bytes in the result.
  size_t calculateTotalResultBytes(
      const SelectivityVector& rows,
      exec::EvalCtx& context,
      std::vector<exec::LocalDecodedVector>& decodedArrays,
      const std::vector<std::string>& constantStrings,
      const std::vector<exec::LocalDecodedVector>& decodedStringArgs,
      const exec::LocalDecodedVector& decodedSeparator) const {
    auto arrayArgNum = decodedArrays.size();
    std::vector<const ArrayVector*> arrayVectors;
    std::vector<DecodedVector*> elementsDecodedVectors;
    for (auto i = 0; i < arrayArgNum; ++i) {
      auto arrayVector = decodedArrays[i].get()->base()->as<ArrayVector>();
      arrayVectors.push_back(arrayVector);
      auto elements = arrayVector->elements();
      exec::LocalSelectivityVector nestedRows(context, elements->size());
      nestedRows.get()->setAll();
      exec::LocalDecodedVector elementsHolder(
          context, *elements, *nestedRows.get());
      elementsDecodedVectors.push_back(elementsHolder.get());
    }

    size_t totalResultBytes = 0;
    rows.applyToSelected([&](auto row) {
      // NULL separator produces NULL result.
      if (!isConstantSeparator() && decodedSeparator->isNullAt(row)) {
        return;
      }
      int32_t allElements = 0;
      // Calculate size for array columns data.
      for (int i = 0; i < arrayArgNum; i++) {
        auto arrayVector = arrayVectors[i];
        auto rawSizes = arrayVector->rawSizes();
        auto rawOffsets = arrayVector->rawOffsets();
        auto indices = decodedArrays[i].get()->indices();
        auto elementsDecoded = elementsDecodedVectors[i];

        auto size = rawSizes[indices[row]];
        auto offset = rawOffsets[indices[row]];
        for (int j = 0; j < size; ++j) {
          if (!elementsDecoded->isNullAt(offset + j)) {
            auto element = elementsDecoded->valueAt<StringView>(offset + j);
            // No matter empty string or not.
            allElements++;
            totalResultBytes += element.size();
          }
        }
      }

      // Calculate size for string arg.
      auto it = decodedStringArgs.begin();
      for (int i = 0; i < constantStrings.size(); i++) {
        StringView value;
        if (!constantStrings[i].empty()) {
          value = StringView(constantStrings[i]);
        } else {
          // Skip NULL.
          if ((*it)->isNullAt(row)) {
            ++it;
            continue;
          }
          value = (*it++)->valueAt<StringView>(row);
        }
        // No matter empty string or not.
        allElements++;
        totalResultBytes += value.size();
      }

      int32_t separatorSize = isConstantSeparator()
          ? separator_.value().size()
          : decodedSeparator->valueAt<StringView>(row).size();

      if (allElements > 1 && separatorSize > 0) {
        totalResultBytes += (allElements - 1) * separatorSize;
      }
    });
    return totalResultBytes;
  }

  void initVectors(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      const exec::EvalCtx& context,
      std::vector<exec::LocalDecodedVector>& decodedArrays,
      std::vector<column_index_t>& argMapping,
      std::vector<std::string>& constantStrings,
      std::vector<exec::LocalDecodedVector>& decodedStringArgs) const {
    for (auto i = 1; i < args.size(); ++i) {
      if (args[i] && args[i]->typeKind() == TypeKind::ARRAY) {
        decodedArrays.emplace_back(context, *args[i], rows);
        continue;
      }
      // Handles string arg.
      argMapping.push_back(i);
      if (!isConstantSeparator()) {
        // Cannot concat consecutive constant string args in advance.
        constantStrings.push_back("");
        continue;
      }
      if (args[i] && args[i]->as<ConstantVector<StringView>>() &&
          !args[i]->as<ConstantVector<StringView>>()->isNullAt(0)) {
        std::ostringstream out;
        out << args[i]->as<ConstantVector<StringView>>()->valueAt(0);
        column_index_t j = i + 1;
        // Concat constant string args in advance.
        for (; j < args.size(); ++j) {
          if (!args[j] || args[j]->typeKind() == TypeKind::ARRAY ||
              !args[j]->as<ConstantVector<StringView>>() ||
              args[j]->as<ConstantVector<StringView>>()->isNullAt(0)) {
            break;
          }
          out << separator_.value()
              << args[j]->as<ConstantVector<StringView>>()->valueAt(0);
        }
        constantStrings.emplace_back(out.str());
        i = j - 1;
      } else {
        constantStrings.push_back("");
      }
    }

    // Number of string columns after combined consecutive constant ones.
    auto numStringCols = constantStrings.size();
    for (auto i = 0; i < numStringCols; ++i) {
      if (constantStrings[i].empty()) {
        auto index = argMapping[i];
        decodedStringArgs.emplace_back(context, *args[index], rows);
      }
    }
  }

  // ConcatWs implementation. It concatenates the arguments with the separator.
  // Mixed using of VARCHAR & ARRAY<VARCHAR> is considered. If separator is
  // constant, concatenate consecutive constant string args in advance. Then,
  // concatenate the intemediate result with neighboring array args or
  // non-constant string args.
  void doApply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    auto& flatResult = *result->asFlatVector<StringView>();
    std::vector<column_index_t> argMapping;
    std::vector<std::string> constantStrings;
    auto numArgs = args.size();
    argMapping.reserve(numArgs - 1);
    // Save intermediate result for consecutive constant string args.
    // They will be concatenated in advance.
    constantStrings.reserve(numArgs - 1);
    std::vector<exec::LocalDecodedVector> decodedArrays;
    decodedArrays.reserve(numArgs - 1);
    // For column string arg decoding.
    std::vector<exec::LocalDecodedVector> decodedStringArgs;
    decodedStringArgs.reserve(numArgs);

    initVectors(
        rows,
        args,
        context,
        decodedArrays,
        argMapping,
        constantStrings,
        decodedStringArgs);
    exec::LocalDecodedVector decodedSeparator(context);
    if (!isConstantSeparator()) {
      decodedSeparator = exec::LocalDecodedVector(context, *args[0], rows);
    }

    auto totalResultBytes = calculateTotalResultBytes(
        rows,
        context,
        decodedArrays,
        constantStrings,
        decodedStringArgs,
        decodedSeparator);

    std::vector<const ArrayVector*> arrayVectors;
    std::vector<DecodedVector*> elementsDecodedVectors;
    for (auto i = 0; i < decodedArrays.size(); ++i) {
      auto arrayVector = decodedArrays[i].get()->base()->as<ArrayVector>();
      arrayVectors.push_back(arrayVector);
      auto elements = arrayVector->elements();
      exec::LocalSelectivityVector nestedRows(context, elements->size());
      nestedRows.get()->setAll();
      exec::LocalDecodedVector elementsHolder(
          context, *elements, *nestedRows.get());
      elementsDecodedVectors.push_back(elementsHolder.get());
    }
    // Allocate a string buffer.
    auto rawBuffer =
        flatResult.getRawStringBufferWithSpace(totalResultBytes, true);
    rows.applyToSelected([&](auto row) {
      // NULL separtor produces NULL result.
      if (!isConstantSeparator() && decodedSeparator->isNullAt(row)) {
        result->setNull(row, true);
        return;
      }
      const char* start = rawBuffer;
      auto isFirst = true;
      // For array arg.
      int32_t i = 0;
      // For string arg.
      int32_t j = 0;
      auto it = decodedStringArgs.begin();

      auto copyToBuffer = [&](StringView value, StringView separator) {
        if (isFirst) {
          isFirst = false;
        } else {
          // Add separator before the current value.
          if (!separator.empty()) {
            memcpy(rawBuffer, separator.data(), separator.size());
            rawBuffer += separator.size();
          }
        }
        if (!value.empty()) {
          memcpy(rawBuffer, value.data(), value.size());
          rawBuffer += value.size();
        }
      };

      for (auto itArgs = args.begin() + 1; itArgs != args.end(); ++itArgs) {
        if ((*itArgs)->typeKind() == TypeKind::ARRAY) {
          auto arrayVector = arrayVectors[i];
          auto rawSizes = arrayVector->rawSizes();
          auto rawOffsets = arrayVector->rawOffsets();
          auto indices = decodedArrays[i].get()->indices();
          auto elementsDecoded = elementsDecodedVectors[i];

          auto size = rawSizes[indices[row]];
          auto offset = rawOffsets[indices[row]];
          for (int k = 0; k < size; ++k) {
            if (!elementsDecoded->isNullAt(offset + k)) {
              auto element = elementsDecoded->valueAt<StringView>(offset + k);
              copyToBuffer(
                  element,
                  isConstantSeparator()
                      ? StringView(separator_.value())
                      : decodedSeparator->valueAt<StringView>(row));
            }
          }
          i++;
          continue;
        }

        if (j >= constantStrings.size()) {
          continue;
        }

        StringView value;
        if (!constantStrings[j].empty()) {
          value = StringView(constantStrings[j]);
        } else {
          // Skip NULL.
          if ((*it)->isNullAt(row)) {
            ++it;
            continue;
          }
          value = (*it++)->valueAt<StringView>(row);
        }
        copyToBuffer(
            value,
            isConstantSeparator() ? StringView(separator_.value())
                                  : decodedSeparator->valueAt<StringView>(row));
        j++;
      }
      flatResult.setNoCopy(row, StringView(start, rawBuffer - start));
    });
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    auto flatResult = result->asFlatVector<StringView>();
    auto numArgs = args.size();
    // If separator is NULL, result is NULL.
    if (isConstantSeparator()) {
      auto constant = args[0]->as<ConstantVector<StringView>>();
      if (constant->isNullAt(0)) {
        auto localResult = BaseVector::createNullConstant(
            outputType, rows.end(), context.pool());
        context.moveOrCopyResult(localResult, rows, result);
        return;
      }
    }
    // If only separator (not a NULL) is provided, result is an empty string.
    if (numArgs == 1) {
      auto decodedSeparator = exec::LocalDecodedVector(context, *args[0], rows);
      //  1. Separator is constant and not a NULL.
      //  2. Separator is column and have no NULL.
      if (isConstantSeparator() || !decodedSeparator->mayHaveNulls()) {
        rows.applyToSelected(
            [&](auto row) { flatResult->setNoCopy(row, StringView("")); });
      } else {
        rows.applyToSelected([&](auto row) {
          if (decodedSeparator->isNullAt(row)) {
            result->setNull(row, true);
          } else {
            flatResult->setNoCopy(row, StringView(""));
          }
        });
      }
      return;
    }
    doApply(rows, args, context, result);
  }

 private:
  // For holding constant separator.
  const std::optional<std::string> separator_;
};
} // namespace

TypePtr ConcatWsCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  return VARCHAR();
}

exec::ExprPtr ConcatWsCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  auto numArgs = args.size();
  VELOX_USER_CHECK_GE(
      numArgs,
      1,
      "concat_ws requires one arguments at least, but got {}.",
      numArgs);
  VELOX_USER_CHECK(
      args[0]->type()->isVarchar(),
      "The first argument of concat_ws must be a varchar.");
  for (size_t i = 1; i < args.size(); i++) {
    VELOX_USER_CHECK(
        args[i]->type()->isVarchar() ||
            (args[i]->type()->isArray() &&
             args[i]->type()->asArray().elementType()->isVarchar()),
        "The 2nd and following arguments for concat_ws should be varchar or array(varchar), but got {}.",
        args[i]->type()->toString());
  }

  std::optional<std::string> separator = std::nullopt;
  auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(args[0]);

  if (constantExpr != nullptr) {
    separator = constantExpr->value()
                    ->asUnchecked<ConstantVector<StringView>>()
                    ->valueAt(0)
                    .str();
  }
  auto concatWsFunction = std::make_shared<ConcatWs>(separator);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(concatWsFunction),
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      kConcatWs,
      trackCpuUsage);
}

} // namespace facebook::velox::functions::sparksql
