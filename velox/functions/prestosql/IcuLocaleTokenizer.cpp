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

#include <unicode/brkiter.h>
#include <unicode/ulocdata.h>
#include <unicode/unistr.h>
#include <string>
#include <unordered_map>

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {

namespace {

class IcuLocaleTokenizer : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    const std::string defaultLocale = std::locale("").name();

    // Cache for BreakIterator instances keyed by locale
    static std::unordered_map<std::string, std::unique_ptr<icu::BreakIterator>>
        iteratorCache;

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* textArg = decodedArgs.at(0);
    auto* localeArg = args.size() > 1 ? decodedArgs.at(1) : nullptr;

    BaseVector::ensureWritable(rows, ARRAY(VARCHAR()), context.pool(), result);
    auto arrayResult = result->as<ArrayVector>();
    auto& arrayElements = arrayResult->elements();
    auto elementsVector = arrayElements->asFlatVector<StringView>();

    rows.applyToSelected([&](vector_size_t row) {
      if (textArg->isNullAt(row) ||
          (textArg->valueAt<StringView>(row).size() == 0)) {
        arrayResult->setNull(row, true);
        return;
      }

      auto text = textArg->valueAt<StringView>(row).str();
      std::string locale = defaultLocale;
      if (localeArg != nullptr && !localeArg->isNullAt(row)) {
        locale = localeArg->valueAt<StringView>(row).str();
      }

      // Clean text: replace tabs and newlines with spaces
      std::replace(text.begin(), text.end(), '\t', ' ');
      std::replace(text.begin(), text.end(), '\n', ' ');

      try {
        auto iterator = iteratorCache.find(locale);
        if (iterator == iteratorCache.end()) {
          UErrorCode status = U_ZERO_ERROR;
          auto breakIterator = std::unique_ptr<icu::BreakIterator>(
              icu::BreakIterator::createWordInstance(
                  icu::Locale(locale.c_str()), status));
          if (U_FAILURE(status)) {
            context.setError(
                row,
                std::make_exception_ptr(std::runtime_error(fmt::format(
                    "Failed to create ICU break iterator: {}",
                    u_errorName(status)))));
            return;
          }
          auto result = iteratorCache.emplace(locale, std::move(breakIterator));
          iterator = result.first;
        }

        // Use a clone to avoid modifying cached iterator
        std::unique_ptr<icu::BreakIterator> boundary(iterator->second->clone());
        if (!boundary) {
          context.setError(
              row,
              std::make_exception_ptr(
                  std::runtime_error("Failed to clone ICU break iterator")));
          return;
        }

        icu::UnicodeString unicodeText = icu::UnicodeString::fromUTF8(text);
        boundary->setText(unicodeText);

        // Collect tokens using ICU's word break rules (status >= 100 for
        // words/numbers/symbols)
        auto offset = arrayResult->offsetAt(row);
        std::vector<std::string> tokens;
        for (int32_t start = boundary->first(), end = boundary->next();
             end != icu::BreakIterator::DONE;
             start = end, end = boundary->next()) {
          // Skip tokens that aren't words, numbers, or symbols
          if (boundary->getRuleStatus() >= 100) {
            std::string token;
            unicodeText.tempSubStringBetween(start, end).toUTF8String(token);

            // Only include tokens with non-whitespace characters
            if (std::any_of(token.begin(), token.end(), [](unsigned char c) {
                  return !std::isspace(c);
                })) {
              tokens.push_back(token);
            }
          }
        }

        // Add tokens to result vector
        const size_t size = tokens.size();
        if (size > 0) {
          elementsVector->resize(offset + size);
          for (size_t i = 0; i < size; ++i) {
            elementsVector->set(offset + i, StringView(tokens[i]));
          }
        }
        arrayResult->setOffsetAndSize(row, offset, size);
      } catch (const std::exception& e) {
        context.setError(
            row,
            std::make_exception_ptr(std::runtime_error(
                fmt::format("Error tokenizing text: {}", e.what()))));
      }
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // varchar, varchar -> array(varchar)
        exec::FunctionSignatureBuilder()
            .returnType("array(varchar)")
            .argumentType("varchar")
            .argumentType("varchar")
            .build(),
        // varchar -> array(varchar)
        exec::FunctionSignatureBuilder()
            .returnType("array(varchar)")
            .argumentType("varchar")
            .build(),
    };
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_icu_locale_tokenizer,
    IcuLocaleTokenizer::signatures(),
    std::make_unique<IcuLocaleTokenizer>());

} // namespace facebook::velox::functions
