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

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/VectorUdfTypeSystem.h"
#include "velox/functions/lib/StringEncodingUtils.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {

using namespace stringCore;
namespace {

/// Check if the input vector's  buffers are single referenced
bool hasSingleReferencedBuffers(const FlatVector<StringView>* vec) {
  for (auto& buffer : vec->getStringBuffers()) {
    if (buffer->refCount() > 1) {
      return false;
    }
  }
  return true;
};
} // namespace

/// Helper function that prepares a string result vector and initializes it.
/// It will use the input argToReuse vector instead of creating new one when
/// possible. Returns true if argToReuse vector was moved to results
VectorPtr emptyVectorPtr;
bool prepareFlatResultsVector(
    VectorPtr* result,
    const SelectivityVector& rows,
    exec::EvalCtx* context,
    VectorPtr& argToReuse = emptyVectorPtr) {
  if (!*result && argToReuse && argToReuse.unique()) {
    // Move input vector to result
    VELOX_CHECK(
        argToReuse.get()->encoding() == VectorEncoding::Simple::FLAT &&
        argToReuse.get()->typeKind() == TypeKind::VARCHAR);
    *result = std::move(argToReuse);
    return true;
  }
  // This will allocate results if not allocated
  BaseVector::ensureWritable(rows, VARCHAR(), context->pool(), result);

  VELOX_CHECK_EQ((*result).get()->encoding(), VectorEncoding::Simple::FLAT);
  return false;
}

namespace {
/**
 * Upper and Lower functions have a fast path for ascii where the functions
 * can be applied in place.
 * */
template <bool isLower /*instantiate for upper or lower*/>
class UpperLowerTemplateFunction : public exec::VectorFunction {
 private:
  /// String encoding wrappable function
  template <bool isAscii>
  struct ApplyInternal {
    static void apply(
        const SelectivityVector& rows,
        const DecodedVector* decodedInput,
        FlatVector<StringView>* results) {
      rows.applyToSelected([&](int row) {
        auto proxy = exec::StringProxy<FlatVector<StringView>>(results, row);
        if constexpr (isLower) {
          stringImpl::lower<isAscii>(
              proxy, decodedInput->valueAt<StringView>(row));
        } else {
          stringImpl::upper<isAscii>(
              proxy, decodedInput->valueAt<StringView>(row));
        }
        proxy.finalize();
      });
    }
  };

  void applyInternalInPlace(
      const SelectivityVector& rows,
      DecodedVector* decodedInput,
      FlatVector<StringView>* results) const {
    rows.applyToSelected([&](int row) {
      auto proxy =
          exec::StringProxy<FlatVector<StringView>, true /*reuseInput*/>(
              results,
              row,
              decodedInput->valueAt<StringView>(row) /*reusedInput*/,
              true /*inPlace*/);
      if constexpr (isLower) {
        stringImpl::lowerAsciiInPlace(proxy);
      } else {
        stringImpl::upperAsciiInPlace(proxy);
      }
      proxy.finalize();
    });
  }

 public:
  bool isDefaultNullBehavior() const override {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VELOX_CHECK(args.size() == 1);
    VELOX_CHECK(args[0]->typeKind() == TypeKind::VARCHAR);

    // Read content before calling prepare results
    BaseVector* inputStringsVector = args[0].get();
    exec::LocalDecodedVector inputHolder(context, *inputStringsVector, rows);
    auto decodedInput = inputHolder.get();

    auto ascii = isAscii(inputStringsVector, rows);

    bool inPlace = ascii &&
        (inputStringsVector->encoding() == VectorEncoding::Simple::FLAT) &&
        hasSingleReferencedBuffers(
                       args.at(0).get()->as<FlatVector<StringView>>());

    if (inPlace) {
      bool inputVectorMoved =
          prepareFlatResultsVector(result, rows, context, args.at(0));
      auto* resultFlatVector = (*result)->as<FlatVector<StringView>>();

      // Move string buffers references to the output vector if we are reusing
      // them and they are not already moved
      if (!inputVectorMoved) {
        resultFlatVector->acquireSharedStringBuffers(inputStringsVector);
      }

      applyInternalInPlace(rows, decodedInput, resultFlatVector);
      return;
    }

    // Not in place path
    VectorPtr emptyVectorPtr;
    prepareFlatResultsVector(result, rows, context, emptyVectorPtr);
    auto* resultFlatVector = (*result)->as<FlatVector<StringView>>();

    StringEncodingTemplateWrapper<ApplyInternal>::apply(
        ascii, rows, decodedInput, resultFlatVector);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // varchar -> varchar
    return {exec::FunctionSignatureBuilder()
                .returnType("varchar")
                .argumentType("varchar")
                .build()};
  }

  bool ensureStringEncodingSetAtAllInputs() const override {
    return true;
  }

  bool propagateStringEncodingFromAllInputs() const override {
    return true;
  }
};

/**
 * concat(string1, ..., stringN) → varchar
 * Returns the concatenation of string1, string2, ..., stringN. This function
 * provides the same functionality as the SQL-standard concatenation operator
 * (||).
 * */
class ConcatFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VectorPtr emptyVectorPtr;
    prepareFlatResultsVector(result, rows, context, emptyVectorPtr);
    auto* resultFlatVector = (*result)->as<FlatVector<StringView>>();

    exec::DecodedArgs decodedArgs(rows, args, context);

    std::vector<StringView> concatInputs(args.size());

    rows.applyToSelected([&](int row) {
      for (int i = 0; i < args.size(); i++) {
        concatInputs[i] = decodedArgs.at(i)->valueAt<StringView>(row);
      }
      auto proxy =
          exec::StringProxy<FlatVector<StringView>>(resultFlatVector, row);
      stringImpl::concatDynamic(proxy, concatInputs);
      proxy.finalize();
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // varchar -> varchar
    return {exec::FunctionSignatureBuilder()
                .returnType("varchar")
                .argumentType("varchar")
                .variableArity()
                .build()};
  }

  bool propagateStringEncodingFromAllInputs() const override {
    return true;
  }
};

/**
 * strpos(string, substring) → bigint
 * Returns the starting position of the first instance of substring in string.
 * Positions start with 1. If not found, 0 is returned.
 *
 * strpos(string, substring, instance) → bigint
 * Returns the position of the N-th instance of substring in string. instance
 * must be a positive number. Positions start with 1. If not found, 0 is
 * returned.
 **/
class StringPosition : public exec::VectorFunction {
 private:
  /// A function that can be wrapped with ascii mode
  template <bool isAscii>
  struct ApplyInternal {
    template <
        typename StringReader,
        typename SubStringReader,
        typename InstanceReader>
    static void apply(
        StringReader stringReader,
        SubStringReader subStringReader,
        InstanceReader instanceReader,
        const SelectivityVector& rows,
        FlatVector<int64_t>* resultFlatVector) {
      rows.applyToSelected([&](int row) {
        auto result = stringImpl::stringPosition<isAscii>(
            stringReader(row), subStringReader(row), instanceReader(row));
        resultFlatVector->set(row, result);
      });
    }
  };

 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedStringInput = decodedArgs.at(0);
    auto decodedSubStringInput = decodedArgs.at(1);

    auto stringArgStringEncoding = isAscii(args.at(0).get(), rows);
    BaseVector::ensureWritable(rows, BIGINT(), context->pool(), result);

    auto* resultFlatVector = (*result)->as<FlatVector<int64_t>>();

    auto stringReader = [&](const vector_size_t row) {
      return decodedStringInput->valueAt<StringView>(row);
    };

    auto substringReader = [&](const vector_size_t row) {
      return decodedSubStringInput->valueAt<StringView>(row);
    };

    // If there's no "instance" parameter.
    if (args.size() <= 2) {
      StringEncodingTemplateWrapper<ApplyInternal>::apply(
          stringArgStringEncoding,
          stringReader,
          substringReader,
          [](const vector_size_t) { return 1L; },
          rows,
          resultFlatVector);
    }
    // If there's an "instance" parameter, check if it's BIGINT or INTEGER.
    else {
      auto decodedInstanceInput = decodedArgs.at(2);

      if (args[2]->typeKind() == TypeKind::BIGINT) {
        auto instanceReader = [&](const vector_size_t row) {
          return decodedInstanceInput->valueAt<int64_t>(row);
        };
        StringEncodingTemplateWrapper<ApplyInternal>::apply(
            stringArgStringEncoding,
            stringReader,
            substringReader,
            instanceReader,
            rows,
            resultFlatVector);
      } else if (args[2]->typeKind() == TypeKind::INTEGER) {
        auto instanceReader = [&](const vector_size_t row) {
          return decodedInstanceInput->valueAt<int32_t>(row);
        };
        StringEncodingTemplateWrapper<ApplyInternal>::apply(
            stringArgStringEncoding,
            stringReader,
            substringReader,
            instanceReader,
            rows,
            resultFlatVector);
      } else {
        VELOX_UNREACHABLE();
      }
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // varchar, varchar -> bigint
        exec::FunctionSignatureBuilder()
            .returnType("bigint")
            .argumentType("varchar")
            .argumentType("varchar")
            .build(),
        // varchar, varchar, integer -> bigint
        exec::FunctionSignatureBuilder()
            .returnType("bigint")
            .argumentType("varchar")
            .argumentType("varchar")
            .argumentType("integer")
            .build(),
        // varchar, varchar, bigint -> bigint
        exec::FunctionSignatureBuilder()
            .returnType("bigint")
            .argumentType("varchar")
            .argumentType("varchar")
            .argumentType("bigint")
            .build(),
    };
  }
};

/**
 * replace(string, search) → varchar
 * Removes all instances of search from string.
 *
 * replace(string, search, replace) → varchar
 * Replaces all instances of search with replace in string.
 * If search is an empty string, inserts replace in front of every character
 *and at the end of the string.
 **/
class Replace : public exec::VectorFunction {
 private:
  template <
      typename StringReader,
      typename SearchReader,
      typename ReplaceReader>
  void applyInternal(
      StringReader stringReader,
      SearchReader searchReader,
      ReplaceReader replaceReader,
      const SelectivityVector& rows,
      FlatVector<StringView>* results) const {
    rows.applyToSelected([&](int row) {
      auto proxy = exec::StringProxy<FlatVector<StringView>>(results, row);
      stringImpl::replace(
          proxy, stringReader(row), searchReader(row), replaceReader(row));
      proxy.finalize();
    });
  }

  template <
      typename StringReader,
      typename SearchReader,
      typename ReplaceReader>
  void applyInPlace(
      StringReader stringReader,
      SearchReader searchReader,
      ReplaceReader replaceReader,
      const SelectivityVector& rows,
      FlatVector<StringView>* results) const {
    rows.applyToSelected([&](int row) {
      auto proxy =
          exec::StringProxy<FlatVector<StringView>, true /*reuseInput*/>(
              results,
              row,
              stringReader(row) /*reusedInput*/,
              true /*inPlace*/);
      stringImpl::replaceInPlace(proxy, searchReader(row), replaceReader(row));
      proxy.finalize();
    });
  }

 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // Read string input
    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();

    // Read search argument
    exec::LocalDecodedVector decodedSearchHolder(context, *args[1], rows);
    auto decodedSearchInput = decodedSearchHolder.get();

    std::optional<StringView> searchArgValue;
    if (decodedSearchInput->isConstantMapping()) {
      searchArgValue = decodedSearchInput->valueAt<StringView>(0);
    }

    // Read replace argument
    exec::LocalDecodedVector decodedReplaceHolder(context);
    auto decodedReplaceInput = decodedReplaceHolder.get();
    std::optional<StringView> replaceArgValue;

    if (args.size() <= 2) {
      replaceArgValue = StringView("");
    } else {
      decodedReplaceInput->decode(*args.at(2), rows);
      if (decodedReplaceInput->isConstantMapping()) {
        replaceArgValue = decodedReplaceInput->valueAt<StringView>(0);
      }
    }

    auto stringReader = [&](const vector_size_t row) {
      return decodedStringInput->valueAt<StringView>(row);
    };

    auto searchReader = [&](const vector_size_t row) {
      return decodedSearchInput->valueAt<StringView>(row);
    };

    auto replaceReader = [&](const vector_size_t row) {
      if (replaceArgValue.has_value()) {
        return replaceArgValue.value();
      } else {
        return decodedReplaceInput->valueAt<StringView>(row);
      }
    };

    // Right now we enable the inplace if 'search' and 'replace' are constants
    // and 'search' size is smaller than or equal to 'replace'.

    // TODO: analyze other options for enabling inplace i.e.:
    // 1. Decide per row.
    // 2. Scan inputs for max lengths and decide based on that. ..etc
    bool inPlace = replaceArgValue.has_value() && searchArgValue.has_value() &&
        (searchArgValue.value().size() <= replaceArgValue.value().size()) &&
        (args.at(0)->encoding() == VectorEncoding::Simple::FLAT) &&
        hasSingleReferencedBuffers(
                       args.at(0).get()->as<FlatVector<StringView>>());

    if (inPlace) {
      bool inputVectorMoved =
          prepareFlatResultsVector(result, rows, context, args.at(0));
      auto* resultFlatVector = (*result)->as<FlatVector<StringView>>();

      // Move string buffers references to the output vector if we are reusing
      // them and they are not already moved.
      if (!inputVectorMoved) {
        resultFlatVector->acquireSharedStringBuffers(args.at(0).get());
      }
      applyInPlace(
          stringReader, searchReader, replaceReader, rows, resultFlatVector);
      return;
    }
    // Not in place path
    VectorPtr emptyVectorPtr;
    prepareFlatResultsVector(result, rows, context, emptyVectorPtr);
    auto* resultFlatVector = (*result)->as<FlatVector<StringView>>();

    applyInternal(
        stringReader, searchReader, replaceReader, rows, resultFlatVector);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // varchar, varchar -> varchar
        exec::FunctionSignatureBuilder()
            .returnType("varchar")
            .argumentType("varchar")
            .argumentType("varchar")
            .build(),
        // varchar, varchar, varchar -> varchar
        exec::FunctionSignatureBuilder()
            .returnType("varchar")
            .argumentType("varchar")
            .argumentType("varchar")
            .argumentType("varchar")
            .build(),
    };
  }

  // Only the original string and the replacement are relevant to the result
  // encoding.
  // TODO: The propagation is a safe approximation here, it might be better
  // for some cases to keep it unset and then rescan.
  std::optional<std::vector<size_t>> propagateStringEncodingFrom()
      const override {
    return {{0, 2}};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_upper,
    UpperLowerTemplateFunction<false /*isLower*/>::signatures(),
    std::make_unique<UpperLowerTemplateFunction<false /*isLower*/>>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_lower,
    UpperLowerTemplateFunction<true /*isLower*/>::signatures(),
    std::make_unique<UpperLowerTemplateFunction<true /*isLower*/>>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_concat,
    ConcatFunction::signatures(),
    std::make_unique<ConcatFunction>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_strpos,
    StringPosition::signatures(),
    std::make_unique<StringPosition>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_replace,
    Replace::signatures(),
    std::make_unique<Replace>());

} // namespace facebook::velox::functions
