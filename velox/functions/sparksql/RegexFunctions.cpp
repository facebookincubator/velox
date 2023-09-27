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
#include "velox/functions/lib/Re2Functions.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/VectorWriters.h"

namespace facebook::velox::functions::sparksql {
namespace {

using ::re2::RE2;

template <typename T>
re2::StringPiece toStringPiece(const T& s) {
  return re2::StringPiece(s.data(), s.size());
}

// If v is a non-null constant vector, returns the constant value. Otherwise
// returns nullopt.
template <typename T>
std::optional<T> getIfConstant(const BaseVector& v) {
  if (v.isConstantEncoding() && !v.isNullAt(0)) {
    return v.as<ConstantVector<T>>()->valueAt(0);
  }
  return std::nullopt;
}

void checkForBadPattern(const RE2& re) {
  if (UNLIKELY(!re.ok())) {
    VELOX_USER_FAIL("invalid regular expression:{}", re.error());
  }
}

/// Validates the provided regex pattern to ensure its compatibility with the
/// system. The function checks if the pattern uses features like character
/// class union, intersection, or difference which are not supported in C++ RE2
/// library but are supported in Java regex.
///
/// This function should be called on the indivdual patterns of a decoded
/// vector. That way when a single pattern in a vector is invalid, we can still
/// operate on the remaining rows.
///
/// @param pattern The regex pattern string to validate.
/// @param functionName (Optional) Name of the calling function to include in
/// error messages.
///
/// @throws VELOX_USER_FAIL If the pattern is found to use unsupported features.
/// @note  Default functionName is "regex_replace" because it uses non-constant
/// patterns so it cannot be checked with "ensureRegexIsCompatible". No
/// other functions work with non-constant patterns, but they may in the future.
///
/// @note Leaving functionName as an optional parameter makes room for
/// other functions to enable non-constant patterns in the future.
void checkForCompatiblePattern(
    const std::string& pattern,
    const char* functionName = "regex_replace") {
  // If in a character class, points to the [ at the beginning of that class.
  const char* charClassStart = nullptr;
  // This minimal regex parser looks just for the class begin/end markers.
  for (const char* c = pattern.data(); c < pattern.data() + pattern.size();
       ++c) {
    if (*c == '\\') {
      ++c;
    } else if (*c == '[') {
      if (charClassStart) {
        VELOX_USER_FAIL(
            "{} does not support character class union, intersection, "
            "or difference ([a[b]], [a&&[b]], [a&&[^b]])",
            functionName);
      }
      charClassStart = c;
      // A ] immediately after a [ does not end the character class, and is
      // instead adds the character ].
    } else if (*c == ']' && charClassStart + 1 != c) {
      charClassStart = nullptr;
    }
  }
}

// Blocks patterns that contain character class union, intersection, or
// difference because these are not understood by RE2 and will be parsed as a
// different pattern than in java.util.regex.
void ensureRegexIsCompatible(
    const char* functionName,
    const VectorPtr& patternVector) {
  if (!patternVector ||
      patternVector->encoding() != VectorEncoding::Simple::CONSTANT) {
    VELOX_USER_FAIL("{} requires a constant pattern.", functionName);
  }
  if (patternVector->isNullAt(0)) {
    return;
  }
  const StringView pattern =
      patternVector->as<ConstantVector<StringView>>()->valueAt(0);
  // Call the factored out function to check the pattern.
  checkForCompatiblePattern(
      std::string(pattern.data(), pattern.size()), functionName);
}

/// @class Re2ReplaceConstantPattern
///
/// @brief Provides functionalities for replacing occurrences of a constant
/// regex pattern in string vectors.
///
/// The Re2ReplaceConstantPattern class is a specialized class designed for
/// regex-based string replacement operations on input string vectors,
/// specifically when the regex pattern to be replaced is constant across the
/// operation. The consistent pattern enables more optimized processing as
/// opposed to scenarios with a dynamic range of patterns.
///
/// The core functionality is encapsulated in the `apply` method, which reads
/// the input string vectors and performs the replacement based on the provided
/// regex pattern. The class supports optional starting positions for
/// replacements, adhering to Spark's behavior of 1-indexing for positions.
/// Additionally, the class ensures optimal performance by minimizing the
/// copying of data between vectors, optimizing memory utilization.
///
/// @param basePattern The regular expression pattern that remains constant
/// throughout the operation.
///        This pattern is used to search for and match substrings within the
///        input strings.
///
/// Method Descriptions:
/// - Constructor: Initializes the object with a base pattern to be used for
/// regex matching.
/// - apply: The primary method that performs the replacement operation on input
/// strings.
///          It replaces occurrences of the basePattern with a specified
///          replacement string starting from an optional position.
///
/// Typical usage scenarios of this class include string preprocessing, data
/// cleaning, and other regex-based string manipulations where the pattern
/// remains unchanged across rows or batches of data.
///
/// Note: The naming "Re2ReplaceConstantPattern" signifies the class's reliance
/// on the RE2 regular expression library for underlying regex operations and
/// the constant nature of the regex pattern.
class Re2ReplaceConstantPattern final : public exec::VectorFunction {
 public:
  /// @brief Constructor for the Re2ReplaceConstantPattern class.
  ///
  /// This class provides a method to replace occurrences of a pattern in a
  /// string starting from a given position.
  ///
  /// @param basePattern The regular expression pattern to search for in the
  /// input string.

  explicit Re2ReplaceConstantPattern(StringView basePattern)
      : basePattern_(basePattern.getString()) {}

  /// @brief Applies the regex replace operation on the input string vectors.
  ///
  /// This function reads the input strings and replaces occurrences of the
  /// regex pattern starting from the given position. If the position is not
  /// provided, it defaults to 1. The result of the replacement operation is
  /// written to the resultRef vector.
  ///
  /// @param rows The rows of data to be processed.
  /// @param args The arguments passed to the function.
  ///             args[0]: The input strings.
  ///             args[1]: The pattern to search for.
  ///             args[2]: The replacement strings.
  ///             args[3]: (Optional) The starting position.
  /// @param context The evaluation context.
  /// @param resultRef The vector to write the result to.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 3 || args.size() == 4);

    exec::LocalDecodedVector input(context, *args[0], rows);

    BaseVector::ensureWritable(
        rows, ARRAY(VARCHAR()), context.pool(), resultRef);
    exec::VectorWriter<Array<Varchar>> resultWriter;
    resultWriter.init(*resultRef->as<ArrayVector>());

    exec::LocalDecodedVector overwriteVec(context, *args[2], rows);

    int64_t position;
    exec::LocalDecodedVector positionVec(context);
    if (args.size() == 4) {
      positionVec = exec::LocalDecodedVector(context, *args[3], rows);
    }

    const re2::RE2 re_(basePattern_, RE2::Quiet);

    try {
      checkForBadPattern(re_);
      checkForCompatiblePattern(basePattern_);
    } catch (const std::exception& e) {
      context.setErrors(rows, std::current_exception());
      return;
    }

    bool mustRefSourceStrings = false;

    rows.applyToSelected([&](vector_size_t row) {
      resultWriter.setOffset(row);
      auto& arrayWriter = resultWriter.current();

      position = 1;
      if (args.size() == 4) {
        position = positionVec->valueAt<int64_t>(row);
      }
      VELOX_CHECK(!(position - 1 < 0));

      const StringView& currentOverwrite =
          overwriteVec->valueAt<StringView>(row);

      // For an unknown reason, when we do not create the StringPiece with
      // .getString() and instead reference the StringView, the line below
      // "strValue.substr" can overwrite the values found in overwrite. Cannot
      // explain this behavior and must default to getString
      std::vector<re2::StringPiece> overwrite = {
          re2::StringPiece(currentOverwrite.getString())};

      const StringView& current = input->valueAt<StringView>(row);
      std::string strValue = current.getString();

      // If position is not 1, adjust the string for regex operation.
      // Spark implementation default position is 1. It is 1-indexed instead of
      // 0-indexed.
      std::string prefix;
      if (position - 1 > 0) {
        prefix = strValue.substr(0, position - 1);
        if (position - 1 > strValue.length()) {
          strValue = "";
        } else {
          strValue = strValue.substr(position - 1);
        }
      }

      int replacements = RE2::GlobalReplace(&strValue, re_, overwrite[0]);

      strValue = prefix + strValue;
      const StringView& answer = StringView(strValue.c_str(), strValue.size());
      arrayWriter.add_item().append(answer);
      resultWriter.commit();
    });
    resultWriter.finish();

    // Ensure that the strings in the result vector share the same string
    // buffers as the input vector. This optimization minimizes the copying of
    // data between vectors.
    resultRef->as<ArrayVector>()
        ->elements()
        ->as<FlatVector<StringView>>()
        ->acquireSharedStringBuffers(args[0].get());
  }

 private:
  // We can store the search pattern because it is constant
  // no matter what row we are processing
  const std::string basePattern_;
};

/// @class Re2Replace
///
/// @brief Facilitates regex-based string replacements for a range of dynamic
/// patterns.
///
/// The Re2Replace class offers regex replacement functionalities for scenarios
/// where the regex pattern can vary across rows. This behavior contrasts with
/// the `Re2ReplaceConstantPattern` class, which operates on a fixed regex
/// pattern. This class is optimized for dynamic replacement patterns, decoding
/// them on-the-fly and performing the necessary replacements.
///
/// One noteworthy feature is that if a constant pattern is detected, it
/// delegates the replacement operation to the `Re2ReplaceConstantPattern` class
/// for more efficient processing. Furthermore, this class includes safety
/// checks for potential bad patterns and ensures compatibility before executing
/// any replacement.
///
/// The class also handles optional positional parameters, maintaining alignment
/// with Spark's 1-indexed default position. This informs the start point for
/// the regex replacement.
///
/// Memory efficiency is ensured by optimizing string buffer sharing, reducing
/// unnecessary data copying between vectors.
///
/// @note The naming "Re2Replace" denotes the class's reliance on the RE2
/// regular expression library for the underlying regex operations and its
/// ability to handle dynamic patterns.
class Re2Replace final : public exec::VectorFunction {
 public:
  explicit Re2Replace() {}

  // Performs regex replacements on given rows with support for variable
  // patterns.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK(args.size() == 3 || args.size() == 4);

    // Optimize for a constant regex pattern by delegating to specialized class.
    if (auto pattern = getIfConstant<StringView>(*args[1])) {
      Re2ReplaceConstantPattern(*pattern).apply(
          rows, args, outputType, context, resultRef);
      return;
    }

    exec::LocalDecodedVector input(context, *args[0], rows);

    BaseVector::ensureWritable(
        rows, ARRAY(VARCHAR()), context.pool(), resultRef);
    exec::VectorWriter<Array<Varchar>> resultWriter;
    resultWriter.init(*resultRef->as<ArrayVector>());

    exec::LocalDecodedVector patternVec(context, *args[1], rows);
    exec::LocalDecodedVector overwriteVec(context, *args[2], rows);

    exec::LocalDecodedVector positionVec(context);
    if (args.size() == 4) {
      positionVec = exec::LocalDecodedVector(context, *args[3], rows);
    }

    int64_t position;
    rows.applyToSelected([&](vector_size_t row) {
      position = 1;
      resultWriter.setOffset(row);
      auto& arrayWriter = resultWriter.current();

      const StringView& currentString = input->valueAt<StringView>(row);
      std::string strValue = currentString.getString();

      const StringView& currentPattern = patternVec->valueAt<StringView>(row);
      const re2::RE2 pattern(currentPattern.getString());

      try {
        checkForBadPattern(pattern);
        checkForCompatiblePattern(currentPattern.getString());
      } catch (const std::exception& e) {
        context.setErrors(rows, std::current_exception());
        return;
      }

      const StringView& currentOverwrite =
          overwriteVec->valueAt<StringView>(row);

      std::vector<re2::StringPiece> overwrite = {
          re2::StringPiece(currentOverwrite.getString())};
      std::string prefix;

      if (args.size() == 4) {
        position = positionVec->valueAt<int64_t>(row);
        VELOX_CHECK(!(position - 1 < 0));

        if (position - 1 > 0) {
          prefix = strValue.substr(0, position - 1);
          if (position - 1 > strValue.length()) {
            strValue = "";
          } else {
            strValue = strValue.substr(position - 1);
          }
        }
      }

      int replacements = RE2::GlobalReplace(&strValue, pattern, overwrite[0]);
      strValue = prefix + strValue;

      const StringView& answer = StringView(strValue.c_str(), strValue.size());

      arrayWriter.add_item().append(answer);
      resultWriter.commit();
    });

    resultWriter.finish();

    // Optimizes memory by sharing buffers.
    resultRef->as<ArrayVector>()
        ->elements()
        ->as<FlatVector<StringView>>()
        ->acquireSharedStringBuffers(args[0].get());
  }
};

} // namespace

// These functions delegate to the RE2-based implementations in
// common/RegexFunctions.h, but check to ensure that syntax that has different
// semantics between Spark (which uses java.util.regex) and RE2 throws an
// error.
std::shared_ptr<exec::VectorFunction> makeRLike(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  // Return any errors from re2Search() first.
  auto result = makeRe2Search(name, inputArgs, config);
  ensureRegexIsCompatible("RLIKE", inputArgs[1].constantValue);
  return result;
}

std::shared_ptr<exec::VectorFunction> makeRegexExtract(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  auto result = makeRe2Extract(name, inputArgs, config, /*emptyNoMatch=*/true);
  ensureRegexIsCompatible("REGEXP_EXTRACT", inputArgs[1].constantValue);
  return result;
}

std::shared_ptr<exec::VectorFunction> makeRegexReplace(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  auto numArgs = inputArgs.size();

  VELOX_USER_CHECK(
      numArgs == 3 || numArgs == 4,
      "{} requires between 2 and 4 arguments, but got {}",
      name,
      numArgs);

  VELOX_USER_CHECK(
      inputArgs[0].type->isVarchar(),
      "{} requires the first argument of type VARCHAR, but got {}",
      name,
      inputArgs[0].type->toString());

  VELOX_USER_CHECK(
      inputArgs[1].type->isVarchar(),
      "{} requires the second argument of type VARCHAR, but got {}",
      name,
      inputArgs[1].type->toString());

  BaseVector* constantPattern = inputArgs[1].constantValue.get();

  VELOX_USER_CHECK(
      inputArgs[2].type->isVarchar(),
      "{} requires the third argument of type VARCHAR, but got {}",
      name,
      inputArgs[1].type->toString());

  if (numArgs == 4) {
    VELOX_USER_CHECK(
        inputArgs[3].type->isInteger() || inputArgs[3].type->isBigint(),
        "{} requires the fourth argument of type INTEGER, but got {}",
        name,
        inputArgs[3].type->toString());
  }

  if (constantPattern != nullptr && !constantPattern->isNullAt(0)) {
    auto pattern =
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0);
    return std::make_shared<Re2ReplaceConstantPattern>(pattern);
  }

  return std::make_shared<Re2Replace>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>> re2ReplaceSignatures() {
  return {

      // regex_replace (str, pattern, replace)
      exec::FunctionSignatureBuilder()
          .returnType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .build(),
      // regex_replace (str, pattern, replace, position)
      exec::FunctionSignatureBuilder()
          .returnType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::VARCHAR))
          .argumentType(mapTypeKindToName(TypeKind::BIGINT))
          .build(),
  };
}

} // namespace facebook::velox::functions::sparksql
