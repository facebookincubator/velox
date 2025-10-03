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

#pragma once

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::cudf_velox {

class CudfFunctionBaseTest: public testing::Test,
                            public velox::test::VectorTestBase {

  protected:

  template <typename TReturn, typename... TArgs>
  std::optional<TReturn> evaluateOnce(
      const std::string& expr,
      std::optional<TArgs>... args) {
    return evaluateOnce<TReturn>(
        expr,
        makeRowVector(unpackEvaluateParams<std::optional<TArgs>...>(
            {}, std::forward<std::optional<TArgs>>(std::move(args))...)));
  }

  // Convenience version to allow API users to specify a single type for
  // functions that take a single parameter, instead of having to wrap them in a
  // initializer_list. For example, it allows users to do:
  //
  //  auto a = evaluateOnce<int64_t>("hour(c0)", DATE(), date);
  //
  // instead of:
  //
  //  auto a = evaluateOnce<int64_t>("hour(c0)", {DATE()}, date);
  //
  // Note that if types are specified, evaluateOnce() will throws if the number
  // of types and parameter do not match.
  template <typename TReturn, typename... TArgs>
  std::optional<TReturn> evaluateOnce(
      const std::string& expr,
      const TypePtr& type,
      std::optional<TArgs>... args) {
    return evaluateOnce<TReturn>(
        expr,
        makeRowVector(unpackEvaluateParams<std::optional<TArgs>...>(
            {type}, std::forward<std::optional<TArgs>>(std::move(args))...)));
  }

  template <typename TReturn>
  std::optional<TReturn> evaluateOnce(
      const std::string& expr,
      const RowVectorPtr rowVectorPtr,
      const TypePtr& resultType = nullptr) {
    auto result =
        evaluate<SimpleVector<facebook::velox::test::EvalType<TReturn>>>(
            expr, rowVectorPtr, resultType);
    return result->isNullAt(0) ? std::optional<TReturn>{}
                               : TReturn(result->valueAt(0));
  }

    /// Evaluate a given expression over a single row of input returning the
  /// result as a std::optional C++ value. Prefer to use the `evaluateOnce()`
  /// helper methods below when testing simple functions instead of manually
  /// handling input and output vectors.
  ///
  /// Arguments should be referenced using c0, c1, .. cn.  Supports integers,
  /// floats, booleans, strings, timestamps, and other primitive types.
  ///
  /// The template parameter type `TReturn` is mandatory and used to cast the
  /// returned value. Always use the C++ type for the returned value (e.g.
  /// `double` not `DOUBLE()`). Either StringView or std::string can be used for
  /// VARCHAR() and VARBINARY().
  ///
  /// If the function returns a custom type, use the physical type that
  /// represent the custom type (e.g. `CustomType<T>::type`). For example, use
  /// `int64_t` to return TIMESTAMP_WITH_TIME_ZONE()
  ///
  /// Input and output parameters should always be wrapped in an std::optional
  /// to allow the representation of null values.
  ///
  /// Example:
  ///
  ///   std::optional<double> exp(std::optional<double> a) {
  ///     return evaluateOnce<double>("exp(c0)", a);
  ///   }
  ///   EXPECT_EQ(1, exp(0));
  ///   EXPECT_EQ(std::nullopt, exp(std::nullopt));
  ///
  ///
  /// Multiple parameters are supported:
  ///
  ///   std::optional<int64_t> xxhash64(
  ///       std::optional<std::string> val,
  ///       std::optional<int64_t> seed) {
  ///     return evaluateOnce<int64_t>("fb_xxhash64(c0, c1)", val, seed);
  ///   }
  ///
  ///
  /// If not specified, input logical types are derived from the C++ type
  /// provided as input using CppToType. In the example above, the `std::string`
  /// C++ type will be translated to `VARCHAR` logical type, and `int64_t` to
  /// `BIGINT`.
  ///
  /// To override the input logical type, use:
  ///
  ///   std::optional<int64_t> hour(std::optional<int32_t> date) {
  ///     return evaluateOnce<int64_t>("hour(c0)", DATE(), date);
  ///   }
  ///
  /// Custom types like TIMESTAMP_WITH_TIMEZONE() and HYPERLOGLOG() are also
  /// supported. For multiple parameters, use a list of logical types:
  ///
  ///   return evaluateOnce<std::string>(
  ///       "hmac_sha1(c0, c1)", {VARBINARY(), VARBINARY()}, arg, key);
  ///
  ///
  /// One can also manually specify additional template parameters, in case
  /// there are problems with template type deduction:
  ///
  ///   return evaluateOnce<int64_t, std::string, int64_t>(
  ///       "fb_xxhash64(c0, c1)", val, seed);
  ///
  template <typename TReturn, typename... TArgs>
  std::optional<TReturn> evaluateOnce(
      const std::string& expr,
      const std::initializer_list<TypePtr>& types,
      std::optional<TArgs>... args) {
    return evaluateOnce<TReturn>(
        expr,
        makeRowVector(unpackEvaluateParams<std::optional<TArgs>...>(
            std::vector<TypePtr>{types},
            std::forward<std::optional<TArgs>>(std::move(args))...)));
  }

  template <typename T>
  std::shared_ptr<T> evaluate(
      const std::string& expression,
      const RowVectorPtr& data,
      const TypePtr& resultType = nullptr) {
    const auto plan = exec::test::PlanBuilder()
                        .setParseOptions(options_)
                        .values({data})
                        .project({expression})
                        .planNode();
    auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool());
    VELOX_CHECK_EQ(result->childrenSize(), 1);
    return castEvaluateResult<T>(result->childAt(0), expression, resultType);
  }

  /// Parses a timestamp string into Timestamp.
  /// Accepts strings formatted as 'YYYY-MM-DD HH:mm:ss[.nnn]'.
  static Timestamp parseTimestamp(const std::string& text) {
    return util::fromTimestampString(
               text.data(), text.size(), util::TimestampParseMode::kPrestoCast)
        .thenOrThrow(folly::identity, [&](const Status& status) {
          VELOX_USER_FAIL("{}", status.message());
        });
  }

  /// Parses a date string into days since epoch.
  /// Accepts strings formatted as 'YYYY-MM-DD'.
  static int32_t parseDate(const std::string& text) {
    return DATE()->toDays(text);
  }

  parse::ParseOptions options_;

  private:

  template <typename T>
  std::shared_ptr<T> castEvaluateResult(
      const VectorPtr& result,
      const std::string& expression,
      const TypePtr& expectedType = nullptr) {
    // reinterpret_cast is used for functions that return logical types
    // like DATE().
    VELOX_CHECK(result, "Expression evaluation result is null: {}", expression);
    if (expectedType != nullptr) {
      VELOX_CHECK_EQ(
          result->type()->kind(),
          expectedType->kind(),
          "Expression evaluation result is not of expected type kind: {} -> {} vector of type {}",
          expression,
          result->encoding(),
          result->type()->kindName());
      auto castedResult = std::reinterpret_pointer_cast<T>(result);
      return castedResult;
    }

    auto castedResult = std::dynamic_pointer_cast<T>(result);
    VELOX_CHECK(
        castedResult,
        "Expression evaluation result is not of expected type: {} -> {} vector of type {}",
        expression,
        result->encoding(),
        result->type()->toString());
    return castedResult;
  }

    // Unpack parameters for evaluateOnce(). Base recursion case.
  template <typename...>
  std::vector<VectorPtr> unpackEvaluateParams(
      const std::vector<TypePtr>& types) {
    VELOX_CHECK(types.empty(), "Wrong number of types passed to evaluateOnce.");
    return {};
  }

  // Recursively unpack input values and types into vectors.
  template <typename T, typename... TArgs>
  std::vector<VectorPtr> unpackEvaluateParams(
      const std::vector<TypePtr>& types,
      T&& value,
      TArgs&&... args) {
    // If there are no input types, let makeNullable figure it out.
    auto output = std::vector<VectorPtr>{
        types.empty()
            ? makeNullableFlatVector(std::vector{value})
            : makeNullableFlatVector(std::vector{value}, types.front()),
    };

    // Recurse starting from the second parameter.
    auto result = unpackEvaluateParams<TArgs...>(
        types.size() > 1 ? std::vector<TypePtr>{types.begin() + 1, types.end()}
                         : std::vector<TypePtr>{},
        std::forward<TArgs>(std::move(args))...);
    output.insert(output.end(), result.begin(), result.end());
    return output;
  }
};
  
}