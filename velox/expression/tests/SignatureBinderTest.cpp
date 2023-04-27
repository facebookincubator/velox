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
#include "velox/expression/SignatureBinder.h"
#include <gtest/gtest.h>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

using namespace facebook::velox;

void testSignatureBinder(
    const std::shared_ptr<exec::FunctionSignature>& signature,
    const std::vector<TypePtr>& actualTypes,
    const TypePtr& expectedReturnType) {
  exec::SignatureBinder binder(*signature, actualTypes);
  ASSERT_TRUE(binder.tryBind());

  auto returnType = binder.tryResolveReturnType();
  ASSERT_TRUE(returnType != nullptr);
  ASSERT_TRUE(expectedReturnType->equivalent(*returnType));
}

void assertCannotResolve(
    const std::shared_ptr<exec::FunctionSignature>& signature,
    const std::vector<TypePtr>& actualTypes) {
  exec::SignatureBinder binder(*signature, actualTypes);
  ASSERT_FALSE(binder.tryBind());
}

TEST(SignatureBinderTest, decimals) {
  // Decimal Add/Subtract.
  {
    auto signature =
        exec::FunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .integerVariable(
                "r_precision",
                "min(38, max(a_precision - a_scale, b_precision - b_scale) + max(a_scale, b_scale) + 1)")
            .integerVariable("r_scale", "max(a_scale, b_scale)")
            .returnType("decimal(r_precision, r_scale)")
            .argumentType("decimal(a_precision, a_scale)")
            .argumentType("decimal(b_precision, b_scale)")
            .build();
    ASSERT_EQ(
        signature->argumentTypes()[0].toString(),
        "decimal(a_precision,a_scale)");
    ASSERT_EQ(
        signature->argumentTypes()[1].toString(),
        "decimal(b_precision,b_scale)");
    testSignatureBinder(
        signature, {DECIMAL(11, 5), DECIMAL(10, 6)}, DECIMAL(13, 6));
  }

  // Decimal Multiply.
  {
    auto signature =
        exec::FunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .integerVariable(
                "r_precision", "min(38, a_precision + b_precision)")
            .integerVariable("r_scale", "a_scale + b_scale")
            .returnType("DECIMAL(r_precision, r_scale)")
            .argumentType("decimal(a_precision, a_scale)")
            .argumentType("decimal(b_precision, b_scale)")
            .build();

    testSignatureBinder(
        signature, {DECIMAL(11, 5), DECIMAL(10, 6)}, LONG_DECIMAL(21, 11));
  }

  // Decimal Divide.
  {
    auto signature =
        exec::FunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .integerVariable(
                "r_precision",
                "min(38, a_precision + b_scale + max(b_scale - a_scale, 0))")
            .integerVariable("r_scale", "max(a_scale, b_scale)")
            .returnType("DECIMAL(r_precision, r_scale)")
            .argumentType("DECIMAL(a_precision, a_scale)")
            .argumentType("DECIMAL(b_precision, b_scale)")
            .build();

    testSignatureBinder(
        signature, {DECIMAL(11, 5), DECIMAL(10, 6)}, SHORT_DECIMAL(18, 6));
  }

  // Decimal Modulus.
  {
    auto signature =
        exec::FunctionSignatureBuilder()
            .integerVariable("a_precision")
            .integerVariable("a_scale")
            .integerVariable("b_precision")
            .integerVariable("b_scale")
            .integerVariable(
                "r_precision",
                "min(b_precision - b_scale, a_precision - a_scale) + max(a_scale, b_scale)")
            .integerVariable("r_scale", "max(a_scale, b_scale)")
            .returnType("DECIMAL(r_precision, r_scale)")
            .argumentType("DECIMAL(a_precision, a_scale)")
            .argumentType("DECIMAL(b_precision, b_scale)")
            .build();

    testSignatureBinder(
        signature,
        {SHORT_DECIMAL(11, 5), SHORT_DECIMAL(10, 6)},
        DECIMAL(10, 6));
  }
  // Aggregate Sum.
  {
    auto signature = exec::AggregateFunctionSignatureBuilder()
                         .integerVariable("a_precision")
                         .integerVariable("a_scale")
                         .argumentType("DECIMAL(a_precision, a_scale)")
                         .intermediateType("DECIMAL(38, a_scale)")
                         .returnType("DECIMAL(38, a_scale)")
                         .build();

    const std::vector<TypePtr> actualTypes{DECIMAL(10, 4)};
    exec::SignatureBinder binder(*signature, actualTypes);
    ASSERT_TRUE(binder.tryBind());

    auto intermediateType =
        binder.tryResolveType(signature->intermediateType());
    ASSERT_TRUE(intermediateType != nullptr);
    ASSERT_TRUE(DECIMAL(38, 4)->equivalent(*intermediateType));
    auto returnType = binder.tryResolveReturnType();
    ASSERT_TRUE(returnType != nullptr);
    ASSERT_TRUE(DECIMAL(38, 4)->equivalent(*returnType));
  }
  // missing constraint returns nullptr.
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .integerVariable("a_precision")
                         .integerVariable("a_scale")
                         .integerVariable("b_precision")
                         .integerVariable("b_scale")
                         .integerVariable("r_precision")
                         .integerVariable("r_scale")
                         .returnType("decimal(r_precision, r_scale)")
                         .argumentType("decimal(a_precision, a_scale)")
                         .argumentType("DECIMAL(b_precision, b_scale)")
                         .build();
    const std::vector<TypePtr> argTypes{DECIMAL(11, 5), DECIMAL(10, 6)};
    exec::SignatureBinder binder(*signature, argTypes);
    ASSERT_TRUE(binder.tryBind());
    ASSERT_TRUE(binder.tryResolveReturnType() == nullptr);
  }
  // Scalar function signature, same precision and scale.
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .integerVariable("a_precision")
                         .integerVariable("a_scale")
                         .returnType("boolean")
                         .argumentType("DECIMAL(a_precision, a_scale)")
                         .argumentType("DECIMAL(a_precision, a_scale)")
                         .build();
    {
      const std::vector<TypePtr> argTypes{DECIMAL(11, 5), DECIMAL(11, 5)};
      exec::SignatureBinder binder(*signature, argTypes);
      ASSERT_TRUE(binder.tryBind());
      auto returnType = binder.tryResolveReturnType();
      ASSERT_TRUE(returnType != nullptr);
      ASSERT_TRUE(BOOLEAN()->equivalent(*returnType));
    }

    {
      const std::vector<TypePtr> argTypes{DECIMAL(28, 5), DECIMAL(28, 5)};
      exec::SignatureBinder binder(*signature, argTypes);
      ASSERT_TRUE(binder.tryBind());
      auto returnType = binder.tryResolveReturnType();
      ASSERT_TRUE(returnType != nullptr);
      ASSERT_TRUE(BOOLEAN()->equivalent(*returnType));
    }

    // Decimal scalar function signature with precision/scale mismatch.
    {
      const std::vector<TypePtr> argTypes{DECIMAL(28, 5), DECIMAL(29, 5)};
      exec::SignatureBinder binder(*signature, argTypes);
      ASSERT_FALSE(binder.tryBind());

      const std::vector<TypePtr> argTypes1{DECIMAL(28, 7), DECIMAL(28, 5)};
      exec::SignatureBinder binder1(*signature, argTypes1);
      ASSERT_FALSE(binder1.tryBind());
    }

    // Resolving invalid ShortDecimal/LongDecimal arguments returns nullptr.
    {
      // Missing constraints.
      const auto typeSignature = signature->argumentTypes()[0];
      ASSERT_EQ(
          exec::SignatureBinder::tryResolveType(typeSignature, {}, {}),
          nullptr);
      ASSERT_EQ(
          exec::SignatureBinder::tryResolveType(
              signature->argumentTypes()[0], {}, {}),
          nullptr);
      // Missing parameters.
      ASSERT_EQ(
          exec::SignatureBinder::tryResolveType(
              exec::TypeSignature("DECIMAL", {}), {}, {}),
          nullptr);
      // Missing constraint value.
      std::unordered_map<std::string, int> integerVariables;
      ASSERT_EQ(
          exec::SignatureBinder::tryResolveType(
              typeSignature, {}, {}, integerVariables),
          nullptr);
    }
    // Type parameter + constraint = error.
    {
      VELOX_ASSERT_THROW(
          exec::SignatureVariable(
              "TypeName", "a = b", exec::ParameterType::kTypeParameter),
          "Type variables cannot have constraints");
    }
  }
}

TEST(SignatureBinderTest, knownOnly) {
  // map(K,V) -> array(K)

  auto signature = exec::FunctionSignatureBuilder()
                       .knownTypeVariable("K")
                       .typeVariable("V")
                       .argumentType("map(K,V)")
                       .returnType("array(K)")
                       .build();
  {
    auto actualTypes = std::vector<TypePtr>{MAP(UNKNOWN(), UNKNOWN())};
    exec::SignatureBinder binder(*signature, actualTypes);
    ASSERT_FALSE(binder.tryBind());
  }

  {
    auto actualTypes = std::vector<TypePtr>{MAP(INTEGER(), UNKNOWN())};
    exec::SignatureBinder binder(*signature, actualTypes);
    ASSERT_TRUE(binder.tryBind());
  }
}

TEST(SignatureBinderTest, generics) {
  // array(T), T -> boolean
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .typeVariable("T")
                         .returnType("boolean")
                         .argumentType("array(T)")
                         .argumentType("T")
                         .build();

    testSignatureBinder(signature, {ARRAY(BIGINT()), BIGINT()}, BOOLEAN());
    testSignatureBinder(
        signature, {ARRAY(DECIMAL(20, 3)), DECIMAL(20, 3)}, BOOLEAN());
    assertCannotResolve(signature, {ARRAY(DECIMAL(20, 3)), DECIMAL(20, 4)});
  }

  // array(array(T)), array(T) -> boolean
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .typeVariable("T")
                         .returnType("boolean")
                         .argumentType("array(array(T))")
                         .argumentType("array(T)")
                         .build();

    testSignatureBinder(
        signature, {ARRAY(ARRAY(BIGINT())), ARRAY(BIGINT())}, BOOLEAN());
  }

  // map(K,V) -> array(K)
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .knownTypeVariable("K")
                         .typeVariable("V")
                         .returnType("array(K)")
                         .argumentType("map(K,V)")
                         .build();

    testSignatureBinder(signature, {MAP(BIGINT(), DOUBLE())}, ARRAY(BIGINT()));
  }

  // map(K,V) -> array(V)
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .knownTypeVariable("K")
                         .typeVariable("V")
                         .returnType("array(V)")
                         .argumentType("map(K,V)")
                         .build();

    testSignatureBinder(signature, {MAP(BIGINT(), DOUBLE())}, ARRAY(DOUBLE()));
  }
}

TEST(SignatureBinderTest, variableArity) {
  // varchar... -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("varchar")
                         .variableArity()
                         .build();

    testSignatureBinder(signature, {}, VARCHAR());
    testSignatureBinder(signature, {VARCHAR()}, VARCHAR());
    testSignatureBinder(signature, {VARCHAR(), VARCHAR()}, VARCHAR());
    testSignatureBinder(
        signature, {VARCHAR(), VARCHAR(), VARCHAR()}, VARCHAR());
  }

  // integer, double... -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("integer")
                         .argumentType("double")
                         .variableArity()
                         .build();

    testSignatureBinder(signature, {INTEGER()}, VARCHAR());
    testSignatureBinder(signature, {INTEGER(), DOUBLE()}, VARCHAR());
    testSignatureBinder(signature, {INTEGER(), DOUBLE(), DOUBLE()}, VARCHAR());
    testSignatureBinder(
        signature, {INTEGER(), DOUBLE(), DOUBLE(), DOUBLE()}, VARCHAR());
  }

  // any... -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("any")
                         .variableArity()
                         .build();

    testSignatureBinder(signature, {}, VARCHAR());
    testSignatureBinder(signature, {INTEGER()}, VARCHAR());
    testSignatureBinder(signature, {INTEGER(), DOUBLE()}, VARCHAR());
    testSignatureBinder(
        signature, {TIMESTAMP(), VARCHAR(), SMALLINT()}, VARCHAR());
  }

  // integer, any... -> timestamp
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("timestamp")
                         .argumentType("integer")
                         .argumentType("any")
                         .variableArity()
                         .build();

    testSignatureBinder(signature, {INTEGER()}, TIMESTAMP());
    testSignatureBinder(signature, {INTEGER(), DOUBLE()}, TIMESTAMP());
    testSignatureBinder(
        signature,
        {INTEGER(), TIMESTAMP(), VARCHAR(), SMALLINT()},
        TIMESTAMP());
  }
}

TEST(SignatureBinderTest, unresolvable) {
  // integer -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("integer")
                         .build();

    // wrong type
    assertCannotResolve(signature, {BIGINT()});

    // wrong number of arguments
    assertCannotResolve(signature, {});
    assertCannotResolve(signature, {INTEGER(), DOUBLE()});
  }

  // integer, double -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("integer")
                         .argumentType("double")
                         .build();

    // wrong type
    assertCannotResolve(signature, {BIGINT()});
    assertCannotResolve(signature, {INTEGER(), INTEGER()});
    assertCannotResolve(signature, {INTEGER(), DOUBLE(), INTEGER()});

    // wrong number of arguments
    assertCannotResolve(signature, {});
  }

  // integer... -> varchar
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("varchar")
                         .argumentType("integer")
                         .variableArity()
                         .build();

    // wrong type
    assertCannotResolve(signature, {BIGINT()});
    assertCannotResolve(signature, {INTEGER(), BIGINT()});
  }

  // row(bigint, varchar) -> bigint
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("bigint")
                         .argumentType("row(bigint, varchar)")
                         .build();

    testSignatureBinder(signature, {ROW({BIGINT(), VARCHAR()})}, BIGINT());

    // wrong type
    assertCannotResolve(signature, {ROW({BIGINT()})});
    assertCannotResolve(signature, {ROW({BIGINT(), VARCHAR(), BOOLEAN()})});
  }

  // array(row(boolean)) -> bigint
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("bigint")
                         .argumentType("array(row(boolean))")
                         .build();

    testSignatureBinder(signature, {ARRAY(ROW({BOOLEAN()}))}, BIGINT());

    // wrong type
    assertCannotResolve(signature, {ARRAY(ROW({BOOLEAN(), BIGINT()}))});
    assertCannotResolve(signature, {ARRAY(ROW({BIGINT()}))});
  }
}

TEST(SignatureBinderTest, tryResolveTypeNullOutput) {
  auto assertNullResult = [&](const std::string& argument) {
    ASSERT_EQ(
        exec::SignatureBinder::tryResolveType(
            exec::parseTypeSignature(argument), {}, {}),
        nullptr);
  };

  assertNullResult("T");
  assertNullResult("any");
  assertNullResult("array(T)");
  assertNullResult("array(any)");
  assertNullResult("map(int, T)");
  assertNullResult("row(int, T)");
}

TEST(SignatureBinderTest, lambda) {
  auto signature = exec::FunctionSignatureBuilder()
                       .typeVariable("T")
                       .typeVariable("S")
                       .typeVariable("R")
                       .returnType("R")
                       .argumentType("array(T)")
                       .argumentType("S")
                       .argumentType("function(S,T,S)")
                       .argumentType("function(S,R)")
                       .build();

  std::vector<TypePtr> inputTypes{ARRAY(BIGINT()), DOUBLE(), nullptr, nullptr};
  exec::SignatureBinder binder(*signature, inputTypes);
  ASSERT_FALSE(binder.tryBind());

  // Resolve inputs for function(S,T,S). We resolve first 2 arguments, since
  // last argument is the return type that requires the signature of the lambda
  // itself for resolution.
  {
    auto lambdaType = signature->argumentTypes()[2];
    ASSERT_EQ(binder.tryResolveType(lambdaType.parameters()[0]), DOUBLE());
    ASSERT_EQ(binder.tryResolveType(lambdaType.parameters()[1]), BIGINT());
  }

  // Resolve inputs for function(S,R).
  {
    auto lambdaType = signature->argumentTypes()[3];
    ASSERT_EQ(binder.tryResolveType(lambdaType.parameters()[0]), DOUBLE());
  }
}

TEST(SignatureBinderTest, logicalType) {
  // Logical type as an argument type.
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("bigint")
                         .argumentType("interval day to second")
                         .build();

    testSignatureBinder(signature, {INTERVAL_DAY_TIME()}, BIGINT());
  }

  // Logical type as an return type.
  {
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("interval day to second")
                         .argumentType("bigint")
                         .build();

    testSignatureBinder(signature, {BIGINT()}, INTERVAL_DAY_TIME());
  }
}

TEST(SignatureBinderTest, customType) {
  registerCustomType(
      "timestamp with time zone",
      std::make_unique<const TimestampWithTimeZoneTypeFactories>());

  // Custom type as an argument type.
  {
    // timestamp with time zone -> bigint
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("bigint")
                         .argumentType("timestamp with time zone")
                         .build();

    testSignatureBinder(signature, {TIMESTAMP_WITH_TIME_ZONE()}, BIGINT());
  }

  {
    // timestamp with time zone -> bigint
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("array(integer)")
                         .argumentType("timestamp with time zone")
                         .argumentType("varchar")
                         .build();

    testSignatureBinder(
        signature, {TIMESTAMP_WITH_TIME_ZONE(), VARCHAR()}, ARRAY(INTEGER()));
  }

  // Custom type as a return type.
  {
    // timestamp with time zone -> bigint
    auto signature = exec::FunctionSignatureBuilder()
                         .returnType("timestamp with time zone")
                         .argumentType("integer")
                         .build();

    testSignatureBinder(signature, {INTEGER()}, TIMESTAMP_WITH_TIME_ZONE());
  }

  // Unknown custom type.
  VELOX_ASSERT_THROW(
      exec::FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("fancy_type")
          .build(),
      "Type doesn't exist: FANCY_TYPE");
}
