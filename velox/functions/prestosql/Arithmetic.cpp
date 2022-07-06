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
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/functions/prestosql/CheckedArithmeticImpl.h"

namespace facebook::velox::functions {
namespace {

template <typename T, typename Operation>
class BaseFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    if (args[0]->isConstantEncoding() &&
        args[1]->encoding() == VectorEncoding::Simple::FLAT) {
      // Fast path for (flat, const).
      auto constant = args[0]->asUnchecked<SimpleVector<T>>()->valueAt(0);
      auto flatValues = args[1]->asUnchecked<FlatVector<T>>();
      auto rawValues = flatValues->mutableRawValues();

      auto rawResults =
          getRawResults(rows, args[1], rawValues, resultType, context, result);

      context->applyToSelectedNoThrow(rows, [&](auto row) {
        rawResults[row] = Operation::apply(constant, rawValues[row]);
      });
    } else if (
        args[0]->encoding() == VectorEncoding::Simple::FLAT &&
        args[1]->isConstantEncoding()) {
      // Fast path for (const, flat).
      auto constant = args[1]->asUnchecked<SimpleVector<T>>()->valueAt(0);
      auto flatValues = args[0]->asUnchecked<FlatVector<T>>();
      auto rawValues = flatValues->mutableRawValues();

      auto rawResults =
          getRawResults(rows, args[0], rawValues, resultType, context, result);

      context->applyToSelectedNoThrow(rows, [&](auto row) {
        rawResults[row] = Operation::apply(rawValues[row], constant);
      });
    } else {
      exec::DecodedArgs decodedArgs(rows, args, context);

      auto a = decodedArgs.at(0);
      auto b = decodedArgs.at(1);

      BaseVector::ensureWritable(rows, resultType, context->pool(), result);
      auto rawResults =
          (*result)->asUnchecked<FlatVector<T>>()->mutableRawValues();

      context->applyToSelectedNoThrow(rows, [&](auto row) {
        rawResults[row] =
            Operation::apply(a->valueAt<T>(row), b->valueAt<T>(row));
      });
    }
  }

 private:
  T* getRawResults(
      const SelectivityVector& rows,
      VectorPtr& flat,
      T* rawValues,
      const TypePtr& resultType,
      exec::EvalCtx* context,
      VectorPtr* result) const {
    // Check if input can be reused for results.
    T* rawResults;
    if (!(*result) && BaseVector::isReusableFlatVector(flat)) {
      rawResults = rawValues;
      *result = std::move(flat);
    } else {
      BaseVector::ensureWritable(rows, resultType, context->pool(), result);
      rawResults = (*result)->asUnchecked<FlatVector<T>>()->mutableRawValues();
    }

    return rawResults;
  }
};

class Addition {
 public:
  template <typename T>
  static T apply(T left, T right) {
    return plus(left, right);
  }
};

class CheckedAddition {
 public:
  template <
      typename T,
      typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return plus(left, right);
  }

  template <
      typename T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return checkedPlus(left, right);
  }
};

class Subtraction {
 public:
  template <typename T>
  static T apply(T left, T right) {
    return minus(left, right);
  }
};

class CheckedSubtraction {
 public:
  template <
      typename T,
      typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return minus(left, right);
  }

  template <
      typename T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return checkedMinus(left, right);
  }
};

class Multiplication {
 public:
  template <typename T>
  static T apply(T left, T right) {
    return multiply(left, right);
  }
};

class CheckedMultiplication {
 public:
  template <
      typename T,
      typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return multiply(left, right);
  }

  template <
      typename T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return checkedMultiply(left, right);
  }
};

class Division {
 public:
  template <typename T>
  static T apply(T left, T right) {
    return divide(left, right);
  }
};

class CheckedDivision {
 public:
  template <
      typename T,
      typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return divide(left, right);
  }

  template <
      typename T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  static T apply(T left, T right) {
    return checkedDivide(left, right);
  }
};

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  for (auto type :
       {"tinyint", "smallint", "integer", "bigint", "real", "double"}) {
    signatures.push_back(exec::FunctionSignatureBuilder()
                             .returnType(type)
                             .argumentType(type)
                             .argumentType(type)
                             .build());
  }
  return signatures;
}

template <typename Operation>
std::shared_ptr<exec::VectorFunction> make(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  auto typeKind = inputArgs[0].type->kind();
  switch (typeKind) {
    case TypeKind::TINYINT:
      return std::make_shared<BaseFunction<int8_t, Operation>>();
    case TypeKind::SMALLINT:
      return std::make_shared<BaseFunction<int16_t, Operation>>();
    case TypeKind::INTEGER:
      return std::make_shared<BaseFunction<int32_t, Operation>>();
    case TypeKind::BIGINT:
      return std::make_shared<BaseFunction<int64_t, Operation>>();
    case TypeKind::REAL:
      return std::make_shared<BaseFunction<float, Operation>>();
    case TypeKind::DOUBLE:
      return std::make_shared<BaseFunction<double, Operation>>();
    default:
      VELOX_UNREACHABLE();
  }
}
} // namespace

void registerPlus(const std::string& name) {
  exec::registerStatefulVectorFunction(name, signatures(), make<Addition>);
}

void registerCheckedPlus(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, signatures(), make<CheckedAddition>);
}

void registerMinus(const std::string& name) {
  exec::registerStatefulVectorFunction(name, signatures(), make<Subtraction>);
}

void registerCheckedMinus(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, signatures(), make<CheckedSubtraction>);
}

void registerMultiply(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, signatures(), make<Multiplication>);
}

void registerCheckedMultiply(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, signatures(), make<CheckedMultiplication>);
}

void registerDivide(const std::string& name) {
  exec::registerStatefulVectorFunction(name, signatures(), make<Division>);
}

void registerCheckedDivide(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, signatures(), make<CheckedDivision>);
}

} // namespace facebook::velox::functions
